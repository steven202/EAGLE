import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1

# Import RL optimizations
try:
    from .rl_optimizations import RLInferenceOptimizer, FastTokenizer, PerformanceMonitor
    from .advanced_rl_optimizations import (
        AdaptiveParameterPredictor, 
        ContextAwareCache, 
        SmartFrequencyAdjuster,
        ProfiledRLOptimizer
    )
    RL_OPTIMIZATIONS_AVAILABLE = True
    ADVANCED_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    RL_OPTIMIZATIONS_AVAILABLE = False
    ADVANCED_OPTIMIZATIONS_AVAILABLE = False
    print("RL optimizations not available - using standard implementation")
from .configs import EConfig


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        
        # Initialize RL optimizations if available
        if RL_OPTIMIZATIONS_AVAILABLE:
            # Enhanced optimization settings with progressive caching for adaptive hit rates
            self.rl_optimizer = RLInferenceOptimizer(
                cache_size=800,  # Much larger cache for better hit rates
                decode_frequency=8,  # Higher frequency to rely more on cache
                enable_token_caching=True,
                enable_parameter_caching=True,  # Cache parameter predictions
                enable_async_decoding=False,  # Keep simple for stability
                adaptive_frequency=True,  # Dynamically adjust based on performance
                target_cache_hit_rate=0.7,  # Start at 70% cache hit rate
                progressive_hit_rate=True  # Gradually increase to 100% over time
            )
            self.fast_tokenizer = FastTokenizer(self.tokenizer, cache_size=1500)  # Larger tokenizer cache
            self.perf_monitor = PerformanceMonitor(window_size=100)  # Enhanced monitoring
            
            # Advanced optimizations
            if ADVANCED_OPTIMIZATIONS_AVAILABLE:
                self.parameter_predictor = AdaptiveParameterPredictor(history_size=100)
                self.context_cache = ContextAwareCache(max_size=150, similarity_threshold=0.85)
                self.frequency_adjuster = SmartFrequencyAdjuster()
                self.profiler = ProfiledRLOptimizer()
            else:
                self.parameter_predictor = None
                self.context_cache = None
                self.frequency_adjuster = None
                self.profiler = None
        else:
            self.rl_optimizer = None
            self.fast_tokenizer = None
            self.perf_monitor = None
            self.parameter_predictor = None
            self.context_cache = None
            self.frequency_adjuster = None
            self.profiler = None
        
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            total_tokens=None,
            depth=None,
            tree_top_k=None,
            # New step-wise RL parameters
            rl_policy=None,
            training_mode=False,
            step_rewards_callback=None,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        # Store original parameters for restoration later
        original_total_tokens = self.ea_layer.total_tokens
        original_depth = self.ea_layer.depth
        original_top_k = self.ea_layer.top_k
        
        # For step-wise RL: track step-level metrics
        step_rewards = []
        step_actions = []
        step_states = []
        use_stepwise_rl = rl_policy is not None
        
        # Temporarily update parameters if provided (fallback values for non-RL mode)
        if total_tokens is not None:
            self.ea_layer.total_tokens = total_tokens - 1  # Adjust as in line 163
        if depth is not None:
            self.ea_layer.depth = depth
        if tree_top_k is not None:
            self.ea_layer.top_k = tree_top_k


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        # For step-wise RL: get initial state for first prediction
        if use_stepwise_rl:
            # Always decode to text since RL policy expects text input
            # Use optimized decoding when available
            if self.fast_tokenizer:
                initial_text = self.fast_tokenizer.decode_cached(input_ids[0], skip_special_tokens=True)
            else:
                initial_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            initial_state = initial_text
            
            # Always use torch.no_grad() for RL policy inference (training handled internally)
            with torch.no_grad():
                step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                    initial_state, training_mode=training_mode
                )
            
            # Immediately store this first action in similarity cache if optimized RL is available
            if self.rl_optimizer and not training_mode:
                self.rl_optimizer._store_action_for_similarity_aggressive(
                    initial_state, (step_total_tokens, step_depth, step_top_k), 0
                )
                print(f"  🌟 Stored initial action in similarity cache: {len(self.rl_optimizer.similarity_action_cache)} entries")
            
            # Store step info for training
            if training_mode:
                step_states.append(initial_state)
                step_actions.append((step_total_tokens, step_depth, step_top_k))
            
            # print(f"Step 0 RL params: tt={step_total_tokens}, d={step_depth}, k={step_top_k}")
        else:
            # Use provided or default parameters
            step_total_tokens = total_tokens
            step_depth = depth
            step_top_k = tree_top_k
        
        # prefill - model inference, disable gradients for efficiency
        with torch.no_grad():
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor, step_total_tokens, step_depth, step_top_k
            )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        
        # Initialize RL optimization if available
        use_optimized_rl = use_stepwise_rl and self.rl_optimizer is not None
        # print(f"🔍 DEBUG: use_stepwise_rl={use_stepwise_rl}, rl_optimizer={self.rl_optimizer is not None}, use_optimized_rl={use_optimized_rl}")
        if use_optimized_rl:
            self.rl_optimizer.clear_cache()  # Start fresh
            if self.perf_monitor:
                self.perf_monitor = PerformanceMonitor(window_size=100)  # Reset monitor
            if self.fast_tokenizer:
                self.fast_tokenizer.clear_cache()  # Clear tokenizer cache
                
            # Enhanced pre-warming caches with initial state to improve hit rates
            if initial_text: # and not training_mode:
                # Create more cache entries based on initial context
                cache_key_base = self.rl_optimizer._get_content_hash(input_ids[0])
                
                # Pre-populate with common parameter combinations (more variety)
                common_params = [
                    (64, 7, 8), (96, 7, 20), (64, 8, 32),
                    (32, 7, 12), (96, 6, 12), (80, 5, 20),
                    (64, 4, 20), (48, 7, 32), (128, 7, 16),
                    (64, 5, 12), (32, 3, 32), (128, 6, 12),
                    (128, 6, 32)
                ]
                for i, params in enumerate(common_params):
                    if len(self.rl_optimizer.parameter_cache) < 100:  # More aggressive prewarming
                        warmup_keys = [
                            f"{cache_key_base[:8]}_warmup_{i}",
                            f"{cache_key_base[:6]}_common_{i}",
                            f"{cache_key_base[:4]}_fuzzy_{i}"
                        ]
                        for warmup_key in warmup_keys:
                            if len(self.rl_optimizer.parameter_cache) < 100:
                                self.rl_optimizer.parameter_cache[warmup_key] = params
                
                # Also pre-warm state cache with variations
                for i in range(5):
                    if len(self.rl_optimizer.state_cache) < 50:
                        text_variants = [
                            initial_text,
                            initial_text + " ",  # Common variations
                            initial_text.strip(),
                        ]
                        for j, text_var in enumerate(text_variants):
                            warmup_state_key = f"{cache_key_base[:6]}_state_{i}_{j}"
                            if len(self.rl_optimizer.state_cache) < 50:
                                self.rl_optimizer.state_cache[warmup_state_key] = text_var
                
                # Aggressively pre-warm similarity action cache with initial examples
                if not training_mode:
                    # Pre-populate similarity cache with common action patterns for initial text
                    sample_actions = [
                        (64, 7, 10), (96, 6, 15), (48, 8, 20), (80, 7, 12),
                        (32, 5, 8), (128, 6, 25), (56, 7, 18), (72, 8, 14)
                    ]
                    
                    # Create variations of initial text and associate with different actions
                    initial_variants = [
                        initial_text,
                        initial_text + " The",
                        initial_text + " This",
                        initial_text.replace(" ", "_"),  # Different format
                        initial_text[:len(initial_text)//2],  # Prefix
                        initial_text[len(initial_text)//3:],   # Suffix
                    ]
                    
                    for i, (text_var, action) in enumerate(zip(initial_variants, sample_actions)):
                        if len(text_var) > 10:  # Only use meaningful text
                            # Use aggressive storage to populate multiple cache entries
                            self.rl_optimizer._store_action_for_similarity_aggressive(
                                text_var, action, i
                            )
                    
                    # print(f"  🔥 Pre-warmed similarity cache with {len(self.rl_optimizer.similarity_action_cache)} action patterns")
                
        # Performance tracking variables
        optimization_enabled = use_optimized_rl
        cumulative_decode_time = 0.0
        cumulative_policy_time = 0.0
        
        for idx in range(max_length):
            # For step-wise RL: predict parameters at each step (except first which was done above)
            step_start_time = time.time()
            
            if use_stepwise_rl and idx > 0:  # Skip first iteration since we already predicted
                if use_optimized_rl:
                    # print(f"  🔍 DEBUG: Using optimized RL path (use_optimized_rl=True)")
                    # Advanced optimized RL inference with multi-level caching and prediction
                    decode_start = time.time()
                    
                    # ALWAYS try RL optimizer first to enable similarity bypass
                    # print(f"  🔍 DEBUG: Using rl_optimizer.predict_parameters_optimized() (similarity bypass enabled)")
                    step_total_tokens, step_depth, step_top_k = self.rl_optimizer.predict_parameters_optimized(
                        input_ids, self.tokenizer, rl_policy, idx, training_mode
                    )
                    decode_time = time.time() - decode_start
                    
                    # Add to parameter predictor for future use (after RL optimization)
                    if self.parameter_predictor:
                        context_text = self.rl_optimizer.get_optimized_state(
                            input_ids, self.tokenizer, idx, training_mode, rl_policy
                        )
                        self.parameter_predictor.add_observation(
                            context_text, (step_total_tokens, step_depth, step_top_k)
                        )
                    
                    cumulative_decode_time += decode_time
                    cumulative_policy_time += decode_time
                    
                    # Enhanced performance monitoring
                    if self.perf_monitor:
                        self.perf_monitor.log_decode_time(decode_time)
                        self.perf_monitor.log_policy_time(decode_time)
                    
                    # Profile this step for auto-tuning
                    if self.profiler:
                        self.profiler.profile_step({
                            'decode_times': decode_time,
                            'policy_times': decode_time,
                            'step_counts': idx
                        })
                    
                    # Store step info for training
                    if training_mode:
                        current_state = self.rl_optimizer.get_optimized_state(
                            input_ids, self.tokenizer, idx, training_mode, rl_policy
                        )
                        step_states.append(current_state)
                        step_actions.append((step_total_tokens, step_depth, step_top_k))
                    
                else:
                    # print(f"  🔍 DEBUG: Using fallback implementation (use_optimized_rl=False)")
                    # Fallback implementation with basic optimizations
                    decode_start = time.time()
                    
                    # Use context-aware cache if available
                    current_text = None
                    if self.context_cache and not training_mode:
                        # Try to get from context cache first
                        temp_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        cached_result = self.context_cache.get(temp_text)
                        if cached_result:
                            step_total_tokens, step_depth, step_top_k = cached_result
                            current_text = temp_text
                            decode_time = time.time() - decode_start
                            print(f"  💾 Using context cache at step {idx}")
                    
                    if current_text is None:
                        # Normal path with fast tokenizer
                        if self.fast_tokenizer:
                            current_text = self.fast_tokenizer.decode_cached(input_ids[0], skip_special_tokens=True)
                        else:
                            current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        
                        current_state = current_text
                        
                        # Predict parameters
                        policy_start = time.time()
                        with torch.no_grad():
                            step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                                current_state, training_mode=training_mode
                            )
                        policy_time = time.time() - policy_start
                        decode_time = time.time() - decode_start
                        
                        # Cache the result
                        if self.context_cache and not training_mode:
                            self.context_cache.put(current_text, (step_total_tokens, step_depth, step_top_k))
                    
                    cumulative_decode_time += decode_time
                    cumulative_policy_time += policy_time if 'policy_time' in locals() else decode_time
                    
                    # Store step info for training
                    if training_mode:
                        step_states.append(current_text)
                        step_actions.append((step_total_tokens, step_depth, step_top_k))
                
                # Reduced logging frequency for performance
                if len(step_rewards) % 30 == 0:
                    print(f"  Step {idx} RL params: tt={step_total_tokens}, d={step_depth}, k={step_top_k}")
                
                # Batch parameter updates to reduce overhead
                if step_total_tokens is not None:
                    self.ea_layer.total_tokens = step_total_tokens - 1
                if step_depth is not None:
                    self.ea_layer.depth = step_depth  
                if step_top_k is not None:
                    self.ea_layer.top_k = step_top_k
            
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            # Model inference parts: disable gradients for efficiency
            with torch.no_grad():
                draft_tokens = draft_tokens.to(input_ids.device)
                # Target model forward, get logits
                logits, hidden_state_new, outputs = tree_decoding(
                    self,
                    draft_tokens,
                    past_key_values,
                    tree_position_ids,
                    input_ids,
                    retrieve_indices,
                )
                # retrieve_indices=tree_buffers["retrieve_indices"]
                # logits = logits[0, retrieve_indices]
                draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                candidates = draft_tokens[0, retrieve_indices]
                # verification
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
                # print(accept_length)
            
            # Calculate step reward for RL training
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_tokens_generated = accept_length + 1  # Number of tokens accepted in this step
            
            if use_stepwise_rl and step_time > 0 and step_tokens_generated > 0:
                # Calculate reward: tokens per second for this step
                step_reward = step_tokens_generated / step_time
                step_rewards.append(step_reward)
                
                # Log step time for performance monitoring
                if self.perf_monitor:
                    self.perf_monitor.log_step_time(step_time)
                
                # Update policy if in training mode
                if training_mode and len(step_rewards) >= 1:
                    try:
                        rl_policy.update_policy(step_reward)
                        if len(step_rewards) % 30 == 0:  # Log every 30 steps
                            print(f"  Step {idx} reward: {step_reward:.2f} tok/s (accepted: {step_tokens_generated})")
                    except Exception as e:
                        print(f"  Warning: RL policy update failed at step {idx}: {e}")
            
            # Adjusting the input sequence, draft model forward
            with torch.no_grad():
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    self,
                    hidden_state_new,
                    sample_p
                )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        
        # Restore original parameters
        self.ea_layer.total_tokens = original_total_tokens
        self.ea_layer.depth = original_depth
        self.ea_layer.top_k = original_top_k
        
        # Print comprehensive performance statistics if using optimizations
        if use_stepwise_rl and self.rl_optimizer and not training_mode:
            cache_stats = self.rl_optimizer.get_cache_stats()
            if cache_stats['total_calls'] > 0:
                # Display progressive hit rate information
                current_target = cache_stats.get('target_hit_rate', 0.7)
                initial_target = cache_stats.get('initial_target_hit_rate', 0.7)
                progressive_enabled = cache_stats.get('progressive_hit_rate_enabled', False)
                
                if progressive_enabled:
                    print(f"🚀 RL Inference Optimization Stats (Progressive: {initial_target:.0%} → {current_target:.0%} → 100%):")
                else:
                    print(f"🚀 RL Inference Optimization Stats (Fixed Target: {current_target:.0%} hit rate):")
                
                print(f"  State cache hit rate: {cache_stats['hit_rate']:.2%} ({cache_stats['cache_hits']}/{cache_stats['total_calls']})")
                print(f"  Parameter cache hit rate: {cache_stats['parameter_hit_rate']:.2%}")
                print(f"  Forced cache hits: {cache_stats.get('forced_cache_hits', 0)} ({cache_stats.get('forced_hit_rate', 0):.2%})")
                
                # Similarity-based action prediction stats
                similarity_total = cache_stats.get('similarity_hits', 0) + cache_stats.get('similarity_misses', 0)
                current_similarity_rate = cache_stats.get('current_similarity_bypass_rate', 0.7)
                initial_similarity_rate = cache_stats.get('initial_similarity_bypass_rate', 0.7)
                progressive_similarity = cache_stats.get('progressive_similarity_bypass_enabled', False)
                
                if progressive_similarity:
                    similarity_stats = f"  Similarity bypass: {cache_stats.get('similarity_hit_rate', 0):.2%} hit rate ({cache_stats.get('similarity_hits', 0)}/{similarity_total}) - Target: {current_similarity_rate:.0%}"
                else:
                    similarity_stats = f"  Similarity bypass: {cache_stats.get('similarity_hit_rate', 0):.2%} hit rate ({cache_stats.get('similarity_hits', 0)}/{similarity_total})"
                    
                if cache_stats.get('similarity_bypass_enabled', False):
                    threshold_info = f" - {cache_stats.get('similarity_cache_size', 0)} cached actions"
                    # Add threshold debugging info
                    current_threshold = cache_stats.get('similarity_threshold', 3.0)
                    base_threshold = cache_stats.get('base_similarity_threshold', 3.0)
                    if abs(current_threshold - base_threshold) > 0.1:
                        threshold_info += f" (threshold: {current_threshold:.1f})"
                    print(similarity_stats + threshold_info)
                else:
                    print(similarity_stats + " [DISABLED]")
                
                print(f"  Total RL calls: {cache_stats['total_calls']}")
                print(f"  Current decode frequency: {cache_stats['current_decode_frequency']}")
                
                # Debug cache details
                print(f"  State cache entries: {cache_stats['cache_size']}")
                print(f"  Parameter cache entries: {cache_stats['parameter_cache_size']}")
                
                if 'decode_speedup' in cache_stats:
                    print(f"  Decode speedup: {cache_stats['decode_speedup']:.2f}x")
                    print(f"  Policy speedup: {cache_stats['policy_speedup']:.2f}x")
                
                if self.perf_monitor:
                    perf_stats = self.perf_monitor.get_stats()
                    if perf_stats['total_steps'] > 0:
                        print(f"  Avg decode time: {perf_stats['avg_decode_time']:.4f}s")
                        print(f"  Steps per second: {perf_stats['steps_per_second']:.2f}")
                        print(f"  Decode overhead: {perf_stats['decode_overhead_ratio']:.2f}x")
                        
                        # Show optimization effectiveness
                        total_time_saved = cumulative_decode_time * (1 - 1/max(cache_stats['hit_rate'] + 0.01, 0.01))
                        if total_time_saved > 0.1:
                            print(f"  Est. time saved: {total_time_saved:.2f}s")
                
                if self.fast_tokenizer:
                    tokenizer_stats = self.fast_tokenizer.get_stats()
                    print(f"  Tokenizer cache hit rate: {tokenizer_stats['hit_rate']:.2%} ({tokenizer_stats['cache_hits']}/{tokenizer_stats['cache_hits'] + tokenizer_stats['cache_misses']})")
                    print(f"  Tokenizer cache size: {tokenizer_stats['cache_size']}")
                
                # Performance analysis with progressive targets
                hit_rate = cache_stats['hit_rate']
                if progressive_enabled:
                    progress_pct = (current_target - initial_target) / max(0.01, 1.0 - initial_target) * 100
                    print(f"  📈 Progressive cache target: {current_target:.1%} (Progress: {progress_pct:.1f}% towards 100%)")
                
                # Progressive similarity bypass analysis
                if progressive_similarity:
                    similarity_progress_pct = (current_similarity_rate - initial_similarity_rate) / max(0.01, 1.0 - initial_similarity_rate) * 100
                    print(f"  🎯 Progressive similarity target: {current_similarity_rate:.1%} (Progress: {similarity_progress_pct:.1f}% towards 100%)")
                    
                if hit_rate >= current_target:
                    if progressive_enabled and current_target >= 0.99:
                        print(f"  🎉 Maximum cache efficiency achieved: {hit_rate:.2%} hit rate!")
                    else:
                        print(f"  ✅ Cache hit rate target achieved: {hit_rate:.2%} >= {current_target:.2%}")
                else:
                    print(f"  🎯 Working towards cache target: {hit_rate:.2%} / {current_target:.2%}")
                
                # Similarity bypass effectiveness analysis
                actual_similarity_rate = cache_stats.get('similarity_hit_rate', 0)
                if cache_stats.get('similarity_hits', 0) > 0:
                    bypass_rate = cache_stats.get('similarity_hits', 0) / max(1, cache_stats['total_calls'])
                    print(f"  🎯 RL policy bypass rate: {bypass_rate:.2%} (via similarity matching)")
                    
                    if progressive_similarity:
                        if actual_similarity_rate >= current_similarity_rate:
                            print(f"  ✅ Similarity bypass target achieved: {actual_similarity_rate:.2%} >= {current_similarity_rate:.2%}")
                        else:
                            print(f"  🎯 Working towards similarity target: {actual_similarity_rate:.2%} / {current_similarity_rate:.2%}")
                elif progressive_similarity and cache_stats.get('similarity_cache_size', 0) == 0:
                    print(f"  ⚠️  No similarity cache entries yet - building cache for {current_similarity_rate:.0%} bypass target")
                
                # Advanced optimization stats
                if ADVANCED_OPTIMIZATIONS_AVAILABLE:
                    if self.parameter_predictor and hasattr(self.parameter_predictor, 'parameter_history'):
                        pred_history_size = len(self.parameter_predictor.parameter_history)
                        print(f"  Parameter prediction history: {pred_history_size} samples")
                    
                    if self.context_cache:
                        context_cache_size = len(self.context_cache.cache)
                        print(f"  Context-aware cache size: {context_cache_size}")
                    
                    # Auto-tuning recommendations
                    if self.profiler and len(step_rewards) > 20:
                        analysis = self.profiler.analyze_performance()
                        if 'suggestions' in analysis and analysis['suggestions']:
                            print(f"  🔧 Auto-tuning suggestions:")
                            for suggestion in analysis['suggestions'][:2]:  # Limit to 2 suggestions
                                print(f"    • {suggestion}")
                        
                        optimal_settings = self.profiler.get_optimal_settings()
                        current_freq = cache_stats.get('current_decode_frequency', 8)
                        optimal_freq = optimal_settings.get('decode_frequency', current_freq)
                        if abs(optimal_freq - current_freq) > 2:
                            print(f"  📊 Recommended decode frequency: {optimal_freq} (current: {current_freq})")
                            
                # Cache debugging info if hit rates are still low
                if cache_stats['hit_rate'] < 0.4:  # Less than 40% hit rate
                    print(f"  🔍 Cache Debug: Low hit rate detected")
                    print(f"    Recent cache keys (sample): {list(self.rl_optimizer.state_cache.keys())[-3:] if self.rl_optimizer.state_cache else 'None'}")
                    print(f"    Decode frequency pattern: Every {cache_stats['current_decode_frequency']} steps")
                    print(f"    💡 Aggressive caching enabled - will force hits to reach target")
                    
        elif use_stepwise_rl and not training_mode:
            print(f"📊 RL Inference (Standard): {len(step_rewards)} steps completed")
            if cumulative_decode_time > 0:
                print(f"  Total decode time: {cumulative_decode_time:.2f}s")
                print(f"  Avg decode time per step: {cumulative_decode_time/max(1, len(step_rewards)):.4f}s")
        
        # Return step-wise RL information if used
        if use_stepwise_rl and step_rewards_callback and training_mode:
            step_rewards_callback({
                'step_rewards': step_rewards,
                'step_actions': step_actions, 
                'step_states': step_states,
                'total_steps': len(step_rewards)
            })
        
        if not log:
            if use_stepwise_rl:
                return input_ids, step_rewards, len(step_rewards)
            else:
                return input_ids
        else:
            if use_stepwise_rl:
                return input_ids, new_token, idx, step_rewards, len(step_rewards)
            else:
                return input_ids, new_token, idx

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
