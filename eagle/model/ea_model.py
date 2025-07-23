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

    @torch.no_grad()
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
            rl_policy_callback=None,  # NEW: Callback for RL policy parameter selection
    ):
        """
        Enhanced EAGLE generation with optional RL-aware parameter selection.
        
        Args:
            rl_policy_callback: Function(context, step_info) -> (total_tokens, depth, top_k)
                If provided, will be called before each draft/verify step for dynamic parameter selection.
        """
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        # Store original parameters for restoration later
        original_total_tokens = self.ea_layer.total_tokens
        original_depth = self.ea_layer.depth
        original_top_k = self.ea_layer.top_k
        
        # Initialize generation state
        generation_state = self._prepare_generation_state(
            input_ids, temperature, top_p, top_k, max_length, is_llama3
        )
        
        try:
            if rl_policy_callback is not None:
                # RL-aware generation with dynamic parameter selection
                result = self._eagenerate_with_rl_policy(
                    generation_state, max_new_tokens, max_length, log,
                    rl_policy_callback, total_tokens, depth, tree_top_k
                )
            else:
                # Traditional generation with fixed parameters
                if total_tokens is not None:
                    self.ea_layer.total_tokens = total_tokens - 1
                if depth is not None:
                    self.ea_layer.depth = depth
                if tree_top_k is not None:
                    self.ea_layer.top_k = tree_top_k
                
                result = self._eagenerate_traditional(
                    generation_state, max_new_tokens, max_length, log
                )
        finally:
            # Restore original parameters
            self.ea_layer.total_tokens = original_total_tokens
            self.ea_layer.depth = original_depth
            self.ea_layer.top_k = original_top_k
        
        return result
    
    def _prepare_generation_state(self, input_ids, temperature, top_p, top_k, max_length, is_llama3):
        """Prepare the generation state (KV cache, logits processor, etc.)"""
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

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
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        return {
            'input_ids': input_ids,
            'input_len': input_len,
            'logits_processor': logits_processor,
            'padding': padding,
            'past_key_values': past_key_values,
            'past_key_values_data': past_key_values_data,
            'current_length_data': current_length_data,
            'is_llama3': is_llama3,
            'stop_token_id': self.tokenizer.convert_tokens_to_ids("<|eot_id|>") if is_llama3 else None,
        }
    
    def _eagenerate_with_rl_policy(self, state, max_new_tokens, max_length, log, rl_policy_callback, 
                                   default_total_tokens, default_depth, default_tree_top_k):
        """RL-aware generation with dynamic parameter selection per draft/verify step"""
        input_ids = state['input_ids']
        input_len = state['input_len']
        new_token = 0
        step_count = 0
        max_length = max_length - 64  # Reserve space for parameter changes
        
        # Track performance for RL feedback
        step_performance = []
        
        while step_count < max_length and new_token < max_new_tokens:
            # Get current context for policy decision
            current_context = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # Step information for policy
            step_info = {
                'step_count': step_count,
                'new_tokens_generated': new_token,
                'current_length': input_ids.shape[1],
                'input_length': input_len,
                'recent_performance': step_performance[-5:] if step_performance else []
            }
            
            # Get parameters from RL policy
            try:
                policy_total_tokens, policy_depth, policy_top_k = rl_policy_callback(current_context, step_info)
                
                # Validate parameters
                if policy_total_tokens <= 0 or policy_depth <= 0 or policy_top_k <= 0:
                    raise ValueError("Invalid policy parameters")
                    
            except Exception as e:
                # Fallback to default parameters if policy fails
                print(f"⚠️ RL policy failed: {e}, using default parameters")
                policy_total_tokens = default_total_tokens or (self.ea_layer.total_tokens + 1)
                policy_depth = default_depth or self.ea_layer.depth
                policy_top_k = default_tree_top_k or self.ea_layer.top_k
            
            # Perform single draft+verify step with selected parameters
            step_start_time = time.time()
            step_result = self._draft_and_verify_step(
                state, policy_total_tokens, policy_depth, policy_top_k
            )
            step_end_time = time.time()
            
            if step_result is None:
                # Generation finished (EOS, etc.)
                break
                
            # Update state with step results
            input_ids = step_result['input_ids']
            new_token = step_result['new_tokens']
            accepted_tokens = step_result['accepted_tokens']
            
            # Calculate step performance for RL feedback
            step_time = step_end_time - step_start_time
            if accepted_tokens > 0 and step_time > 0:
                tokens_per_second = accepted_tokens / step_time
                step_performance.append({
                    'step': step_count,
                    'tokens_per_second': tokens_per_second,
                    'accepted_tokens': accepted_tokens,
                    'parameters': (policy_total_tokens, policy_depth, policy_top_k),
                    'step_time': step_time
                })
            
            step_count += 1
            
            # Check termination conditions
            if state['is_llama3'] and state['stop_token_id'] in input_ids[0, input_len:].tolist():
                break
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if input_ids.shape[1] > max_length:
                break
        
        # Update final state
        state['input_ids'] = input_ids
        
        if not log:
            return input_ids
        else:
            return input_ids, new_token, step_count
    
    def _eagenerate_traditional(self, state, max_new_tokens, max_length, log):
        """Traditional generation with fixed parameters (original implementation)"""
        input_ids = state['input_ids']
        input_len = state['input_len']
        
        # Initialize tree with current parameters (traditional approach)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, state['past_key_values'], state['logits_processor']
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        
        for idx in range(max_length):
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                state['past_key_values'],
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            
            draft_tokens = torch.cat((draft_tokens, state['padding']), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, state['logits_processor']
            )
            
            # Update inputs
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                state['logits_processor'],
                new_token,
                state['past_key_values_data'],
                state['current_length_data'],
                self,
                hidden_state_new,
                sample_p
            )

            # Check termination conditions
            if state['is_llama3'] and state['stop_token_id'] in input_ids[0, input_len:].tolist():
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
    
    def _draft_and_verify_step(self, state, total_tokens, depth, top_k):
        """Perform a single draft+verify step with given parameters"""
        try:
            # Temporarily update model parameters
            original_total_tokens = self.ea_layer.total_tokens
            original_depth = self.ea_layer.depth
            original_top_k = self.ea_layer.top_k
            
            self.ea_layer.total_tokens = total_tokens - 1  # Adjust as needed
            self.ea_layer.depth = depth
            self.ea_layer.top_k = top_k
            
            # Re-initialize tree with new parameters
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                state['input_ids'], self, state['past_key_values'], state['logits_processor'], 
                total_tokens, depth, top_k
            )
            
            # Single draft+verify step
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(state['input_ids'].device)
            
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                state['past_key_values'],
                tree_position_ids,
                state['input_ids'],
                retrieve_indices,
            )
            
            draft_tokens = torch.cat((draft_tokens, state['padding']), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, state['logits_processor']
            )
            
            # Update inputs
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                state['input_ids'],
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                state['logits_processor'],
                state.get('new_tokens_so_far', 0),
                state['past_key_values_data'],
                state['current_length_data'],
                self,
                hidden_state_new,
                sample_p
            )
            
            # Restore original parameters
            self.ea_layer.total_tokens = original_total_tokens
            self.ea_layer.depth = original_depth
            self.ea_layer.top_k = original_top_k
            
            # Update state
            state['input_ids'] = input_ids
            
            return {
                'input_ids': input_ids,
                'new_tokens': new_token,
                'accepted_tokens': accept_length,
                'draft_tokens': draft_tokens,
                'retrieve_indices': retrieve_indices,
                'tree_mask': tree_mask,
                'tree_position_ids': tree_position_ids,
                'hidden_state': hidden_state,
                'sample_token': sample_token
            }
            
        except Exception as e:
            print(f"⚠️ Draft+verify step failed with params ({total_tokens}, {depth}, {top_k}): {e}")
            return None

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
