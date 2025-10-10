"""Generate answers with local models - PROFILING VERSION

This version focuses on measuring overhead breakdown for the optimized SB3 discrete PPO policy.
Only supports: OptimizedSB3DiscretePPOOnlineTreePolicy from optimized_sb3_discrete_ppo_online_rl_policy_ofl

Usage:
Same as gen_ea_answer_llama3chat_rl.py but with detailed timing measurements and overhead breakdown.
"""
import argparse
import json
import os
import sys
import time
import threading
import contextlib
import torch
import numpy as np
from collections import defaultdict, deque

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, '..'))

from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
    from ..model import utils as eagle_utils
    # Import specific functions to have references for monkey-patching
    from ..model.utils import initialize_tree, tree_decoding, evaluate_posterior, update_inference_inputs
    # ONLY SUPPORT OPTIMIZED POLICY FOR SIMPLICITY
    from .optimized_sb3_discrete_ppo_online_rl_policy_ofl import OptimizedSB3DiscretePPOOnlineTreePolicy as OptimizedSB3DiscretePPOOnlineTreePolicyOFL
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *
    from eagle.model import utils as eagle_utils
    # Import specific functions to have references for monkey-patching
    from eagle.model.utils import initialize_tree, tree_decoding, evaluate_posterior, update_inference_inputs
    # ONLY SUPPORT OPTIMIZED POLICY FOR SIMPLICITY
    from eagle.evaluation.optimized_sb3_discrete_ppo_online_rl_policy_ofl import OptimizedSB3DiscretePPOOnlineTreePolicy as OptimizedSB3DiscretePPOOnlineTreePolicyOFL


class ProfilingTimer:
    """Context manager for timing code blocks with detailed breakdown"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timings = {}
        self.stack = []
    
    def start_timer(self, name):
        """Start timing a specific operation"""
        self.stack.append(name)
        self.current_timings[name] = time.perf_counter()
    
    def end_timer(self, name):
        """End timing a specific operation"""
        if name in self.current_timings:
            elapsed = time.perf_counter() - self.current_timings[name]
            self.timings[name].append(elapsed)
            del self.current_timings[name]
            if self.stack and self.stack[-1] == name:
                self.stack.pop()
    
    def get_stats(self):
        """Get timing statistics with breakdown"""
        stats = {}
        total_time = 0
        
        for name, times in self.timings.items():
            if times:
                avg_time = np.mean(times)
                total_time_for_op = np.sum(times)
                total_time += total_time_for_op
                stats[name] = {
                    'avg_time': avg_time,
                    'total_time': total_time_for_op,
                    'count': len(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                }
        
        # Calculate percentages
        for name in stats:
            if total_time > 0:
                stats[name]['percentage'] = (stats[name]['total_time'] / total_time) * 100
            else:
                stats[name]['percentage'] = 0
        
        return stats, total_time
    
    def print_report(self):
        """Print detailed timing report"""
        stats, total_time = self.get_stats()
        
        print("\n" + "="*80)
        print("PROFILING REPORT - OVERHEAD BREAKDOWN")
        print("="*80)
        print(f"Total execution time: {total_time:.4f}s")
        print("\nBreakdown by component:")
        print("-" * 80)
        print(f"{'Component':<40} {'Total(s)':<10} {'Avg(s)':<10} {'Count':<8} {'%':<8}")
        print("-" * 80)
        
        # Sort by total time (highest first) - show ALL components, even small ones
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for name, stat in sorted_stats:
            print(f"{name:<40} {stat['total_time']:<10.4f} {stat['avg_time']:<10.4f} {stat['count']:<8} {stat['percentage']:<8.1f}")
        
        print("-" * 80)
        print("\nDetailed Analysis:")
        
        # Show EAGLE-specific components first
        eagle_components = [name for name, _ in sorted_stats if 'eagle_' in name.lower()]
        if eagle_components:
            print(f"\n🦅 EAGLE Components:")
            for comp in eagle_components:
                stat = stats[comp]
                print(f"  • {comp}: {stat['percentage']:.3f}% ({stat['total_time']:.4f}s, {stat['count']} calls)")
        
        # Categorize components for better analysis
        categories = {
            'RL Policy': [name for name, _ in sorted_stats if any(kw in name.lower() for kw in ['rl_policy', 'policy'])],
            'EAGLE Tree Construction': [name for name, _ in sorted_stats if any(kw in name.lower() for kw in ['tree_construction'])],
            'EAGLE Drafting': [name for name, _ in sorted_stats if any(kw in name.lower() for kw in ['drafting_process'])],
            'EAGLE Verification': [name for name, _ in sorted_stats if any(kw in name.lower() for kw in ['verification_process'])],
            'EAGLE Tree Update': [name for name, _ in sorted_stats if any(kw in name.lower() for kw in ['tree_update'])],
            'EAGLE Core': [name for name, _ in sorted_stats if any(kw in name.lower() for kw in ['eagle_generation_core'])],
            'Model Generation': [name for name, _ in sorted_stats if any(kw in name.lower() for kw in ['model_generation', 'generation_step'])],
            'Data Processing': [name for name, _ in sorted_stats if any(kw in name.lower() for kw in ['tokenization', 'post_processing', 'conversation_preparation'])],
            'Question Processing': [name for name, _ in sorted_stats if any(kw in name.lower() for kw in ['single_question_processing', 'turn_processing'])]
        }
        
        for category, components in categories.items():
            if components:
                category_total = sum(stats[comp]['percentage'] for comp in components)
                category_time = sum(stats[comp]['total_time'] for comp in components)
                print(f"\n{category}: {category_total:.1f}% ({category_time:.4f}s)")
                for comp in components:
                    print(f"  • {comp}: {stats[comp]['percentage']:.1f}%")
        
        print("\nKey Insights:")
        if sorted_stats:
            top_component = sorted_stats[0]
            print(f"• Highest overhead: {top_component[0]} ({top_component[1]['percentage']:.1f}%)")
            
            # Specific analysis for EAGLE components
            tree_construction_comps = ['eagle_tree_construction']
            drafting_comps = ['eagle_drafting_process']
            verification_comps = ['eagle_verification_process']
            tree_update_comps = ['eagle_tree_update']
            
            tree_construction_total = sum(stats.get(comp, {}).get('percentage', 0) for comp in tree_construction_comps)
            drafting_total = sum(stats.get(comp, {}).get('percentage', 0) for comp in drafting_comps)
            verification_total = sum(stats.get(comp, {}).get('percentage', 0) for comp in verification_comps)
            tree_update_total = sum(stats.get(comp, {}).get('percentage', 0) for comp in tree_update_comps)
            
            if tree_construction_total > 0:
                print(f"• EAGLE tree construction: {tree_construction_total:.1f}%")
            if drafting_total > 0:
                print(f"• EAGLE drafting process: {drafting_total:.1f}%")
            if verification_total > 0:
                print(f"• EAGLE verification process: {verification_total:.1f}%")
            if tree_update_total > 0:
                print(f"• EAGLE tree update: {tree_update_total:.1f}%")
                
            # Analysis for generation steps
            generation_steps = [name for name in stats.keys() if 'generation_step_' in name]
            if generation_steps:
                steps_total = sum(stats[comp]['percentage'] for comp in generation_steps)
                print(f"• Generation steps ({len(generation_steps)} steps): {steps_total:.1f}%")
                
                # Find the most expensive generation step
                if generation_steps:
                    most_expensive_step = max(generation_steps, key=lambda x: stats[x]['total_time'])
                    print(f"• Most expensive step: {most_expensive_step} ({stats[most_expensive_step]['percentage']:.1f}%)")
            
            # RL policy specific analysis
            rl_components = [name for name in stats.keys() if 'rl_policy' in name.lower()]
            if rl_components:
                rl_total = sum(stats[comp]['percentage'] for comp in rl_components)
                print(f"• RL Policy total overhead: {rl_total:.1f}%")
                
                if 'rl_policy_prediction' in stats:
                    pred_pct = stats['rl_policy_prediction']['percentage']
                    pred_count = stats['rl_policy_prediction']['count']
                    print(f"  - Prediction: {pred_pct:.1f}% ({pred_count} calls)")
                
                if 'rl_policy_update' in stats:
                    update_pct = stats['rl_policy_update']['percentage']
                    update_count = stats['rl_policy_update']['count']
                    print(f"  - Training updates: {update_pct:.1f}% ({update_count} calls)")
        
        print(f"\n{'='*80}")
        print("💡 OPTIMIZATION SUGGESTIONS:")
        
        # Provide optimization suggestions based on results
        if sorted_stats:
            top_overhead = sorted_stats[0]
            if top_overhead[1]['percentage'] > 30:
                print(f"• Consider optimizing '{top_overhead[0]}' (>30% overhead)")
            
            # Check for RL overhead
            rl_total = sum(stats.get(comp, {}).get('percentage', 0) for comp in stats.keys() if 'rl_policy' in comp.lower())
            if rl_total > 20:
                print(f"• RL Policy overhead is high ({rl_total:.1f}%) - consider action caching or model compression")
            
            # Check for EAGLE component overhead
            tree_construction_total = sum(stats.get(comp, {}).get('percentage', 0) for comp in ['eagle_tree_construction'])
            drafting_total = sum(stats.get(comp, {}).get('percentage', 0) for comp in ['eagle_drafting_process'])
            verification_total = sum(stats.get(comp, {}).get('percentage', 0) for comp in ['eagle_verification_process'])
            tree_update_total = sum(stats.get(comp, {}).get('percentage', 0) for comp in ['eagle_tree_update'])
            
            if tree_construction_total > 10:
                print(f"• EAGLE tree construction overhead is high ({tree_construction_total:.1f}%) - consider optimizing tree initialization")
            if drafting_total > 20:
                print(f"• EAGLE drafting overhead is high ({drafting_total:.1f}%) - consider optimizing draft model inference")
            if verification_total > 20:
                print(f"• EAGLE verification overhead is high ({verification_total:.1f}%) - consider optimizing posterior evaluation")
            if tree_update_total > 10:
                print(f"• EAGLE tree update overhead is high ({tree_update_total:.1f}%) - consider optimizing tree state management")
            
            # Overall EAGLE vs other components
            eagle_total = tree_construction_total + drafting_total + verification_total + tree_update_total
            if eagle_total > 50:
                print(f"• EAGLE processes dominate ({eagle_total:.1f}%) - this is expected but could be optimized")
        
        print(f"{'='*80}")


# Global profiling timer
profiler = ProfilingTimer()


def patch_eagle_functions_with_timing(model):
    """Patch EAGLE internal functions with detailed timing instrumentation for drafting and verification"""
    # Import the utils module to patch its functions
    try:
        from ..model import utils as eagle_utils
    except:
        from eagle.model import utils as eagle_utils
    
    # Store original functions
    original_functions = {
        'initialize_tree': eagle_utils.initialize_tree,
        'tree_decoding': eagle_utils.tree_decoding,
        'evaluate_posterior': eagle_utils.evaluate_posterior,
        'update_inference_inputs': eagle_utils.update_inference_inputs,
    }
    
    # Create timing-wrapped versions with specific names for drafting/verification
    def timed_initialize_tree(*args, **kwargs):
        profiler.start_timer('eagle_tree_initialization')
        result = original_functions['initialize_tree'](*args, **kwargs)
        profiler.end_timer('eagle_tree_initialization')
        return result
    
    def timed_tree_decoding(*args, **kwargs):
        profiler.start_timer('eagle_drafting_process')  # This is the drafting step
        result = original_functions['tree_decoding'](*args, **kwargs)
        profiler.end_timer('eagle_drafting_process')
        return result
    
    def timed_evaluate_posterior(*args, **kwargs):
        profiler.start_timer('eagle_verification_process')  # This is the verification step
        result = original_functions['evaluate_posterior'](*args, **kwargs)
        profiler.end_timer('eagle_verification_process')
        return result
    
    def timed_update_inference_inputs(*args, **kwargs):
        profiler.start_timer('eagle_input_update')
        result = original_functions['update_inference_inputs'](*args, **kwargs)
        profiler.end_timer('eagle_input_update')
        return result
    
    # Patch the functions
    eagle_utils.initialize_tree = timed_initialize_tree
    eagle_utils.tree_decoding = timed_tree_decoding
    eagle_utils.evaluate_posterior = timed_evaluate_posterior
    eagle_utils.update_inference_inputs = timed_update_inference_inputs
    
    return original_functions


def restore_eagle_functions(model, original_functions):
    """Restore original EAGLE functions after profiling"""
    try:
        from ..model import utils as eagle_utils
    except:
        from eagle.model import utils as eagle_utils
    
    # Restore original functions
    eagle_utils.initialize_tree = original_functions['initialize_tree']
    eagle_utils.tree_decoding = original_functions['tree_decoding']
    eagle_utils.evaluate_posterior = original_functions['evaluate_posterior']
    eagle_utils.update_inference_inputs = original_functions['update_inference_inputs']


def create_detailed_eagenerate_wrapper(original_eagenerate):
    """Create a wrapper for eagenerate with detailed internal timing"""
    def wrapped_eagenerate(self, *args, **kwargs):
        profiler.start_timer('eagenerate_setup')
        
        # Call original function and capture the result
        result = original_eagenerate(self, *args, **kwargs)
        
        profiler.end_timer('eagenerate_setup')
        
        return result
    
    return wrapped_eagenerate


def detect_actual_gpu_usage():
    """Detect if multiple GPUs are actually being used"""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        visible_gpus = [x.strip() for x in cuda_visible.split(',') if x.strip()]
        if len(visible_gpus) > 1:
            print(f"🔍 Multiple GPU setup detected: {visible_gpus}")
            return True
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"🔍 Multiple GPU setup detected: {gpu_count} GPUs")
            return True
    
    print(f"🔍 Single GPU or CPU setup detected")
    return False


def wandb_login_with_timeout(api_key, timeout=10):
    """Attempt wandb login with timeout to avoid hanging on DNS issues."""
    def login_worker(result_container):
        try:
            import wandb
            wandb.login(key=api_key)
            result_container['success'] = True
        except Exception as e:
            result_container['error'] = str(e)
    
    result = {'success': False, 'error': None}
    thread = threading.Thread(target=login_worker, args=(result,))
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"⏱️  wandb login timed out after {timeout}s (likely DNS/network issue)")
        return False
    
    if result['success']:
        return True
    else:
        print(f"❌ wandb login failed: {result.get('error', 'Unknown error')}")
        return False


def parse_ppo_net_arch_args(net_arch_str, policy_version="ofl"):
    """Parse network architecture string for PPO policies (OFL version only)"""
    net_arch_str = net_arch_str.strip().strip('"').strip("'")
    if not net_arch_str or net_arch_str.strip() == "":
        return [128, 128]  # Default for OFL version
    
    # Parse OFL version format: "pi_layers;vf_layers" or just "layers"
    if ";" in net_arch_str:
        pi_str, vf_str = net_arch_str.split(";", 1)
        pi_layers = [int(x.strip()) for x in pi_str.split(",") if x.strip()]
        vf_layers = [int(x.strip()) for x in vf_str.split(",") if x.strip()]
        return {"pi": pi_layers, "vf": vf_layers}
    else:
        layers = [int(x.strip()) for x in net_arch_str.split(",") if x.strip()]
        return layers  # Same for both pi and vf


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    profiler.start_timer('total_execution')
    
    # Load questions (no profiling - one-time setup)
    questions = load_questions(question_file, question_begin, question_end)
    
    # Enhanced question handling for online RL training vs inference
    if args.use_online_rl and not args.online_inference_only:
        original_count = len(questions)
        repeat_factor = args.online_repeat_factor
        questions = questions * repeat_factor
        
        import random
        training_seed = args.training_seed if hasattr(args, 'training_seed') and args.training_seed else 42
        random.seed(training_seed)
        random.shuffle(questions)
        
        print(f"🔄 Online RL Training Mode: Expanded {original_count} → {len(questions)} questions (repeat={repeat_factor}, seed={training_seed})")
    else:
        print(f"📋 Standard Mode: Using {len(questions)} questions in original order")

    # Only support single GPU for profiling simplicity
    assert num_gpus_total == num_gpus_per_model == 1, "Profiling version only supports single GPU"
    
    get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
    )
    
    profiler.end_timer('total_execution')
    profiler.print_report()


def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # Initialize model (no profiling - one-time setup)
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=args.use_eagle3,
    )

    tokenizer = model.get_tokenizer()
    max_length_param = 2200 if (args.bench_name == "sum" or "sum" in args.question_file) and args.online_inference_only else 2048

    # Initialize RL policy (ONLY OPTIMIZED SB3 DISCRETE PPO OFL) - no profiling, one-time setup
    online_policy = None
    
    if args.use_online_rl:
        print("Initializing Optimized SB3 Discrete PPO Policy (OFL version) for profiling...")
        
        # Setup configurations (no profiling - one-time setup)
        checkpoint_dir = args.checkpoint_dir if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir else "checkpoints"
        multi_gpu_detected = detect_actual_gpu_usage()
        use_wandb = not args.online_inference_only and not args.no_wandb
        wandb_login_successful = False
        
        if use_wandb:
            wandb_api_key = os.environ.get('WANDB_API_KEY')
            if wandb_api_key:
                try:
                    wandb_login_successful = wandb_login_with_timeout(wandb_api_key, timeout=args.wandb_timeout)
                except Exception as e:
                    print(f"❌ wandb login failed: {e}")
                    wandb_login_successful = False
            else:
                print("⚠️  WANDB_API_KEY not set, skipping wandb authentication")
                use_wandb = False
                wandb_login_successful = False
        
        wandb_run_name = f"eagle-optimized-profiling-{args.model_id}-{int(time.time())}" if use_wandb else None
        
        # Parse network architecture
        net_arch = parse_ppo_net_arch_args(args.ppo_net_arch, policy_version="ofl")
        
        # Initialize optimized SB3 discrete PPO policy (OFL version)
        online_policy = OptimizedSB3DiscretePPOOnlineTreePolicyOFL(
            learning_rate=args.online_lr,
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            n_epochs=args.ppo_epochs,
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_gae_lambda,
            clip_range=args.ppo_clip_range,
            ent_coef=args.ppo_ent_coef,
            vf_coef=args.ppo_vf_coef,
            max_grad_norm=args.max_grad_norm,
            enable_max_entropy=args.enable_max_entropy,
            max_entropy_ent_coef=args.max_entropy_ent_coef,
            inference_temperature=args.inference_temperature,
            max_entropy_inference=args.max_entropy_inference,
            action_cache_steps=args.action_cache_steps,
            action_cache_enabled=args.action_cache_enabled,
            hidden_size=args.hidden_size,
            use_eagle3_features=args.use_eagle3_features,
            use_context_only_state=getattr(args, 'use_context_only_state', False),
            net_arch=net_arch,
            use_wandb=use_wandb and wandb_login_successful,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_run_name,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=args.checkpoint_freq,
            max_checkpoints=args.max_checkpoints
        )
        
        # Resume mechanism
        resumed = False
        if not args.online_inference_only:
            if getattr(args, 'resume_training', True) and not getattr(args, 'no_resume', False):
                try:
                    online_policy.load_checkpoint()
                    resumed = True
                    print(f"✅ Resumed from checkpoint")
                except Exception as e:
                    print(f"⚠️  Could not resume from checkpoint: {e}")
        
        # Fallback to explicit policy path
        if not resumed and hasattr(args, 'online_policy_path') and args.online_policy_path:
            try:
                online_policy.load(args.online_policy_path)
                print(f"✅ Loaded policy from {args.online_policy_path}")
            except Exception as e:
                print(f"⚠️  Could not load policy from {args.online_policy_path}: {e}")
        
        # Set training seed and mode
        training_seed = getattr(args, 'training_seed', 42)
        online_policy.set_training_seed(training_seed)
        online_policy.set_training_mode(not args.online_inference_only)

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # Warmup (no profiling - one-time setup)
    question = questions[0]
    for _ in range(3):
        torch.manual_seed(0)
        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({"role": "user", "content": qs})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer([prompt]).input_ids
            if len(input_ids[0]) >= max_length_param:
                break
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Get parameters from policy for warmup
            full_context = " ".join([msg["content"] for msg in messages])
            
            if online_policy is not None:
                # Check if this is an optimized policy
                is_optimized_policy = hasattr(online_policy, 'use_eagle3_features') and online_policy.use_eagle3_features
                
                if is_optimized_policy:
                    # Optimized RL: predict parameters using EAGLE-3 features (inference mode during warmup)
                    # Note: hidden_states not available during warmup, so pass None
                    # predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                        # context=full_context, hidden_states=None, training_mode=False  # No exploration during warmup
                    # )
                    predicted_total_tokens, predicted_depth, predicted_top_k = (96, 8, 20)
                else:
                    # Traditional Online RL: predict parameters (inference mode during warmup)
                    predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                        full_context, training_mode=False  # No exploration during warmup
                    )
                print(f"Online RL predicted params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
            else:
                # Fallback to defaults
                predicted_total_tokens = args.total_token
                predicted_depth = args.depth
                predicted_top_k = args.top_k
            
            # Use no_grad for warmup since it's always inference mode
            try:
                with torch.no_grad():
                    if args.use_stepwise_rl and online_policy is not None:
                        # Step-wise RL mode: pass RL policy to eagenerate
                        result = model.eagenerate(
                            torch.as_tensor(input_ids).cuda(),
                            temperature=temperature,
                            log=True,
                            is_llama3=True,
                            total_tokens=predicted_total_tokens,  # Fallback values
                            depth=predicted_depth,
                            tree_top_k=predicted_top_k,
                            rl_policy=online_policy,
                            training_mode=False,  # No training during warmup
                            max_length=max_length_param,
                        )
                        # Handle variable return values from step-wise RL
                        if len(result) == 5:  # Step-wise RL with log=True
                            output_ids, new_token, idx, step_rewards, step_count = result
                            print(f"Warmup with step-wise RL completed: {new_token} tokens, {step_count} steps")
                        else:  # Fallback to traditional
                            output_ids, new_token, idx = result
                            print(f"Warmup completed: {new_token} tokens generated")
                    else:
                        # Traditional mode: fixed parameters for entire generation
                        output_ids, new_token, idx = model.eagenerate(
                            torch.as_tensor(input_ids).cuda(),
                            temperature=temperature,
                            log=True,
                            is_llama3=True,
                            total_tokens=predicted_total_tokens,
                            depth=predicted_depth,
                            tree_top_k=predicted_top_k,
                            max_length=max_length_param,
                        )
            except RuntimeError as e:
                if ("selected index k out of range" in str(e) or "exceeds dimension size" in str(e) or 
                    "start" in str(e) or "KV cache buffer overflow" in str(e) or 
                    "CUDA out of memory" in str(e) or "out of memory" in str(e)):
                    print(f"❌ Warmup error with params: tt={predicted_total_tokens}, d={predicted_depth}, k={predicted_top_k}")
                    print(f"   Error: {e}")
                    print(f"   Falling back to ultra-conservative warmup parameters...")
                    
                    # Use ultra-safe parameters for warmup that definitely won't overflow
                    safe_total_tokens = 60
                    safe_depth = 5
                    safe_top_k = 10
                
                    try:
                        with torch.no_grad():
                            if args.use_stepwise_rl and online_policy is not None:
                                # Step-wise RL fallback mode
                                result = model.eagenerate(
                                    torch.as_tensor(input_ids).cuda(),
                                    temperature=temperature,
                                    log=True,
                                    is_llama3=True,
                                    total_tokens=safe_total_tokens,  # Conservative fallback
                                    depth=safe_depth,
                                    tree_top_k=safe_top_k,
                                    rl_policy=online_policy,
                                    training_mode=False,  # No training during fallback
                                    max_length=max_length_param,
                                )
                                # Handle variable return values from step-wise RL
                                if len(result) == 5:  # Step-wise RL with log=True
                                    output_ids, new_token, idx, step_rewards, step_count = result
                                    print(f"Fallback warmup with step-wise RL: {new_token} tokens, {step_count} steps")
                                else:  # Fallback to traditional
                                    output_ids, new_token, idx = result
                            else:
                                # Traditional fallback mode  
                                output_ids, new_token, idx = model.eagenerate(
                                    torch.as_tensor(input_ids).cuda(),
                                    temperature=temperature,
                                    log=True,
                                    is_llama3=True,
                                    total_tokens=safe_total_tokens,
                                    depth=safe_depth,
                                    tree_top_k=safe_top_k,
                                    max_length=max_length_param,
                                )
                    except RuntimeError as e2:
                        print(f"❌ Even ultra-conservative warmup failed: {e2}")
                        print("   Skipping warmup - proceeding with standard generation")
                        # Set dummy values to avoid UnboundLocalError
                        device = next(model.parameters()).device
                        output_ids = torch.tensor([[tokenizer.eos_token_id]]).to(device)
                        new_token = torch.tensor(0).to(device)
                        idx = torch.tensor(0).to(device)
                        total_time = 0.0
                else:
                    raise e  # Re-raise if it's a different error
            
            # Only process output_ids if warmup was successful
            if 'output_ids' in locals() and output_ids is not None:
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                
                # Remove special tokens
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            else:
                # Warmup failed completely, add empty response
                print("   ⚠️  Warmup failed - adding empty response")
                turns.append("")
                idxs.append(0)
                new_tokens.append(0)
                wall_time.append(0.0)
                messages.append({
                    "role": "assistant",
                    "content": ""
                })
            
            torch.cuda.synchronize()
            end_time = time.time()
            break
    
    print('Warmup done')

    # Process questions with detailed profiling
    question_count = 0
    for question in tqdm(questions, desc="Processing questions (PROFILING)"):
        
        question_count += 1
        if online_policy is not None and not args.online_inference_only:
            online_policy.increment_questions_processed()

        choices = []
        question_failed = False
        
        for i in range(num_choices):
            profiler.start_timer('single_question_processing')
            
            torch.manual_seed(i)
            
            # Prepare conversation
            profiler.start_timer('conversation_preparation')
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            profiler.end_timer('conversation_preparation')
            
            # Process each turn
            for j in range(len(question["turns"])):
                profiler.start_timer('turn_processing')
                
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                
                profiler.start_timer('tokenization')
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
                profiler.end_timer('tokenization')
                
                if len(input_ids[0]) >= max_length_param:
                    print(f"⚠️  Input too long ({len(input_ids[0])} >= {max_length_param}), skipping turn {j}")
                    break

                torch.cuda.synchronize()
                start_time = time.time()
                
                try:
                    # RL Policy parameter prediction with detailed timing
                    if online_policy is not None:
                        profiler.start_timer('rl_policy_prediction')
                        
                        # Get context for RL policy
                        context = prompt if hasattr(args, 'use_context_only_state') and args.use_context_only_state else ""
                        hidden_states = None
                        
                        # For EAGLE-3 features, we need to get hidden states from model
                        if args.use_eagle3_features and not getattr(args, 'use_context_only_state', False):
                            profiler.start_timer('hidden_states_extraction')
                            # This is a simplified version - in real implementation, 
                            # hidden states would come from the model's forward pass
                            hidden_states = torch.randn(args.hidden_size * 3)  # Mock for profiling
                            profiler.end_timer('hidden_states_extraction')
                        
                        # Predict parameters using RL policy
                        total_token_rl, depth_rl, top_k_rl = online_policy.predict_parameters(
                            context=context,
                            hidden_states=hidden_states,
                            training_mode=not args.online_inference_only
                        )
                        
                        profiler.end_timer('rl_policy_prediction')
                        
                        # Use RL-predicted parameters
                        current_total_token = total_token_rl
                        current_depth = depth_rl
                        current_top_k = top_k_rl
                    else:
                        # Use fixed parameters
                        current_total_token = args.total_token
                        current_depth = args.depth
                        current_top_k = args.top_k

                    # Model generation with detailed timing
                    profiler.start_timer('model_generation')
                    
                    with torch.no_grad():
                        # Patch EAGLE functions with timing instrumentation
                        original_functions = patch_eagle_functions_with_timing(model)
                        
                        # Store original eagenerate method for restoration
                        original_eagenerate = model.eagenerate
                        # Note: Temporarily disabled eagenerate wrapper to avoid signature conflicts
                        # model.eagenerate = create_detailed_eagenerate_wrapper(original_eagenerate).__get__(model, type(model))
                        
                        try:
                            # This is where the main EAGLE generation happens
                            profiler.start_timer('eagle_generation_core')
                            
                            # Temporarily monkey-patch the EAGLE functions with more detailed timing
                            import eagle.model.utils as eagle_utils
                            # Save references to the original functions (from both module and globals)
                            original_initialize_tree = eagle_utils.initialize_tree
                            original_tree_decoding = eagle_utils.tree_decoding
                            original_evaluate_posterior = eagle_utils.evaluate_posterior
                            original_update_inference_inputs = eagle_utils.update_inference_inputs
                            
                            # Also save from current module's globals
                            original_global_initialize_tree = globals()['initialize_tree']
                            original_global_tree_decoding = globals()['tree_decoding']
                            original_global_evaluate_posterior = globals()['evaluate_posterior']
                            original_global_update_inference_inputs = globals()['update_inference_inputs']
                            
                            # Import and patch in the ea_model module
                            try:
                                from ..model import ea_model
                            except:
                                from eagle.model import ea_model
                            
                            # Save original functions from the ea_model module 
                            original_ea_initialize_tree = getattr(ea_model, 'initialize_tree', None)
                            original_ea_tree_decoding = getattr(ea_model, 'tree_decoding', None)
                            original_ea_evaluate_posterior = getattr(ea_model, 'evaluate_posterior', None)
                            original_ea_update_inference_inputs = getattr(ea_model, 'update_inference_inputs', None)
                            
                            def timed_initialize_tree(*args, **kwargs):
                                profiler.start_timer('eagle_tree_construction')
                                result = original_initialize_tree(*args, **kwargs)
                                profiler.end_timer('eagle_tree_construction')
                                return result
                            
                            def timed_tree_decoding(*args, **kwargs):
                                profiler.start_timer('eagle_drafting_process')
                                result = original_tree_decoding(*args, **kwargs)
                                profiler.end_timer('eagle_drafting_process')
                                return result
                            
                            def timed_evaluate_posterior(*args, **kwargs):
                                profiler.start_timer('eagle_verification_process')
                                result = original_evaluate_posterior(*args, **kwargs)
                                profiler.end_timer('eagle_verification_process')
                                return result
                            
                            def timed_update_inference_inputs(*args, **kwargs):
                                profiler.start_timer('eagle_tree_update')
                                result = original_update_inference_inputs(*args, **kwargs)
                                profiler.end_timer('eagle_tree_update')
                                return result
                            
                            # Apply the monkey patches to multiple places
                            eagle_utils.initialize_tree = timed_initialize_tree
                            eagle_utils.tree_decoding = timed_tree_decoding
                            eagle_utils.evaluate_posterior = timed_evaluate_posterior
                            eagle_utils.update_inference_inputs = timed_update_inference_inputs
                            
                            globals()['initialize_tree'] = timed_initialize_tree
                            globals()['tree_decoding'] = timed_tree_decoding
                            globals()['evaluate_posterior'] = timed_evaluate_posterior
                            globals()['update_inference_inputs'] = timed_update_inference_inputs
                            
                            # Patch in ea_model module if functions exist there
                            if original_ea_initialize_tree:
                                setattr(ea_model, 'initialize_tree', timed_initialize_tree)
                            if original_ea_tree_decoding:
                                setattr(ea_model, 'tree_decoding', timed_tree_decoding)
                            if original_ea_evaluate_posterior:
                                setattr(ea_model, 'evaluate_posterior', timed_evaluate_posterior)
                            if original_ea_update_inference_inputs:
                                setattr(ea_model, 'update_inference_inputs', timed_update_inference_inputs)
                            
                            try:
                                # Use eagenerate with correct arguments (following the original file pattern)
                                if args.use_stepwise_rl and online_policy is not None:
                                    # Step-wise RL mode: pass RL policy to eagenerate
                                    result = model.eagenerate(
                                        torch.as_tensor(input_ids).cuda(),
                                        temperature=temperature,
                                        log=True,
                                        is_llama3=True,
                                        total_tokens=current_total_token,
                                        depth=current_depth,
                                        tree_top_k=current_top_k,
                                        rl_policy=online_policy,
                                        training_mode=not args.online_inference_only,
                                        max_length=max_length_param,
                                    )
                                    # Handle variable return values from step-wise RL
                                    if len(result) == 5:  # Step-wise RL with log=True
                                        output_ids, new_token, idx, step_rewards, step_count = result
                                    else:  # Fallback to traditional
                                        output_ids, new_token, idx = result
                                else:
                                    # Traditional mode: fixed parameters for entire generation
                                    output_ids, new_token, idx = model.eagenerate(
                                        torch.as_tensor(input_ids).cuda(),
                                        temperature=temperature,
                                        log=True,
                                        is_llama3=True,
                                        total_tokens=current_total_token,
                                        depth=current_depth,
                                        tree_top_k=current_top_k,
                                        max_length=max_length_param,
                                    )
                            finally:
                                # Restore original functions to all places
                                eagle_utils.initialize_tree = original_initialize_tree
                                eagle_utils.tree_decoding = original_tree_decoding
                                eagle_utils.evaluate_posterior = original_evaluate_posterior
                                eagle_utils.update_inference_inputs = original_update_inference_inputs
                                
                                globals()['initialize_tree'] = original_global_initialize_tree
                                globals()['tree_decoding'] = original_global_tree_decoding
                                globals()['evaluate_posterior'] = original_global_evaluate_posterior
                                globals()['update_inference_inputs'] = original_global_update_inference_inputs
                                
                                # Restore in ea_model module if they existed there
                                if original_ea_initialize_tree:
                                    setattr(ea_model, 'initialize_tree', original_ea_initialize_tree)
                                if original_ea_tree_decoding:
                                    setattr(ea_model, 'tree_decoding', original_ea_tree_decoding)
                                if original_ea_evaluate_posterior:
                                    setattr(ea_model, 'evaluate_posterior', original_ea_evaluate_posterior)
                                if original_ea_update_inference_inputs:
                                    setattr(ea_model, 'update_inference_inputs', original_ea_update_inference_inputs)
                            
                            profiler.end_timer('eagle_generation_core')
                            
                        finally:
                            # Restore original functions
                            model.eagenerate = original_eagenerate
                            restore_eagle_functions(model, original_functions)
                    
                    profiler.end_timer('model_generation')
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    # Post-processing
                    profiler.start_timer('post_processing')
                    
                    # Handle eagenerate output - it returns tensor output_ids directly
                    if hasattr(output_ids, '__len__') and len(output_ids.shape) > 1:
                        output_ids = output_ids[0][len(input_ids[0]):]
                    else:
                        # Fallback case
                        output_ids = output_ids
                    
                    # Handle EOS tokens (following original file pattern)
                    stop_token_ids = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    if stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    
                    # Remove special tokens
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()
                    
                    if output == "":
                        output = "I don't know."
                    
                    turn_wall_time = end_time - start_time
                    wall_time.append(turn_wall_time)
                    new_tokens.append(int(new_token))
                    turns.append(output)
                    idxs.append(int(idx))
                    
                    profiler.end_timer('post_processing')
                    
                    # Update RL policy with reward
                    if online_policy is not None:
                        profiler.start_timer('rl_policy_update')
                        
                        # Calculate reward (tokens per second)
                        generation_time = turn_wall_time
                        num_new_tokens = int(new_token)
                        reward = num_new_tokens / generation_time if generation_time > 0 else 0.0
                        
                        # Update policy
                        online_policy.update_policy(
                            reward=reward,
                            generation_time=generation_time,
                            new_tokens=num_new_tokens,
                            training_mode=not args.online_inference_only
                        )
                        
                        profiler.end_timer('rl_policy_update')
                    
                    messages.append({"role": "assistant", "content": output})
                    
                except Exception as e:
                    print(f"❌ Error processing turn {j}: {e}")
                    question_failed = True
                    break
                
                profiler.end_timer('turn_processing')

            if not question_failed:
                ans = {
                    "question_id": question["question_id"],
                    "answer_id": f"{model_id}_q{question['question_id']}_c{i}",
                    "model_id": model_id,
                    "choices": [{"index": i, "turns": turns}],
                    "tstamp": time.time(),
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "idx": idxs,
                }
                choices.append(ans)
            
            profiler.end_timer('single_question_processing')

        # Save answers for successfully processed questions
        if choices and not question_failed:
            # Save answers (no profiling - just I/O)
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(answer_file, "a") as fout:
                for choice in choices:
                    fout.write(json.dumps(choice) + "\n")

    # Final policy save and cleanup (no profiling - one-time cleanup)
    if online_policy is not None:
        if args.online_inference_only:
            print("🔍 Inference-only mode: Policy state preserved")
        else:
            online_policy.save(args.online_policy_save_path)
            print(f"💾 Saved trained policy to {args.online_policy_save_path}")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    # No profiling - one-time cleanup operation
    
    # Check if file exists before trying to reorganize
    if not os.path.exists(answer_file):
        print(f"⚠️  Answer file {answer_file} does not exist, skipping reorganization")
        return
    
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAGLE RL Profiling - Overhead Breakdown Analysis")
    
    # Model paths
    parser.add_argument("--ea-model-path", type=str, default="/home/lyh/weights/hf/eagle3/llama31chat/8B/")
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama31chat/8B/")
    parser.add_argument("--model-id", type=str, default="llama38b2_40_profiling")
    
    # Data parameters
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-file", type=str, help="Custom question file path")
    parser.add_argument("--question-begin", type=int, help="Begin index of questions")
    parser.add_argument("--question-end", type=int, help="End index of questions")
    parser.add_argument("--answer-file", type=str, help="Output answer file")
    
    # Generation parameters
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    
    # Hardware parameters
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    
    # EAGLE parameters
    parser.add_argument("--use-eagle3", action="store_true")
    
    # Online RL parameters (ONLY OPTIMIZED SB3 DISCRETE PPO OFL SUPPORTED)
    parser.add_argument("--use-online-rl", action="store_true")
    parser.add_argument("--use-optimized-sb3-discrete-ppo", action="store_true", help="Use optimized SB3 discrete PPO (required for profiling)")
    parser.add_argument("--optimized-policy-version", type=str, choices=["ofl"], default="ofl", help="Only OFL version supported for profiling")
    parser.add_argument("--online-lr", type=float, default=3e-4)
    parser.add_argument("--online-inference-only", action="store_true")
    parser.add_argument("--online-repeat-factor", type=int, default=1)
    
    # PPO parameters
    parser.add_argument("--ppo-n-steps", type=int, default=64)
    parser.add_argument("--ppo-batch-size", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-gamma", type=float, default=0.95)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.9)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-net-arch", type=str, default="128,128")
    
    # Max-entropy parameters
    parser.add_argument("--enable-max-entropy", action="store_true", default=True)
    parser.add_argument("--max-entropy-ent-coef", type=float, default=0.1)
    parser.add_argument("--inference-temperature", type=float, default=1.5)
    parser.add_argument("--max-entropy-inference", action="store_true", default=True)
    
    # Optimization parameters
    parser.add_argument("--action-cache-steps", type=int, default=10)
    parser.add_argument("--action-cache-enabled", action="store_true", default=True)
    parser.add_argument("--use-eagle3-features", action="store_true", default=True)
    parser.add_argument("--use-context-only-state", action="store_true", default=False)
    parser.add_argument("--hidden-size", type=int, default=4096)
    
    # Step-wise RL
    parser.add_argument("--use-stepwise-rl", action="store_true")
    
    # Checkpoint and wandb parameters
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=100)
    parser.add_argument("--max-checkpoints", type=int, default=1)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--resume-training", type=bool, default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--online-policy-save-path", type=str, default="optimized_ppo_policy_profiling.zip")
    parser.add_argument("--wandb-project", type=str, default="eagle-optimized-profiling")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-timeout", type=int, default=10)

    args = parser.parse_args()
    
    # Validate profiling requirements
    if not args.use_online_rl:
        print("❌ Error: --use-online-rl is required for profiling")
        sys.exit(1)
    
    if not args.use_optimized_sb3_discrete_ppo:
        print("❌ Error: --use-optimized-sb3-discrete-ppo is required for profiling")
        sys.exit(1)
    
    if args.optimized_policy_version != "ofl":
        print("❌ Error: Only OFL version (--optimized-policy-version ofl) is supported for profiling")
        sys.exit(1)

    # Print configuration
    print("\n" + "="*60)
    print("EAGLE RL PROFILING MODE")
    print("="*60)
    print("Only supports: OptimizedSB3DiscretePPOOnlineTreePolicy (OFL)")
    print(f"Model: {args.base_model_path}")
    print(f"Questions: {args.question_begin or 0} to {args.question_end or 'end'}")
    print(f"Training mode: {not args.online_inference_only}")
    print("="*60)

    # Set up file paths
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    question_file = args.question_file if args.question_file else f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{parent_dir}/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    # Run evaluation with profiling
    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
    
    print("\n🎯 Profiling completed! Check the detailed breakdown above.")