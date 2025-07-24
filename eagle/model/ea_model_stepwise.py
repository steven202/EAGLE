"""
Step-wise EAGLE Model for Real-time RL Parameter Optimization

This module provides a step-wise refactor of the EAGLE generation process,
allowing RL policies to select parameters before each draft/verify step
rather than just once per generation call.

Key improvements:
- Exposes each draft/verify step for RL intervention
- Enables real-time parameter adaptation during generation
- Supports immediate reward feedback for better learning
"""

import copy
import time
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .ea_model import EaModel
from .utils import *
from .kv_cache import initialize_past_key_values


class StepwiseEaModel(EaModel):
    """
    Step-wise EAGLE Model that exposes individual draft/verify steps
    for real-time RL parameter optimization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # State for step-wise generation
        self.generation_state = None
    
    def initialize_stepwise_generation(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            is_llama3=False,
    ):
        """
        Initialize step-wise generation state.
        
        Returns:
            generation_state: Dictionary containing all necessary state for step-wise generation
        """
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        else:
            stop_token_id = None

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        # Prepare input
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

        # Store all state needed for step-wise generation
        generation_state = {
            'input_ids': input_ids,
            'input_len': input_len,
            'past_key_values': past_key_values,
            'past_key_values_data': past_key_values_data,
            'current_length_data': current_length_data,
            'logits_processor': logits_processor,
            'padding': padding,
            'new_token': 0,
            'step_count': 0,
            'max_new_tokens': max_new_tokens,
            'max_length': max_length - self.ea_layer.total_tokens - 10,
            'stop_token_id': stop_token_id,
            'is_llama3': is_llama3,
            'completed': False,
            'draft_tokens': None,
            'retrieve_indices': None,
            'tree_mask': None,
            'tree_position_ids': None,
            'logits': None,
            'hidden_state': None,
            'sample_token': None,
            'initialized': False
        }
        
        self.generation_state = generation_state
        return generation_state
    
    def stepwise_draft_step(
            self,
            generation_state: Dict[str, Any],
            total_tokens: Optional[int] = None,
            depth: Optional[int] = None,
            tree_top_k: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a single draft step with the given parameters.
        
        Args:
            generation_state: Current generation state
            total_tokens: Total tokens parameter for this step
            depth: Depth parameter for this step
            tree_top_k: Top-k parameter for this step
            
        Returns:
            updated_state: Updated generation state
            draft_metrics: Metrics from this draft step (for reward calculation)
        """
        if generation_state['completed']:
            return generation_state, {'completed': True}
        
        # Store original parameters
        original_total_tokens = self.ea_layer.total_tokens
        original_depth = self.ea_layer.depth
        original_top_k = self.ea_layer.top_k
        
        # Temporarily update parameters if provided
        if total_tokens is not None:
            self.ea_layer.total_tokens = total_tokens - 1
        if depth is not None:
            self.ea_layer.depth = depth
        if tree_top_k is not None:
            self.ea_layer.top_k = tree_top_k

        start_time = time.time()
        
        try:
            # Initialize tree on first step
            if not generation_state['initialized']:
                print(f"🔧 Initializing step-wise generation tree...")
                # prefill
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                    generation_state['input_ids'], 
                    self, 
                    generation_state['past_key_values'], 
                    generation_state['logits_processor'],
                    total_tokens, 
                    depth, 
                    tree_top_k
                )
                
                generation_state.update({
                    'draft_tokens': draft_tokens,
                    'retrieve_indices': retrieve_indices,
                    'tree_mask': tree_mask,
                    'tree_position_ids': tree_position_ids,
                    'logits': logits,
                    'hidden_state': hidden_state,
                    'sample_token': sample_token,
                    'initialized': True
                })
                
                print(f"✅ Step-wise generation tree initialized successfully")
                
                draft_time = time.time() - start_time
                
                # Draft metrics for reward calculation
                draft_metrics = {
                    'draft_time': draft_time,
                    'total_tokens': total_tokens or self.ea_layer.total_tokens + 1,
                    'depth': depth or self.ea_layer.depth,
                    'tree_top_k': tree_top_k or self.ea_layer.top_k,
                    'step_type': 'initialize',
                    'tree_size': draft_tokens.shape[1] if draft_tokens is not None else 0,
                    'completed': False
                }
                
                return generation_state, draft_metrics
            
            # Regular draft step - generate new draft tree with current parameters
            else:
                # Ensure hidden_state is available
                if generation_state['hidden_state'] is None:
                    # print(f"⚠️  Hidden state is None, re-initializing with fresh KV cache...")
                    # Reset KV cache to avoid dimension mismatches
                    self.ea_layer.reset_kv()
                    generation_state['current_length_data'].zero_()
                    reset_tree_mode(self)
                    
                    # Use initialize_tree to properly set up the state
                    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                        generation_state['input_ids'], 
                        self, 
                        generation_state['past_key_values'], 
                        generation_state['logits_processor'],
                        total_tokens, 
                        depth, 
                        tree_top_k
                    )
                    
                    generation_state.update({
                        'draft_tokens': draft_tokens,
                        'retrieve_indices': retrieve_indices,
                        'tree_mask': tree_mask,
                        'tree_position_ids': tree_position_ids,
                        'logits': logits,
                        'hidden_state': hidden_state,
                        'sample_token': sample_token,
                    })
                    
                    draft_time = time.time() - start_time
                    
                    # Draft metrics for reward calculation
                    draft_metrics = {
                        'draft_time': draft_time,
                        'total_tokens': total_tokens or self.ea_layer.total_tokens + 1,
                        'depth': depth or self.ea_layer.depth,
                        'tree_top_k': tree_top_k or self.ea_layer.top_k,
                        'step_type': 're-initialize',
                        'tree_size': draft_tokens.shape[1] if draft_tokens is not None else 0,
                        'completed': False
                    }
                    
                    return generation_state, draft_metrics
                
                # Get current tree parameters based on what was passed or current state
                current_total_tokens = total_tokens or (self.ea_layer.total_tokens + 1)
                current_depth = depth or self.ea_layer.depth  
                current_tree_top_k = tree_top_k or self.ea_layer.top_k
                
                # Re-generate draft tree with new parameters
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_genrate(
                    generation_state['hidden_state'],
                    generation_state['input_ids'],
                    self.base_model.lm_head,
                    generation_state['logits_processor'],
                    total_tokens=current_total_tokens-1 if total_tokens else None,
                    depth=current_depth,
                    top_k=current_tree_top_k
                )
                
                generation_state.update({
                    'draft_tokens': draft_tokens,
                    'retrieve_indices': retrieve_indices,
                    'tree_mask': tree_mask,
                    'tree_position_ids': tree_position_ids,
                })
                
                draft_time = time.time() - start_time
                
                # Draft metrics for reward calculation
                draft_metrics = {
                    'draft_time': draft_time,
                    'total_tokens': current_total_tokens,
                    'depth': current_depth,
                    'tree_top_k': current_tree_top_k,
                    'step_type': 'draft',
                    'tree_size': draft_tokens.shape[1] if draft_tokens is not None else 0,
                    'completed': False
                }
                
                return generation_state, draft_metrics
                
        finally:
            # Restore original parameters
            self.ea_layer.total_tokens = original_total_tokens
            self.ea_layer.depth = original_depth
            self.ea_layer.top_k = original_top_k
    
    def stepwise_verify_step(
            self,
            generation_state: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform verification step for the current draft.
        
        Args:
            generation_state: Current generation state with draft tokens
            
        Returns:
            updated_state: Updated generation state
            verify_metrics: Metrics from verification (tokens accepted, etc.)
        """
        if generation_state['completed'] or not generation_state['initialized']:
            return generation_state, {'completed': True}
        
        start_time = time.time()
        
        # Set tree mask for base model
        self.base_model.model.tree_mask = generation_state['tree_mask']
        
        draft_tokens = generation_state['draft_tokens'].to(generation_state['input_ids'].device)
        
        # Target model forward, get logits
        logits, hidden_state_new, outputs = tree_decoding(
            self,
            draft_tokens,
            generation_state['past_key_values'],
            generation_state['tree_position_ids'],
            generation_state['input_ids'],
            generation_state['retrieve_indices'],
        )
        
        # Prepare candidates
        draft_tokens = torch.cat((draft_tokens, generation_state['padding']), dim=1)
        candidates = draft_tokens[0, generation_state['retrieve_indices']]
        
        # Verification
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits, candidates, generation_state['logits_processor']
        )
        
        verify_time = time.time() - start_time
        
        # Update inference inputs
        start_update_time = time.time()
        
        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
            generation_state['input_ids'],
            candidates,
            best_candidate,
            accept_length,
            generation_state['retrieve_indices'],
            generation_state['logits_processor'],
            generation_state['new_token'],
            generation_state['past_key_values_data'],
            generation_state['current_length_data'],
            self,
            hidden_state_new,
            sample_p
        )
        
        update_time = time.time() - start_update_time
        total_verify_time = time.time() - start_time
        
        # Check for completion conditions
        completed = False
        if generation_state['is_llama3'] and generation_state['stop_token_id']:
            if generation_state['stop_token_id'] in input_ids[0, generation_state['input_len']:].tolist():
                completed = True
        
        if self.tokenizer.eos_token_id in input_ids[0, generation_state['input_len']:].tolist():
            completed = True
        if new_token > generation_state['max_new_tokens']:
            completed = True
        if input_ids.shape[1] > generation_state['max_length']:
            completed = True
        
        # Update generation state
        generation_state.update({
            'input_ids': input_ids,
            'draft_tokens': draft_tokens,
            'retrieve_indices': retrieve_indices,
            'tree_mask': tree_mask,
            'tree_position_ids': tree_position_ids,
            'new_token': new_token,
            'hidden_state': hidden_state,
            'sample_token': sample_token,
            'step_count': generation_state['step_count'] + 1,
            'completed': completed,
            'logits': logits,
        })
        
        # Verification metrics for reward calculation
        verify_metrics = {
            'verify_time': verify_time,
            'update_time': update_time,
            'total_verify_time': total_verify_time,
            'accept_length': accept_length,
            'tokens_accepted': accept_length + 1,  # +1 for the sampled token
            'step_type': 'verify',
            'candidates_count': len(candidates),
            'best_candidate': best_candidate,
            'step_count': generation_state['step_count'],
            'total_tokens_generated': new_token,
            'completed': completed
        }
        
        return generation_state, verify_metrics
    
    def stepwise_draft_verify_step(
            self,
            generation_state: Dict[str, Any],
            total_tokens: Optional[int] = None,
            depth: Optional[int] = None,
            tree_top_k: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a complete draft+verify step with the given parameters.
        
        This is a convenience method that combines draft and verify steps.
        For fine-grained RL control, use draft_step and verify_step separately.
        
        Args:
            generation_state: Current generation state
            total_tokens: Total tokens parameter for this step
            depth: Depth parameter for this step
            tree_top_k: Top-k parameter for this step
            
        Returns:
            updated_state: Updated generation state
            step_metrics: Combined metrics from draft+verify
        """
        # Draft step
        generation_state, draft_metrics = self.stepwise_draft_step(
            generation_state, total_tokens, depth, tree_top_k
        )
        
        if draft_metrics.get('completed', False):
            return generation_state, draft_metrics
        
        # Verify step
        generation_state, verify_metrics = self.stepwise_verify_step(generation_state)
        
        # Combine metrics
        combined_metrics = {
            **draft_metrics,
            **verify_metrics,
            'step_type': 'draft_verify',
            'total_step_time': draft_metrics.get('draft_time', 0) + verify_metrics.get('total_verify_time', 0),
        }
        
        return generation_state, combined_metrics
    
    @torch.no_grad()
    def stepwise_eagenerate(
            self,
            input_ids,
            rl_policy=None,
            reward_function=None,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            default_total_tokens=None,
            default_depth=None,
            default_tree_top_k=None,
    ):
        """
        Step-wise EAGLE generation with RL policy control at each step.
        
        Args:
            input_ids: Input token ids
            rl_policy: RL policy to select parameters at each step
            reward_function: Function to calculate rewards from step metrics
            temperature, top_p, top_k: Sampling parameters
            max_new_tokens, max_length: Generation limits
            log: Whether to return detailed logs
            is_llama3: Whether using LLaMA 3
            default_*: Default parameters when RL policy not provided
            
        Returns:
            input_ids: Generated token sequence
            generation_log: Detailed step-by-step log (if log=True)
        """
        # Initialize generation state
        generation_state = self.initialize_stepwise_generation(
            input_ids, temperature, top_p, top_k, max_new_tokens, max_length, is_llama3
        )
        
        generation_log = {
            'steps': [],
            'total_steps': 0,
            'total_tokens_generated': 0,
            'total_time': 0,
            'avg_accept_length': 0,
            'rl_decisions': []
        }
        
        start_generation_time = time.time()
        
        # Generation loop - each iteration is one draft/verify step
        while not generation_state['completed']:
            step_start_time = time.time()
            
            # RL policy selects parameters for this step
            if rl_policy is not None:
                # Create context for RL policy (could include recent performance, question context, etc.)
                rl_context = self._create_rl_context(generation_state, generation_log)
                
                # Get parameters from RL policy
                rl_params = rl_policy.predict_parameters(
                    rl_context, 
                    training_mode=(not getattr(rl_policy, 'inference_only', False))
                )
                
                total_tokens = rl_params.get('total_tokens', default_total_tokens)
                depth = rl_params.get('depth', default_depth)
                tree_top_k = rl_params.get('tree_top_k', default_tree_top_k)
                
                generation_log['rl_decisions'].append({
                    'step': generation_state['step_count'],
                    'total_tokens': total_tokens,
                    'depth': depth,
                    'tree_top_k': tree_top_k,
                    'context_length': len(rl_context) if isinstance(rl_context, str) else 0
                })
            else:
                # Use default parameters
                total_tokens = default_total_tokens
                depth = default_depth
                tree_top_k = default_tree_top_k
            
            # Perform draft+verify step
            generation_state, step_metrics = self.stepwise_draft_verify_step(
                generation_state, total_tokens, depth, tree_top_k
            )
            
            step_time = time.time() - step_start_time
            step_metrics['step_total_time'] = step_time
            
            # Calculate reward if function provided
            if reward_function is not None and rl_policy is not None:
                reward_result = reward_function(
                    step_metrics.get('total_verify_time', 0),
                    step_metrics.get('tokens_accepted', 0),
                    total_tokens or self.ea_layer.total_tokens + 1,
                    depth or self.ea_layer.depth,
                    tree_top_k or self.ea_layer.top_k
                )
                
                # Handle both old (float) and new (tuple) reward function formats
                if isinstance(reward_result, tuple):
                    reward, detailed_metrics = reward_result
                    step_metrics.update(detailed_metrics)  # Add detailed metrics to step log
                else:
                    reward = reward_result
                
                step_metrics['reward'] = reward
                
                # Update RL policy with immediate feedback
                rl_policy.update_policy(
                    reward, 
                    step_metrics.get('total_verify_time', 0),
                    step_metrics.get('tokens_accepted', 0)
                )
            
            # Log step
            generation_log['steps'].append(step_metrics)
            generation_log['total_steps'] += 1
            
            # Check if we've exceeded maximum steps (safety)
            if generation_state['step_count'] > generation_state['max_length']:
                print(f"⚠️ Maximum steps ({generation_state['max_length']}) exceeded, stopping generation")
                break
        
        # Finalize generation log
        total_generation_time = time.time() - start_generation_time
        generation_log['total_time'] = total_generation_time
        
        # Ensure total_tokens_generated is a scalar
        total_tokens_generated = generation_state['new_token']
        if hasattr(total_tokens_generated, 'cpu'):
            total_tokens_generated = int(total_tokens_generated.cpu().item())
        else:
            total_tokens_generated = int(total_tokens_generated)
        generation_log['total_tokens_generated'] = total_tokens_generated
        
        if generation_log['steps']:
            # Convert tensor values to scalars for numpy operations
            tokens_accepted_values = []
            for s in generation_log['steps']:
                val = s.get('tokens_accepted', 0)
                if hasattr(val, 'cpu'):
                    tokens_accepted_values.append(int(val.cpu().item()))
                else:
                    tokens_accepted_values.append(int(val))
            
            avg_accept = np.mean(tokens_accepted_values) if tokens_accepted_values else 0
            generation_log['avg_accept_length'] = avg_accept
            
            # Also convert verify times to scalars
            verify_time_values = []
            for s in generation_log['steps']:
                val = s.get('total_verify_time', 0)
                if hasattr(val, 'cpu'):
                    verify_time_values.append(float(val.cpu().item()))
                else:
                    verify_time_values.append(float(val))
            
            total_verify_time = sum(verify_time_values)
            if total_verify_time > 0:
                # Ensure total_tokens_generated is also a scalar
                total_tokens = generation_state['new_token']
                if hasattr(total_tokens, 'cpu'):
                    total_tokens = int(total_tokens.cpu().item())
                else:
                    total_tokens = int(total_tokens)
                generation_log['tokens_per_second'] = total_tokens / total_verify_time
            else:
                generation_log['tokens_per_second'] = 0
        
        print(f"🎯 Step-wise generation completed: {generation_log['total_steps']} steps, "
              f"{generation_log['total_tokens_generated']} tokens, "
              f"{generation_log['tokens_per_second']:.2f} tokens/sec")
        
        if log:
            return generation_state['input_ids'], generation_log
        else:
            return generation_state['input_ids']
    
    def _create_rl_context(self, generation_state: Dict[str, Any], generation_log: Dict[str, Any]) -> str:
        """
        Create context string for RL policy decision making.
        
        This could include:
        - Recent performance metrics
        - Current position in generation
        - Question context
        - Historical patterns
        """
        # Basic context with recent performance
        recent_steps = generation_log['steps'][-3:] if generation_log['steps'] else []
        
        context_parts = []
        
        # Add current generation progress
        context_parts.append(f"Generation progress: {generation_state['new_token']}/{generation_state['max_new_tokens']} tokens")
        context_parts.append(f"Current step: {generation_state['step_count']}")
        
        # Add recent performance metrics
        if recent_steps:
            recent_accept_lengths = [s.get('tokens_accepted', 0) for s in recent_steps]
            recent_times = [s.get('total_verify_time', 0) for s in recent_steps]
            
            context_parts.append(f"Recent accept lengths: {recent_accept_lengths}")
            if any(t > 0 for t in recent_times):
                recent_speeds = [al/t if t > 0 else 0 for al, t in zip(recent_accept_lengths, recent_times)]
                context_parts.append(f"Recent speeds (tokens/sec): {[f'{s:.2f}' for s in recent_speeds]}")
        
        # Add current input length for context
        context_parts.append(f"Current sequence length: {generation_state['input_ids'].shape[1]}")
        
        return " | ".join(context_parts)
