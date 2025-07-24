"""
Step-wise RL Policy Interface for EAGLE Parameter Optimization

This module provides RL policies that can select parameters at each draft/verify step
rather than just once per generation call, enabling real-time adaptation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
import time

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sentence_transformers import SentenceTransformer
    import gym
    from gym import spaces
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: Stable Baselines 3 not available. Install with: pip install stable-baselines3")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class StepwiseEagleParameterEnv(gym.Env):
    """Custom Gym environment for step-wise EAGLE parameter optimization"""
    
    def __init__(self):
        super(StepwiseEagleParameterEnv, self).__init__()
        
        # Parameter bins (same as before, but now used per step)
        self.total_tokens_bins = [32, 48, 64, 80, 96, 128]  # 6 options
        self.depth_bins = [3, 4, 5, 6, 7, 8]  # 6 options  
        self.top_k_bins = [8, 12, 16, 20, 32]  # 5 options
        
        # Action space dimensions
        self.n_total_tokens = len(self.total_tokens_bins)
        self.n_depth = len(self.depth_bins)
        self.n_top_k = len(self.top_k_bins)
        self.total_actions = self.n_total_tokens * self.n_depth * self.n_top_k
        
        # Precompute valid actions to avoid constraint violations
        self.valid_actions = self._precompute_valid_actions()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.valid_actions))  # Only valid actions
        # Extended observation space for step-wise context
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(400,), dtype=np.float32)  # SBERT + step context
        
        # Initialize SBERT for state encoding
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Environment state for step-wise operation
        self.current_context = ""
        self.step_count = 0
        self.episode_rewards = []
        self.step_history = deque(maxlen=10)  # Keep recent step performance
        
        print(f"StepwiseEagleParameterEnv initialized:")
        print(f"  - Total parameter combinations: {self.total_actions}")
        print(f"  - Valid parameter combinations: {len(self.valid_actions)}")
        print(f"  - Valid coverage: {len(self.valid_actions)/self.total_actions*100:.1f}%")
        print(f"  - Constraint: total_tokens ≤ top_k^(depth-1)")
        print(f"  - Extended observation space for step-wise context")
    
    def _precompute_valid_actions(self):
        """Precompute valid action indices based on constraint: total_tokens <= top_k^(depth-1)"""
        valid_actions = []
        for action in range(self.total_actions):
            total_tokens, depth, top_k = self._action_to_params(action)
            if self._is_valid_combination(total_tokens, depth, top_k):
                valid_actions.append(action)
        return valid_actions
    
    def _is_valid_combination(self, total_tokens, depth, top_k):
        """Check if parameter combination satisfies constraints"""
        max_tokens_constraint = top_k ** (depth - 1)
        basic_constraint = total_tokens <= max_tokens_constraint
        return basic_constraint
    
    def _action_to_params(self, action):
        """Convert discrete action to parameter values"""
        total_tokens_idx = action // (self.n_depth * self.n_top_k)
        remaining = action % (self.n_depth * self.n_top_k)
        depth_idx = remaining // self.n_top_k
        top_k_idx = remaining % self.n_top_k
        
        return (
            self.total_tokens_bins[total_tokens_idx],
            self.depth_bins[depth_idx],
            self.top_k_bins[top_k_idx]
        )
    
    def _encode_step_context(self, context, step_info=None):
        """Encode step context with additional step-wise information"""
        # Base SBERT embedding
        base_embedding = self.sbert_model.encode(context)
        
        # Additional step-wise features (16 additional features)
        step_features = np.zeros(16, dtype=np.float32)
        
        if step_info:
            # Recent performance features
            step_features[0] = step_info.get('step_count', 0) / 100.0  # Normalized step count
            step_features[1] = step_info.get('total_tokens_generated', 0) / 1000.0  # Normalized tokens
            step_features[2] = step_info.get('avg_tokens_per_second', 0) / 100.0  # Normalized speed
            
            # Recent step performance (last 3 steps)
            recent_rewards = step_info.get('recent_rewards', [])
            for i, reward in enumerate(recent_rewards[-3:]):
                if i < 3:
                    step_features[3 + i] = reward / 100.0  # Normalized recent rewards
            
            # Recent acceptance rates
            recent_accepts = step_info.get('recent_accept_lengths', [])
            for i, accept in enumerate(recent_accepts[-3:]):
                if i < 3:
                    step_features[6 + i] = accept / 10.0  # Normalized recent accepts
            
            # Generation progress
            step_features[9] = step_info.get('generation_progress', 0)  # 0-1 progress
            
            # Recent parameter effectiveness
            recent_params = step_info.get('recent_param_effectiveness', {})
            step_features[10] = recent_params.get('total_tokens_avg_reward', 0) / 100.0
            step_features[11] = recent_params.get('depth_avg_reward', 0) / 100.0
            step_features[12] = recent_params.get('top_k_avg_reward', 0) / 100.0
            
            # Trend indicators
            step_features[13] = step_info.get('reward_trend', 0)  # -1 to 1
            step_features[14] = step_info.get('speed_trend', 0)  # -1 to 1
            step_features[15] = step_info.get('accept_trend', 0)  # -1 to 1
        
        # Combine base embedding with step features
        full_embedding = np.concatenate([base_embedding, step_features])
        return full_embedding.astype(np.float32)
    
    def reset(self):
        """Reset environment state"""
        self.current_context = ""
        self.step_count = 0
        self.step_history.clear()
        # Return zero state - will be set properly when predict_parameters is called
        return np.zeros(400, dtype=np.float32)
    
    def step(self, action):
        """Execute one step with the given action"""
        # Convert from valid action index to actual action
        actual_action = self.valid_actions[action]
        
        # Convert action to parameters
        total_tokens, depth, top_k = self._action_to_params(actual_action)
        
        # Store last parameters for external reward calculation
        self.last_params = (total_tokens, depth, top_k)
        
        # The reward will be set externally by the policy
        reward = 0.0  # Placeholder
        done = True   # Each step is episodic in the step-wise setting
        info = {
            'total_tokens': total_tokens,
            'depth': depth,
            'top_k': top_k,
            'valid_action': actual_action,
            'step_count': self.step_count
        }
        
        self.step_count += 1
        
        # Return next observation (will be updated externally)
        next_obs = np.zeros(400, dtype=np.float32)
        
        return next_obs, reward, done, info


class StepwiseSB3DiscretePPOPolicy:
    """
    Step-wise Stable Baselines 3 PPO-based Policy for EAGLE parameter optimization
    
    This policy selects parameters at each draft/verify step during generation,
    enabling real-time adaptation based on immediate feedback.
    """
    
    def __init__(self, 
                 learning_rate=3e-4,
                 n_steps=64,
                 batch_size=32,
                 n_epochs=4,
                 gamma=0.95,
                 gae_lambda=0.9,
                 clip_range=0.2,
                 ent_coef=0.1,        # Max-entropy by default
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 # Max-entropy configuration
                 enable_max_entropy=True,
                 inference_temperature=1.5,
                 max_entropy_inference=True,
                 # Step-wise specific parameters
                 step_memory_size=1000,
                 use_step_context=True,
                 reward_smoothing=0.1,
                 # Logging and checkpointing
                 use_wandb=True,
                 wandb_project="eagle-stepwise-rl",
                 wandb_run_name=None,
                 checkpoint_dir="stepwise_checkpoints",
                 checkpoint_freq=50):
        
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines 3 not available. Install with: pip install stable-baselines3")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Step-wise configuration
        self.step_memory_size = step_memory_size
        self.use_step_context = use_step_context
        self.reward_smoothing = reward_smoothing
        self.inference_only = False  # Will be set during inference
        
        # Max-entropy configuration
        self.enable_max_entropy = enable_max_entropy
        self.inference_temperature = inference_temperature if enable_max_entropy else 1.0
        self.max_entropy_inference = max_entropy_inference and enable_max_entropy
        
        # Step-wise tracking
        self.step_history = deque(maxlen=step_memory_size)
        self.recent_rewards = deque(maxlen=20)
        self.recent_accept_lengths = deque(maxlen=20)
        self.recent_parameters = deque(maxlen=50)
        self.parameter_effectiveness = {}
        
        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.steps_taken = 0
        self.current_checkpoint_name = None
        
        # Initialize wandb
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            mode_name = "Max-Entropy PPO" if enable_max_entropy else "Standard PPO"
            if not wandb.run:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or f"stepwise-{mode_name.lower().replace(' ', '-')}",
                    config={
                        "learning_rate": learning_rate,
                        "n_steps": n_steps,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "ent_coef": ent_coef,
                        "enable_max_entropy": enable_max_entropy,
                        "inference_temperature": self.inference_temperature,
                        "step_memory_size": step_memory_size,
                        "mode": "stepwise",
                        "device": str(self.device)
                    }
                )
        
        # Create step-wise environment
        self.env = StepwiseEagleParameterEnv()
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=1,
            device=self.device,
        )
        
        mode_name = "Max-Entropy PPO" if enable_max_entropy else "Standard PPO"
        print(f"StepwiseSB3DiscretePPOPolicy initialized:")
        print(f"  - Mode: {mode_name} (step-wise)")
        print(f"  - Environment: StepwiseEagleParameterEnv")
        print(f"  - Action space: {self.env.action_space.n} valid actions")
        print(f"  - Extended observation space: {self.env.observation_space.shape}")
        print(f"  - Step memory size: {step_memory_size}")
        print(f"  - Reward smoothing: {reward_smoothing}")
        print(f"  - Use step context: {use_step_context}")
        if enable_max_entropy:
            print(f"  - Max-entropy inference temperature: {self.inference_temperature}")
        print(f"  - Device: {self.device}")
    
    def predict_parameters(self, context: str, training_mode: bool = True) -> Dict[str, Any]:
        """
        Predict parameters for the current step.
        
        Args:
            context: Context string for this step
            training_mode: Whether in training mode (vs inference)
            
        Returns:
            Dictionary with parameter predictions
        """
        self.inference_only = not training_mode
        
        # Create step-wise context information
        step_info = self._create_step_info() if self.use_step_context else None
        
        # Encode state with step context
        state = self.env._encode_step_context(context, step_info)
        
        # Get action from policy
        if training_mode:
            # Training mode: use policy exploration
            action, _ = self.model.predict(state, deterministic=False)
        else:
            # Inference mode
            if self.max_entropy_inference:
                # Max-entropy inference: sample with temperature
                action = self._sample_with_temperature(state, self.inference_temperature)
            else:
                # Deterministic inference
                action, _ = self.model.predict(state, deterministic=True)
        
        # Convert action to parameters
        actual_action = self.env.valid_actions[action]
        total_tokens, depth, tree_top_k = self.env._action_to_params(actual_action)
        
        # Store current state for potential learning
        self.current_state = state
        self.current_action = action
        self.current_action_params = {
            'total_tokens': total_tokens,
            'depth': depth,
            'tree_top_k': tree_top_k
        }
        
        # Track parameter usage
        param_key = (total_tokens, depth, tree_top_k)
        if param_key not in self.parameter_effectiveness:
            self.parameter_effectiveness[param_key] = {'rewards': [], 'count': 0}
        self.parameter_effectiveness[param_key]['count'] += 1
        
        return {
            'total_tokens': total_tokens,
            'depth': depth,
            'tree_top_k': tree_top_k,
            'action': action,
            'context_length': len(context),
            'step_info': step_info
        }
    
    def _sample_with_temperature(self, state, temperature):
        """Sample action with temperature for max-entropy inference"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            distribution = self.model.policy.get_distribution(obs_tensor)
            
            # Apply temperature to logits
            logits = distribution.distribution.logits / temperature
            tempered_dist = torch.distributions.Categorical(logits=logits)
            
            action = tempered_dist.sample()
            return action.cpu().numpy()[0]
    
    def _create_step_info(self) -> Dict[str, Any]:
        """Create step context information for enhanced state representation"""
        step_info = {
            'step_count': self.steps_taken,
            'total_tokens_generated': 0,  # Will be filled from history
            'avg_tokens_per_second': 0,
            'recent_rewards': list(self.recent_rewards),
            'recent_accept_lengths': list(self.recent_accept_lengths),
            'generation_progress': 0,  # Approximate progress
            'recent_param_effectiveness': self._calculate_param_effectiveness(),
            'reward_trend': self._calculate_trend(self.recent_rewards),
            'speed_trend': 0,  # Will calculate if we have timing data
            'accept_trend': self._calculate_trend(self.recent_accept_lengths)
        }
        
        # Calculate aggregated metrics from recent history
        if self.step_history:
            recent_steps = list(self.step_history)[-10:]  # Last 10 steps
            total_tokens = sum(s.get('tokens_accepted', 0) for s in recent_steps)
            total_time = sum(s.get('verify_time', 0) for s in recent_steps)
            
            step_info['total_tokens_generated'] = total_tokens
            if total_time > 0:
                step_info['avg_tokens_per_second'] = total_tokens / total_time
        
        return step_info
    
    def _calculate_param_effectiveness(self) -> Dict[str, float]:
        """Calculate average effectiveness of recent parameter choices"""
        effectiveness = {
            'total_tokens_avg_reward': 0,
            'depth_avg_reward': 0,
            'top_k_avg_reward': 0
        }
        
        if not self.parameter_effectiveness:
            return effectiveness
        
        # Group by parameter and calculate averages
        total_tokens_rewards = []
        depth_rewards = []
        top_k_rewards = []
        
        for (tt, d, tk), data in self.parameter_effectiveness.items():
            avg_reward = np.mean(data['rewards']) if data['rewards'] else 0
            total_tokens_rewards.append(avg_reward)
            depth_rewards.append(avg_reward)
            top_k_rewards.append(avg_reward)
        
        if total_tokens_rewards:
            effectiveness['total_tokens_avg_reward'] = np.mean(total_tokens_rewards)
        if depth_rewards:
            effectiveness['depth_avg_reward'] = np.mean(depth_rewards)
        if top_k_rewards:
            effectiveness['top_k_avg_reward'] = np.mean(top_k_rewards)
        
        return effectiveness
    
    def _calculate_trend(self, values_deque) -> float:
        """Calculate trend (-1 to 1) from recent values"""
        if len(values_deque) < 3:
            return 0.0
        
        values = list(values_deque)[-10:]  # Use last 10 values
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to -1 to 1 range (approximate)
        normalized_slope = np.tanh(slope / np.std(y) if np.std(y) > 0 else 0)
        
        return float(normalized_slope)
    
    def update_policy(self, reward: float, generation_time: float = None, new_tokens: int = None):
        """
        Update policy with immediate step feedback.
        
        Args:
            reward: Reward for the current step
            generation_time: Time taken for this step
            new_tokens: Number of tokens accepted in this step
        """
        if not hasattr(self, 'current_state') or not hasattr(self, 'current_action'):
            return  # No current step to update
        
        # Ensure reward is a Python scalar (convert from tensor if needed)
        if hasattr(reward, 'cpu'):
            reward = float(reward.cpu().item())
        else:
            reward = float(reward)
        
        # Apply reward smoothing
        smoothed_reward = reward
        if self.recent_rewards:
            # Convert recent rewards to Python scalars for numpy operations
            recent_rewards_scalars = []
            for r in list(self.recent_rewards)[-5:]:
                if hasattr(r, 'cpu'):
                    recent_rewards_scalars.append(float(r.cpu().item()))
                else:
                    recent_rewards_scalars.append(float(r))
            recent_avg = np.mean(recent_rewards_scalars)
            smoothed_reward = (1 - self.reward_smoothing) * reward + self.reward_smoothing * recent_avg
        
        # Ensure all values are scalars
        if generation_time is not None and hasattr(generation_time, 'cpu'):
            generation_time = float(generation_time.cpu().item())
        if new_tokens is not None and hasattr(new_tokens, 'cpu'):
            new_tokens = int(new_tokens.cpu().item())
        
        # Store step information
        step_data = {
            'reward': smoothed_reward,
            'original_reward': reward,
            'verify_time': generation_time or 0,
            'tokens_accepted': new_tokens or 0,
            'step': self.steps_taken
        }
        self.step_history.append(step_data)
        
        # Update tracking (store as scalars)
        self.recent_rewards.append(smoothed_reward)
        if new_tokens is not None:
            self.recent_accept_lengths.append(new_tokens)
        
        # Update parameter effectiveness
        current_params = getattr(self, 'current_params', None)
        if current_params:
            param_key = current_params
            if param_key in self.parameter_effectiveness:
                self.parameter_effectiveness[param_key]['rewards'].append(smoothed_reward)
                # Keep only recent rewards
                if len(self.parameter_effectiveness[param_key]['rewards']) > 20:
                    self.parameter_effectiveness[param_key]['rewards'] = self.parameter_effectiveness[param_key]['rewards'][-20:]
        
        # For SB3, we would need to implement a custom buffer or use the learn method differently
        # This is a simplified version - in practice, you might want to collect experiences
        # and call model.learn() periodically with batches of step experiences
        
        self.steps_taken += 1
        
        # Log to wandb (all values are now guaranteed to be scalars)
        if self.use_wandb and wandb.run:
            # Convert recent rewards to scalars for numpy mean calculation
            recent_rewards_for_avg = []
            for r in list(self.recent_rewards)[-10:]:
                if hasattr(r, 'cpu'):
                    recent_rewards_for_avg.append(float(r.cpu().item()))
                else:
                    recent_rewards_for_avg.append(float(r))
            
            # Calculate tokens per second for this step
            tokens_per_second = 0.0
            if generation_time is not None and generation_time > 0 and new_tokens is not None and new_tokens > 0:
                tokens_per_second = new_tokens / generation_time
            
            # Calculate recent average tokens per second
            recent_tps_values = []
            for step_data in list(self.step_history)[-10:]:
                step_time = step_data.get('verify_time', 0)
                step_tokens = step_data.get('tokens_accepted', 0)
                if step_time > 0 and step_tokens > 0:
                    recent_tps_values.append(step_tokens / step_time)
            
            recent_avg_tps = np.mean(recent_tps_values) if recent_tps_values else 0.0
            
            # Get current parameters if available
            current_params = getattr(self, 'current_action_params', {})
            
            wandb.log({
                # Reward metrics
                'step_reward': smoothed_reward,
                'original_reward': reward,
                'recent_avg_reward': np.mean(recent_rewards_for_avg) if recent_rewards_for_avg else 0,
                
                # Performance metrics
                'step_tokens': new_tokens or 0,
                'step_time': generation_time or 0,
                'tokens_per_second': tokens_per_second,
                'recent_avg_tokens_per_second': recent_avg_tps,
                
                # Step tracking
                'step_count': self.steps_taken,
                'total_step_history_length': len(self.step_history),
                
                # Parameter choices (if available)
                'param_total_tokens': current_params.get('total_tokens', 0),
                'param_depth': current_params.get('depth', 0),
                'param_tree_top_k': current_params.get('tree_top_k', 0),
                
                # Efficiency metrics
                'acceptance_rate': new_tokens / max(current_params.get('total_tokens', 1), 1) if new_tokens else 0,
                'parameter_efficiency': tokens_per_second / max(current_params.get('total_tokens', 1), 1) if tokens_per_second > 0 else 0,
                
                # Trend indicators
                'reward_trend': self._calculate_trend(self.recent_rewards),
                'accept_trend': self._calculate_trend(self.recent_accept_lengths) if self.recent_accept_lengths else 0,
            })
        
        # Checkpoint periodically
        if self.checkpoint_freq > 0 and self.steps_taken % self.checkpoint_freq == 0:
            self.save_checkpoint()
    
    def save_checkpoint(self, checkpoint_name: str = None):
        """Save model checkpoint"""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if checkpoint_name is None:
            checkpoint_name = f"stepwise_ppo_step_{self.steps_taken}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.zip")
        self.model.save(checkpoint_path)
        
        # Save additional step-wise state
        state_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_state.pt")
        torch.save({
            'step_history': list(self.step_history),
            'recent_rewards': list(self.recent_rewards),
            'recent_accept_lengths': list(self.recent_accept_lengths),
            'parameter_effectiveness': self.parameter_effectiveness,
            'steps_taken': self.steps_taken
        }, state_path)
        
        # Delete previous checkpoint if exists
        if self.current_checkpoint_name is not None:
            old_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.current_checkpoint_name}.zip")
            old_state_path = os.path.join(self.checkpoint_dir, f"{self.current_checkpoint_name}_state.pt")
            
            if os.path.exists(old_checkpoint_path):
                os.remove(old_checkpoint_path)
                print(f"🗑️  Deleted old checkpoint: {old_checkpoint_path}")
            
            if os.path.exists(old_state_path):
                os.remove(old_state_path)
                print(f"🗑️  Deleted old state: {old_state_path}")
        
        self.current_checkpoint_name = checkpoint_name
        print(f"💾 Stepwise checkpoint saved: {checkpoint_path}")
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('stepwise_ppo_step_') and f.endswith('.zip')]
        if not checkpoint_files:
            return None
        
        # Sort by step number to get latest
        def extract_step(filename):
            try:
                return int(filename.replace('stepwise_ppo_step_', '').replace('.zip', ''))
            except:
                return 0
        
        latest_file = max(checkpoint_files, key=extract_step)
        return os.path.join(self.checkpoint_dir, latest_file.replace('.zip', ''))
    
    def load_checkpoint(self, checkpoint_path: str = None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path + ".zip"):
            print(f"❌ No stepwise checkpoint found to load")
            return False
        
        try:
            self.model = PPO.load(checkpoint_path, env=self.vec_env, device=self.device)
            
            # Load additional state if available
            state_path = checkpoint_path + "_state.pt"
            if os.path.exists(state_path):
                state = torch.load(state_path)
                self.step_history = deque(state['step_history'], maxlen=self.step_memory_size)
                self.recent_rewards = deque(state['recent_rewards'], maxlen=20)
                self.recent_accept_lengths = deque(state['recent_accept_lengths'], maxlen=20)
                self.parameter_effectiveness = state['parameter_effectiveness']
                self.steps_taken = state['steps_taken']
                print(f"📂 Stepwise state loaded from: {state_path}")
            
            print(f"✅ Stepwise checkpoint loaded: {checkpoint_path}")
            print(f"   Resuming from step {self.steps_taken}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading stepwise checkpoint: {e}")
            return False
    
    def save(self, path: str):
        """Save the complete policy"""
        self.save_checkpoint(path.replace('.zip', ''))
    
    def load(self, path: str):
        """Load the complete policy"""
        if not path.endswith('.zip'):
            path += '.zip'
        self.load_checkpoint(path)
    
    def set_training_seed(self, seed: int):
        """Set training seed for reproducibility"""
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        print(f"🌱 Step-wise RL training seed set to: {seed}")
    
    def increment_questions_processed(self):
        """Increment the count of processed questions (for compatibility)"""
        # This is mainly for compatibility with the standard policy interface
        # In step-wise RL, we track steps rather than questions
        pass
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.recent_rewards:
            return {
                'total_steps': self.steps_taken,
                'total_updates': 0,
                'avg_reward': 0.0,
                'recent_avg_reward': 0.0,
                'best_reward': 0.0,
                'step_count': self.steps_taken
            }
        
        # Convert any tensor rewards to scalars for numpy operations
        reward_scalars = []
        for r in self.recent_rewards:
            if hasattr(r, 'cpu'):
                reward_scalars.append(float(r.cpu().item()))
            else:
                reward_scalars.append(float(r))
        
        # Get all rewards from step history
        all_rewards = []
        for step_data in self.step_history:
            reward = step_data.get('reward', 0)
            if hasattr(reward, 'cpu'):
                all_rewards.append(float(reward.cpu().item()))
            else:
                all_rewards.append(float(reward))
        
        # Use recent rewards if we don't have full history
        if not all_rewards:
            all_rewards = reward_scalars
        
        recent_rewards = reward_scalars[-100:] if len(reward_scalars) > 100 else reward_scalars
        
        stats = {
            'total_steps': self.steps_taken,
            'total_updates': len(all_rewards),
            'avg_reward': np.mean(all_rewards) if all_rewards else 0.0,
            'recent_avg_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'best_reward': max(all_rewards) if all_rewards else 0.0,
            'step_count': self.steps_taken
        }
        
        # Add timing information from step history
        if self.step_history:
            verify_times = []
            tokens_accepted = []
            tokens_per_second_values = []
            
            for step_data in self.step_history:
                vt = step_data.get('verify_time', 0)
                ta = step_data.get('tokens_accepted', 0)
                if hasattr(vt, 'cpu'):
                    vt = float(vt.cpu().item())
                else:
                    vt = float(vt)
                if hasattr(ta, 'cpu'):
                    ta = int(ta.cpu().item())
                else:
                    ta = int(ta)
                
                verify_times.append(vt)
                tokens_accepted.append(ta)
                
                # Calculate tokens per second for each step
                if vt > 0 and ta > 0:
                    tokens_per_second_values.append(ta / vt)
            
            if verify_times and tokens_accepted:
                total_time = sum(verify_times)
                total_tokens = sum(tokens_accepted)
                if total_time > 0:
                    stats.update({
                        'avg_tokens_per_second': total_tokens / total_time,
                        'recent_avg_tokens_per_second': np.mean(tokens_per_second_values[-10:]) if tokens_per_second_values else 0,
                        'best_tokens_per_second': max(tokens_per_second_values) if tokens_per_second_values else 0,
                        'total_generation_time': total_time,
                        'total_tokens_processed': total_tokens,
                        'avg_step_time': np.mean(verify_times) if verify_times else 0,
                        'avg_tokens_per_step': np.mean(tokens_accepted) if tokens_accepted else 0,
                    })
        
        # Add step-wise specific stats
        if self.recent_accept_lengths:
            accept_scalars = []
            for a in self.recent_accept_lengths:
                if hasattr(a, 'cpu'):
                    accept_scalars.append(int(a.cpu().item()))
                else:
                    accept_scalars.append(int(a))
            
            stats.update({
                'recent_step_rewards': reward_scalars[-10:],  # Last 10 step rewards
                'avg_step_reward': np.mean(reward_scalars) if reward_scalars else 0.0,
                'recent_accept_lengths': accept_scalars[-10:],
                'avg_accept_length': np.mean(accept_scalars) if accept_scalars else 0.0
            })
        
        return stats


def calculate_stepwise_reward(generation_time: float, new_tokens: int, total_tokens: int, depth: int, top_k: int) -> Tuple[float, Dict[str, float]]:
    """
    Calculate reward for a single step in step-wise generation.
    
    This reward function focuses on immediate step performance rather than overall generation.
    
    Args:
        generation_time: Time taken for this verification step
        new_tokens: Number of tokens accepted in this step
        total_tokens: Total tokens parameter used
        depth: Depth parameter used
        top_k: Top-k parameter used
        
    Returns:
        Tuple of (reward_value, detailed_metrics_dict)
    """
    if generation_time <= 0 or new_tokens <= 0:
        return -10.0, {
            'tokens_per_second': 0.0,
            'acceptance_rate': 0.0,
            'parameter_efficiency': 0.0,
            'complexity_penalty': 0.0,
            'acceptance_bonus': 0.0,
            'base_reward': -10.0
        }
    
    # Primary reward: tokens per second for this step
    tokens_per_second = new_tokens / generation_time
    
    # Scale reward to reasonable range
    base_reward = tokens_per_second
    
    # Bonus for high acceptance (more tokens accepted per step is better)
    acceptance_bonus = min(new_tokens * 2, 20)  # Bonus up to 20 for high acceptance
    
    # Small penalty for overly complex parameters (encourages efficiency)
    complexity_penalty = (total_tokens / 100) + (depth / 10) + (top_k / 50)
    
    # Calculate acceptance rate and parameter efficiency
    acceptance_rate = new_tokens / max(total_tokens, 1)
    parameter_efficiency = tokens_per_second / max(total_tokens, 1)
    
    # Final reward
    reward = base_reward + acceptance_bonus - complexity_penalty
    
    # Detailed metrics for logging
    detailed_metrics = {
        'tokens_per_second': tokens_per_second,
        'acceptance_rate': acceptance_rate,
        'parameter_efficiency': parameter_efficiency,
        'complexity_penalty': complexity_penalty,
        'acceptance_bonus': acceptance_bonus,
        'base_reward': base_reward,
        'total_tokens_used': total_tokens,
        'depth_used': depth,
        'top_k_used': top_k,
        'generation_time': generation_time,
        'tokens_accepted': new_tokens
    }
    
    return reward, detailed_metrics
