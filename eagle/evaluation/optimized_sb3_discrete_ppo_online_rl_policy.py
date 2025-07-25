"""
Optimized RL Policy implementations with:
1. Layer Feature Concatenation (EAGLE-3 features instead of SBERT)
2. Action Generation Frequency optimization (caching actions for N steps)
"""

import numpy as np
import torch
import torch.nn as nn
import os
import json
from collections import deque
import random
import gym
from gym import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Error: Stable Baselines 3 not available. Install with: pip install stable-baselines3")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class OptimizedEagleParameterEnv(gym.Env):
    """Optimized Custom Gym environment for EAGLE parameter optimization
    
    Optimizations:
    1. Uses EAGLE-3 layer features instead of SBERT text embeddings
    2. Supports action caching to reduce computation frequency
    """
    
    def __init__(self, hidden_size=4096):
        super(OptimizedEagleParameterEnv, self).__init__()
        
        # Parameter bins (6×6×5 = 180 total combinations)
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
        
        # OPTIMIZATION 1: Use EAGLE-3 layer features instead of SBERT
        # The hidden_size is k (model hidden size), and we expect 3k features from EAGLE-3
        self.hidden_size = hidden_size
        self.feature_dim = hidden_size  # After FC layer reduction: 3k -> k
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.valid_actions))  # Only valid actions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.feature_dim,), dtype=np.float32
        )  # EAGLE-3 features
        
        # Environment state
        self.current_hidden_states = None
        self.step_count = 0
        
        print(f"OptimizedEagleParameterEnv initialized:")
        print(f"  - Total parameter combinations: {self.total_actions}")
        print(f"  - Valid parameter combinations: {len(self.valid_actions)}")
        print(f"  - Valid coverage: {len(self.valid_actions)/self.total_actions*100:.1f}%")
        print(f"  - Feature dimension: {self.feature_dim} (EAGLE-3 layer features)")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Constraint: total_tokens ≤ top_k^(depth-1)")
    
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
    
    def _encode_state_from_hidden_states(self, hidden_states):
        """OPTIMIZATION 1: Encode state from EAGLE-3 layer features instead of SBERT
        
        Args:
            hidden_states: Concatenated hidden states from EAGLE-3 (3k dimensions)
                          or already reduced features (k dimensions)
        
        Returns:
            numpy array of shape (feature_dim,) ready for RL policy
        """
        if hidden_states is None:
            # Return zero state if no hidden states available
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Convert to numpy and ensure correct shape
        if isinstance(hidden_states, torch.Tensor):
            features = hidden_states.detach().cpu().numpy()
        else:
            features = np.array(hidden_states)
        
        # Handle different input shapes
        if len(features.shape) > 1:
            # If batch dimension exists, take the first item
            if features.shape[0] == 1:
                features = features[0]  # Remove batch dimension
            else:
                features = features[-1]  # Take last sequence position
        
        # Handle different feature dimensions
        if features.shape[-1] == self.hidden_size * 3:
            # This is the 3k-dimensional concatenated features
            # In practice, this should be reduced by the FC layer in EAGLE-3
            # For now, we'll take the mean across the 3 feature groups
            features = features.reshape(3, self.hidden_size).mean(axis=0)
        elif features.shape[-1] == self.hidden_size:
            # This is already the k-dimensional reduced features
            pass
        else:
            # Unexpected dimension, pad or truncate to match expected size
            if features.shape[-1] > self.hidden_size:
                features = features[:self.hidden_size]
            else:
                padded = np.zeros(self.hidden_size)
                padded[:features.shape[-1]] = features
                features = padded
        
        return features.astype(np.float32)
    
    def reset(self):
        """Reset environment state"""
        self.current_hidden_states = None
        self.step_count = 0
        # Return zero state - will be set properly when predict_parameters is called
        return np.zeros(self.feature_dim, dtype=np.float32)
    
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
        done = True   # Each inference is episodic
        info = {
            'total_tokens': total_tokens,
            'depth': depth,
            'top_k': top_k,
            'valid_action': actual_action
        }
        
        self.step_count += 1
        
        # Return next observation (will be updated externally)
        next_obs = np.zeros(self.feature_dim, dtype=np.float32)
        
        return next_obs, reward, done, info

class OptimizedSB3DiscretePPOOnlineTreePolicy:
    """Optimized Stable Baselines 3 PPO-based Online RL Policy
    
    Optimizations:
    1. Uses EAGLE-3 layer features instead of SBERT text embeddings
    2. Action caching to reduce computation frequency (generate action every N steps)
    
    Supports both standard PPO and max-entropy PPO modes:
    - Standard PPO: Low entropy coefficient, deterministic inference
    - Max-Entropy PPO: High entropy coefficient, temperature-based inference
    """
    
    def __init__(self, 
                 learning_rate=3e-4,
                 n_steps=64,
                 batch_size=32,
                 n_epochs=4,
                 gamma=0.95,
                 gae_lambda=0.9,
                 clip_range=0.2,
                 ent_coef=0.1,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 # Max-entropy specific parameters
                 enable_max_entropy=True,
                 max_entropy_ent_coef=0.1,
                 inference_temperature=1.5,
                 max_entropy_inference=True,
                 # OPTIMIZATION 2: Action caching parameters
                 action_cache_steps=10,        # Generate action every N steps
                 action_cache_enabled=True,    # Enable action caching
                 # OPTIMIZATION 1: EAGLE-3 feature parameters
                 hidden_size=4096,             # Model hidden size (k)
                 use_eagle3_features=True,     # Use EAGLE-3 features instead of SBERT
                 use_wandb=True,
                 wandb_project="eagle-optimized-sb3-discrete-ppo",
                 wandb_run_name=None,
                 checkpoint_dir="optimized_sb3_discrete_ppo_checkpoints",
                 checkpoint_freq=100,
                 max_checkpoints=3):
        
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines 3 not available. Install with: pip install stable-baselines3")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # OPTIMIZATION 1: EAGLE-3 features configuration
        self.use_eagle3_features = use_eagle3_features
        self.hidden_size = hidden_size
        
        # OPTIMIZATION 2: Action caching configuration
        self.action_cache_enabled = action_cache_enabled
        self.action_cache_steps = action_cache_steps
        self.cached_action = None
        self.cached_params = None
        self.cache_step_counter = 0
        self.cache_hidden_states = None
        
        # Max-entropy RL configuration
        self.enable_max_entropy = enable_max_entropy
        self.max_entropy_ent_coef = max_entropy_ent_coef
        self.inference_temperature = inference_temperature
        self.max_entropy_inference = max_entropy_inference
        
        # Resume mechanism configuration
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self.questions_processed = 0
        self.training_seed = None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize wandb logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb_run_name = wandb_run_name or f"optimized-eagle-sb3-ppo-{int(torch.randint(0, 10000, (1,)).item())}"
            wandb.init(project=wandb_project, name=wandb_run_name, reinit=True)
            wandb.config.update({
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "enable_max_entropy": enable_max_entropy,
                "action_cache_steps": action_cache_steps,
                "action_cache_enabled": action_cache_enabled,
                "use_eagle3_features": use_eagle3_features,
                "hidden_size": hidden_size
            })
        else:
            print("Warning: Wandb logging disabled")
        
        # Initialize environment with optimizations
        self.env = OptimizedEagleParameterEnv(hidden_size=hidden_size)
        
        # Determine entropy coefficient based on mode
        actual_ent_coef = max_entropy_ent_coef if enable_max_entropy else ent_coef
        
        # Initialize SB3 PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=actual_ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=0,
            device=self.device,
            tensorboard_log=None
        )
        
        # Initialize training tracking
        self.reward_history = deque(maxlen=1000)
        self.last_state = None
        self.last_action = None
        self.last_params = None
        self.step_count = 0
        
        print(f"OptimizedSB3DiscretePPOOnlineTreePolicy initialized:")
        print(f"  - Action space: {self.env.action_space.n} discrete actions")
        print(f"  - Observation space: {self.env.observation_space.shape}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - PPO n_steps: {n_steps}")
        print(f"  - PPO batch_size: {batch_size}")
        print(f"  - PPO n_epochs: {n_epochs}")
        print(f"  - Entropy coefficient: {actual_ent_coef} ({'high for diversity' if enable_max_entropy else 'standard'})")
        print(f"  - OPTIMIZATION 1 - EAGLE-3 features: {'enabled' if use_eagle3_features else 'disabled'}")
        print(f"  - OPTIMIZATION 2 - Action caching: {'enabled' if action_cache_enabled else 'disabled'}")
        if action_cache_enabled:
            print(f"    - Cache steps: {action_cache_steps} (generate action every {action_cache_steps} steps)")
        if enable_max_entropy:
            print(f"  - Inference temperature: {self.inference_temperature}")
            print(f"  - Max-entropy inference: {self.max_entropy_inference}")
        else:
            print(f"  - Inference mode: Deterministic/Standard Stochastic")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
        
        if enable_max_entropy:
            print(f"🌟 OPTIMIZED MAX-ENTROPY MODE: EAGLE-3 features + action caching + higher exploration")
        else:
            print(f"⚙️  OPTIMIZED STANDARD PPO MODE: EAGLE-3 features + action caching + standard exploration")
    
    def predict_parameters(self, context=None, hidden_states=None, training_mode=True):
        """OPTIMIZED: Predict parameters using EAGLE-3 features with action caching
        
        Args:
            context: Text context (used for fallback if hidden_states not available)
            hidden_states: EAGLE-3 layer features (preferred input)
            training_mode: Whether in training mode
        
        Returns:
            tuple: (total_tokens, depth, top_k)
        """
        # OPTIMIZATION 2: Action caching logic
        if self.action_cache_enabled and not training_mode:
            # Check if we can use cached action
            if (self.cached_params is not None and 
                self.cache_step_counter < self.action_cache_steps):
                
                self.cache_step_counter += 1
                
                # Log cache hit
                if self.use_wandb and self.cache_step_counter % 5 == 0:
                    wandb.log({
                        "cache_hit": 1,
                        "cache_step": self.cache_step_counter,
                        "cached_params": {
                            "total_tokens": self.cached_params[0],
                            "depth": self.cached_params[1], 
                            "top_k": self.cached_params[2]
                        }
                    })
                
                return self.cached_params
        
        # OPTIMIZATION 1: Use EAGLE-3 features if available
        if self.use_eagle3_features and hidden_states is not None:
            # Encode state from EAGLE-3 layer features
            state = self.env._encode_state_from_hidden_states(hidden_states)
            # Store hidden states for future reference
            self.env.current_hidden_states = hidden_states
        else:
            # Fallback to context-based encoding (if needed)
            if context is not None:
                # For backward compatibility, we'll still support text context
                # but this should be rarely used with EAGLE-3 features
                from sentence_transformers import SentenceTransformer
                if not hasattr(self, '_sbert_model'):
                    self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = self._sbert_model.encode(context)
                # Adapt SBERT embedding to expected feature dimension
                if embedding.shape[0] != self.env.feature_dim:
                    if embedding.shape[0] > self.env.feature_dim:
                        state = embedding[:self.env.feature_dim]
                    else:
                        state = np.zeros(self.env.feature_dim)
                        state[:embedding.shape[0]] = embedding
                else:
                    state = embedding
                state = state.astype(np.float32)
            else:
                # No input available, use zero state
                state = np.zeros(self.env.feature_dim, dtype=np.float32)
        
        self.env.current_context = context if context else ""
        
        # Choose prediction strategy based on mode and phase
        if self.enable_max_entropy and self.max_entropy_inference and not training_mode:
            # MAX-ENTROPY INFERENCE: Use temperature-based sampling for diversity
            if self.inference_temperature != 1.0:
                try:
                    action = self._sample_with_temperature(state, self.inference_temperature)
                    exploration_mode = "MAX-ENTROPY"
                except Exception as e:
                    print(f"⚠️  Temperature sampling failed: {e}")
                    print("   Using enhanced stochastic sampling instead")
                    action = self._enhanced_stochastic_sampling(state)
                    exploration_mode = "ENHANCED-STOCHASTIC"
            else:
                action, _ = self.model.predict(state, deterministic=False) 
                exploration_mode = "STOCHASTIC"
        elif training_mode:
            # TRAINING MODE: Always use stochastic for both modes (exploration)
            action, _ = self.model.predict(state, deterministic=False)
            exploration_mode = "EXPLORE"
        else:
            # STANDARD INFERENCE: Use deterministic prediction
            action, _ = self.model.predict(state, deterministic=True)
            exploration_mode = "DETERMINISTIC"
        
        # Convert to actual parameters
        actual_action = self.env.valid_actions[action]
        total_tokens, depth, top_k = self.env._action_to_params(actual_action)
        
        # OPTIMIZATION 2: Update action cache
        if self.action_cache_enabled and not training_mode:
            self.cached_params = (total_tokens, depth, top_k)
            self.cached_action = action
            self.cache_step_counter = 1  # Reset counter
            self.cache_hidden_states = hidden_states
            
            # Log cache update
            if self.use_wandb:
                wandb.log({
                    "cache_update": 1,
                    "new_cached_params": {
                        "total_tokens": total_tokens,
                        "depth": depth,
                        "top_k": top_k
                    }
                })
        
        # Store for training
        if training_mode:
            self.last_state = state
            self.last_action = action
            self.last_params = (total_tokens, depth, top_k)
        
        self.step_count += 1
        
        return total_tokens, depth, top_k
    
    def _sample_with_temperature(self, state, temperature):
        """Sample action using temperature-based softmax for max-entropy exploration"""
        obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            try:
                # Get logits from policy network
                features = self.model.policy.extract_features(obs_tensor)
                if hasattr(self.model.policy, 'mlp_extractor'):
                    latent_pi, _ = self.model.policy.mlp_extractor(features)
                    logits = self.model.policy.action_net(latent_pi)
                else:
                    logits = self.model.policy.action_net(features)
                
                # Apply temperature scaling
                if temperature > 0:
                    scaled_logits = logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Sample from the distribution
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    
                    return action.item()
                else:
                    return torch.argmax(logits, dim=-1).item()
                    
            except Exception as e:
                print(f"Temperature sampling error: {e}")
                # Fallback to standard prediction
                action, _ = self.model.predict(state, deterministic=False)
                return action
    
    def _enhanced_stochastic_sampling(self, state):
        """Enhanced stochastic sampling as fallback"""
        action, _ = self.model.predict(state, deterministic=False)
        return action
    
    def update_policy(self, reward, generation_time=None, new_tokens=None):
        """Update policy with received reward"""
        if self.last_state is not None and self.last_action is not None:
            # Add to reward history
            self.reward_history.append(reward)
            
            # Store experience for SB3 learning
            # Note: SB3 handles the experience collection automatically during training
            
            # Log to wandb
            if self.use_wandb:
                log_data = {
                    "reward": reward,
                    "step": self.step_count,
                    "avg_reward_100": torch.mean(torch.tensor(list(self.reward_history)[-100:])).item(),
                }
                
                if generation_time is not None:
                    log_data["generation_time"] = generation_time
                if new_tokens is not None:
                    log_data["new_tokens"] = new_tokens
                    if generation_time is not None and generation_time > 0:
                        log_data["tokens_per_second"] = new_tokens / generation_time
                
                # Add parameter info
                if self.last_params:
                    log_data.update({
                        "total_tokens": self.last_params[0],
                        "depth": self.last_params[1],
                        "top_k": self.last_params[2]
                    })
                
                wandb.log(log_data)
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save model checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.questions_processed}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.zip")
        
        # Save SB3 model
        self.model.save(checkpoint_path)
        
        # Save additional state
        state_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_state.json")
        state = {
            "questions_processed": self.questions_processed,
            "training_seed": self.training_seed,
            "step_count": self.step_count,
            "reward_history": list(self.reward_history),
            "cache_step_counter": self.cache_step_counter,
            "cached_params": self.cached_params,
            "enable_max_entropy": self.enable_max_entropy,
            "action_cache_enabled": self.action_cache_enabled,
            "action_cache_steps": self.action_cache_steps,
            "use_eagle3_features": self.use_eagle3_features
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✅ Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Load SB3 model
            self.model = PPO.load(checkpoint_path, env=self.env, device=self.device)
            
            # Load additional state
            state_path = checkpoint_path.replace('.zip', '_state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.questions_processed = state.get("questions_processed", 0)
                self.training_seed = state.get("training_seed", None)
                self.step_count = state.get("step_count", 0)
                self.reward_history = deque(state.get("reward_history", []), maxlen=1000)
                self.cache_step_counter = state.get("cache_step_counter", 0)
                self.cached_params = tuple(state["cached_params"]) if state.get("cached_params") else None
            
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
            print(f"   Questions processed: {self.questions_processed}")
            print(f"   Step count: {self.step_count}")
            return True
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.zip')]
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
        return os.path.join(self.checkpoint_dir, checkpoints[0])
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.zip')]
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.max_checkpoints:]:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
            state_path = checkpoint_path.replace('.zip', '_state.json')
            
            try:
                os.remove(checkpoint_path)
                if os.path.exists(state_path):
                    os.remove(state_path)
                print(f"🗑️  Removed old checkpoint: {checkpoint}")
            except Exception as e:
                print(f"Warning: Could not remove {checkpoint}: {e}")
    
    def should_save_checkpoint(self):
        """Check if we should save a checkpoint"""
        return self.questions_processed % self.checkpoint_freq == 0
    
    def set_training_seed(self, seed):
        """Set training seed for reproducible shuffling"""
        self.training_seed = seed
    
    def get_resume_info(self):
        """Get information for resuming training"""
        return {
            "questions_processed": self.questions_processed,
            "training_seed": self.training_seed
        }
    
    def increment_questions_processed(self, count=1):
        """Increment the number of questions processed"""
        self.questions_processed += count
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if len(self.reward_history) == 0:
            return {"avg_reward": 0.0, "reward_std": 0.0, "total_steps": self.step_count}
        
        rewards = list(self.reward_history)
        return {
            "avg_reward": torch.mean(torch.tensor(rewards)).item(),
            "reward_std": torch.std(torch.tensor(rewards)).item(),
            "total_steps": self.step_count,
            "total_questions": self.questions_processed,
            "cache_hit_rate": (self.cache_step_counter / max(self.step_count, 1)) if self.action_cache_enabled else 0.0
        }
    
    def save(self, path):
        """Save the entire policy"""
        self.model.save(path)
        
        # Save additional metadata
        metadata_path = path.replace('.zip', '_metadata.json')
        metadata = {
            "questions_processed": self.questions_processed,
            "step_count": self.step_count,
            "enable_max_entropy": self.enable_max_entropy,
            "action_cache_enabled": self.action_cache_enabled,
            "action_cache_steps": self.action_cache_steps,
            "use_eagle3_features": self.use_eagle3_features,
            "hidden_size": self.hidden_size
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved optimized policy to: {path}")
    
    def load(self, path):
        """Load the entire policy"""
        if os.path.exists(path):
            self.model = PPO.load(path, env=self.env, device=self.device)
            
            # Load metadata if available
            metadata_path = path.replace('.zip', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.questions_processed = metadata.get("questions_processed", 0)
                self.step_count = metadata.get("step_count", 0)
            
            print(f"📂 Loaded optimized policy from: {path}")
            return True
        else:
            print(f"❌ Policy file not found: {path}")
            return False

def calculate_optimized_sb3_discrete_ppo_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """Calculate reward for optimized SB3 discrete PPO learning with appropriate scale"""
    # Primary reward: tokens per second (speed)
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Reward is simply tokens per second
    tokens_per_second = new_tokens / generation_time
    return tokens_per_second
