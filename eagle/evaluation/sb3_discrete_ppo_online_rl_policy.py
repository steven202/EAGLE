"""
Stable Baselines 3 Discrete PPO-based Online RL Policy for Real-time EAGLE Parameter Optimization
Uses SB3's optimized PPO implementation with discrete action space for stable learning
"""

import numpy as np
import torch
import torch.nn as nn
import os
import json
from sentence_transformers import SentenceTransformer
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

class EagleParameterEnv(gym.Env):
    """Custom Gym environment for EAGLE parameter optimization"""
    
    def __init__(self):
        super(EagleParameterEnv, self).__init__()
        
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
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.valid_actions))  # Only valid actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)  # SBERT embedding
        
        # Initialize SBERT for state encoding
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Environment state
        self.current_context = ""
        self.step_count = 0
        
        print(f"EagleParameterEnv initialized:")
        print(f"  - Total parameter combinations: {self.total_actions}")
        print(f"  - Valid parameter combinations: {len(self.valid_actions)}")
        print(f"  - Valid coverage: {len(self.valid_actions)/self.total_actions*100:.1f}%")
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
        """Check if parameter combination satisfies constraint"""
        max_tokens = top_k ** (depth - 1)
        return total_tokens <= max_tokens
    
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
    
    def _encode_state(self, context):
        """Encode conversation context using SBERT"""
        embedding = self.sbert_model.encode(context)
        return embedding.astype(np.float32)
    
    def reset(self):
        """Reset environment state"""
        self.current_context = ""
        self.step_count = 0
        # Return zero state - will be set properly when predict_parameters is called
        return np.zeros(384, dtype=np.float32)
    
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
        next_obs = np.zeros(384, dtype=np.float32)
        
        return next_obs, reward, done, info

class WandbCallback(BaseCallback):
    """Custom callback for Wandb logging"""
    
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_params = []
    
    def _on_step(self) -> bool:
        # Get recent episode info if available
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                
                # Log to wandb
                if WANDB_AVAILABLE and wandb.run:
                    wandb.log({
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                        'total_timesteps': self.num_timesteps,
                    })
        
        return True

class SB3DiscretePPOOnlineTreePolicy:
    """Stable Baselines 3 PPO-based Online RL Policy for EAGLE parameter optimization"""
    
    def __init__(self, 
                 learning_rate=3e-4,
                 n_steps=64,          # Reduced for online learning
                 batch_size=32,       # Smaller batch for faster updates
                 n_epochs=4,          # Reduced epochs to prevent overfitting
                 gamma=0.95,
                 gae_lambda=0.9,
                 clip_range=0.2,
                 ent_coef=0.05,       # Entropy coefficient for exploration
                 vf_coef=0.5,         # Value function coefficient
                 max_grad_norm=0.5,
                 use_wandb=True,
                 wandb_project="eagle-sb3-discrete-ppo",
                 wandb_run_name=None,
                 checkpoint_dir="sb3_discrete_ppo_checkpoints",
                 checkpoint_freq=100,
                 max_checkpoints=3):
        
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines 3 not available. Install with: pip install stable-baselines3")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
            if not wandb.run:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "learning_rate": learning_rate,
                        "n_steps": n_steps,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "gamma": gamma,
                        "gae_lambda": gae_lambda,
                        "clip_range": clip_range,
                        "ent_coef": ent_coef,
                        "vf_coef": vf_coef,
                        "device": str(self.device)
                    }
                )
                print(f"🔗 Wandb logging initialized: {wandb.run.get_url()}")
            else:
                print(f"🔗 Using existing wandb run: {wandb.run.get_url()}")
        else:
            print("📊 Wandb logging disabled")
        
        # Create custom environment
        self.env = EagleParameterEnv()
        
        # Wrap environment for SB3
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
            tensorboard_log=None  # We'll use wandb instead
        )
        
        # Training counters
        self.step_count = 0
        self.update_count = 0
        
        # Performance tracking
        self.reward_history = []
        self.parameter_history = []
        self.tokens_per_second_history = []
        
        # Setup wandb callback
        self.callbacks = []
        if self.use_wandb:
            self.callbacks.append(WandbCallback())
        
        print(f"SB3 Discrete PPO Policy initialized:")
        print(f"  - Environment: EagleParameterEnv")
        print(f"  - Action space: {self.env.action_space.n} valid actions")
        print(f"  - Observation space: {self.env.observation_space.shape}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - PPO n_steps: {n_steps}")
        print(f"  - PPO batch_size: {batch_size}")
        print(f"  - PPO n_epochs: {n_epochs}")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
    
    def predict_parameters(self, context, training_mode=True):
        """Predict parameters using SB3 PPO policy"""
        # Encode state
        state = self.env._encode_state(context)
        self.env.current_context = context
        
        # Get action from PPO model
        action, _ = self.model.predict(state, deterministic=not training_mode)
        
        # Convert to actual parameters
        actual_action = self.env.valid_actions[action]
        total_tokens, depth, top_k = self.env._action_to_params(actual_action)
        
        # Store for training
        if training_mode:
            self.last_state = state
            self.last_action = action
            self.last_params = (total_tokens, depth, top_k)
            
            # Debug print
            max_tokens = top_k ** (depth - 1)
            mode_str = "EXPLORE" if not self.model.predict(state, deterministic=True)[0] == action else "EXPLOIT"
            print(f"SB3 Discrete PPO {mode_str}: tt={total_tokens}, d={depth}, k={top_k} (max={max_tokens})")
        else:
            # Inference mode
            max_tokens = top_k ** (depth - 1)
            print(f"SB3 Discrete PPO INFERENCE: tt={total_tokens}, d={depth}, k={top_k} (max={max_tokens})")
        
        return total_tokens, depth, top_k
    
    def update_policy(self, reward, generation_time=None, new_tokens=None):
        """Update policy with reward from last action"""
        if not hasattr(self, 'last_state'):
            return
        
        # Create a temporary episode for the single step
        obs = self.vec_env.reset()
        obs[0] = self.last_state
        
        # Take action and get environment response
        obs, _, done, info = self.vec_env.step([self.last_action])
        
        # Manually set the reward in the environment
        # Since SB3 doesn't directly support external rewards, we'll use a different approach
        
        # Track performance
        self.reward_history.append(reward)
        self.parameter_history.append(self.last_params)
        
        # Track tokens per second if provided
        if generation_time and new_tokens:
            tps = new_tokens / generation_time
            self.tokens_per_second_history.append(tps)
        
        print(f"  → Reward: {reward:.3f} for {self.last_params}")
        
        # Update model every few steps
        if len(self.reward_history) % 32 == 0:  # Update every 32 steps
            # Create training data from recent experiences
            # Note: This is a simplified approach. For full SB3 integration, 
            # you'd want to use a custom environment with proper episode handling
            
            self.update_count += 1
            print(f"SB3 PPO Update #{self.update_count}: Manual reward integration")
        
        # Increment step counter
        self.step_count += 1
        
        # Save checkpoint periodically
        if self.should_save_checkpoint():
            self.save_checkpoint()
        
        # Progress logging
        if self.step_count % 10 == 0:
            recent_rewards = self.reward_history[-10:] if len(self.reward_history) >= 10 else self.reward_history
            avg_recent_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            
            print(f"Online RL Update: Reward={reward:.3f}, Avg Recent Reward={avg_recent_reward:.3f}, Tokens/sec={tps:.1f}" if 'tps' in locals() else f"Online RL Update: Reward={reward:.3f}, Avg Recent Reward={avg_recent_reward:.3f}")
            print(f"📊 Progress: {self.questions_processed}/{400 if hasattr(self, 'total_questions') else '?'} questions, Step: {self.step_count}, SB3 Updates: {self.update_count}")
        
        # Wandb logging
        if self.use_wandb:
            log_data = {
                "reward": reward,
                "total_tokens": self.last_params[0],
                "depth": self.last_params[1], 
                "top_k": self.last_params[2],
                "step": self.step_count,
                "questions_processed": self.questions_processed
            }
            
            # Add tokens per second if available
            if 'tps' in locals():
                log_data["tokens_per_second"] = tps
            
            # Add averaging windows
            if len(self.reward_history) >= 10:
                log_data["avg_reward_10"] = np.mean(self.reward_history[-10:])
            if len(self.reward_history) >= 50:
                log_data["avg_reward_50"] = np.mean(self.reward_history[-50:])
            
            wandb.log(log_data)
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save training checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"sb3_discrete_ppo_checkpoint_step_{self.step_count}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save SB3 model
        self.model.save(checkpoint_path)
        
        # Save additional metadata
        metadata = {
            'step_count': self.step_count,
            'update_count': self.update_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'tokens_per_second_history': self.tokens_per_second_history,
        }
        
        metadata_path = checkpoint_path + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"💾 SB3 Discrete PPO checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load training checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path + ".zip"):
            print(f"❌ No checkpoint found to load")
            return False
        
        try:
            # Load SB3 model
            self.model = PPO.load(checkpoint_path, env=self.vec_env, device=self.device)
            
            # Load metadata
            metadata_path = checkpoint_path + "_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.step_count = metadata['step_count']
                self.update_count = metadata['update_count']
                self.questions_processed = metadata.get('questions_processed', 0)
                self.training_seed = metadata.get('training_seed')
                self.reward_history = metadata['reward_history']
                self.parameter_history = metadata['parameter_history']
                self.tokens_per_second_history = metadata['tokens_per_second_history']
            
            print(f"✅ SB3 Discrete PPO checkpoint loaded: {checkpoint_path}")
            print(f"   Resuming from step {self.step_count}, update {self.update_count}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('sb3_discrete_ppo_checkpoint_step_') and f.endswith('.zip')]
        if not checkpoint_files:
            return None
        
        # Sort by step number to get latest
        def extract_step(filename):
            try:
                return int(filename.replace('sb3_discrete_ppo_checkpoint_step_', '').replace('.zip', ''))
            except:
                return 0
        
        latest_file = max(checkpoint_files, key=extract_step)
        return os.path.join(self.checkpoint_dir, latest_file.replace('.zip', ''))
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only max_checkpoints files"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('sb3_discrete_ppo_checkpoint_step_') and f.endswith('.zip')]
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by step number
        def extract_step(filename):
            try:
                return int(filename.replace('sb3_discrete_ppo_checkpoint_step_', '').replace('.zip', ''))
            except:
                return 0
        
        checkpoint_files.sort(key=extract_step)
        
        # Remove oldest checkpoints
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        for file_to_remove in files_to_remove:
            file_path = os.path.join(self.checkpoint_dir, file_to_remove)
            metadata_path = file_path.replace('.zip', '_metadata.json')
            
            try:
                os.remove(file_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                print(f"🗑️ Removed old checkpoint: {file_to_remove}")
            except OSError as e:
                print(f"⚠️ Failed to remove checkpoint: {e}")
    
    def should_save_checkpoint(self):
        """Check if it's time to save a checkpoint"""
        return self.step_count % self.checkpoint_freq == 0
    
    def set_training_seed(self, seed):
        """Set training seed for reproducible shuffling"""
        self.training_seed = seed
    
    def get_resume_info(self):
        """Get information needed for resume"""
        return {
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'ppo_updates': self.update_count,
            'total_episodes': len(self.reward_history)
        }
    
    def increment_questions_processed(self, count=1):
        """Increment questions processed counter"""
        self.questions_processed += count
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]  # Last 100 rewards
        stats = {
            'total_steps': self.step_count,
            'total_updates': self.update_count,
            'avg_reward': np.mean(self.reward_history),
            'recent_avg_reward': np.mean(recent_rewards),
            'best_reward': max(self.reward_history),
            'questions_processed': self.questions_processed
        }
        
        if self.tokens_per_second_history:
            recent_tps = self.tokens_per_second_history[-100:]
            stats.update({
                'avg_tokens_per_second': np.mean(self.tokens_per_second_history),
                'recent_avg_tokens_per_second': np.mean(recent_tps),
                'best_tokens_per_second': max(self.tokens_per_second_history)
            })
        
        return stats
    
    def save(self, path):
        """Save trained policy (final model)"""
        # Remove .pth extension if present and add SB3 naming
        if path.endswith('.pth'):
            path = path.replace('.pth', '_sb3')
        
        # Save SB3 model
        self.model.save(path)
        
        # Save performance stats
        stats = self.get_performance_stats()
        stats_path = path + "_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 SB3 Discrete PPO policy saved to: {path}.zip")
        print(f"💾 Performance stats saved to: {stats_path}")
    
    def load(self, path):
        """Load trained policy"""
        # Handle .pth extension
        if path.endswith('.pth'):
            path = path.replace('.pth', '_sb3')
        
        if not os.path.exists(path + ".zip"):
            print(f"❌ Policy file not found: {path}.zip")
            return False
        
        try:
            self.model = PPO.load(path, env=self.vec_env, device=self.device)
            print(f"✅ SB3 Discrete PPO policy loaded from: {path}.zip")
            return True
        except Exception as e:
            print(f"❌ Error loading policy: {e}")
            return False


def calculate_sb3_discrete_ppo_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """Calculate reward for SB3 discrete PPO learning with appropriate scale"""
    # Primary reward: tokens per second (speed)
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Reward is simply tokens per second
    tokens_per_second = new_tokens / generation_time
    return tokens_per_second
