"""
Online RL Policy for Real-time EAGLE Parameter Optimization
Learns and adapts tree parameters dynamically during inference
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from collections import deque
import random
import json
import os
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class OnlineTreePolicy:
    """Online RL Policy that learns tree parameters in real-time during evaluation"""
    
    def __init__(self, 
                 learning_rate=3e-4,
                 epsilon_start=0.9,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 memory_size=1000,
                 batch_size=32,
                 target_update_freq=100,
                 use_wandb=True,
                 wandb_project="eagle-online-rl",
                 wandb_run_name=None,
                 checkpoint_dir="checkpoints",
                 checkpoint_freq=100,
                 max_checkpoints=3):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Resume mechanism configuration
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self.questions_processed = 0  # Track processed questions for resume
        self.training_seed = None  # Store training seed for reproducible shuffling
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize wandb logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if not wandb.run:  # Only init if not already initialized
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "learning_rate": learning_rate,
                        "epsilon_start": epsilon_start,
                        "epsilon_end": epsilon_end,
                        "epsilon_decay": epsilon_decay,
                        "memory_size": memory_size,
                        "batch_size": batch_size,
                        "target_update_freq": target_update_freq,
                        "device": str(self.device)
                    }
                )
                print(f"🔗 Wandb logging initialized: {wandb.run.get_url()}")
            else:
                print(f"🔗 Using existing wandb run: {wandb.run.get_url()}")
        else:
            print("📊 Wandb logging disabled")
        
        # Parameter bins for discrete actions - expanded ranges
        self.total_tokens_bins = [32, 48, 64, 80, 96]  # 5 options
        self.depth_bins = [3, 4, 5, 6, 7]  # 5 options
        self.top_k_bins = [4, 8, 12, 16, 20]  # 5 options
        
        # Action space dimensions
        self.n_total_tokens = len(self.total_tokens_bins)
        self.n_depth = len(self.depth_bins)
        self.n_top_k = len(self.top_k_bins)
        self.total_actions = self.n_total_tokens * self.n_depth * self.n_top_k
        
        # Precompute valid actions to avoid runtime constraint violations
        # Constraint: total_tokens <= top_k^(depth-1)
        self.valid_actions = self._precompute_valid_actions()
        print(f"Action space: {self.total_actions} total, {len(self.valid_actions)} valid")
        
        # Initialize SBERT for state encoding
        print("Loading SBERT model for state representation...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.state_dim = 384  # SBERT embedding dimension
        
        # Initialize Q-network and target network
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Training counters
        self.step_count = 0
        self.target_update_freq = target_update_freq
        
        # Performance tracking with wandb integration
        self.reward_history = []
        self.parameter_history = []
        self.loss_history = []
        self.tokens_per_second_history = []
        
        # Wandb logging setup
        if self.use_wandb:
            # Log initial configuration
            wandb.config.update({
                "total_actions": self.total_actions,
                "valid_actions": len(self.valid_actions),
                "valid_coverage": len(self.valid_actions)/self.total_actions,
                "total_tokens_bins": self.total_tokens_bins,
                "depth_bins": self.depth_bins,
                "top_k_bins": self.top_k_bins
            })
        
        print(f"Online RL Policy initialized:")
        print(f"  - State dim: {self.state_dim}")
        print(f"  - Action space: {self.total_actions} total ({self.n_total_tokens}×{self.n_depth}×{self.n_top_k})")
        print(f"  - Valid actions: {len(self.valid_actions)} ({len(self.valid_actions)/self.total_actions*100:.1f}%)")
        print(f"  - Constraint: total_tokens ≤ top_k^(depth-1)")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
        
        # Show some example valid/invalid combinations
        print(f"  - Example valid: tt≤k^(d-1)")
        for i, action in enumerate(self.valid_actions[:3]):
            tt, d, k = self._action_to_params(action)
            max_tt = k**(d-1)
            print(f"    • tt={tt}, d={d}, k={k} → max_tt={max_tt} ✓")
        
        if len(self.valid_actions) < self.total_actions:
            print(f"  - Example invalid:")
            invalid_count = 0
            for action in range(self.total_actions):
                if action not in self.valid_actions:
                    tt, d, k = self._action_to_params(action)
                    max_tt = k**(d-1)
                    print(f"    • tt={tt}, d={d}, k={k} → max_tt={max_tt} ✗")
                    invalid_count += 1
                    if invalid_count >= 2:  # Show max 2 examples
                        break
    
    def _build_network(self):
        """Build Q-network architecture with better initialization"""
        network = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.total_actions)
        )
        
        # Better weight initialization to encourage exploration
        for layer in network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)  # Small positive bias
        
        return network
    
    def _encode_state(self, context):
        """Encode conversation context using SBERT"""
        embedding = self.sbert_model.encode(context)
        return torch.FloatTensor(embedding).to(self.device)
    
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
    
    def _params_to_action(self, total_tokens, depth, top_k):
        """Convert parameters to discrete action"""
        try:
            total_tokens_idx = self.total_tokens_bins.index(total_tokens)
            depth_idx = self.depth_bins.index(depth)
            top_k_idx = self.top_k_bins.index(top_k)
            return total_tokens_idx * (self.n_depth * self.n_top_k) + depth_idx * self.n_top_k + top_k_idx
        except ValueError:
            # If parameters not in bins, return random action
            return random.randint(0, self.total_actions - 1)
    
    def predict_parameters(self, context, training_mode=True):
        """Predict parameters using epsilon-greedy policy with enhanced exploration and safety constraints"""
        state = self._encode_state(context)
        
        # Enhanced epsilon-greedy action selection from VALID actions only
        if training_mode and random.random() < self.epsilon:
            # Exploration: random VALID action
            action = random.choice(self.valid_actions)
            exploration_used = True
        else:
            # Exploitation: best VALID action from Q-network
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                
                # Mask invalid actions by setting their Q-values to -inf
                masked_q_values = q_values.clone()
                for i in range(self.total_actions):
                    if i not in self.valid_actions:
                        masked_q_values[0, i] = float('-inf')
                
                action = masked_q_values.argmax().item()
            exploration_used = False
        
        # Convert action to parameters
        total_tokens, depth, top_k = self._action_to_params(action)
        
        # Safety backup: clamp parameters if somehow invalid (should never happen with valid actions)
        total_tokens, depth, top_k = self._safe_clamp_params(total_tokens, depth, top_k)
        
        # Store state-action for learning
        if training_mode:
            self.last_state = state
            self.last_action = action
            self.last_params = (total_tokens, depth, top_k)
            self.last_exploration = exploration_used
            
            # Debug print for training mode with constraint validation
            mode_str = "EXPLORE" if exploration_used else "EXPLOIT"
            max_tokens = top_k ** (depth - 1)
            # valid_mark = "✓" if total_tokens <= max_tokens else "✗"
            # print(f"Online RL {mode_str} {valid_mark} (ε={self.epsilon:.3f}): tt={total_tokens}, d={depth}, k={top_k} (max={max_tokens})")
        else:
            # Inference mode - just use best action
            max_tokens = top_k ** (depth - 1)
            # valid_mark = "✓" if total_tokens <= max_tokens else "✗"
            # print(f"Online RL INFERENCE {valid_mark}: tt={total_tokens}, d={depth}, k={top_k} (max={max_tokens})")
        
        return total_tokens, depth, top_k
    
    def update_policy(self, reward, generation_time=None, new_tokens=None):
        """Update policy with reward from last action"""
        if not hasattr(self, 'last_state'):
            return
        
        # Store experience in replay buffer
        experience = {
            'state': self.last_state.detach().cpu(),  # Detach to avoid gradient issues
            'action': self.last_action,
            'reward': reward,
            'done': True  # Each inference is treated as terminal
        }
        self.memory.append(experience)
        
        # Track performance
        self.reward_history.append(reward)
        self.parameter_history.append(self.last_params)
        
        # Track tokens per second if provided
        if generation_time and new_tokens:
            tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0
            self.tokens_per_second_history.append(tokens_per_sec)
        
        # Debug info for learning
        explore_str = "EXPLORE" if hasattr(self, 'last_exploration') and self.last_exploration else "EXPLOIT"
        print(f"  → Reward: {reward:.3f} for {self.last_params} ({explore_str})")
        
        # Learn from experience if we have enough samples
        if len(self.memory) >= self.batch_size:
            loss = self._learn_from_batch()
            self.loss_history.append(loss)
            print(f"  → Policy updated! Memory: {len(self.memory)}/{self.memory.maxlen}")
        
        # Update exploration rate
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if abs(old_epsilon - self.epsilon) > 0.001:
            print(f"  → Exploration decreased: {old_epsilon:.3f} → {self.epsilon:.3f}")
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"Step {self.step_count}: Updated target network, epsilon={self.epsilon:.3f}")
        
        # Wandb logging for real-time tracking
        if self.use_wandb:
            log_dict = {
                "step": self.step_count,
                "reward": reward,
                "epsilon": self.epsilon,
                "memory_size": len(self.memory),
                "total_tokens": self.last_params[0],
                "depth": self.last_params[1], 
                "top_k": self.last_params[2],
                "exploration": 1 if (hasattr(self, 'last_exploration') and self.last_exploration) else 0
            }
            
            # Add tokens per second if available
            if generation_time and new_tokens:
                log_dict["tokens_per_second"] = tokens_per_sec
                log_dict["generation_time"] = generation_time
                log_dict["new_tokens"] = new_tokens
            
            # Add loss if available
            if hasattr(self, 'loss_history') and self.loss_history:
                log_dict["loss"] = self.loss_history[-1]
            
            # Add recent performance metrics
            if len(self.reward_history) >= 10:
                log_dict["avg_reward_10"] = np.mean(self.reward_history[-10:])
            if len(self.reward_history) >= 50:
                log_dict["avg_reward_50"] = np.mean(self.reward_history[-50:])
            
            wandb.log(log_dict)
            
        # Show statistics every 10 steps
        if self.step_count % 10 == 0:
            recent_rewards = self.reward_history[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"Step {self.step_count}: Recent avg reward: {avg_reward:.3f}, ε={self.epsilon:.3f}")
            
            # Show parameter diversity
            recent_params = self.parameter_history[-10:]
            unique_params = len(set(recent_params))
            print(f"  → Parameter diversity: {unique_params}/10 unique combinations")
            
    def save_checkpoint(self, checkpoint_name=None):
        """Save training checkpoint with automatic cleanup of old checkpoints"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.step_count}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save complete training state
        checkpoint_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'loss_history': self.loss_history,
            'tokens_per_second_history': self.tokens_per_second_history,
            'memory': list(self.memory),  # Convert deque to list for serialization
            'total_tokens_bins': self.total_tokens_bins,
            'depth_bins': self.depth_bins,
            'top_k_bins': self.top_k_bins,
            'valid_actions': self.valid_actions,
            # Training configuration for consistency
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory.maxlen,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'checkpoint_freq': self.checkpoint_freq,
            'max_checkpoints': self.max_checkpoints
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Log checkpoint to wandb if enabled
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"training_checkpoint_{self.step_count}",
                type="checkpoint",
                description=f"Training checkpoint at step {self.step_count}"
            )
            artifact.add_file(checkpoint_path)
            artifact.metadata = {
                "step_count": self.step_count,
                "questions_processed": self.questions_processed,
                "epsilon": self.epsilon,
                "avg_reward_recent": np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else np.mean(self.reward_history) if self.reward_history else 0
            }
            wandb.log_artifact(artifact)
            print(f"🔗 Checkpoint logged to wandb: {artifact.name}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load training checkpoint to resume from interruption"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"❌ No checkpoint found at {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Restore model states
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training progress
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.questions_processed = checkpoint.get('questions_processed', 0)
            self.training_seed = checkpoint.get('training_seed', None)
            
            # Restore history
            self.reward_history = checkpoint['reward_history']
            self.parameter_history = checkpoint['parameter_history']
            self.loss_history = checkpoint.get('loss_history', [])
            self.tokens_per_second_history = checkpoint.get('tokens_per_second_history', [])
            
            # Restore experience replay memory
            if 'memory' in checkpoint:
                self.memory.clear()
                for experience in checkpoint['memory']:
                    self.memory.append(experience)
            
            # Restore valid actions and parameter bins
            if 'valid_actions' in checkpoint:
                self.valid_actions = checkpoint['valid_actions']
            if 'total_tokens_bins' in checkpoint:
                self.total_tokens_bins = checkpoint['total_tokens_bins']
                self.depth_bins = checkpoint['depth_bins']
                self.top_k_bins = checkpoint['top_k_bins']
            
            print(f"✅ Checkpoint loaded: {checkpoint_path}")
            print(f"   Step: {self.step_count}, Questions processed: {self.questions_processed}")
            print(f"   Epsilon: {self.epsilon:.3f}, Memory size: {len(self.memory)}")
            print(f"   Training seed: {self.training_seed}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            return False
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file in checkpoint directory"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_step_') and f.endswith('.pth')]
        if not checkpoint_files:
            return None
        
        # Sort by step number to get latest
        def extract_step(filename):
            try:
                return int(filename.split('_')[2].split('.')[0])
            except:
                return 0
        
        latest_file = max(checkpoint_files, key=extract_step)
        return os.path.join(self.checkpoint_dir, latest_file)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only max_checkpoints files"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_step_') and f.endswith('.pth')]
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by step number
        def extract_step(filename):
            try:
                return int(filename.split('_')[2].split('.')[0])
            except:
                return 0
        
        checkpoint_files.sort(key=extract_step)
        
        # Remove oldest checkpoints
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        for filename in files_to_remove:
            old_path = os.path.join(self.checkpoint_dir, filename)
            try:
                os.remove(old_path)
                print(f"🗑️  Removed old checkpoint: {filename}")
            except OSError as e:
                print(f"⚠️  Failed to remove old checkpoint {filename}: {e}")
    
    def should_save_checkpoint(self):
        """Check if it's time to save a checkpoint"""
        return self.step_count > 0 and self.step_count % self.checkpoint_freq == 0
    
    def set_training_seed(self, seed):
        """Set the training seed for reproducible shuffling"""
        self.training_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"🎲 Training seed set to: {seed}")
    
    def get_resume_info(self):
        """Get information needed for resuming training"""
        return {
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'epsilon': self.epsilon,
            'total_episodes': len(self.reward_history)
        }
    
    def increment_questions_processed(self, count=1):
        """Track processed questions for resume capability"""
        self.questions_processed += count
        
        # Auto-save checkpoint if needed
        if self.should_save_checkpoint():
            self.save_checkpoint()
    
    def _learn_from_batch(self):
        """Learn from a batch of experiences using DQN with valid action masking"""
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert states to numpy array first, then to tensor (more efficient)
        state_data = np.array([exp['state'].detach().numpy() for exp in batch])
        states = torch.FloatTensor(state_data).to(self.device)  # No requires_grad needed for inputs
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        
        # Current Q-values (Q-network parameters will have gradients automatically)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values with valid action masking (detached, no gradients needed)
        with torch.no_grad():
            target_q_all = self.target_network(states)
            
            # Mask invalid actions in target network
            for i in range(target_q_all.shape[0]):  # For each sample in batch
                for j in range(self.total_actions):
                    if j not in self.valid_actions:
                        target_q_all[i, j] = float('-inf')
            
            # Use rewards directly as targets (episodic setting)
            target_q_values = rewards.unsqueeze(1)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.step_count % 50 == 0:
            avg_reward = np.mean(self.reward_history[-50:]) if self.reward_history else 0
            print(f"Step {self.step_count}: Loss={loss.item():.4f}, Avg Reward={avg_reward:.4f}, Valid Actions: {len(self.valid_actions)}")
            
            # Show constraint compliance
            recent_params = self.parameter_history[-20:] if len(self.parameter_history) >= 20 else self.parameter_history
            violations = sum(1 for tt, d, k in recent_params if tt > k**(d-1))
            print(f"  → Constraint violations in last {len(recent_params)} actions: {violations} ({violations/len(recent_params)*100:.1f}%)")
        
        return loss.item()  # Return loss value for logging
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]
        recent_params = self.parameter_history[-100:]
        
        # Parameter usage statistics
        param_stats = {}
        for i, (total_tokens, depth, top_k) in enumerate(recent_params):
            key = f"{total_tokens}-{depth}-{top_k}"
            param_stats[key] = param_stats.get(key, 0) + 1
        
        return {
            'total_episodes': len(self.reward_history),
            'avg_reward_recent': np.mean(recent_rewards),
            'avg_reward_overall': np.mean(self.reward_history),
            'epsilon': self.epsilon,
            'most_used_params': sorted(param_stats.items(), key=lambda x: x[1], reverse=True)[:5],
            'reward_trend': recent_rewards[-10:] if len(recent_rewards) >= 10 else recent_rewards
        }
    
    def save(self, path):
        """Save the trained policy with valid actions and wandb artifacts (final save)"""
        # Use checkpoint save for consistency but mark as final
        final_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'loss_history': self.loss_history,
            'tokens_per_second_history': self.tokens_per_second_history,
            'total_tokens_bins': self.total_tokens_bins,
            'depth_bins': self.depth_bins,
            'top_k_bins': self.top_k_bins,
            'valid_actions': self.valid_actions,
            'is_final_model': True  # Mark as final trained model
        }
        
        torch.save(final_data, path)
        print(f"✅ Final trained policy saved to {path} (with {len(self.valid_actions)} valid actions)")
        
        # Save as wandb artifact
        if self.use_wandb:
            artifact = wandb.Artifact(
                name="final_online_rl_policy",
                type="model",
                description="Final trained EAGLE Online RL Policy with constraint validation"
            )
            artifact.add_file(path)
            
            # Add metadata
            artifact.metadata = {
                "final_epsilon": self.epsilon,
                "total_steps": self.step_count,
                "questions_processed": self.questions_processed,
                "final_avg_reward": np.mean(self.reward_history[-50:]) if len(self.reward_history) >= 50 else np.mean(self.reward_history),
                "parameter_diversity": len(set(self.parameter_history[-100:])) if len(self.parameter_history) >= 100 else len(set(self.parameter_history)),
                "valid_actions": len(self.valid_actions),
                "total_actions": self.total_actions,
                "training_seed": self.training_seed
            }
            
            wandb.log_artifact(artifact)
            print(f"🔗 Final policy saved as wandb artifact: {artifact.name}")
            
            artifact.add_file(path)
            
            # Add metadata
            artifact.metadata = {
                "final_epsilon": self.epsilon,
                "total_steps": self.step_count,
                "final_avg_reward": np.mean(self.reward_history[-50:]) if len(self.reward_history) >= 50 else np.mean(self.reward_history),
                "parameter_diversity": len(set(self.parameter_history[-100:])) if len(self.parameter_history) >= 100 else len(set(self.parameter_history)),
                "valid_actions": len(self.valid_actions),
                "total_actions": self.total_actions
            }
            
            wandb.log_artifact(artifact)
            print(f"🔗 Policy saved as wandb artifact: {artifact.name}")
    
    def load(self, path):
        """Load a trained policy with valid actions (supports both checkpoints and final models)"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load core model state
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            
            # Load resume information if available
            self.questions_processed = checkpoint.get('questions_processed', 0)
            self.training_seed = checkpoint.get('training_seed', None)
            
            # Load training history
            self.reward_history = checkpoint['reward_history']
            self.parameter_history = checkpoint['parameter_history']
            
            # Load additional history if available
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
            if 'tokens_per_second_history' in checkpoint:
                self.tokens_per_second_history = checkpoint['tokens_per_second_history']
            
            # Restore experience replay memory if available (for checkpoints)
            if 'memory' in checkpoint:
                self.memory.clear()
                for experience in checkpoint['memory']:
                    self.memory.append(experience)
            
            # Handle backward compatibility for valid_actions
            if 'valid_actions' in checkpoint:
                self.valid_actions = checkpoint['valid_actions']
                model_type = "final model" if checkpoint.get('is_final_model', False) else "checkpoint"
                print(f"✅ Online policy loaded from {path} ({model_type} with {len(self.valid_actions)} valid actions)")
                if self.training_seed:
                    print(f"   Training seed: {self.training_seed}, Questions processed: {self.questions_processed}")
            else:
                # Recompute if not saved (backward compatibility)
                self.valid_actions = self._precompute_valid_actions()
                print(f"✅ Online policy loaded from {path} (legacy format, recomputed {len(self.valid_actions)} valid actions)")
            
            return True
        return False
    
    def _precompute_valid_actions(self):
        """Precompute valid actions that satisfy: total_tokens <= top_k^(depth-1)"""
        valid_actions = []
        constraint_violations = 0
        
        for action in range(self.total_actions):
            total_tokens, depth, top_k = self._action_to_params(action)
            max_possible_tokens = top_k ** (depth - 1)
            
            if total_tokens <= max_possible_tokens:
                valid_actions.append(action)
            else:
                constraint_violations += 1
        
        print(f"Constraint analysis: {constraint_violations}/{self.total_actions} actions violate tt <= k^(d-1)")
        print(f"Valid action coverage: {len(valid_actions)/self.total_actions*100:.1f}%")
        
        return valid_actions
    
    def _is_valid_combination(self, total_tokens, depth, top_k):
        """Check if parameter combination satisfies constraint: total_tokens <= top_k^(depth-1)"""
        max_possible_tokens = top_k ** (depth - 1)
        return total_tokens <= max_possible_tokens
    
    def _safe_clamp_params(self, total_tokens, depth, top_k):
        """Safely clamp total_tokens to satisfy constraint as backup safety measure"""
        max_possible_tokens = top_k ** (depth - 1)
        if total_tokens > max_possible_tokens:
            clamped_total_tokens = max_possible_tokens
            print(f"⚠️  Safety clamp: {total_tokens} → {clamped_total_tokens} (depth={depth}, top_k={top_k})")
            return clamped_total_tokens, depth, top_k
        return total_tokens, depth, top_k


def calculate_online_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """
    Calculate reward for online learning
    Focuses on speed (tokens/second) with penalties for extreme parameters
    """
    if generation_time <= 0 or new_tokens <= 0:
        return -1.0
    
    # Primary reward: tokens per second (normalized)
    tokens_per_second = new_tokens / generation_time
    speed_reward = min(tokens_per_second / 100.0, 1.0)  # Normalize to [0,1]
    
    # Efficiency penalty for using too many resources unnecessarily
    resource_penalty = 0.0
    if total_tokens > 65:  # Penalize excessive token budgets
        resource_penalty += 0.1
    if depth > 5:  # Penalize excessive depth
        resource_penalty += 0.1
    if top_k > 10:  # Penalize excessive top_k
        resource_penalty += 0.1
    
    # Bonus for good performance
    if tokens_per_second > 50:  # Bonus for fast generation
        speed_reward += 0.2
    
    final_reward = speed_reward - resource_penalty
    return max(final_reward, -1.0)  # Clamp to reasonable range
