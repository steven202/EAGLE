"""
Optimized RL Policy for EAGLE with Layer Feature Concatenation and Reduced Action Frequency
Implements two key optimizations for faster RL inference:
1. Uses EAGLE-3's 3k-dimensional concatenated layer features as state instead of SBERT
2. Generates actions every N steps instead of every step (configurable)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


class OptimizedOnlineTreePolicy:
    """
    Optimized Online RL Policy for EAGLE with:
    1. Layer feature concatenation as state (no SBERT)
    2. Configurable action generation frequency
    """
    
    def __init__(self, 
                 learning_rate=0.001,
                 epsilon_start=0.9,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=32,
                 target_update_freq=10,
                 temperature=1.0,
                 entropy_weight=0.1,
                 inference_temperature=1.5,
                 max_entropy_inference=True,
                 action_generation_freq=10,  # NEW: Generate action every N steps
                 use_wandb=True,
                 wandb_project="eagle-optimized-rl",
                 wandb_run_name=None,
                 checkpoint_dir="optimized_rl_checkpoints",
                 checkpoint_freq=100,
                 max_checkpoints=3):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.inference_temperature = inference_temperature
        self.max_entropy_inference = max_entropy_inference
        
        # NEW: Action generation frequency optimization
        self.action_generation_freq = action_generation_freq
        self.step_counter = 0
        self.cached_action = None  # Cache last predicted action
        self.cached_params = None  # Cache last predicted parameters
        
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
                        "epsilon_start": epsilon_start,
                        "epsilon_end": epsilon_end,
                        "epsilon_decay": epsilon_decay,
                        "memory_size": memory_size,
                        "batch_size": batch_size,
                        "target_update_freq": target_update_freq,
                        "temperature": temperature,
                        "entropy_weight": entropy_weight,
                        "inference_temperature": inference_temperature,
                        "max_entropy_inference": max_entropy_inference,
                        "action_generation_freq": action_generation_freq,
                        "device": str(self.device)
                    }
                )
                print(f"🔗 Wandb logging initialized: {wandb.run.get_url()}")
            else:
                print(f"🔗 Using existing wandb run: {wandb.run.get_url()}")
        else:
            print("📊 Wandb logging disabled")
        
        # Parameter bins for discrete actions
        self.total_tokens_bins = [32, 48, 64, 80, 96, 128]
        self.top_k_bins = [8, 12, 16, 20, 32]
        self.depth_bins = [3, 4, 5, 6, 7, 8]
        
        # Action space dimensions
        self.n_total_tokens = len(self.total_tokens_bins)
        self.n_depth = len(self.depth_bins)
        self.n_top_k = len(self.top_k_bins)
        self.total_actions = self.n_total_tokens * self.n_depth * self.n_top_k
        
        # Precompute valid actions
        self.valid_actions = self._precompute_valid_actions()
        print(f"Action space: {self.total_actions} total, {len(self.valid_actions)} valid")
        
        # NEW: Use concatenated layer features as state instead of SBERT
        # EAGLE-3 concatenates 3 layers: layer 2, middle layer, last-3 layer
        # Each layer has hidden_size dimensions, so total is 3 * hidden_size
        # We'll detect the actual dimension dynamically from the first input
        self.state_dim = None  # Will be set dynamically (typically 3 * 4096 = 12288 for Llama)
        self.q_network = None  # Will be built after first state
        self.target_network = None
        self.optimizer = None
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training counters
        self.step_count = 0
        
        # Performance tracking
        self.reward_history = []
        self.parameter_history = []
        self.loss_history = []
        self.tokens_per_second_history = []
        self.entropy_history = []
        self.action_diversity_history = []
        
        print(f"✅ Optimized Online RL Policy initialized with:")
        print(f"   - Layer feature concatenation (no SBERT)")
        print(f"   - Action generation frequency: every {action_generation_freq} steps")
        print(f"   - Valid actions: {len(self.valid_actions)}/{self.total_actions}")
        print(f"   - Max-entropy inference: {max_entropy_inference}")
    
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
        return total_tokens <= top_k ** (depth - 1)
    
    def _build_network(self, state_dim):
        """Build Q-network architecture optimized for high-dimensional layer features"""
        # Optimized architecture for high-dimensional concatenated features
        network = nn.Sequential(
            # First layer: reduce high dimensionality
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Second layer: further processing
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third layer: feature refinement
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Fourth layer: final processing
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Output layer: action values
            nn.Linear(128, len(self.valid_actions))
        )
        
        # Optimized weight initialization for high-dimensional inputs
        for layer in network:
            if isinstance(layer, nn.Linear):
                # Use He initialization for ReLU networks
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)  # Small positive bias
        
        return network
    
    def _encode_state(self, layer_features):
        """
        Encode state using EAGLE-3's concatenated layer features instead of SBERT
        
        Args:
            layer_features: Concatenated features from EAGLE-3 layers (3k-dimensional)
                          Shape: [seq_len, 3 * hidden_size] or [batch, seq_len, 3 * hidden_size]
        
        Returns:
            torch.Tensor: State representation for RL policy
        """
        if layer_features is None:
            # Fallback: return zero state if no features provided
            if self.state_dim is None:
                # Assume default Llama hidden size
                self.state_dim = 3 * 4096
            return torch.zeros(self.state_dim, device=self.device)
        
        # Convert to tensor if numpy
        if isinstance(layer_features, np.ndarray):
            layer_features = torch.from_numpy(layer_features)
        
        # Move to device
        layer_features = layer_features.to(self.device)
        
        # Handle different input shapes
        if layer_features.dim() == 3:  # [batch, seq_len, features]
            # Take the last token's features
            layer_features = layer_features[0, -1, :]
        elif layer_features.dim() == 2:  # [seq_len, features]
            # Take the last token's features
            layer_features = layer_features[-1, :]
        elif layer_features.dim() == 1:  # [features]
            # Already the right shape
            pass
        else:
            raise ValueError(f"Unexpected layer_features shape: {layer_features.shape}")
        
        # Initialize network on first call
        if self.state_dim is None:
            self.state_dim = layer_features.shape[-1]
            print(f"🔧 Detected state dimension: {self.state_dim}")
            
            # Build networks
            self.q_network = self._build_network(self.state_dim).to(self.device)
            self.target_network = self._build_network(self.state_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Initialize optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            
            print(f"🏗️  Networks initialized for state dimension: {self.state_dim}")
        
        return layer_features.float()
    
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
        # Find closest bins
        total_tokens_idx = np.argmin([abs(x - total_tokens) for x in self.total_tokens_bins])
        depth_idx = np.argmin([abs(x - depth) for x in self.depth_bins])
        top_k_idx = np.argmin([abs(x - top_k) for x in self.top_k_bins])
        
        action = total_tokens_idx * (self.n_depth * self.n_top_k) + depth_idx * self.n_top_k + top_k_idx
        return action
    
    def predict_parameters(self, layer_features, training_mode=True):
        """
        Predict parameters with optimized action generation frequency
        
        Args:
            layer_features: EAGLE-3 concatenated layer features (3k-dimensional)
            training_mode: Whether in training mode
        
        Returns:
            tuple: (total_tokens, depth, top_k)
        """
        if self.q_network is None:
            # Initialize with dummy state to build network
            dummy_state = self._encode_state(layer_features)
        
        # NEW: Action generation frequency optimization
        self.step_counter += 1
        
        # Only generate new action if:
        # 1. No cached action exists, OR
        # 2. Step counter is divisible by action_generation_freq
        if self.cached_params is None or (self.step_counter % self.action_generation_freq) == 0:
            # Generate new action
            state = self._encode_state(layer_features)
            
            if training_mode:
                # Training mode with exploration
                if self.max_entropy_inference:
                    action = self._sample_action_with_temperature(state, self.inference_temperature)
                else:
                    # Epsilon-greedy exploration
                    if random.random() < self.epsilon:
                        action = random.choice(self.valid_actions)
                    else:
                        with torch.no_grad():
                            q_values = self.q_network(state.unsqueeze(0))
                            valid_q_values = q_values[0]
                            action_idx = torch.argmax(valid_q_values).item()
                            action = self.valid_actions[action_idx]
            else:
                # Inference mode: deterministic or temperature-based
                if self.max_entropy_inference:
                    action = self._sample_action_with_temperature(state, self.inference_temperature)
                else:
                    with torch.no_grad():
                        q_values = self.q_network(state.unsqueeze(0))
                        valid_q_values = q_values[0]
                        action_idx = torch.argmax(valid_q_values).item()
                        action = self.valid_actions[action_idx]
            
            # Convert action to parameters and cache
            self.cached_action = action
            self.cached_params = self._action_to_params(action)
            
            if training_mode:
                # Store current state for potential policy update
                self.current_state = state
                self.current_action = action
            
            # Decay epsilon for training
            if training_mode and self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            
            print(f"🎯 Generated new action at step {self.step_counter}: {self.cached_params}")
        else:
            # Use cached action
            print(f"♻️  Using cached action at step {self.step_counter}: {self.cached_params}")
        
        return self.cached_params
    
    def _sample_action_with_temperature(self, state, temperature):
        """Sample action using temperature-based softmax for max-entropy exploration"""
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            valid_q_values = q_values[0]
            
            # Apply temperature scaling
            scaled_logits = valid_q_values / temperature
            
            # Softmax sampling
            probs = torch.softmax(scaled_logits, dim=0)
            action_idx = torch.multinomial(probs, 1).item()
            
            return self.valid_actions[action_idx]
    
    def update_policy(self, reward, generation_time=None, new_tokens=None):
        """Update policy with reward from last action including entropy regularization"""
        if not hasattr(self, 'current_state') or not hasattr(self, 'current_action'):
            return  # No previous experience to update
        
        # Store experience in replay buffer
        # Note: We store the index in valid_actions, not the raw action
        action_idx = self.valid_actions.index(self.current_action)
        self.memory.append((
            self.current_state,
            action_idx,
            reward,
            None,  # next_state (will be set when we have it)
            False  # done
        ))
        
        # Performance tracking
        self.reward_history.append(reward)
        total_tokens, depth, top_k = self._action_to_params(self.current_action)
        self.parameter_history.append([total_tokens, depth, top_k])
        
        if generation_time is not None and new_tokens is not None:
            tokens_per_second = new_tokens / max(generation_time, 0.001)
            self.tokens_per_second_history.append(tokens_per_second)
        
        # Calculate entropy for max-entropy RL
        if hasattr(self, 'current_state'):
            with torch.no_grad():
                q_values = self.q_network(self.current_state.unsqueeze(0))
                probs = torch.softmax(q_values[0], dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                self.entropy_history.append(entropy)
        
        # Log to wandb
        if self.use_wandb:
            log_data = {
                "reward": reward,
                "epsilon": self.epsilon,
                "step_count": self.step_count,
                "total_tokens": total_tokens,
                "depth": depth,
                "top_k": top_k,
                "action_generation_step": self.step_counter
            }
            
            if generation_time is not None:
                log_data["generation_time"] = generation_time
            if new_tokens is not None:
                log_data["new_tokens"] = new_tokens
                if generation_time is not None:
                    log_data["tokens_per_second"] = new_tokens / max(generation_time, 0.001)
            
            if len(self.entropy_history) > 0:
                log_data["entropy"] = self.entropy_history[-1]
            
            wandb.log(log_data)
        
        # Perform learning update if we have enough experience
        if len(self.memory) >= self.batch_size:
            self._dqn_update()
        
        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.step_count += 1
    
    def _dqn_update(self):
        """Perform DQN update with entropy regularization for max-entropy RL"""
        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch], device=self.device)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32, device=self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (simplified for now - using rewards directly)
        target_q_values = rewards.unsqueeze(1)
        
        # Compute loss with entropy regularization
        td_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Add entropy regularization for max-entropy RL
        q_values_all = self.q_network(states)
        probs = torch.softmax(q_values_all, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        
        # Total loss with entropy bonus
        loss = td_loss - self.entropy_weight * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Track loss
        self.loss_history.append(loss.item())
        
        if self.use_wandb:
            wandb.log({
                "loss": loss.item(),
                "td_loss": td_loss.item(),
                "entropy_regularization": entropy.item(),
            })
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save training checkpoint with automatic cleanup of old checkpoints"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.step_count}_q_{self.questions_processed}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        if self.q_network is None:
            print("⚠️  Warning: Cannot save checkpoint - networks not initialized")
            return
        
        checkpoint_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'step_counter': self.step_counter,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'loss_history': self.loss_history,
            'tokens_per_second_history': self.tokens_per_second_history,
            'entropy_history': self.entropy_history,
            'action_diversity_history': self.action_diversity_history,
            'total_tokens_bins': self.total_tokens_bins,
            'depth_bins': self.depth_bins,
            'top_k_bins': self.top_k_bins,
            'valid_actions': self.valid_actions,
            'state_dim': self.state_dim,
            'action_generation_freq': self.action_generation_freq,
            'cached_params': self.cached_params
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({"checkpoint_step": self.step_count})
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load training checkpoint and resume training"""
        if checkpoint_path is None:
            # Find most recent checkpoint
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
            if not checkpoints:
                print("⚠️  No checkpoints found for resuming")
                return False
            
            # Sort by modification time
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)))
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoints[-1])
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            
            # Restore hyperparameters
            self.total_tokens_bins = checkpoint_data.get('total_tokens_bins', self.total_tokens_bins)
            self.depth_bins = checkpoint_data.get('depth_bins', self.depth_bins)
            self.top_k_bins = checkpoint_data.get('top_k_bins', self.top_k_bins)
            self.valid_actions = checkpoint_data.get('valid_actions', self.valid_actions)
            self.state_dim = checkpoint_data.get('state_dim', self.state_dim)
            self.action_generation_freq = checkpoint_data.get('action_generation_freq', self.action_generation_freq)
            self.cached_params = checkpoint_data.get('cached_params', self.cached_params)
            
            # Build networks if not already built
            if self.q_network is None and self.state_dim is not None:
                self.q_network = self._build_network(self.state_dim).to(self.device)
                self.target_network = self._build_network(self.state_dim).to(self.device)
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            
            if self.q_network is not None:
                # Load network states
                self.q_network.load_state_dict(checkpoint_data['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint_data['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Restore training state
            self.epsilon = checkpoint_data.get('epsilon', self.epsilon)
            self.step_count = checkpoint_data.get('step_count', 0)
            self.step_counter = checkpoint_data.get('step_counter', 0)
            self.questions_processed = checkpoint_data.get('questions_processed', 0)
            self.training_seed = checkpoint_data.get('training_seed', None)
            
            # Restore history
            self.reward_history = checkpoint_data.get('reward_history', [])
            self.parameter_history = checkpoint_data.get('parameter_history', [])
            self.loss_history = checkpoint_data.get('loss_history', [])
            self.tokens_per_second_history = checkpoint_data.get('tokens_per_second_history', [])
            self.entropy_history = checkpoint_data.get('entropy_history', [])
            self.action_diversity_history = checkpoint_data.get('action_diversity_history', [])
            
            print(f"✅ Checkpoint loaded: {checkpoint_path}")
            print(f"   Resumed at step {self.step_count}, questions processed: {self.questions_processed}")
            print(f"   Action generation frequency: {self.action_generation_freq}")
            print(f"   Cached params: {self.cached_params}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint from {checkpoint_path}: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        if len(checkpoints) > self.max_checkpoints:
            # Sort by modification time
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)))
            
            # Remove oldest checkpoints
            for old_checkpoint in checkpoints[:-self.max_checkpoints]:
                old_path = os.path.join(self.checkpoint_dir, old_checkpoint)
                os.remove(old_path)
                print(f"🗑️  Removed old checkpoint: {old_checkpoint}")
    
    def should_save_checkpoint(self):
        """Check if it's time to save a checkpoint"""
        return self.questions_processed % self.checkpoint_freq == 0
    
    def increment_questions_processed(self, count=1):
        """Track processed questions for resume capability"""
        self.questions_processed += count
        
        # Auto-save checkpoint if needed
        if self.should_save_checkpoint():
            self.save_checkpoint()
    
    def save(self, path):
        """Save the trained policy (final save)"""
        if self.q_network is None:
            print("⚠️  Warning: Cannot save policy - networks not initialized")
            return
        
        final_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'step_counter': self.step_counter,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'loss_history': self.loss_history,
            'tokens_per_second_history': self.tokens_per_second_history,
            'entropy_history': self.entropy_history,
            'action_diversity_history': self.action_diversity_history,
            'total_tokens_bins': self.total_tokens_bins,
            'depth_bins': self.depth_bins,
            'top_k_bins': self.top_k_bins,
            'valid_actions': self.valid_actions,
            'state_dim': self.state_dim,
            'action_generation_freq': self.action_generation_freq,
            'cached_params': self.cached_params,
            
            # Save policy configuration for reproducibility
            'config': {
                'learning_rate': self.learning_rate,
                'epsilon_start': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'temperature': self.temperature,
                'entropy_weight': self.entropy_weight,
                'inference_temperature': self.inference_temperature,
                'max_entropy_inference': self.max_entropy_inference,
                'action_generation_freq': self.action_generation_freq
            }
        }
        
        torch.save(final_data, path)
        print(f"💾 Final policy saved: {path}")
        
        # Upload to wandb if available
        if self.use_wandb:
            wandb.save(path)
            wandb.log({
                "final_step_count": self.step_count,
                "final_questions_processed": self.questions_processed,
                "avg_reward": np.mean(self.reward_history) if self.reward_history else 0,
                "avg_tokens_per_second": np.mean(self.tokens_per_second_history) if self.tokens_per_second_history else 0
            })
    
    def load(self, load_path):
        """Load a trained policy"""
        try:
            data = torch.load(load_path, map_location=self.device)
            
            # Load configuration if available
            if 'config' in data:
                config = data['config']
                self.action_generation_freq = config.get('action_generation_freq', self.action_generation_freq)
                print(f"📋 Loaded configuration with action generation frequency: {self.action_generation_freq}")
            
            # Restore hyperparameters
            self.total_tokens_bins = data.get('total_tokens_bins', self.total_tokens_bins)
            self.depth_bins = data.get('depth_bins', self.depth_bins)
            self.top_k_bins = data.get('top_k_bins', self.top_k_bins)
            self.valid_actions = data.get('valid_actions', self.valid_actions)
            self.state_dim = data.get('state_dim', self.state_dim)
            self.cached_params = data.get('cached_params', self.cached_params)
            
            # Build networks if state_dim is known
            if self.state_dim is not None:
                self.q_network = self._build_network(self.state_dim).to(self.device)
                self.target_network = self._build_network(self.state_dim).to(self.device)
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
                
                # Load weights
                self.q_network.load_state_dict(data['q_network_state_dict'])
                self.target_network.load_state_dict(data['target_network_state_dict'])
                
                print(f"✅ Optimized RL policy loaded from {load_path}")
                print(f"   State dimension: {self.state_dim}")
                print(f"   Action generation frequency: {self.action_generation_freq}")
                print(f"   Valid actions: {len(self.valid_actions)}")
                return True
            else:
                print(f"⚠️  Policy loaded but networks will be built on first use")
                return True
                
        except Exception as e:
            print(f"❌ Failed to load optimized RL policy from {load_path}: {e}")
            return False


def calculate_optimized_online_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """
    Calculate reward for the optimized online RL policy.
    Same as the original but with emphasis on efficiency gains from optimizations.
    """
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Primary metrics
    tokens_per_second = new_tokens / generation_time
    
    # Reward components
    speed_reward = np.log(tokens_per_second + 1) * 3.0  # Increased weight for speed
    efficiency_bonus = 2.0 if tokens_per_second > 15 else 0.0  # Bonus for high efficiency
    
    # Parameter efficiency penalty (encourage simpler configurations)
    complexity_penalty = (total_tokens + depth * 5 + top_k) * 0.01
    
    # Optimization bonus: reward faster inference from reduced action generation
    optimization_bonus = 1.0  # Constant bonus for using optimized policy
    
    total_reward = speed_reward + efficiency_bonus - complexity_penalty + optimization_bonus
    
    return max(0.0, total_reward)  # Ensure non-negative reward
