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

class OnlineTreePolicy:
    """Online RL Policy that learns tree parameters in real-time during evaluation"""
    
    def __init__(self, 
                 learning_rate=3e-4,
                 epsilon_start=0.9,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 memory_size=1000,
                 batch_size=32,
                 target_update_freq=100):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parameter bins for discrete actions
        self.total_tokens_bins = [55, 60, 65, 70]
        self.depth_bins = [4, 5, 6]
        self.top_k_bins = [8, 10, 12]
        
        # Action space dimensions
        self.n_total_tokens = len(self.total_tokens_bins)
        self.n_depth = len(self.depth_bins)
        self.n_top_k = len(self.top_k_bins)
        self.total_actions = self.n_total_tokens * self.n_depth * self.n_top_k
        
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
        
        # Performance tracking
        self.reward_history = []
        self.parameter_history = []
        
        print(f"Online RL Policy initialized:")
        print(f"  - State dim: {self.state_dim}")
        print(f"  - Action space: {self.total_actions} combinations")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Device: {self.device}")
    
    def _build_network(self):
        """Build Q-network architecture"""
        return nn.Sequential(
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
        """Predict parameters using epsilon-greedy policy"""
        state = self._encode_state(context)
        
        # Epsilon-greedy action selection
        if training_mode and random.random() < self.epsilon:
            # Exploration: random action
            action = random.randint(0, self.total_actions - 1)
        else:
            # Exploitation: best action from Q-network
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                action = q_values.argmax().item()
        
        # Convert action to parameters
        total_tokens, depth, top_k = self._action_to_params(action)
        
        # Store state-action for learning
        if training_mode:
            self.last_state = state
            self.last_action = action
            self.last_params = (total_tokens, depth, top_k)
        
        return total_tokens, depth, top_k
    
    def update_policy(self, reward):
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
        
        # Learn from experience if we have enough samples
        if len(self.memory) >= self.batch_size:
            self._learn_from_batch()
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"Step {self.step_count}: Updated target network, epsilon={self.epsilon:.3f}")
    
    def _learn_from_batch(self):
        """Learn from a batch of experiences using DQN"""
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert states to numpy array first, then to tensor (more efficient)
        state_data = np.array([exp['state'].detach().numpy() for exp in batch])
        states = torch.FloatTensor(state_data).to(self.device)  # No requires_grad needed for inputs
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        
        # Current Q-values (Q-network parameters will have gradients automatically)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values (detached, no gradients needed)
        with torch.no_grad():
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
            print(f"Step {self.step_count}: Loss={loss.item():.4f}, Avg Reward={avg_reward:.4f}")
    
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
        """Save the trained policy"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'total_tokens_bins': self.total_tokens_bins,
            'depth_bins': self.depth_bins,
            'top_k_bins': self.top_k_bins
        }, path)
        print(f"Online policy saved to {path}")
    
    def load(self, path):
        """Load a trained policy"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.reward_history = checkpoint['reward_history']
            self.parameter_history = checkpoint['parameter_history']
            print(f"Online policy loaded from {path}")
            return True
        return False


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
