"""
Additional RL Policy Optimizations
Advanced techniques for further optimizing RL policy inference.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import time

class BatchedRLPolicyOptimizer:
    """
    Optimizer that batches similar RL policy requests for better throughput.
    """
    
    def __init__(self, batch_size: int = 4, timeout: float = 0.01):
        self.batch_size = batch_size
        self.timeout = timeout
        self.request_queue = queue.Queue()
        self.response_dict = {}
        self.processing = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def predict_batch(self, states: List[str], rl_policy, training_mode: bool) -> List[Tuple[int, int, int]]:
        """Batch predict parameters for multiple states."""
        results = []
        for state in states:
            with torch.no_grad():
                result = rl_policy.predict_parameters(state, training_mode=training_mode)
                results.append(result)
        return results
    
    def predict_async(self, state: str, rl_policy, training_mode: bool, request_id: str):
        """Asynchronous prediction with batching."""
        # Add to queue
        self.request_queue.put((state, rl_policy, training_mode, request_id))
        
        # Process batch if enough requests or timeout
        if self.request_queue.qsize() >= self.batch_size or not self.processing:
            self._process_batch()
    
    def _process_batch(self):
        """Process a batch of requests."""
        if self.processing:
            return
            
        self.processing = True
        batch = []
        
        # Collect batch
        while len(batch) < self.batch_size and not self.request_queue.empty():
            try:
                item = self.request_queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                break
        
        if batch:
            # Process batch
            states = [item[0] for item in batch]
            rl_policy = batch[0][1]  # Assume same policy
            training_mode = batch[0][2]  # Assume same mode
            
            results = self.predict_batch(states, rl_policy, training_mode)
            
            # Store results
            for i, (_, _, _, request_id) in enumerate(batch):
                self.response_dict[request_id] = results[i]
        
        self.processing = False
    
    def get_result(self, request_id: str, timeout: float = 0.1) -> Optional[Tuple[int, int, int]]:
        """Get result for a request."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.response_dict:
                result = self.response_dict.pop(request_id)
                return result
            time.sleep(0.001)
        return None


class AdaptiveParameterPredictor:
    """
    Predicts parameters based on patterns in previous predictions.
    """
    
    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self.parameter_history = []
        self.context_features = []
        
    def extract_features(self, context: str) -> np.ndarray:
        """Extract simple features from context for pattern recognition."""
        features = np.zeros(10)  # Simple feature vector
        
        # Length-based features
        features[0] = len(context) / 1000.0  # Normalized length
        features[1] = len(context.split()) / 100.0  # Word count
        
        # Content-based features
        features[2] = context.count('?') / max(1, len(context))  # Question ratio
        features[3] = context.count('.') / max(1, len(context))  # Period ratio
        features[4] = context.count(',') / max(1, len(context))  # Comma ratio
        
        # Character type ratios
        alpha_count = sum(1 for c in context if c.isalpha())
        digit_count = sum(1 for c in context if c.isdigit())
        features[5] = alpha_count / max(1, len(context))
        features[6] = digit_count / max(1, len(context))
        
        # Complexity indicators
        features[7] = len(set(context.split())) / max(1, len(context.split()))  # Vocabulary diversity
        features[8] = context.count('\n') / max(1, len(context))  # Line breaks
        features[9] = len(context.strip()) / max(1, len(context))  # Content density
        
        return features
    
    def add_observation(self, context: str, parameters: Tuple[int, int, int]):
        """Add a new observation to the history."""
        features = self.extract_features(context)
        
        self.context_features.append(features)
        self.parameter_history.append(parameters)
        
        # Maintain history size
        if len(self.parameter_history) > self.history_size:
            self.parameter_history = self.parameter_history[-self.history_size:]
            self.context_features = self.context_features[-self.history_size:]
    
    def predict_parameters(self, context: str) -> Optional[Tuple[int, int, int]]:
        """Predict parameters based on historical patterns."""
        if len(self.parameter_history) < 5:
            return None
        
        current_features = self.extract_features(context)
        
        # Simple similarity-based prediction
        similarities = []
        for hist_features in self.context_features:
            # Cosine similarity
            dot_product = np.dot(current_features, hist_features)
            norm_product = np.linalg.norm(current_features) * np.linalg.norm(hist_features)
            similarity = dot_product / max(norm_product, 1e-8)
            similarities.append(similarity)
        
        # Use parameters from most similar context
        if similarities:
            best_idx = np.argmax(similarities)
            if similarities[best_idx] > 0.8:  # High similarity threshold
                return self.parameter_history[best_idx]
        
        return None


class ContextAwareCache:
    """
    Cache that considers context similarity for better hit rates.
    """
    
    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.9):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache = {}  # context_hash -> (context, result, timestamp)
        self.context_embeddings = {}  # For similarity calculation
        
    def _compute_similarity(self, context1: str, context2: str) -> float:
        """Compute similarity between two contexts."""
        # Simple word-based similarity (can be enhanced with embeddings)
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union
    
    def get(self, context: str) -> Optional[Any]:
        """Get cached result for similar context."""
        context_hash = hash(context)
        
        # Direct hit
        if context_hash in self.cache:
            return self.cache[context_hash][1]
        
        # Similarity-based search
        for cached_hash, (cached_context, result, timestamp) in self.cache.items():
            similarity = self._compute_similarity(context, cached_context)
            if similarity >= self.similarity_threshold:
                return result
        
        return None
    
    def put(self, context: str, result: Any):
        """Store result in cache."""
        context_hash = hash(context)
        timestamp = time.time()
        
        # Manage cache size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_hash = min(self.cache.keys(), key=lambda k: self.cache[k][2])
            del self.cache[oldest_hash]
        
        self.cache[context_hash] = (context, result, timestamp)


class SmartFrequencyAdjuster:
    """
    Intelligently adjusts decode frequency based on multiple factors.
    """
    
    def __init__(self):
        self.decode_times = []
        self.policy_times = []
        self.cache_hit_rates = []
        self.current_frequency = 5
        
    def should_decode(self, step_idx: int, context_length: int, training_mode: bool) -> bool:
        """Smart decision on whether to decode at this step."""
        if training_mode:
            return True  # Always decode in training
        
        # Base frequency adjustment
        base_freq = self.current_frequency
        
        # Adjust based on context length (longer contexts change less frequently)
        if context_length > 1000:
            base_freq = int(base_freq * 1.5)
        elif context_length < 100:
            base_freq = max(1, int(base_freq * 0.7))
        
        # Early steps need more precision
        if step_idx < 10:
            base_freq = max(1, int(base_freq * 0.5))
        
        return step_idx % base_freq == 0
    
    def update_stats(self, decode_time: float, policy_time: float, cache_hit_rate: float):
        """Update statistics and adjust frequency."""
        self.decode_times.append(decode_time)
        self.policy_times.append(policy_time)
        self.cache_hit_rates.append(cache_hit_rate)
        
        # Keep recent history
        max_history = 20
        self.decode_times = self.decode_times[-max_history:]
        self.policy_times = self.policy_times[-max_history:]
        self.cache_hit_rates = self.cache_hit_rates[-max_history:]
        
        # Adjust frequency based on performance
        if len(self.decode_times) >= 10:
            avg_decode_time = sum(self.decode_times[-10:]) / 10
            avg_policy_time = sum(self.policy_times[-10:]) / 10
            avg_hit_rate = sum(self.cache_hit_rates[-10:]) / 10
            
            # If decode time is much larger than policy time, reduce frequency
            if avg_decode_time > avg_policy_time * 3 and avg_hit_rate < 0.5:
                self.current_frequency = min(self.current_frequency + 1, 15)
            elif avg_decode_time < avg_policy_time and avg_hit_rate > 0.8:
                self.current_frequency = max(self.current_frequency - 1, 2)


class ProfiledRLOptimizer:
    """
    RL optimizer with profiling and auto-tuning capabilities.
    """
    
    def __init__(self):
        self.profiler_data = {
            'decode_times': [],
            'policy_times': [],
            'cache_operations': [],
            'memory_usage': [],
            'step_counts': []
        }
        self.optimization_suggestions = []
        
    def profile_step(self, step_data: Dict[str, Any]):
        """Profile a single step and collect metrics."""
        for key, value in step_data.items():
            if key in self.profiler_data:
                self.profiler_data[key].append(value)
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze collected performance data and suggest optimizations."""
        if not self.profiler_data['decode_times']:
            return {}
        
        analysis = {}
        
        # Decode time analysis
        decode_times = self.profiler_data['decode_times']
        analysis['avg_decode_time'] = sum(decode_times) / len(decode_times)
        analysis['decode_time_variance'] = np.var(decode_times)
        
        # Policy time analysis
        policy_times = self.profiler_data['policy_times']
        if policy_times:
            analysis['avg_policy_time'] = sum(policy_times) / len(policy_times)
            analysis['policy_time_variance'] = np.var(policy_times)
            analysis['decode_to_policy_ratio'] = analysis['avg_decode_time'] / analysis['avg_policy_time']
        
        # Generate suggestions
        suggestions = []
        if analysis.get('decode_to_policy_ratio', 0) > 5:
            suggestions.append("Consider increasing decode frequency to reduce tokenization overhead")
        
        if analysis.get('decode_time_variance', 0) > analysis.get('avg_decode_time', 0):
            suggestions.append("High decode time variance - consider context-aware caching")
        
        analysis['suggestions'] = suggestions
        return analysis
    
    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal settings based on profiling data."""
        analysis = self.analyze_performance()
        
        settings = {
            'decode_frequency': 5,  # Default
            'cache_size': 200,      # Default
            'enable_parameter_caching': True
        }
        
        # Adjust based on analysis
        if analysis.get('decode_to_policy_ratio', 0) > 3:
            settings['decode_frequency'] = min(15, settings['decode_frequency'] * 2)
        
        if len(self.profiler_data.get('decode_times', [])) > 50:
            # More aggressive caching for longer runs
            settings['cache_size'] = min(500, settings['cache_size'] * 2)
        
        return settings
