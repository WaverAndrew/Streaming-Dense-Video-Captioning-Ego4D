"""
Streaming Memory Module for Dense Video Captioning.

Implements the core innovation from the paper: K-Means based memory
that maintains a fixed-size buffer of token centroids, enabling
processing of arbitrarily long videos with constant memory.

Streaming Methods:
- kmeans: K-Means clustering (recommended, matches paper)
- ema: Exponential Moving Average
- temporal_mean_pool: Mean pooling over time
- adjacent_tome: Token merging (ToMe-style)
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.default_config import MemoryConfig


def kmeans_iteration(
    centers: torch.Tensor,
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single K-Means iteration with weighted updates.
    
    Args:
        centers: (B, K, D) current cluster centers
        data: (B, N, D) data points to cluster
        weights: (B, N) optional weights for each data point
    
    Returns:
        new_centers: (B, K, D) updated cluster centers
        new_counts: (B, K) count of points per cluster
    """
    B, K, D = centers.shape
    N = data.shape[1]
    
    if weights is None:
        weights = torch.ones(B, N, device=data.device, dtype=data.dtype)
    
    # Compute distances: (B, N, K)
    # Using einsum for efficiency
    dists = torch.cdist(data, centers, p=2)  # (B, N, K)
    
    # Assign points to nearest center
    assignments = dists.argmin(dim=-1)  # (B, N)
    
    # One-hot encoding of assignments
    one_hot = F.one_hot(assignments, num_classes=K).float()  # (B, N, K)
    
    # Weight the one-hot assignments
    weighted_one_hot = one_hot * weights.unsqueeze(-1)  # (B, N, K)
    
    # Sum weighted data points per cluster
    # (B, K, N) @ (B, N, D) -> (B, K, D)
    weighted_data = weighted_one_hot.permute(0, 2, 1) @ data
    
    # Count points per cluster
    new_counts = weighted_one_hot.sum(dim=1)  # (B, K)
    
    # Compute new centers (avoid division by zero)
    new_centers = weighted_data / (new_counts.unsqueeze(-1) + 1e-8)
    
    # Keep old centers for empty clusters
    empty_mask = new_counts < 1e-6
    new_centers = torch.where(
        empty_mask.unsqueeze(-1).expand_as(new_centers),
        centers,
        new_centers,
    )
    
    return new_centers, new_counts


def kmeans(
    init_centers: torch.Tensor,
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    num_iters: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-Means clustering with fixed number of iterations.
    
    Args:
        init_centers: (B, K, D) initial cluster centers
        data: (B, N, D) data points to cluster
        weights: (B, N) optional weights for each data point
        num_iters: number of K-Means iterations
    
    Returns:
        centers: (B, K, D) final cluster centers
        counts: (B, K) final count of points per cluster
    """
    centers = init_centers
    counts = None
    
    for _ in range(num_iters):
        centers, counts = kmeans_iteration(centers, data, weights)
    
    return centers, counts


class KMeansMemory(nn.Module):
    """
    K-Means based streaming memory module.
    
    Maintains a fixed number of centroids (buffer_size) that summarize
    all observed frames. When new frames arrive, their tokens are merged
    into the memory using weighted K-Means updates.
    
    This is the core innovation of the Streaming DVC paper, enabling
    processing of arbitrarily long videos with constant memory.
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.buffer_size = config.buffer_size
        self.num_iters = config.kmeans_num_iters
        self.use_momentum = config.use_momentum
    
    def init_memory(
        self,
        visual_features: torch.Tensor,
        num_init_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize memory from the first few frames.
        
        Args:
            visual_features: (B, T, N, D) features from all frames
            num_init_frames: number of initial frames to use for init
        
        Returns:
            memory: (B, buffer_size, D) initial memory centroids
            counts: (B, buffer_size) initial counts (all ones)
        """
        B, T, N, D = visual_features.shape
        
        # Use first num_init_frames frames to initialize
        init_features = visual_features[:, :num_init_frames]  # (B, num_init, N, D)
        init_features = init_features.reshape(B, -1, D)  # (B, num_init * N, D)
        
        # Take first buffer_size tokens as initial centers
        memory = init_features[:, :self.buffer_size]  # (B, buffer_size, D)
        counts = torch.ones(B, self.buffer_size, device=memory.device)
        
        return memory, counts
    
    def update(
        self,
        memory: torch.Tensor,
        counts: torch.Tensor,
        new_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update memory with new frame tokens using K-Means.
        
        Args:
            memory: (B, buffer_size, D) current memory centroids
            counts: (B, buffer_size) count per centroid
            new_tokens: (B, N_new, D) tokens from new frame(s)
        
        Returns:
            updated_memory: (B, buffer_size, D) updated centroids
            updated_counts: (B, buffer_size) updated counts
        """
        B, K, D = memory.shape
        N_new = new_tokens.shape[1]
        
        # Concatenate memory and new tokens
        all_data = torch.cat([memory, new_tokens], dim=1)  # (B, K + N_new, D)
        
        # Create weights: memory points have their counts, new points have 1
        if self.use_momentum:
            new_weights = torch.ones(B, N_new, device=memory.device)
            weights = torch.cat([counts, new_weights], dim=1)  # (B, K + N_new)
        else:
            weights = torch.ones(B, K + N_new, device=memory.device)
        
        # Run K-Means to cluster back to buffer_size centroids
        updated_memory, updated_counts = kmeans(
            init_centers=memory,
            data=all_data,
            weights=weights,
            num_iters=self.num_iters,
        )
        
        return updated_memory, updated_counts
    
    def forward(
        self,
        visual_features: torch.Tensor,
        return_per_frame: bool = False,
    ) -> torch.Tensor:
        """
        Process all frames and build streaming memory.
        
        Args:
            visual_features: (B, T*N, D) flattened visual features
                or (B, T, N, D) per-frame features
            return_per_frame: if True, return memory state after each frame
        
        Returns:
            if return_per_frame:
                (B, T, buffer_size, D) memory states at each frame
            else:
                (B, buffer_size, D) final memory state
        """
        # Reshape if flattened
        if visual_features.dim() == 3:
            B, TN, D = visual_features.shape
            # Assume we need to infer T and N from buffer_size
            # This is a simplification; in practice, T should be passed
            raise ValueError(
                "Pass 4D tensor (B, T, N, D) or use process_streaming method"
            )
        
        B, T, N, D = visual_features.shape
        
        # Determine number of initial frames for memory init
        num_init_frames = self.buffer_size // N
        num_init_frames = max(1, min(num_init_frames, T))
        
        # Initialize memory
        memory, counts = self.init_memory(visual_features, num_init_frames)
        
        if return_per_frame:
            memory_states = [memory]
            
            # Process remaining frames one at a time
            for t in range(num_init_frames, T):
                frame_tokens = visual_features[:, t]  # (B, N, D)
                memory, counts = self.update(memory, counts, frame_tokens)
                memory_states.append(memory)
            
            # Pad initial frames with their init memory state
            memory_states = [memory_states[0]] * num_init_frames + memory_states[1:]
            return torch.stack(memory_states, dim=1)  # (B, T, buffer_size, D)
        else:
            # Just compute final memory state
            for t in range(num_init_frames, T):
                frame_tokens = visual_features[:, t]  # (B, N, D)
                memory, counts = self.update(memory, counts, frame_tokens)
            
            return memory  # (B, buffer_size, D)


class EMAMemory(nn.Module):
    """
    Exponential Moving Average memory module.
    
    Simpler alternative to K-Means that maintains a running average
    of visual tokens with decay.
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.decay = config.ema_decay
    
    def forward(
        self,
        visual_features: torch.Tensor,
        return_per_frame: bool = False,
    ) -> torch.Tensor:
        """
        Process frames with EMA.
        
        Args:
            visual_features: (B, T, N, D) per-frame features
        
        Returns:
            (B, N, D) or (B, T, N, D) memory states
        """
        B, T, N, D = visual_features.shape
        
        # Initialize with first frame
        memory = visual_features[:, 0]  # (B, N, D)
        
        if return_per_frame:
            memory_states = [memory]
            
            for t in range(1, T):
                memory = self.decay * memory + (1 - self.decay) * visual_features[:, t]
                memory_states.append(memory)
            
            return torch.stack(memory_states, dim=1)  # (B, T, N, D)
        else:
            for t in range(1, T):
                memory = self.decay * memory + (1 - self.decay) * visual_features[:, t]
            
            return memory  # (B, N, D)


class TemporalMeanPoolMemory(nn.Module):
    """Simple temporal mean pooling (baseline)."""
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        visual_features: torch.Tensor,
        return_per_frame: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, T, N, D) per-frame features
        
        Returns:
            (B, N, D) or (B, T, N, D) cumulative means
        """
        B, T, N, D = visual_features.shape
        
        if return_per_frame:
            # Cumulative mean at each frame
            cumsum = torch.cumsum(visual_features, dim=1)  # (B, T, N, D)
            counts = torch.arange(1, T + 1, device=visual_features.device).float()
            counts = counts.view(1, T, 1, 1)
            return cumsum / counts  # (B, T, N, D)
        else:
            return visual_features.mean(dim=1)  # (B, N, D)


class StreamingMemory(nn.Module):
    """
    Wrapper that selects the appropriate streaming memory method.
    
    Supported methods:
    - 'kmeans': K-Means clustering (paper default)
    - 'ema': Exponential Moving Average
    - 'temporal_mean_pool': Simple mean over time
    - 'none': No memory, just flatten
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.method = config.streaming_method
        
        if self.method == "kmeans":
            self.memory = KMeansMemory(config)
        elif self.method == "ema":
            self.memory = EMAMemory(config)
        elif self.method == "temporal_mean_pool":
            self.memory = TemporalMeanPoolMemory(config)
        elif self.method == "none":
            self.memory = None
        else:
            raise ValueError(f"Unknown streaming method: {self.method}")
    
    def forward(
        self,
        visual_features: torch.Tensor,
        return_per_frame: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, T, N, D) per-frame features
                or (B, T*N, D) flattened features
        
        Returns:
            Streaming memory features
        """
        if self.memory is None:
            # No streaming, just return flattened
            if visual_features.dim() == 4:
                B, T, N, D = visual_features.shape
                return visual_features.reshape(B, T * N, D)
            return visual_features
        
        return self.memory(visual_features, return_per_frame=return_per_frame)
