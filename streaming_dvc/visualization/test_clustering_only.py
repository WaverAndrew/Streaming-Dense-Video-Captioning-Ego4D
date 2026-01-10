import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional, Tuple
import os

from streaming_dvc.models.memory_module import StreamingMemory
from streaming_dvc.config.default_config import MemoryConfig

class ClusteringTestPipeline:
    """
    Pipeline to test Streaming Memory clustering on extracted features.
    """
    
    def __init__(self, buffer_size: int = 64, feature_dim: int = 768):
        self.config = MemoryConfig(
            buffer_size=buffer_size,
            kmeans_num_iters=2,
            streaming_method="kmeans"
        )
        self.memory_module = StreamingMemory(self.config)
        self.feature_dim = feature_dim
        
    def run(self, features: torch.Tensor, output_dir: str = "clustering_viz"):
        """
        Run clustering on features and visualize.
        
        Args:
            features: (B, T, N, D) or (B, T, D) tensor of extracted features.
            output_dir: Directory to save visualizations.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle input shape
        if features.dim() == 3:
            # (B, T, D) -> (B, T, 1, D)
            features = features.unsqueeze(2)
            
        B, T, N, D = features.shape
        assert D == self.feature_dim, f"Feature dim mismatch: got {D}, expected {self.feature_dim}"
        
        print(f"Processing {B} clips of length {T} with {N} tokens/frame...")
        
        # Run Memory Module
        # We want per-frame states to visualize evolution
        # memory_states: (B, T, buffer_size, D)
        with torch.no_grad():
            memory_states = self.memory_module(features, return_per_frame=True)
            
        # Visualize for the first clip in batch
        self.visualize_pca_evolution(features[0], memory_states[0], os.path.join(output_dir, "pca_evolution.png"))
        self.visualize_assignment_timeline(features[0], memory_states[0], os.path.join(output_dir, "timeline.png"))
        
        print(f"Visualizations saved to {output_dir}")
        
    def visualize_pca_evolution(self, features: torch.Tensor, memory_states: torch.Tensor, save_path: str):
        """
        Visualize how memory centers evolve in PCA space vs input features.
        """
        T, N, D = features.shape
        buffer_size = memory_states.shape[1]
        
        # Flatten features for PCA fitting: (T*N, D)
        flat_features = features.reshape(-1, D).cpu().numpy()
        
        # Fit PCA on all features to get a common space
        pca = PCA(n_components=2)
        pca.fit(flat_features)
        
        # Transform features
        features_2d = pca.transform(flat_features)
        
        # Select a few timestamps to visualize (e.g., Start, Middle, End)
        checkpoints = [0, T//2, T-1]
        
        fig, axes = plt.subplots(1, len(checkpoints), figsize=(15, 5))
        
        for i, t in enumerate(checkpoints):
            ax = axes[i]
            
            # Plot all features (grey background)
            ax.scatter(features_2d[:, 0], features_2d[:, 1], c='lightgrey', alpha=0.1, s=1, label='All Features')
            
            # Plot current frame features (blue)
            current_frame_feats = features[t].cpu().numpy()
            current_2d = pca.transform(current_frame_feats)
            ax.scatter(current_2d[:, 0], current_2d[:, 1], c='blue', alpha=0.5, s=10, label='Current Frame')
            
            # Plot memory centers at time t (red)
            mem_at_t = memory_states[t].cpu().numpy()
            mem_2d = pca.transform(mem_at_t)
            ax.scatter(mem_2d[:, 0], mem_2d[:, 1], c='red', marker='x', s=50, linewidth=2, label='Memory Centers')
            
            ax.set_title(f"Time t={t}")
            if i == 0:
                ax.legend()
                
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize_assignment_timeline(self, features: torch.Tensor, memory_states: torch.Tensor, save_path: str):
        """
        Visualize which memory cluster each frame is assigned to.
        """
        T, N, D = features.shape
        buffer_size = memory_states.shape[1]
        
        # For each frame, find the closest memory center
        # We use the memory state at time t (or t-1?) to assign frame t
        # Usually we assign frame t to the memory *after* update, or *before*?
        # Let's use the memory state at time t (which includes frame t info)
        
        assignments = []
        for t in range(T):
            # (N, D)
            frame_feats = features[t]
            # (buffer_size, D)
            mem_centers = memory_states[t]
            
            # Compute distances
            dists = torch.cdist(frame_feats, mem_centers) # (N, buffer_size)
            
            # Assign each token to closest center
            token_assignments = dists.argmin(dim=1).cpu().numpy() # (N,)
            assignments.append(token_assignments)
            
        assignments = np.array(assignments) # (T, N)
        
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        # We want to show distribution of assignments over time
        # X-axis: Time, Y-axis: Cluster ID
        
        # Count how many tokens assigned to each cluster at each time
        counts = np.zeros((buffer_size, T))
        for t in range(T):
            for k in assignments[t]:
                counts[k, t] += 1
                
        plt.imshow(counts, aspect='auto', cmap='viridis', interpolation='nearest')
        # plt.colorbar(label='Token Count') # Removed as requested
        plt.xlabel('Time (Frame Index)')
        plt.ylabel('Memory Cluster ID')
        plt.title('Cluster Assignments Over Time')
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    # Mock data example
    print("Running with mock data...")
    pipeline = ClusteringTestPipeline(buffer_size=64, feature_dim=128) # Reverted to 64
    # (B=1, T=100, N=4, D=128)
    mock_features = torch.randn(1, 100, 4, 128)
    # Make some temporal structure
    mock_features[:, 50:] += 2.0 
    
    pipeline.run(mock_features)
