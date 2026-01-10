import torch
import tarfile
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from streaming_dvc.visualization.test_clustering_only import ClusteringTestPipeline

def load_features_from_tar(tar_path: str, max_clips: int = 4):
    """
    Load features from a tar file.
    Returns:
        features: (B, T_max, 1, D) tensor
        lengths: (B,) tensor of original lengths
    """
    features_list = []
    
    print(f"Loading features from {tar_path}...")
    with tarfile.open(tar_path, "r") as tar:
        count = 0
        for member in tar:
            if member.name.endswith(".pt"):
                f = tar.extractfile(member)
                if f:
                    try:
                        # Load tensor: (T, D)
                        feat = torch.load(f, map_location='cpu')
                        if isinstance(feat, torch.Tensor):
                            # Ensure (T, D)
                            if feat.dim() == 1:
                                feat = feat.unsqueeze(0)
                            
                            # Add spatial dim: (T, 1, D)
                            feat = feat.unsqueeze(1)
                            features_list.append(feat)
                            count += 1
                            if count >= max_clips:
                                break
                    except Exception as e:
                        print(f"Error loading {member.name}: {e}")
                        
    if not features_list:
        raise ValueError("No features found in tar file.")
        
    # Pad sequences
    # features_list is list of (T, 1, D)
    # pad_sequence expects (T, ...) so it works
    padded_features = pad_sequence(features_list, batch_first=True) # (B, T_max, 1, D)
    
    return padded_features

def main():
    tar_path = "streaming_dvc/data/dataset/egovlp_egonce"
    output_dir = "clustering_viz_egovlp"
    
    # Load data
    features = load_features_from_tar(tar_path, max_clips=10) # Load 10 clips
    print(f"Loaded batch shape: {features.shape}")
    
    # Initialize pipeline
    B, T, N, D = features.shape
    
    pipeline = ClusteringTestPipeline(
        buffer_size=64, # Reverted to 64 as requested
        feature_dim=D
    )
    
    # Run for each clip
    for i in range(B):
        clip_output_dir = os.path.join(output_dir, f"clip_{i}")
        print(f"Visualizing clip {i} to {clip_output_dir}...")
        # Pass single clip as batch of 1: (1, T, N, D)
        pipeline.run(features[i:i+1], output_dir=clip_output_dir)

if __name__ == "__main__":
    main()
