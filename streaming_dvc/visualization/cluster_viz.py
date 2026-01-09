import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any, Optional
import cv2
from PIL import Image

from streaming_dvc.models.streaming_model import StreamingDenseVideoCaptioning
from streaming_dvc.config.default_config import get_default_config

def visualize_memory_timeline(
    model: StreamingDenseVideoCaptioning,
    video_path: str,
    captions: List[Dict[str, Any]],
    output_path: str = "memory_timeline.png",
    num_checkpoints: int = 4,
    num_clusters_to_show: int = 5
):
    """
    Visualizes memory evolution and semantic meaning.
    
    Args:
        model: Trained model.
        video_path: Path to video file.
        captions: List of dicts with 'timestamp' and 'text'.
        output_path: Output image path.
        num_checkpoints: Number of temporal checkpoints to visualize.
        num_clusters_to_show: Number of top clusters to show per checkpoint.
    """
    # 1. Load Video and Sample Frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    # Sample frames for model input (e.g., 64 frames)
    model_num_frames = model.config.training.num_frames
    indices = np.linspace(0, total_frames - 1, model_num_frames).astype(int)
    
    frames = []
    original_frames = [] # Keep original for visualization
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(frame_rgb)
            # Preprocess for model (resize, normalize)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_resized)
    cap.release()
    
    video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() # (T, C, H, W)
    
    # 2. Run Model to get Memory States
    model.eval()
    with torch.no_grad():
        # (1, T, N, D)
        visual_features = model.video_encoder(video_tensor.unsqueeze(0))
        N = model.video_encoder.num_tokens_per_frame
        T = video_tensor.shape[0]
        visual_features = visual_features.reshape(1, T, N, -1)
        visual_features = model.visual_projection(visual_features)
        
        # Get memory states at all frames: (1, T, buffer_size, D)
        memory_states = model.memory_module(visual_features, return_per_frame=True)
        
        # Frame features for matching: (T, D)
        frame_features = visual_features.mean(dim=2).squeeze(0)
        frame_features_norm = torch.nn.functional.normalize(frame_features, dim=1)

    # 3. Setup Visualization
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(num_checkpoints + 1, 1, height_ratios=[1] + [4]*num_checkpoints)
    
    # --- Timeline Plot ---
    ax_timeline = plt.subplot(gs[0])
    ax_timeline.set_title(f"Video Timeline ({duration:.1f}s)", fontsize=14, fontweight='bold')
    ax_timeline.set_xlim(0, duration)
    ax_timeline.set_yticks([])
    
    # Plot captions on timeline
    colors = plt.cm.tab10(np.linspace(0, 1, len(captions)))
    for i, cap in enumerate(captions):
        ts = cap['timestamp']
        ax_timeline.axvline(x=ts, color=colors[i % 10], linestyle='--', alpha=0.7)
        ax_timeline.text(ts, 0.5, f" {cap['text']}", rotation=0, verticalalignment='center', fontsize=10)
        
    # --- Checkpoint Visualizations ---
    checkpoint_indices = np.linspace(T//num_checkpoints, T-1, num_checkpoints).astype(int)
    
    for i, cp_idx in enumerate(checkpoint_indices):
        cp_time = (cp_idx / T) * duration
        
        # Get memory at this checkpoint
        memory = memory_states[0, cp_idx] # (buffer_size, D)
        memory_norm = torch.nn.functional.normalize(memory, dim=1)
        
        # Find nearest frames for each memory token
        # (buffer_size, T_current) - only look at history up to cp_idx
        similarity = torch.mm(memory_norm, frame_features_norm[:cp_idx+1].t())
        best_frame_indices = torch.argmax(similarity, dim=1).cpu().numpy()
        
        # Select "interesting" clusters (e.g., most distinct frames)
        # For simplicity, we just pick unique frames or random ones
        unique_frames = np.unique(best_frame_indices)
        if len(unique_frames) > num_clusters_to_show:
            selected_frames = np.random.choice(unique_frames, num_clusters_to_show, replace=False)
        else:
            selected_frames = unique_frames
            
        # Create sub-grid for this checkpoint
        ax_cp_container = plt.subplot(gs[i+1])
        ax_cp_container.axis('off')
        ax_cp_container.set_title(f"Memory State at {cp_time:.1f}s (Frame {cp_idx})", fontsize=12, loc='left')
        
        # Inner grid for frames
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, len(selected_frames), subplot_spec=gs[i+1], wspace=0.1)
        
        for j, frame_idx in enumerate(selected_frames):
            ax_frame = plt.subplot(inner_gs[j])
            
            # Show frame
            ax_frame.imshow(original_frames[frame_idx])
            ax_frame.axis('off')
            
            # Find caption
            frame_time = (frame_idx / T) * duration
            caption_text = "No Caption"
            min_dist = float('inf')
            for cap in captions:
                dist = abs(cap['timestamp'] - frame_time)
                if dist < min_dist and dist < 5.0: # 5s window
                    min_dist = dist
                    caption_text = cap['text']
            
            # Color code border based on caption (concept)
            rect = plt.Rectangle((0,0), original_frames[frame_idx].shape[1], original_frames[frame_idx].shape[0], 
                                 linewidth=3, edgecolor=colors[j % 10], facecolor='none')
            ax_frame.add_patch(rect)
            
            # Add text
            ax_frame.set_title(f"Cluster {j}\n{caption_text}\n(t={frame_time:.1f}s)", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Dummy run
    print("This script requires a real video path to run fully.")
    print("Usage: visualize_memory_timeline(model, 'video.mp4', captions)")
