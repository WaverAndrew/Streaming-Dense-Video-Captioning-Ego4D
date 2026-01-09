"""
Ego4D Dataset Adapter for Streaming Dense Video Captioning.

Converts Ego4D narrations into dense captioning format with estimated durations.
"""
import json
import os
import random
from typing import List, Dict, Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ..config.default_config import DataConfig, TrainingConfig


class Ego4DDenseCapDataset(Dataset):
    """
    Ego4D dataset for dense video captioning.
    
    Reads pre-processed pilot dataset (JSON) containing clips and narrations.
    Each item is a video clip with multiple narration events.
    """
    
    def __init__(
        self,
        config: DataConfig,
        training_config: TrainingConfig,
        split: str = "train",
        transform=None,
    ):
        self.config = config
        self.training_config = training_config
        self.split = split
        self.transform = transform
        
        # Load pilot dataset
        pilot_path = os.path.join(config.data_root, "pilot_dataset.json")
        if not os.path.exists(pilot_path):
            # Fallback to local file if data_root is not set correctly
            pilot_path = "pilot_dataset.json"
            
        if not os.path.exists(pilot_path):
            raise FileNotFoundError(f"Could not find {pilot_path}. Run pre_process_ego4d.py first.")
            
        with open(pilot_path, "r") as f:
            self.data = json.load(f)
            
        # Group by clip_uid to form dense captioning samples
        self.clips = {}
        for item in self.data:
            clip_uid = item["clip_uid"]
            if clip_uid not in self.clips:
                self.clips[clip_uid] = {
                    "video_path": item["video_file"],
                    "narrations": []
                }
            self.clips[clip_uid]["narrations"].append({
                "text": item["text"],
                "timestamp": item["timestamp_sec"]
            })
            
        self.clip_ids = list(self.clips.keys())
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Special tokens for time bins
        # We add them to the tokenizer or handle them manually
        # Here we'll handle them manually in the collate function
        
    def __len__(self):
        return len(self.clip_ids)
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load and sample video frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Return dummy tensor if video fails
            return torch.zeros(self.training_config.num_frames, 3, self.config.crop_size, self.config.crop_size)
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        indices = torch.linspace(0, total_frames - 1, self.training_config.num_frames).long()
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.config.crop_size, self.config.crop_size))
                frames.append(frame)
            else:
                # Pad if read fails
                frames.append(np.zeros((self.config.crop_size, self.config.crop_size, 3), dtype=np.uint8))
                
        cap.release()
        
        # Stack and normalize: (T, H, W, C) -> (T, C, H, W)
        video = torch.from_numpy(np.stack(frames))
        video = video.permute(0, 3, 1, 2).float()
        # Note: Normalization usually happens in the encoder or transform
        
        return video

    def _quantize_time(self, timestamp: float, duration: float) -> int:
        """Quantize timestamp to bin index."""
        if duration <= 0: return 0
        normalized = timestamp / duration
        bin_idx = int(normalized * 100)  # 100 bins
        return max(0, min(99, bin_idx))

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        clip_data = self.clips[clip_id]
        
        # 1. Load Video
        video = self._load_video(clip_data["video_path"])
        
        # 2. Process Narrations
        # We need to align narrations to the intermediate checkpoints
        # For simplicity in this first version, we'll just take the narrations
        # and assign them to the nearest checkpoint
        
        narrations = clip_data["narrations"]
        # Sort by timestamp
        narrations.sort(key=lambda x: x["timestamp"])
        
        # Get video duration (approximate from last timestamp + buffer if not known)
        # Ideally we'd read this from metadata
        duration = max(n["timestamp"] for n in narrations) + 5.0
        
        # Prepare text targets for each dense output
        num_outputs = self.training_config.num_dense_outputs
        captions_per_output = [""] * num_outputs
        
        # Assign narrations to time segments
        # Each output k covers time range [0, k * duration / num_outputs]
        # But in "streaming" mode, output k is responsible for events ending in its specific window?
        # The paper says: "Each intermediate checkpoint is in charge of segments ending between last checkpoint to this checkpoint."
        
        segment_duration = duration / num_outputs
        
        for narr in narrations:
            ts = narr["timestamp"]
            # Find which segment this falls into
            segment_idx = int(ts / segment_duration)
            segment_idx = min(segment_idx, num_outputs - 1)
            
            # Append to that segment's caption
            # Format: [TIME_START] [TIME_END] Caption
            # For point events, we estimate a small window around the timestamp
            start_bin = self._quantize_time(max(0, ts - 2.0), duration)
            end_bin = self._quantize_time(ts + 2.0, duration)
            
            # We use special tokens for time: <time=N>
            # Since we haven't added them to tokenizer yet, we'll use placeholders
            # and handle conversion in collate or model
            time_str = f"<time={start_bin}> <time={end_bin}> "
            
            captions_per_output[segment_idx] += time_str + narr["text"] + " . "
            
        # Tokenize
        encoded = self.tokenizer(
            captions_per_output,
            padding="max_length",
            truncation=True,
            max_length=128, # Shorter than config max to save memory
            return_tensors="pt"
        )
        
        # Prepare context tokens (previous caption)
        # Shift captions by 1: context for segment k is caption of segment k-1
        context_tokens_list = []
        for i in range(num_outputs):
            if i == 0:
                context_text = "" # No context for first segment
            else:
                context_text = captions_per_output[i-1]
                
            # Randomly mask context during training
            if self.split == "train" and random.random() < self.training_config.context_mask_ratio:
                context_text = ""
                
            context_tokens_list.append(context_text)
            
        # Tokenize context
        encoded_context = self.tokenizer(
            context_tokens_list,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "video": video,
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "context_tokens": encoded_context.input_ids, # Pass context tokens
            "clip_id": clip_id
        }

import numpy as np
