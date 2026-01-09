"""
Main Streaming Dense Video Captioning Model.

Integrates:
1. Video Encoder (ViT/V-JEPA)
2. Streaming Memory (K-Means)
3. Text Decoder (Transformer)
"""
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn

from ..config.default_config import StreamingDVCConfig
from .video_encoder import get_video_encoder
from .memory_module import StreamingMemory
from .text_decoder import TransformerTextDecoder


class StreamingDenseVideoCaptioning(nn.Module):
    """
    Streaming Dense Video Captioning Model.
    
    Processes video frames sequentially and maintains a constant-size memory.
    Can generate dense captions at any point in the stream.
    """
    
    def __init__(self, config: StreamingDVCConfig):
        super().__init__()
        self.config = config
        
        # 1. Video Encoder
        self.video_encoder = get_video_encoder(config.encoder)
        
        # Projection layer if encoder dim != decoder dim
        self.encoder_dim = config.encoder.embed_dim
        self.decoder_dim = config.decoder.hidden_size
        
        if self.encoder_dim != self.decoder_dim:
            self.visual_projection = nn.Linear(self.encoder_dim, self.decoder_dim)
        else:
            self.visual_projection = nn.Identity()
            
        # 2. Streaming Memory
        self.memory_module = StreamingMemory(config.memory)
        
        # 3. Text Decoder
        if config.decoder.decoder_type == "t5":
            from .text_decoder import Vid2SeqDecoder
            self.text_decoder = Vid2SeqDecoder(config.decoder)
            self.decoder_dim = self.text_decoder.hidden_size
        else:
            from .text_decoder import TransformerTextDecoder
            self.text_decoder = TransformerTextDecoder(config.decoder)
            self.decoder_dim = config.decoder.hidden_size
            
        # Update projection if needed (re-check dimensions)
        if self.encoder_dim != self.decoder_dim:
            self.visual_projection = nn.Linear(self.encoder_dim, self.decoder_dim)
        else:
            self.visual_projection = nn.Identity()
        
        # Training settings
        self.num_dense_outputs = config.training.num_dense_outputs
        
    def forward(
        self,
        video: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        checkpoint_indices: Optional[torch.Tensor] = None,
        context_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.
        
        Args:
            video: (B, T, C, H, W) video frames
            text_tokens: (B, num_dense_outputs, L) ground truth captions
            checkpoint_indices: (B, num_dense_outputs) indices of frames to predict at
            context_tokens: (B, num_dense_outputs, C) previous captions as context
            
        Returns:
            dict with 'loss' and 'logits'
        """
        B, T = video.shape[:2]
        
        # 1. Encode Video
        # (B, T*N, D)
        visual_features = self.video_encoder(video)
        
        # Reshape to (B, T, N, D) for memory processing
        N = self.video_encoder.num_tokens_per_frame
        visual_features = visual_features.reshape(B, T, N, -1)
        
        # Project to decoder dimension
        visual_features = self.visual_projection(visual_features)
        
        # 2. Streaming Memory Processing
        # We need memory states at specific checkpoints for dense supervision
        # memory_states: (B, T, buffer_size, D)
        memory_states = self.memory_module(
            visual_features, 
            return_per_frame=True
        )
        
        # 3. Select Memory States for Decoding
        if checkpoint_indices is None:
            # Default: evenly spaced checkpoints
            # e.g. if T=64, num_dense_outputs=16, then indices=[3, 7, 11, ...]
            stride = T // self.num_dense_outputs
            checkpoint_indices = torch.arange(
                stride - 1, T, stride, device=video.device
            ).unsqueeze(0).expand(B, -1)
            
        # Gather memory at checkpoints
        # (B, num_dense_outputs, buffer_size, D)
        batch_indices = torch.arange(B, device=video.device).unsqueeze(1).expand(-1, self.num_dense_outputs)
        selected_memory = memory_states[batch_indices, checkpoint_indices]
        
        # 4. Decode Text
        # We treat each checkpoint as a separate sample for the decoder
        # Flatten batch and checkpoints: (B * num_dense_outputs, buffer_size, D)
        flat_memory = selected_memory.reshape(-1, self.config.memory.buffer_size, self.decoder_dim)
        
        if text_tokens is not None:
            # Flatten text tokens: (B * num_dense_outputs, L)
            flat_text = text_tokens.reshape(-1, text_tokens.shape[-1])
            
            # Flatten context tokens if provided
            flat_context = None
            if context_tokens is not None:
                flat_context = context_tokens.reshape(-1, context_tokens.shape[-1])
            
            outputs = self.text_decoder(
                input_ids=flat_text,
                encoder_hidden_states=flat_memory,
                labels=flat_text,
                context_input_ids=flat_context
            )
            
            # Reshape output for clarity
            outputs["logits"] = outputs["logits"].reshape(
                B, self.num_dense_outputs, -1, self.text_decoder.total_vocab_size
            )
            
            return outputs
            
        return {"memory_states": memory_states}

    @torch.no_grad()
    def stream_inference(
        self,
        video_stream_iterator,
        generate_at_indices: Optional[List[int]] = None,
    ):
        """
        Simulate streaming inference.
        
        Args:
            video_stream_iterator: yields (1, C, H, W) frames
            generate_at_indices: list of frame indices to generate captions at
        """
        # Initialize memory
        # This requires a bit of care since our memory module expects batch mode
        # We'll handle state manually or adapt the module
        pass 
        # TODO: Implement true streaming inference loop
