"""
Video Encoder implementations for Streaming Dense Video Captioning.

Supports:
- GIT ViT-L encoder (original paper implementation)
- V-JEPA encoder (HuggingFace implementation)

Use --encoder flag to switch between them.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.default_config import EncoderConfig


# Pixel normalization constants (from GIT paper)
GIT_PIXEL_MEAN = (0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
GIT_PIXEL_STD = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GITViTEncoder(nn.Module):
    """
    GIT Vision Transformer encoder.
    
    Matches the original paper's ViT-L configuration:
    - embed_dim: 1024
    - depth: 24
    - num_heads: 16
    - patch_size: 14
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim)
        )
        
        # Pre-norm layer norm
        self.ln_pre = nn.LayerNorm(config.embed_dim, eps=1e-5)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
            )
            for _ in range(config.depth)
        ])
        
        # Post-norm layer norm
        self.ln_post = nn.LayerNorm(config.embed_dim, eps=1e-5)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor, pixel values in [0, 255]
        Returns:
            (B, num_patches + 1, embed_dim) visual features
        """
        B = x.shape[0]
        
        # Patch embed
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Pre-norm
        x = self.ln_pre(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Post-norm
        x = self.ln_post(x)
        
        return x
    
    @property
    def num_tokens_per_frame(self) -> int:
        """Number of output tokens per frame (including CLS)."""
        return self.patch_embed.num_patches + 1


class VideoGITViTEncoder(nn.Module):
    """
    Video encoder wrapping GIT ViT with temporal embedding support.
    
    Processes frames independently and adds temporal position embeddings.
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # Image encoder
        self.image_encoder = GITViTEncoder(config)
        
        # Pixel normalization
        self.register_buffer(
            "pixel_mean",
            torch.tensor(GIT_PIXEL_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(GIT_PIXEL_STD).view(1, 3, 1, 1),
        )
        
        # Temporal position embedding (will be lazily initialized)
        self.temporal_embed = None
        self.use_temporal_embedding = config.use_temporal_embedding
    
    def _get_temporal_embed(self, num_frames: int) -> torch.Tensor:
        """Get or create temporal position embedding."""
        if self.temporal_embed is None or self.temporal_embed.shape[0] != num_frames:
            self.temporal_embed = nn.Parameter(
                torch.zeros(num_frames, 1, 1, self.config.embed_dim,
                           device=self.pixel_mean.device),
            )
            nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        return self.temporal_embed
    
    def normalize_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values."""
        return (x - self.pixel_mean) / self.pixel_std
    
    def forward(
        self,
        video: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            video: (B, T, C, H, W) or (B, T, H, W, C) video tensor
            normalize: whether to normalize pixel values
        Returns:
            (B, T * num_patches_per_frame, embed_dim) visual features
        """
        # Handle different input formats
        if video.shape[-1] == 3:  # (B, T, H, W, C)
            video = video.permute(0, 1, 4, 2, 3)  # -> (B, T, C, H, W)
        
        B, T, C, H, W = video.shape
        
        # Flatten time to batch
        video_flat = video.reshape(B * T, C, H, W)
        
        # Normalize if needed
        if normalize:
            video_flat = self.normalize_pixels(video_flat)
        
        # Get visual features
        visual_features = self.image_encoder(video_flat)  # (B*T, N, D)
        
        # Reshape to (B, T, N, D)
        N, D = visual_features.shape[1], visual_features.shape[2]
        visual_features = visual_features.reshape(B, T, N, D)
        
        # Add temporal embedding
        if self.use_temporal_embedding:
            temp_embed = self._get_temporal_embed(T)
            visual_features = visual_features + temp_embed
        
        # Flatten temporal and spatial: (B, T*N, D)
        visual_features = visual_features.reshape(B, T * N, D)
        
        return visual_features
    
    @property
    def num_tokens_per_frame(self) -> int:
        return self.image_encoder.num_tokens_per_frame


class VJEPAVideoEncoder(nn.Module):
    """
    V-JEPA video encoder wrapper using HuggingFace transformers.
    
    This allows swapping in V-JEPA for experiments while keeping
    the same interface as the GIT ViT encoder.
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        try:
            from transformers import AutoVideoProcessor, AutoModel
            
            self.processor = AutoVideoProcessor.from_pretrained(
                config.vjepa_model_name
            )
            self.model = AutoModel.from_pretrained(
                config.vjepa_model_name
            )
            
            # Get feature dimension from model config
            self.feature_dim = self.model.config.hidden_size
            
        except ImportError:
            raise ImportError(
                "transformers library required for V-JEPA encoder. "
                "Install with: pip install transformers"
            )
    
    def forward(
        self,
        video: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            video: (B, T, C, H, W) or (B, T, H, W, C) video tensor
            normalize: preprocessing is handled by processor
        Returns:
            (B, num_tokens, embed_dim) visual features
        """
        # Handle different input formats
        if video.dim() == 5 and video.shape[-1] == 3:  # (B, T, H, W, C)
            # V-JEPA processor expects (T, H, W, C) numpy arrays per video
            pass
        
        # For simplicity, we assume input is already preprocessed
        # In practice, you'd use self.processor
        
        outputs = self.model(pixel_values_videos=video)
        
        if hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state
        else:
            features = outputs[0]
        
        return features
    
    @property
    def num_tokens_per_frame(self) -> int:
        # V-JEPA has different tokenization
        return self.model.config.num_patches_per_frame


def get_video_encoder(config: EncoderConfig) -> nn.Module:
    """
    Factory function to get video encoder based on config.
    
    Args:
        config: EncoderConfig with encoder_type set
    
    Returns:
        Video encoder module (VideoGITViTEncoder or VJEPAVideoEncoder)
    """
    if config.encoder_type == "git_vit":
        encoder = VideoGITViTEncoder(config)
    elif config.encoder_type == "vjepa":
        encoder = VJEPAVideoEncoder(config)
    else:
        raise ValueError(f"Unknown encoder type: {config.encoder_type}")
    
    # Load pretrained weights if provided
    if config.pretrained_weights:
        state_dict = torch.load(config.pretrained_weights, map_location="cpu")
        encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded encoder weights from {config.pretrained_weights}")
    
    # Freeze if requested
    if config.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        print("Encoder frozen (no gradients)")
    
    return encoder
