"""
Default configuration for Streaming Dense Video Captioning.
Matches the original paper's hyperparameters as closely as possible.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class EncoderConfig:
    """Video encoder configuration."""
    # Encoder type: 'git_vit' (original) or 'vjepa'
    encoder_type: str = "git_vit"
    
    # GIT ViT-L configuration (matches paper)
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    patch_size: int = 14
    image_size: int = 224
    
    # Temporal settings
    use_temporal_embedding: bool = True
    
    # V-JEPA specific (only used if encoder_type == 'vjepa')
    vjepa_model_name: str = "facebook/vjepa2-vitl-fpc64-256"
    
    # Pretrained weights path
    pretrained_weights: Optional[str] = None
    freeze_encoder: bool = True


@dataclass
class MemoryConfig:
    """Streaming memory module configuration."""
    # Memory method: 'kmeans', 'ema', 'temporal_mean_pool', 'adjacent_tome'
    streaming_method: str = "kmeans"
    
    # Buffer size: number of memory tokens (paper uses 514 = (256+1)*2)
    buffer_size: int = 514
    
    # K-means iterations (paper uses 2)
    kmeans_num_iters: int = 2
    
    # EMA decay (if using EMA method)
    ema_decay: float = 0.9
    
    # Whether to use momentum in memory updates
    use_momentum: bool = True


@dataclass
class DecoderConfig:
    """Text decoder configuration."""
    # Decoder type: 'transformer' (scratch) or 't5' (pretrained)
    decoder_type: str = "t5"
    
    # T5/Vid2Seq settings - use t5-base to match VidChapters weights
    model_name: str = "t5-base"
    pretrained: bool = True
    
    # Common settings
    vocab_size: int = 32128  # T5 vocab size
    hidden_size: int = 768   # T5-base hidden size
    num_layers: int = 12     # T5-base layers
    num_heads: int = 12
    intermediate_size: int = 2048
    
    # Special tokens (T5 defaults)
    bos_token_id: int = 0    # usually pad/eos in T5, but we'll handle it
    eos_token_id: int = 1
    pad_token_id: int = 0
    
    # Caption settings
    max_caption_length: int = 256
    
    # Time bins for timestamp prediction
    num_time_bins: int = 100
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Decoding settings
    decode_method: str = "beam"  # 'greedy' or 'beam'
    beam_size: int = 4
    length_penalty: float = 0.6


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Number of frames per video
    num_frames: int = 64
    
    # Number of intermediate decoding outputs (streaming outputs)
    num_dense_outputs: int = 16
    num_dense_outputs_test: int = 2
    
    # Streaming output settings
    early_segments_as_context: bool = True
    normalize_early_timestamps: bool = True
    
    # Loss weights
    localization_loss_weight: float = 0.5
    
    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 250
    
    # Training
    batch_size: int = 32
    num_training_steps: int = 5000
    
    # Learning rate schedule
    lr_decay_steps: Tuple[int, ...] = (4000,)
    lr_decay_factor: float = 0.1
    
    # Checkpointing
    checkpoint_steps: int = 500
    eval_steps: int = 500
    
    # Data augmentation
    context_mask_ratio: float = 0.5


@dataclass
class DataConfig:
    """Data configuration."""
    # Dataset name
    dataset_name: str = "ego4d"
    
    # Paths (to be set by user)
    data_root: str = ""
    annotations_path: str = ""
    clips_dir: str = ""
    
    # Preprocessing
    crop_size: int = 224
    
    # Tokenizer
    tokenizer_name: str = "bert-base-uncased"


@dataclass
class StreamingDVCConfig:
    """Full configuration for Streaming Dense Video Captioning."""
    experiment_name: str = "streaming_dvc_ego4d"
    
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Output directory
    output_dir: str = "./outputs"
    
    # Random seed
    seed: int = 42
    
    # Device
    device: str = "cuda"
    
    # Evaluation only mode
    eval_only: bool = False


def get_default_config() -> StreamingDVCConfig:
    """Returns default configuration matching the paper."""
    return StreamingDVCConfig()
