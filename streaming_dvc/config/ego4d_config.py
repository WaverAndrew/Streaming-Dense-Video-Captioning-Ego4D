"""
Ego4D-specific configuration for Streaming Dense Video Captioning.
"""
from .default_config import (
    StreamingDVCConfig,
    EncoderConfig,
    MemoryConfig, 
    DecoderConfig,
    TrainingConfig,
    DataConfig,
)


def get_ego4d_config(
    data_root: str = "ego4d_pilot",
    encoder_type: str = "git_vit",
) -> StreamingDVCConfig:
    """
    Get configuration for Ego4D dataset.
    
    Args:
        data_root: Path to Ego4D data directory
        encoder_type: 'git_vit' or 'vjepa'
    
    Returns:
        StreamingDVCConfig configured for Ego4D
    """
    config = StreamingDVCConfig(
        experiment_name=f"streaming_dvc_ego4d_{encoder_type}",
        
        encoder=EncoderConfig(
            encoder_type=encoder_type,
            freeze_encoder=True,  # Freeze backbone as in paper
        ),
        
        memory=MemoryConfig(
            streaming_method="kmeans",
            buffer_size=514,
            kmeans_num_iters=2,
        ),
        
        decoder=DecoderConfig(
            max_caption_length=256,
            num_time_bins=100,
            decode_method="beam",
            beam_size=4,
        ),
        
        training=TrainingConfig(
            num_frames=64,
            num_dense_outputs=16,
            num_dense_outputs_test=2,
            early_segments_as_context=True,
            normalize_early_timestamps=True,
            batch_size=8,  # Reduced for single GPU
            learning_rate=1e-5,
            num_training_steps=5000,
        ),
        
        data=DataConfig(
            dataset_name="ego4d",
            data_root=data_root,
            clips_dir=f"{data_root}/v1/clips",
            annotations_path=f"{data_root}/v1/annotations",
        ),
    )
    
    return config
