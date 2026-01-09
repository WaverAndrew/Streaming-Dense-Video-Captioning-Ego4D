"""
Test script to verify Vid2SeqDecoder integration with VidChapters weights.
Run from project root: venv/bin/python -m tests.test_decoder_integration
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch


def test_decoder_integration():
    print("Testing Vid2SeqDecoder integration...")
    
    # Import here to avoid issues with relative imports
    from streaming_dvc.config.default_config import DecoderConfig
    from streaming_dvc.models.text_decoder import Vid2SeqDecoder
    
    # Create config
    config = DecoderConfig()
    print(f"Using model: {config.model_name}")
    
    # Initialize decoder
    print("Initializing Vid2SeqDecoder...")
    decoder = Vid2SeqDecoder(config)
    print(f"Decoder initialized. Hidden size: {decoder.hidden_size}, Vocab size: {decoder.total_vocab_size}")
    
    # Load VidChapters weights
    weights_path = os.path.join(project_root, "weights/vid2seq_htmchaptersyoucook.pth")
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        msg = decoder.load_vidchapters_weights(weights_path)
        print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        if msg.missing_keys:
            print(f"Missing keys sample: {msg.missing_keys[:5]}")
    else:
        print(f"Weights file not found at {weights_path}")
    
    # Test forward pass with streaming-compatible interface
    print("\nTesting forward pass (streaming interface)...")
    B, buffer_size, D = 2, 514, decoder.hidden_size  # Match streaming memory buffer size
    L = 32  # Caption length
    
    encoder_hidden_states = torch.randn(B, buffer_size, D)
    labels = torch.randint(0, 100, (B, L))
    
    outputs = decoder(
        input_ids=labels,  # Not used when labels provided
        encoder_hidden_states=encoder_hidden_states,
        labels=labels
    )
    
    print(f"Forward pass successful!")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    # Test generation
    print("\nTesting generation...")
    generated = decoder.generate(
        encoder_hidden_states=encoder_hidden_states,
        max_length=20,
        num_beams=2
    )
    decoded = decoder.decode_output(generated)
    print(f"Generated tokens shape: {generated.shape}")
    print(f"Decoded output sample: {decoded[0][:100]}...")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_decoder_integration()
