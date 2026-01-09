"""
Standalone test for Vid2SeqDecoder integration.
Uses importlib to directly import vidchapters_t5 without triggering package __init__.py
"""
import os
import sys
import importlib.util

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directly load vidchapters_t5 module without going through models/__init__.py
spec = importlib.util.spec_from_file_location(
    "vidchapters_t5", 
    os.path.join(project_root, "models", "vidchapters_t5.py")
)
vidchapters_t5 = importlib.util.module_from_spec(spec)
sys.modules["vidchapters_t5"] = vidchapters_t5
spec.loader.exec_module(vidchapters_t5)

import torch
import torch.nn as nn
from transformers import T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

T5ForConditionalGeneration = vidchapters_t5.T5ForConditionalGeneration


class Vid2SeqDecoderTest(nn.Module):
    """
    Standalone version of Vid2SeqDecoder for testing.
    """
    
    def __init__(self, model_name: str = "t5-base", num_time_bins: int = 100, label_smoothing: float = 0.1):
        super().__init__()
        
        # Load T5 with VidChapters modifications
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            encoder_dropout=0.1,
            decoder_dropout=0.1,
            label_smoothing=label_smoothing,
            is_gated_act="v1_1" in model_name
        )
        
        # Setup tokenizer with time tokens
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.num_time_bins = num_time_bins
        
        # Add time tokens to tokenizer
        time_tokens = [f"<time={i}>" for i in range(self.num_time_bins)]
        self.tokenizer.add_tokens(time_tokens)
        
        # Resize embeddings
        original_vocab = len(self.tokenizer) - self.num_time_bins
        self.t5.resize_token_embeddings(original_vocab)
        self.t5.resize_token_embeddings(len(self.tokenizer))
        
        self.total_vocab_size = len(self.tokenizer)
        self.hidden_size = self.t5.config.d_model
        
    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask=None, labels=None):
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.long,
                device=encoder_hidden_states.device
            )
        
        if labels is not None:
            targets = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        else:
            targets = None
        
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=targets,
            return_dict=True,
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if targets is not None else None
        }
    
    def load_vidchapters_weights(self, checkpoint_path: str, strict: bool = False):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("t5_model."):
                new_key = "t5." + key[len("t5_model."):]
                new_state_dict[new_key] = value
            elif key.startswith("visual_encoder.") or key.startswith("proj_v2t."):
                continue
            else:
                new_state_dict[key] = value
        
        msg = self.load_state_dict(new_state_dict, strict=strict)
        return msg


def test_decoder():
    print("Testing Vid2SeqDecoder integration...")
    
    # Initialize decoder
    print("Initializing Vid2SeqDecoder with t5-base...")
    decoder = Vid2SeqDecoderTest(model_name="t5-base")
    print(f"Decoder initialized. Hidden size: {decoder.hidden_size}, Vocab size: {decoder.total_vocab_size}")
    
    # Load VidChapters weights
    weights_path = os.path.join(project_root, "weights/vid2seq_htmchaptersyoucook.pth")
    if os.path.exists(weights_path):
        print(f"\nLoading weights from {weights_path}...")
        msg = decoder.load_vidchapters_weights(weights_path)
        print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        if msg.missing_keys:
            print(f"Missing keys sample: {msg.missing_keys[:5]}")
        if msg.unexpected_keys:
            print(f"Unexpected keys sample: {msg.unexpected_keys[:5]}")
    else:
        print(f"Weights file not found at {weights_path}")
    
    # Test forward pass with streaming-compatible interface
    print("\nTesting forward pass (streaming interface)...")
    B, buffer_size, D = 2, 514, decoder.hidden_size
    L = 32
    
    encoder_hidden_states = torch.randn(B, buffer_size, D)
    labels = torch.randint(0, 100, (B, L))
    
    outputs = decoder(
        input_ids=labels,
        encoder_hidden_states=encoder_hidden_states,
        labels=labels
    )
    
    print(f"Forward pass successful!")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_decoder()
