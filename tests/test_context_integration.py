import unittest
import torch
import sys
import os
import importlib.util

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streaming_dvc.models.text_decoder import Vid2SeqDecoder
from streaming_dvc.config.default_config import DecoderConfig

class TestContextIntegration(unittest.TestCase):
    def setUp(self):
        self.config = DecoderConfig(
            model_name="t5-small", # Use small for faster testing
            vocab_size=32128,
            hidden_size=512,
            intermediate_size=2048,
            num_layers=6,
            num_heads=8,
            max_caption_length=20
        )
        self.decoder = Vid2SeqDecoder(self.config)
        self.device = torch.device("cpu")
        self.decoder.to(self.device)

    def test_forward_with_context(self):
        batch_size = 2
        num_visual_tokens = 10
        hidden_size = self.decoder.hidden_size
        seq_len = 15
        context_len = 5
        
        # Dummy inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        encoder_hidden_states = torch.randn(batch_size, num_visual_tokens, hidden_size).to(self.device)
        labels = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        
        # Context tokens
        context_input_ids = torch.randint(0, 1000, (batch_size, context_len)).to(self.device)
        
        # Forward pass WITH context
        outputs_with_context = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels,
            context_input_ids=context_input_ids
        )
        
        self.assertIn("logits", outputs_with_context)
        self.assertIn("loss", outputs_with_context)
        self.assertEqual(outputs_with_context["logits"].shape, (batch_size, seq_len, self.decoder.total_vocab_size))
        
        # Forward pass WITHOUT context
        outputs_no_context = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels,
            context_input_ids=None
        )
        
        # Check that logits are different (context should affect generation)
        # Note: T5 is an encoder-decoder model. The encoder sees the context.
        # So the decoder attending to the encoder output should see different things.
        self.assertFalse(torch.allclose(outputs_with_context["logits"], outputs_no_context["logits"]),
                         "Logits should differ when context is provided")

    def test_generate_with_context(self):
        batch_size = 1
        num_visual_tokens = 10
        hidden_size = self.decoder.hidden_size
        context_len = 5
        
        encoder_hidden_states = torch.randn(batch_size, num_visual_tokens, hidden_size).to(self.device)
        context_input_ids = torch.randint(0, 1000, (batch_size, context_len)).to(self.device)
        
        # Generate with context
        generated_ids = self.decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            context_input_ids=context_input_ids,
            max_length=10
        )
        
        self.assertTrue(generated_ids.shape[0] == batch_size)
        self.assertTrue(generated_ids.shape[1] <= 10)

if __name__ == "__main__":
    unittest.main()
