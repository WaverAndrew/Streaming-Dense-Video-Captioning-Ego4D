"""
Text Decoder for Streaming Dense Video Captioning.

Implements:
1. TransformerTextDecoder: Scratch-trained transformer (baseline)
2. Vid2SeqDecoder: Pretrained T5-based decoder (Vid2Seq architecture)
"""
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

from ..config.default_config import DecoderConfig


class TransformerTextDecoder(nn.Module):
    """
    Transformer decoder for generating captions with timestamps.
    Trained from scratch.
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # Vocabulary size = vocab + time bins
        self.vocab_size = config.vocab_size
        self.num_time_bins = config.num_time_bins
        self.total_vocab_size = self.vocab_size + self.num_time_bins
        
        # Embeddings
        self.word_embeddings = nn.Embedding(
            self.total_vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_caption_length, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, self.total_vocab_size)
        self.output_projection.weight = self.word_embeddings.weight
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.output_projection.bias, 0.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        B, L = input_ids.shape
        
        # Embeddings
        position_ids = torch.arange(L, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        token_type_ids = torch.zeros_like(input_ids)
        
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Causal Mask
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=input_ids.device), diagonal=1
        )
        
        # Decoder Forward
        if encoder_attention_mask is not None:
            memory_key_padding_mask = (encoder_attention_mask == 0)
        else:
            memory_key_padding_mask = None
            
        hidden_states = self.decoder(
            tgt=embeddings,
            memory=encoder_hidden_states,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        logits = self.output_projection(hidden_states)
        output = {"logits": logits}
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing,
                ignore_index=self.config.pad_token_id
            )
            loss = loss_fct(
                shift_logits.view(-1, self.total_vocab_size),
                shift_labels.view(-1)
            )
            output["loss"] = loss
            
        return output


class Vid2SeqDecoder(nn.Module):
    """
    T5-based decoder for Vid2Seq.
    Uses pretrained T5 weights and adds time tokens.
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # Load T5
        if config.pretrained:
            self.t5 = T5ForConditionalGeneration.from_pretrained(config.model_name)
        else:
            t5_config = T5Config.from_pretrained(config.model_name)
            self.t5 = T5ForConditionalGeneration(t5_config)
            
        # Resize embeddings for time tokens
        # T5 vocab is usually 32128. We add num_time_bins.
        self.original_vocab_size = self.t5.config.vocab_size
        self.num_time_bins = config.num_time_bins
        self.total_vocab_size = self.original_vocab_size + self.num_time_bins
        
        self.t5.resize_token_embeddings(self.total_vocab_size)
        
        # Initialize new time token embeddings
        # We can initialize them to be similar to digit embeddings or random
        with torch.no_grad():
            self.t5.shared.weight[self.original_vocab_size:] = torch.normal(
                mean=0.0, 
                std=self.t5.config.initializer_factor,
                size=(self.num_time_bins, self.t5.config.d_model),
                device=self.t5.device
            )
            
        # Project visual features to T5 dimension if needed
        # (Though usually this is done in the main model, we can check here)
        self.hidden_size = self.t5.config.d_model
        
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: (B, L) input tokens (for teacher forcing or generation start)
            encoder_hidden_states: (B, N, D) visual features
            labels: (B, L) target tokens
        """
        # T5 expects encoder_outputs as a tuple/BaseModelOutput
        # We wrap our visual features
        
        # T5 forward
        # If labels are provided, T5 computes loss automatically
        # Note: T5 shifts labels internally for decoder_input_ids if not provided
        
        outputs = self.t5(
            input_ids=input_ids if labels is None else None, # If training, let T5 handle input_ids from labels
            labels=labels,
            encoder_outputs=(encoder_hidden_states,),
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None
        }

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate captions using T5's generate method.
        """
        max_length = max_length or self.config.max_caption_length
        num_beams = num_beams or self.config.beam_size
        
        # T5 generate expects encoder_outputs
        # We need to wrap it in a class that mimics BaseModelOutput or just pass as tuple if supported
        # Transformers generate() is a bit tricky with custom encoder_outputs
        # easier to use the model wrapper
        
        # We need to create a dummy encoder wrapper or pass inputs_embeds?
        # Actually T5ForConditionalGeneration.generate supports encoder_outputs
        
        # We need to ensure encoder_hidden_states matches T5 d_model
        
        outputs = self.t5.generate(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            use_cache=True,
        )
        
        return outputs
