"""
Text Decoder for Streaming Dense Video Captioning.

Implements:
1. TransformerTextDecoder: Scratch-trained transformer (baseline)
2. Vid2SeqDecoder: Pretrained T5-based decoder (VidChapters implementation)
"""
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import T5Tokenizer

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
    T5-based decoder for Vid2Seq using VidChapters implementation.
    Uses pretrained T5 weights with time token extension.
    Compatible with VidChapters pretrained checkpoints.
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # Use VidChapters T5 implementation (supports dropout/label_smoothing params)
        from .vidchapters_t5 import T5ForConditionalGeneration
        
        # Load T5 with VidChapters modifications
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.model_name,
            encoder_dropout=0.1,
            decoder_dropout=0.1,
            label_smoothing=config.label_smoothing,
            is_gated_act="v1_1" in config.model_name
        )
        
        # Setup tokenizer with time tokens
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        self.num_time_bins = config.num_time_bins
        
        # Add time tokens to tokenizer
        time_tokens = [f"<time={i}>" for i in range(self.num_time_bins)]
        self.tokenizer.add_tokens(time_tokens)
        
        # Resize embeddings: first trim unused tokens, then add time tokens
        # T5 tokenizer has 32128 tokens but T5 model uses 32100
        original_vocab = len(self.tokenizer) - self.num_time_bins
        self.t5.resize_token_embeddings(original_vocab)
        self.t5.resize_token_embeddings(len(self.tokenizer))
        
        self.total_vocab_size = len(self.tokenizer)
        self.hidden_size = self.t5.config.d_model
        
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        context_input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass compatible with streaming model interface.
        
        Args:
            input_ids: (B, L) input tokens (unused if labels provided, T5 handles internally)
            encoder_hidden_states: (B, N, D) visual memory features
            encoder_attention_mask: (B, N) attention mask for encoder features
            labels: (B, L) target tokens for training
            context_input_ids: (B, C) previous captions tokens to use as context
        """
        from transformers.modeling_outputs import BaseModelOutput
        
        # 1. Embed context tokens if provided
        if context_input_ids is not None:
            # Embed context tokens using T5 encoder
            # (B, C, D)
            context_embeds = self.t5.encoder.embed_tokens(context_input_ids)
            
            # Concatenate context with visual features
            # (B, C + N, D)
            encoder_hidden_states = torch.cat([context_embeds, encoder_hidden_states], dim=1)
            
            # Update attention mask
            if encoder_attention_mask is None:
                # Create mask for visual features (all ones)
                # (B, N)
                visual_mask = torch.ones(
                    (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1] - context_embeds.shape[1]),
                    dtype=torch.long,
                    device=encoder_hidden_states.device
                )
            else:
                visual_mask = encoder_attention_mask
                
            # Create mask for context (1 for valid, 0 for pad)
            # (B, C)
            context_mask = (context_input_ids != self.tokenizer.pad_token_id).long()
            
            # Concatenate masks
            # (B, C + N)
            encoder_attention_mask = torch.cat([context_mask, visual_mask], dim=1)
            
        # Wrap encoder_hidden_states as T5 encoder output
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        # Create attention mask if not provided (and no context)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.long,
                device=encoder_hidden_states.device
            )
        
        # Prepare labels: mask pad tokens with -100
        if labels is not None:
            targets = labels.masked_fill(
                labels == self.tokenizer.pad_token_id, -100
            )
        else:
            targets = None
        
        # T5 forward pass
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

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        context_input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate captions using beam search.
        """
        from transformers.modeling_outputs import BaseModelOutput
        
        max_length = max_length or self.config.max_caption_length
        num_beams = num_beams or self.config.beam_size
        length_penalty = length_penalty or self.config.length_penalty
        
        # 1. Embed context tokens if provided
        if context_input_ids is not None:
            context_embeds = self.t5.encoder.embed_tokens(context_input_ids)
            encoder_hidden_states = torch.cat([context_embeds, encoder_hidden_states], dim=1)
            
            if encoder_attention_mask is None:
                visual_mask = torch.ones(
                    (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1] - context_embeds.shape[1]),
                    dtype=torch.long,
                    device=encoder_hidden_states.device
                )
            else:
                visual_mask = encoder_attention_mask
                
            context_mask = (context_input_ids != self.tokenizer.pad_token_id).long()
            encoder_attention_mask = torch.cat([context_mask, visual_mask], dim=1)
        
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.long,
                device=encoder_hidden_states.device
            )
        
        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_new_tokens=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            use_cache=True,
        )
        
        return outputs
    
    def decode_output(self, token_ids: torch.Tensor) -> list:
        """Decode token IDs to text strings."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    def load_vidchapters_weights(self, checkpoint_path: str, strict: bool = False):
        """
        Load weights from a VidChapters checkpoint.
        
        Args:
            checkpoint_path: Path to VidChapters .pth file
            strict: Whether to require exact key match
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        
        # Filter and rename keys: t5_model.* -> t5.*
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("t5_model."):
                new_key = "t5." + key[len("t5_model."):]
                new_state_dict[new_key] = value
            elif key.startswith("visual_encoder.") or key.startswith("proj_v2t."):
                # Skip visual encoder weights (we use our own encoder)
                continue
            else:
                new_state_dict[key] = value
        
        msg = self.load_state_dict(new_state_dict, strict=strict)
        return msg
