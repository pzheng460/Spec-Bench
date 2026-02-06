# coding=utf-8
# MTP (Multi-Token Prediction) Layer Architecture
# Key difference from EAGLE: separate RMSNorm for embeddings and hidden states

import copy
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    from .configs import MTPConfig
    from .utils_c import generate_tree_buffers
    from ..eagle.choices import mc_sim_7b_63
except ImportError:
    from configs import MTPConfig

TOPK = 10  # topk for sparse tree


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
):
    """Make causal mask used for bi-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
            dim=-1
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat k/v heads for GQA."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2RMSNorm(nn.Module):
    """RMSNorm for Qwen2/MTP."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    """Rotary position embedding for Qwen2."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class Qwen2Attention(nn.Module):
    """Multi-headed attention for Qwen2/MTP with bias support."""

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Qwen2 attention uses bias for q/k/v projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2MLP(nn.Module):
    """MLP for Qwen2/MTP."""

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    """Decoder layer for Qwen2/MTP (used inside each MTP layer)."""

    def __init__(self, config: MTPConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config)
        self.mlp = Qwen2MLP(config)
        self.layer_idx = layer_idx

        # All MTP layers have input_layernorm
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MTPLayer(nn.Module):
    """
    Single MTP layer that predicts a specific future token position.

    Key difference from EAGLE: separate RMSNorm for embeddings (enorm) and hidden states (hnorm).

    Architecture:
        embeddings → enorm → ┐
                             ├─→ concat → eh_proj → decoder_layer → final_layernorm → output
        hidden_states → hnorm → ┘
    """

    def __init__(self, config: MTPConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # MTP-specific: separate RMSNorm for embeddings and hidden states
        self.enorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Projection layer after concatenation: 2*hidden_size → hidden_size
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # Single decoder layer
        self.decoder = Qwen2DecoderLayer(config, layer_idx=0)

        # Final layer norm
        self.final_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for a single MTP layer.

        Args:
            hidden_states: Output from base model, shape (batch, seq_len, hidden_size)
            input_ids: Input token IDs, shape (batch, seq_len)
            embed_tokens: Embedding layer from base model
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Cached key/value states
            use_cache: Whether to return cached key/value states

        Returns:
            Tuple of (output_hidden_states, present_key_value)
        """
        # Get the device of this layer
        layer_device = self.enorm.weight.device

        # Get embeddings
        with torch.no_grad():
            inputs_embeds = embed_tokens(input_ids)

        # Move inputs to layer device
        inputs_embeds = inputs_embeds.to(device=layer_device, dtype=hidden_states.dtype)
        hidden_states = hidden_states.to(device=layer_device)

        # Apply separate RMSNorm to embeddings and hidden states
        inputs_embeds_norm = self.enorm(inputs_embeds)
        hidden_states_norm = self.hnorm(hidden_states)

        # Concatenate and project
        combined = torch.cat([inputs_embeds_norm, hidden_states_norm], dim=-1)
        hidden = self.eh_proj(combined)

        # Pass through decoder layer
        layer_outputs = self.decoder(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        hidden = layer_outputs[0]
        present_key_value = layer_outputs[-1] if use_cache else None

        # Apply final layer norm
        output = self.final_layernorm(hidden)

        return output, present_key_value


class MTPDraftModel(nn.Module):
    """
    MTP Draft Model containing all 7 MTP layers.

    Each MTP layer predicts a different future token position:
    - Layer 0: predicts token at position t+1
    - Layer 1: predicts token at position t+2
    - ...
    - Layer 6: predicts token at position t+7
    """

    def __init__(self, config: MTPConfig, load_emb: bool = False, path: str = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_mtp_layers = config.num_mtp_layers

        # Embedding layer (shared with base model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        if load_emb and path:
            self._load_embeddings(path)

        # Freeze embeddings
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        # Create MTP layers
        self.mtp_layers = nn.ModuleList([
            MTPLayer(config, layer_idx=i) for i in range(self.num_mtp_layers)
        ])

        # Tree buffers for speculative decoding
        self.tree_mask = None
        self.tree_buffer = None
        self.stable_kv = None
        self.diff_device = False
        self.headweight = None

    def _load_embeddings(self, path: str):
        """Load embeddings from base model."""
        from safetensors import safe_open
        import json

        try:
            with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
            with safe_open(os.path.join(path, emb_path), framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice("model.embed_tokens.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()
        except Exception:
            try:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            except Exception:
                return

        self.embed_tokens.weight.data = tensor

    def init_tree(self):
        """Initialize tree buffers for speculative decoding."""
        from ..eagle.choices import mc_sim_7b_63
        from ..eagle.utils_c import generate_tree_buffers as eagle_generate_tree_buffers
        self.tree = mc_sim_7b_63
        # Use EAGLE's tree buffer format for topK_generate compatibility
        self.tree_buffer = eagle_generate_tree_buffers(self.tree, self.embed_tokens.weight.device)

    def reset(self):
        """Reset tree mask."""
        self.tree_mask = None

    def reset_kv(self):
        """Reset KV cache."""
        self.stable_kv = None

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length
    ):
        """Prepare the attention mask with optional tree mask."""
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        # Apply tree mask if present
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
            ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Forward pass through a specific MTP layer.

        Args:
            hidden_states: Output from base model
            input_ids: Input token IDs
            layer_idx: Which MTP layer to use (0-6)
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached key/value states
            use_cache: Whether to return cached states
        """
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # Get embeddings for attention mask preparation
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length
        )

        # Get past_key_value for this layer
        past_key_value = past_key_values[layer_idx] if past_key_values is not None else None

        # Forward through the MTP layer
        output, present_key_value = self.mtp_layers[layer_idx](
            hidden_states,
            input_ids,
            self.embed_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        if use_cache:
            # Return as tuple of tuples for compatibility with EAGLE-style cache
            return output, (present_key_value,)
        return output

    def forward_all_layers(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ) -> Tuple[List[torch.Tensor], Optional[List]]:
        """
        Forward pass through all MTP layers (for training or parallel inference).

        Returns outputs from all 7 MTP layers.
        """
        outputs = []
        next_cache = [] if use_cache else None

        for layer_idx in range(self.num_mtp_layers):
            past_kv = past_key_values[layer_idx] if past_key_values else None

            if use_cache:
                output, present_kv = self.forward(
                    hidden_states, input_ids, layer_idx,
                    attention_mask, position_ids,
                    [past_kv] if past_kv else None, use_cache
                )
                outputs.append(output)
                next_cache.append(present_kv)
            else:
                output = self.forward(
                    hidden_states, input_ids, layer_idx,
                    attention_mask, position_ids, None, False
                )
                outputs.append(output)

        return outputs, next_cache

    @torch.no_grad()
    def repeat_hidden(self, hidden_state, repeat_num):
        """Repeat hidden states for tree structure."""
        new_hidden = []
        for id, i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:, id:id+1].repeat(1, i, 1))
        return torch.cat(new_hidden, dim=1)

    def sample(self, logits, logits_processor, k=1, replacement=False):
        """Sample from logits with optional logits processing."""
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]),
            dim=-1
        )

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1
        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

        return sampled_indices, sampled_probs, probabilities

    @torch.no_grad()
    def topK_generate(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        head: nn.Linear,
        logits_processor=None,
        max_length: int = 4,
        use_cache: bool = True
    ):
        """
        Generate top-k candidates using the MTP layers with tree structure.

        This method generates a tree of candidate tokens where each MTP layer
        predicts the next token at its corresponding position.
        """
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        ss_token, ss_prob, ss_op = [], [], []
        len_posi = input_ids.shape[1]
        self.reset()

        if use_cache:
            # Use stable KV cache if available
            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                kv_len = self.stable_kv[0][0].shape[2]
                # Forward through layer 0 with partial input
                out_hidden, past_key_values = self.forward(
                    hidden_states,
                    input_ids[:, kv_len:],
                    layer_idx=0,
                    past_key_values=self.stable_kv,
                    use_cache=True
                )
            else:
                out_hidden, past_key_values = self.forward(
                    hidden_states,
                    input_ids,
                    layer_idx=0,
                    use_cache=True
                )

            self.stable_kv = past_key_values

            last_hidden = out_hidden[:, -1]

            # Get logits from lm_head
            if not self.diff_device:
                last_headout = head(last_hidden)
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(last_hidden)
                    last_headout = last_headout.to(self.layer_device)
                else:
                    last_headout = F.linear(last_hidden, self.headweight)

            # Generate tree structure
            for i in range(len(self.tree_buffer['tree_indices'])):
                if logits_processor is not None:
                    topk_index, topk_prob, op = self.sample(last_headout, logits_processor, k=TOPK)
                else:
                    top = torch.topk(last_headout, TOPK, dim=-1)
                    topk_index, topk_prob = top.indices, top.values
                    op = None

                ss_token.append(topk_index)
                ss_prob.append(topk_prob)
                ss_op.append(op)

                topk_index = topk_index.view(-1)
                select_index = topk_index[self.tree_buffer['tree_indices'][i]]
                input_ids = select_index[None, :]

                if i == 0:
                    hidden_states = out_hidden[:, -1:]
                else:
                    hidden_states = out_hidden

                hidden_states = self.repeat_hidden(hidden_states, self.tree_buffer["repeat_nums"][i])
                self.tree_mask = self.tree_buffer['attn_mask'][i]
                position_ids = len_posi + self.tree_buffer["position_ids"][i]

                # Use the appropriate MTP layer for this tree level
                # Each level uses layer 0 in this implementation (following EAGLE pattern)
                out_hidden, past_key_values = self.forward(
                    hidden_states,
                    input_ids,
                    layer_idx=0,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                len_posi += 1

                if not self.diff_device:
                    last_headout = head(out_hidden[0])
                else:
                    if hasattr(self, "layer_device"):
                        last_headout = head(out_hidden[0])
                        last_headout = last_headout.to(self.layer_device)
                    else:
                        last_headout = F.linear(out_hidden[0], self.headweight)

            # Final sampling
            if logits_processor is not None:
                topk_index, topk_prob, op = self.sample(last_headout, logits_processor, k=TOPK)
            else:
                top = torch.topk(last_headout, TOPK, dim=-1)
                topk_index, topk_prob = top.indices, top.values
                op = None

            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)

        return (torch.cat(ss_token), torch.cat(ss_prob), ss_op)


def count_parameters(model):
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = MTPConfig()
    model = MTPDraftModel(config)
    print(f"Total parameters: {count_parameters(model):,}")
