# KV-cache optimized OpenPanGu model for MTP speculative decoding
# Architecture: PanguProMoEV2ForCausalLM
# Features: Sink Attention, MoE, Sandwich Norm, Asymmetric KV cache

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .kv_cache import KVCache

logger = logging.get_logger(__name__)


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
            dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class PanguRMSNorm(nn.Module):
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


class PanguRotaryEmbedding(nn.Module):
    """Rotary embedding for the rope portion of Sink Attention keys/queries.

    Uses GPT-J (non-interleaved) style where pairs of adjacent elements
    are rotated together: (x0, x1), (x2, x3), etc.
    """

    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # GPT-J style: repeat-interleave so cos/sin align with (x0,x1), (x2,x3), ...
        emb = freqs.repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """GPT-J style rotation: rotate pairs of adjacent elements."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb_partial(q_rope, k_rope, cos, sin, position_ids):
    """Apply rotary embedding to the rope portion of Q and K only."""
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q_rope * cos) + (rotate_half(q_rope) * sin)
    k_embed = (k_rope * cos) + (rotate_half(k_rope) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PanguSinkAttention(nn.Module):
    """
    Sink Attention for OpenPanGu with asymmetric K/V dimensions.

    K: (num_kv_heads, qk_nope_dim + qk_rope_dim) = (4, 192)
    V: (num_kv_heads, v_channels) = (4, 128)
    Q: (num_heads, qk_nope_dim + qk_rope_dim) = (64, 192)

    Features:
    - Fused QKV projection split into Q, K, V
    - K has k_layernorm applied
    - Only rope portion gets rotary embedding
    - Parametric sink key/value prepended to KV cache
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.qk_nope_dim = config.qk_nope_dim
        self.qk_rope_dim = config.qk_rope_dim
        self.head_dim = self.qk_nope_dim + self.qk_rope_dim  # 192
        self.v_channels = config.v_channels  # 128

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_channels

        self.param_sink_number = getattr(config, 'param_sink_number', 0)
        self.param_sink_with_value = getattr(config, 'param_sink_with_value', False)

        # Fused QKV projection
        self.qkv_proj = nn.Linear(
            self.hidden_size, self.q_size + self.k_size + self.v_size, bias=False
        )
        # Output projection: num_heads * v_channels -> hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.v_channels, self.hidden_size, bias=False)

        # K layernorm
        self.k_layernorm = PanguRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Rotary embedding for the rope portion only
        self.rotary_emb = PanguRotaryEmbedding(
            self.qk_rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Parametric sink tokens
        if self.param_sink_number > 0:
            self.param_sink_key = nn.Parameter(
                torch.zeros(self.param_sink_number, self.num_key_value_heads, self.head_dim)
            )
            if self.param_sink_with_value:
                self.param_sink_value = nn.Parameter(
                    torch.zeros(self.param_sink_number, self.num_key_value_heads, self.v_channels)
                )
            else:
                self.register_buffer(
                    "param_sink_value",
                    torch.zeros(self.param_sink_number, self.num_key_value_heads, self.v_channels)
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

        # Fused QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        # Reshape Q: (bsz, q_len, num_heads, head_dim) -> (bsz, num_heads, q_len, head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Reshape K and apply k_layernorm
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        k = self.k_layernorm(k)
        k = k.transpose(1, 2)  # (bsz, num_kv_heads, q_len, head_dim)

        # Reshape V: (bsz, q_len, num_kv_heads, v_channels) -> (bsz, num_kv_heads, q_len, v_channels)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.v_channels).transpose(1, 2)

        # Split Q and K into rope and nope portions
        # Weight layout is [rope(qk_rope_dim=64), nope(qk_nope_dim=128)]
        # vLLM applies partial_rotary_factor to the FIRST dimensions
        q_rope = q[..., :self.qk_rope_dim]
        q_nope = q[..., self.qk_rope_dim:]
        k_rope = k[..., :self.qk_rope_dim]
        k_nope = k[..., self.qk_rope_dim:]

        # Apply rotary embeddings to rope portions
        kv_seq_len = k.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(q_rope, seq_len=kv_seq_len)
        q_rope, k_rope = apply_rotary_pos_emb_partial(q_rope, k_rope, cos, sin, position_ids)

        # Recombine rope and nope portions (same order as original layout)
        q = torch.cat([q_rope, q_nope], dim=-1)
        k = torch.cat([k_rope, k_nope], dim=-1)

        # KV-cache handling: two paths
        # 1. KVCache objects (base model): in-place mutation, return None
        # 2. Plain tensors (MTP draft model): torch.cat, return updated tensors
        if past_key_value is not None:
            if isinstance(past_key_value[0], KVCache):
                k = past_key_value[0].cat(k, dim=2)
                v = past_key_value[1].cat(v, dim=2)
                past_key_value = None
            else:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
                past_key_value = (k, v) if use_cache else None
        elif use_cache:
            past_key_value = (k, v)

        # Prepend parametric sink tokens (k_layernorm applied to sink keys, matching vLLM)
        if self.param_sink_number > 0:
            sink_k = self.k_layernorm(self.param_sink_key)
            sink_k = sink_k.unsqueeze(0).expand(bsz, -1, -1, -1).transpose(1, 2)
            sink_v = self.param_sink_value.unsqueeze(0).expand(bsz, -1, -1, -1).transpose(1, 2)
            # sink_k: (bsz, num_kv_heads, sink_number, head_dim)
            # sink_v: (bsz, num_kv_heads, sink_number, v_channels)
            k_with_sink = torch.cat([sink_k.to(k.dtype), k], dim=2)
            v_with_sink = torch.cat([sink_v.to(v.dtype), v], dim=2)
        else:
            k_with_sink = k
            v_with_sink = v

        # Repeat k/v heads for GQA
        k_with_sink = repeat_kv(k_with_sink, self.num_key_value_groups)
        v_with_sink = repeat_kv(v_with_sink, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(q, k_with_sink.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply attention mask (need to account for sink tokens)
        if attention_mask is not None:
            if self.param_sink_number > 0:
                # Extend mask for sink tokens (they're always attended to)
                sink_mask = torch.zeros(
                    bsz, 1, q_len, self.param_sink_number,
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask_with_sink = torch.cat([sink_mask, attention_mask], dim=-1)
                attn_weights = attn_weights + attention_mask_with_sink
            else:
                attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_with_sink)

        # attn_output: (bsz, num_heads, q_len, v_channels)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_channels)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PanguMLP(nn.Module):
    """Dense MLP for layers 0 to first_k_dense_replace-1."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PanguExpert(nn.Module):
    """Single expert MLP."""

    def __init__(self, hidden_size, moe_intermediate_size, hidden_act="silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(moe_intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PanguMoE(nn.Module):
    """
    Mixture of Experts for OpenPanGu.

    80 routed experts + 2 shared experts, top-8 routing with sigmoid scoring.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob

        # Router gate
        self.gate = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)

        # Expert bias for routing
        if getattr(config, 'router_enable_expert_bias', False):
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(self.n_routed_experts)
            )
        else:
            self.e_score_correction_bias = None

        # Routed experts
        self.experts = nn.ModuleList([
            PanguExpert(self.hidden_size, config.moe_intermediate_size, config.hidden_act)
            for _ in range(self.n_routed_experts)
        ])

        # Shared experts
        if self.n_shared_experts > 0:
            shared_intermediate = config.moe_intermediate_size * self.n_shared_experts
            self.shared_experts = PanguMLP.__new__(PanguMLP)
            nn.Module.__init__(self.shared_experts)
            self.shared_experts.hidden_size = self.hidden_size
            self.shared_experts.intermediate_size = shared_intermediate
            self.shared_experts.gate_proj = nn.Linear(self.hidden_size, shared_intermediate, bias=False)
            self.shared_experts.up_proj = nn.Linear(self.hidden_size, shared_intermediate, bias=False)
            self.shared_experts.down_proj = nn.Linear(shared_intermediate, self.hidden_size, bias=False)
            self.shared_experts.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router
        router_logits = self.gate(hidden_states_flat)

        # Sigmoid scoring
        scores = torch.sigmoid(router_logits)

        # Apply expert bias for top-k SELECTION only (not for weighting)
        if self.e_score_correction_bias is not None:
            scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        else:
            scores_for_choice = scores

        # Top-k selection using biased scores
        _, topk_indices = torch.topk(
            scores_for_choice, self.num_experts_per_tok, dim=-1
        )

        # Gather UNBIASED sigmoid scores for the selected experts
        topk_weights = scores.gather(1, topk_indices)

        # Normalize if configured
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Scale
        topk_weights = topk_weights * self.routed_scaling_factor

        # Compute expert outputs (Mixtral-style one_hot + index_add_)
        topk_weights = topk_weights.to(hidden_states_flat.dtype)

        final_hidden = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states_flat.dtype,
            device=hidden_states_flat.device,
        )

        # One-hot encode selected experts to create an expert mask
        # expert_mask shape: (n_routed_experts, num_experts_per_tok, batch*seq)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.n_routed_experts
        ).permute(2, 1, 0)

        for expert_idx in range(self.n_routed_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # In torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states_flat[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state)
            current_hidden_states = current_hidden_states * topk_weights[top_x_list, idx_list, None]

            final_hidden.index_add_(
                0, top_x, current_hidden_states.to(hidden_states_flat.dtype)
            )

        # Add shared expert output
        if self.n_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states_flat)
            final_hidden = final_hidden + shared_output

        return final_hidden.view(batch_size, seq_len, hidden_dim)


class PanguDecoderLayer(nn.Module):
    """
    Decoder layer for OpenPanGu with Sink Attention, MoE/Dense MLP, and Sandwich Norm.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        self.sandwich_norm = getattr(config, 'sandwich_norm', False)

        self.self_attn = PanguSinkAttention(config, layer_idx=layer_idx)

        # Dense MLP for layers < first_k_dense_replace, MoE otherwise
        first_k = getattr(config, 'first_k_dense_replace', config.num_hidden_layers)
        if layer_idx < first_k or not hasattr(config, 'n_routed_experts'):
            self.mlp = PanguMLP(config)
            self.is_moe = False
        else:
            self.mlp = PanguMoE(config, layer_idx=layer_idx)
            self.is_moe = True

        self.input_layernorm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.sandwich_norm:
            self.pre_mlp_layernorm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_mlp_layernorm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, ...]:
        # Pre-attention: input_layernorm + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # Pre-MLP: sandwich norm or standard pre-norm
        # vLLM reference (openpangu.py:995-1014):
        #   sandwich: post_attention_layernorm(attn_out) -> residual + normed -> pre_mlp_layernorm -> mlp -> post_mlp_layernorm
        #   standard: residual + attn_out -> post_attention_layernorm -> mlp
        if self.sandwich_norm:
            # Sandwich: norm attn output BEFORE adding to residual
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.pre_mlp_layernorm(hidden_states)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP (dense or MoE)
        hidden_states = self.mlp(hidden_states)

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PanguPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PanguDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class PanguModel(PanguPreTrainedModel):
    """OpenPanGu transformer model outputting raw hidden-states."""

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            PanguDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.norm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # Tree mask support
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][tree_mask == 0] = combined_attention_mask.min()

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KVPanguForCausalLM(PanguPreTrainedModel, GenerationMixin):
    """KV-cache optimized OpenPanGu model with language modeling head."""

    _tied_weights_keys = []  # lm_head.weight is NOT tied in OpenPanGu

    def __init__(self, config):
        super().__init__(config)
        self.model = PanguModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs
