# MTP (Multi-Token Prediction) Draft Model for OpenPanGu
# Architecture: PanguProMoEV2 with Sink Attention + MoE + Sandwich Norm
# Key: Uses full PanGu decoder (not simplified) for the MTP draft layer

import math
import os
import json
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .modeling_pangu_kv import (
    PanguRMSNorm,
    PanguSinkAttention,
    PanguMLP,
    PanguMoE,
    PanguDecoderLayer,
    _make_causal_mask,
    _expand_mask,
)
from .configs import PanguMTPConfig

try:
    from ..mtp.utils_c import generate_tree_buffers
    from ..eagle.choices import mc_sim_7b_63
except ImportError:
    pass

TOPK = 10  # topk for sparse tree


class PanguMTPLayer(nn.Module):
    """
    Single MTP layer for OpenPanGu with full Sink Attention + MoE decoder.

    Architecture:
        embeddings -> enorm -> |
                                |-> concat -> eh_proj -> PanguDecoderLayer -> final_layernorm -> output
        hidden_states -> hnorm -> |

    Weight structure from layer 50:
        - enorm, hnorm, eh_proj: MTP-specific projection
        - shared_head.{head, norm}: shared output head
        - self_attn.*, mlp.*, input_layernorm, post_attention_layernorm,
          pre_mlp_layernorm, post_mlp_layernorm: decoder sub-layer
    """

    def __init__(self, config: PanguMTPConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # MTP-specific norms and projection
        self.enorm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # Full PanGu decoder layer (Sink Attention + MoE + Sandwich Norm)
        # The MTP layer at index 50 uses MoE (layer_idx >= first_k_dense_replace=4)
        # We pass a high layer_idx to ensure MoE is used
        self.decoder = PanguDecoderLayer(config, layer_idx=config.first_k_dense_replace)

        # Final layer norm (maps to shared_head.norm in weights)
        self.final_layernorm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        layer_device = self.enorm.weight.device

        # Get embeddings
        with torch.no_grad():
            inputs_embeds = embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(device=layer_device, dtype=hidden_states.dtype)
        hidden_states = hidden_states.to(device=layer_device)

        # Apply separate RMSNorm
        inputs_embeds_norm = self.enorm(inputs_embeds)
        hidden_states_norm = self.hnorm(hidden_states)

        # Concatenate and project: [enorm(emb), hnorm(h)] -> eh_proj -> hidden
        combined = torch.cat([inputs_embeds_norm, hidden_states_norm], dim=-1)
        hidden = self.eh_proj(combined)

        # Pass through full PanGu decoder layer
        layer_outputs = self.decoder(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        hidden = layer_outputs[0]
        present_key_value = layer_outputs[-1] if use_cache else None

        # Apply final layer norm (shared_head.norm)
        output = self.final_layernorm(hidden)

        return output, present_key_value


class PanguMTPDraftModel(nn.Module):
    """
    MTP Draft Model for OpenPanGu with tree-based speculative decoding.

    OpenPanGu has num_nextn_predict_layers=1 (single MTP layer at layer 50).
    Uses EAGLE-style tree buffers for candidate generation.
    """

    def __init__(self, config: PanguMTPConfig, load_emb: bool = False, path: str = None):
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
            PanguMTPLayer(config, layer_idx=i) for i in range(self.num_mtp_layers)
        ])

        # Shared head (lm_head equivalent for MTP, maps to shared_head.head in weights)
        self.shared_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tree buffers for speculative decoding
        self.tree_mask = None
        self.tree_buffer = None
        self.stable_kv = None
        self.diff_device = False
        self.headweight = None

    def _load_embeddings(self, path: str):
        """Load embeddings from base model."""
        from safetensors import safe_open

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
        past_key_values_length,
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
                device=device,
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
                device=hidden_states.device,
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
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
            return output, (present_key_value,)
        return output

    @torch.no_grad()
    def repeat_hidden(self, hidden_state, repeat_num):
        """Repeat hidden states for tree structure."""
        new_hidden = []
        for id, i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:, id:id + 1].repeat(1, i, 1))
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
            dim=-1,
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
        use_cache: bool = True,
    ):
        """
        Generate top-k candidates using MTP layers with tree structure.

        Uses shared_head (self.shared_head) for logit computation instead of
        the base model's lm_head, but accepts head parameter for API compatibility.
        For PanGu MTP, we use self.shared_head which maps to shared_head.head weights.
        """
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        ss_token, ss_prob, ss_op = [], [], []
        len_posi = input_ids.shape[1]
        self.reset()

        # Determine which head to use
        # Use shared_head if available and on same device, otherwise use passed head
        if not self.diff_device:
            use_head = head
        else:
            if hasattr(self, "headweight") and self.headweight is not None:
                use_head = None  # will use F.linear with headweight
            else:
                use_head = head

        if use_cache:
            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                kv_len = self.stable_kv[0][0].shape[2]
                out_hidden, past_key_values = self.forward(
                    hidden_states,
                    input_ids[:, kv_len:],
                    layer_idx=0,
                    past_key_values=self.stable_kv,
                    use_cache=True,
                )
            else:
                out_hidden, past_key_values = self.forward(
                    hidden_states,
                    input_ids,
                    layer_idx=0,
                    use_cache=True,
                )

            self.stable_kv = past_key_values
            last_hidden = out_hidden[:, -1]

            if use_head is not None:
                last_headout = use_head(last_hidden)
            else:
                last_headout = F.linear(last_hidden, self.headweight)

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

                out_hidden, past_key_values = self.forward(
                    hidden_states,
                    input_ids,
                    layer_idx=0,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                len_posi += 1

                if use_head is not None:
                    last_headout = use_head(out_hidden[0])
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
