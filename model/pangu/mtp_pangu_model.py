"""
MTP (Multi-Token Prediction) Model for OpenPanGu Speculative Decoding

This module provides the MTPPanguModel class that wraps a base OpenPanGu model
(KVPanguForCausalLM) and 1 MTP draft layer for speculative decoding.

Key differences from Qwen2 MTP:
1. Sink Attention with asymmetric K/V dims (K=192, V=128)
2. MoE with 80 routed experts + 2 shared experts in the MTP layer
3. Sandwich norm (4 norms per decoder layer)
4. Parametric sink key/value tokens
5. Weight loading from model.layers.50.* in safetensors

Weight mapping for MTP layer 50:
  - model.layers.50.enorm.weight -> mtp_layers.0.enorm.weight
  - model.layers.50.hnorm.weight -> mtp_layers.0.hnorm.weight
  - model.layers.50.eh_proj.weight -> mtp_layers.0.eh_proj.weight
  - model.layers.50.embed_tokens.weight -> embed_tokens.weight
  - model.layers.50.shared_head.norm.weight -> mtp_layers.0.final_layernorm.weight
  - model.layers.50.shared_head.head.weight -> shared_head.weight
  - model.layers.50.self_attn.* -> mtp_layers.0.decoder.self_attn.*
  - model.layers.50.mlp.* -> mtp_layers.0.decoder.mlp.*
  - model.layers.50.{input,post_attention,pre_mlp,post_mlp}_layernorm.* -> mtp_layers.0.decoder.*
"""

import json
import os
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from .modeling_pangu_kv import KVPanguForCausalLM
from .cnets import PanguMTPDraftModel
from .configs import PanguMTPConfig
from .kv_cache import initialize_past_key_values
from ..mtp.utils import (
    generate_tree_buffers,
    initialize_tree,
    reset_tree_mode,
    generate_candidates,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    prepare_logits_processor,
)

try:
    from ..eagle.choices import mc_sim_7b_63
except ImportError:
    from model.eagle.choices import mc_sim_7b_63


class MTPPanguModel(nn.Module):
    """
    MTP Model for speculative decoding with OpenPanGu.

    Combines a base KVPanguForCausalLM model with 1 MTP draft layer.
    The MTP layer (layer 50) predicts the next token using tree-based
    speculative decoding.
    """

    def __init__(
        self,
        base_model,
        base_model_name_or_path: str,
        mtp_model_path: str,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path, trust_remote_code=True
        )

        # Create MTP config from base config
        mtp_config = self._create_mtp_config(mtp_model_path)

        # Create MTP draft model
        self.mtp_layer = PanguMTPDraftModel(
            mtp_config, load_emb=True, path=base_model_name_or_path
        )

        # Handle multi-device scenarios
        low_memory = False
        device = base_model.model.layers[-1].self_attn.qkv_proj.weight.device

        if device != base_model.lm_head.weight.device:
            self.mtp_layer.diff_device = True
            if not low_memory:
                self.mtp_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.mtp_layer.layer_device = device
        else:
            self.mtp_layer.diff_device = False

        # Move MTP layers to correct device and dtype
        self.mtp_layer.to(self.base_model.dtype).to(device)
        self.mtp_layer.init_tree()

    def _create_mtp_config(self, mtp_model_path: str) -> PanguMTPConfig:
        """Create PanguMTPConfig from config file."""
        if os.path.isdir(mtp_model_path):
            config_path = os.path.join(mtp_model_path, "config.json")
        else:
            config_path = mtp_model_path

        try:
            with open(config_path, "r") as f:
                config_dict = json.loads(f.read())

            num_mtp_layers = config_dict.get("num_nextn_predict_layers", 1)

            return PanguMTPConfig(
                vocab_size=config_dict.get("vocab_size", self.vocab_size),
                hidden_size=config_dict.get("hidden_size", self.hidden_size),
                intermediate_size=config_dict.get("intermediate_size", 10240),
                num_hidden_layers=1,
                num_mtp_layers=num_mtp_layers,
                num_attention_heads=config_dict.get("num_attention_heads", 64),
                num_key_value_heads=config_dict.get("num_key_value_heads", 4),
                hidden_act=config_dict.get("hidden_act", "silu"),
                max_position_embeddings=config_dict.get("max_position_embeddings", 4096),
                rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
                rope_theta=config_dict.get("rope_theta", 10000.0),
                qk_nope_dim=config_dict.get("qk_nope_dim", 128),
                qk_rope_dim=config_dict.get("qk_rope_dim", 64),
                v_channels=config_dict.get("v_channels", 128),
                param_sink_number=config_dict.get("param_sink_number", 128),
                param_sink_with_value=config_dict.get("param_sink_with_value", True),
                n_routed_experts=config_dict.get("n_routed_experts", 80),
                n_shared_experts=config_dict.get("n_shared_experts", 2),
                moe_intermediate_size=config_dict.get("moe_intermediate_size", 1280),
                num_experts_per_tok=config_dict.get("num_experts_per_tok", 8),
                first_k_dense_replace=config_dict.get("first_k_dense_replace", 4),
                routed_scaling_factor=config_dict.get("routed_scaling_factor", 2.5),
                norm_topk_prob=config_dict.get("norm_topk_prob", True),
                router_enable_expert_bias=config_dict.get("router_enable_expert_bias", True),
                sandwich_norm=config_dict.get("sandwich_norm", True),
            )
        except Exception:
            return PanguMTPConfig.from_pangu_config(self.config, num_mtp_layers=1)

    def get_tokenizer(self):
        """Get the tokenizer of the base model."""
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        mtp_model_path: str,
        **kwargs,
    ):
        """
        Load MTPPanguModel from pretrained weights.

        For OpenPanGu, both the base model (layers 0-49) and MTP layer (layer 50)
        are stored in the same model directory. The mtp_model_path is typically the
        same as base_model_path.

        Args:
            base_model_path: Path to base OpenPanGu model weights
            mtp_model_path: Path to MTP weights (same dir or separate)
            **kwargs: Additional arguments for model loading
        """
        # Load config
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

        # Override num_hidden_layers to only load base layers (0-49)
        base_num_layers = config.num_hidden_layers
        mtp_layers = getattr(config, 'num_nextn_predict_layers', 1)
        if mtp_layers > 0:
            # The config reports total layers including MTP
            # For base model, only load the non-MTP layers
            pass  # KVPanguForCausalLM uses config.num_hidden_layers directly

        # Create a config for the base model with the correct number of layers
        base_config = PanguMTPConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=base_num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 10000.0),
            qk_nope_dim=getattr(config, 'qk_nope_dim', 128),
            qk_rope_dim=getattr(config, 'qk_rope_dim', 64),
            v_channels=getattr(config, 'v_channels', 128),
            param_sink_number=getattr(config, 'param_sink_number', 128),
            param_sink_with_value=getattr(config, 'param_sink_with_value', True),
            n_routed_experts=getattr(config, 'n_routed_experts', 80),
            n_shared_experts=getattr(config, 'n_shared_experts', 2),
            moe_intermediate_size=getattr(config, 'moe_intermediate_size', 1280),
            num_experts_per_tok=getattr(config, 'num_experts_per_tok', 8),
            first_k_dense_replace=getattr(config, 'first_k_dense_replace', 4),
            routed_scaling_factor=getattr(config, 'routed_scaling_factor', 2.5),
            norm_topk_prob=getattr(config, 'norm_topk_prob', True),
            router_enable_expert_bias=getattr(config, 'router_enable_expert_bias', True),
            sandwich_norm=getattr(config, 'sandwich_norm', True),
        )

        # Load base model via accelerate (meta-device init to avoid OOM)
        dtype = kwargs.get('torch_dtype', torch.float16)
        device_map = kwargs.get('device_map', "auto")
        max_memory = kwargs.get('max_memory', None)

        print("Loading base model with accelerate (meta-device init)...")
        base_model = KVPanguForCausalLM.from_pretrained(
            base_model_path,
            config=base_config,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )

        # Get config path for MTP
        if os.path.isdir(mtp_model_path):
            config_path = os.path.join(mtp_model_path, "config.json")
        else:
            config_path = mtp_model_path

        # Create model wrapper
        model = cls(base_model, base_model_path, config_path)

        # Load MTP weights
        model._load_mtp_weights(mtp_model_path)

        return model

    def _load_mtp_weights(self, mtp_model_path: str):
        """Load MTP layer weights (layer 50) from safetensors."""
        from safetensors import safe_open

        if not os.path.isdir(mtp_model_path):
            return

        index_path = os.path.join(mtp_model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            return

        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        num_base_layers = self.config.num_hidden_layers

        # Find files containing MTP layer weights
        mtp_files = set()
        for name, filename in weight_map.items():
            if "layers." in name:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                if layer_idx >= num_base_layers:
                    mtp_files.add(filename)

        # Load MTP weights
        print(f"Loading MTP weights from {len(mtp_files)} shard(s)...")
        mtp_weights = {}
        for filename in sorted(mtp_files):
            filepath = os.path.join(mtp_model_path, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "layers." in key:
                        layer_idx = int(key.split("layers.")[1].split(".")[0])
                        if layer_idx >= num_base_layers:
                            mtp_weights[key] = f.get_tensor(key)
        print(f"Loaded {len(mtp_weights)} MTP weight tensors")

        # Map weights to MTP draft model
        self._map_mtp_weights(mtp_weights, num_base_layers)

    def _map_mtp_weights(self, mtp_weights: dict, mtp_start_layer: int):
        """
        Map loaded MTP weights to the PanguMTPDraftModel structure.

        Weight mapping for layer 50:
          model.layers.50.enorm.weight -> mtp_layers.0.enorm.weight
          model.layers.50.hnorm.weight -> mtp_layers.0.hnorm.weight
          model.layers.50.eh_proj.weight -> mtp_layers.0.eh_proj.weight
          model.layers.50.embed_tokens.weight -> embed_tokens.weight
          model.layers.50.shared_head.norm.weight -> mtp_layers.0.final_layernorm.weight
          model.layers.50.shared_head.head.weight -> shared_head.weight
          model.layers.50.self_attn.* -> mtp_layers.0.decoder.self_attn.*
          model.layers.50.mlp.* -> mtp_layers.0.decoder.mlp.*
          model.layers.50.{input,post_attention,pre_mlp,post_mlp}_layernorm.* -> mtp_layers.0.decoder.*
        """
        # Top-level MTP weight names (not part of the decoder sub-layer)
        spec_layer_weight_names = ["embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head"]
        # Decoder sub-layer weight prefixes
        decoder_prefixes = [
            "self_attn.", "mlp.", "input_layernorm.", "post_attention_layernorm.",
            "pre_mlp_layernorm.", "post_mlp_layernorm.",
        ]

        state_dict = {}

        for name, tensor in mtp_weights.items():
            if "layers." not in name:
                continue

            layer_idx = int(name.split("layers.")[1].split(".")[0])
            mtp_idx = layer_idx - mtp_start_layer

            if mtp_idx < 0 or mtp_idx >= self.mtp_layer.num_mtp_layers:
                continue

            # Get the rest of the key after "model.layers.{layer_idx}."
            prefix = f"model.layers.{layer_idx}."
            rest = name[len(prefix):]

            # Check if this is a top-level MTP weight
            is_spec_weight = False
            for spec_name in spec_layer_weight_names:
                if rest.startswith(spec_name):
                    is_spec_weight = True
                    break

            if is_spec_weight:
                if rest.startswith("embed_tokens."):
                    # embed_tokens is shared, goes to top-level
                    new_name = rest  # embed_tokens.weight
                elif rest.startswith("shared_head.norm."):
                    # shared_head.norm -> mtp_layers.{i}.final_layernorm
                    sub = rest[len("shared_head.norm."):]
                    new_name = f"mtp_layers.{mtp_idx}.final_layernorm.{sub}"
                elif rest.startswith("shared_head.head."):
                    # shared_head.head -> shared_head (top-level)
                    sub = rest[len("shared_head.head."):]
                    new_name = f"shared_head.{sub}"
                elif rest.startswith("enorm.") or rest.startswith("hnorm.") or rest.startswith("eh_proj."):
                    new_name = f"mtp_layers.{mtp_idx}.{rest}"
                else:
                    new_name = f"mtp_layers.{mtp_idx}.{rest}"
            else:
                # Decoder sub-layer weights
                new_name = f"mtp_layers.{mtp_idx}.decoder.{rest}"

            state_dict[new_name] = tensor

        # Load the mapped weights
        missing, unexpected = self.mtp_layer.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"MTP weight loading - Missing ({len(missing)}): {missing[:10]}...")
        if unexpected:
            print(f"MTP weight loading - Unexpected ({len(unexpected)}): {unexpected[:10]}...")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values=None,
        output_orig: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        init: bool = True,
        logits_processor=None,
    ):
        """
        Forward pass through base model and MTP draft layers.

        Returns:
            If init=True: (mtp_logits, hidden_states, token) or
                          (mtp_logits, outputs, orig, hidden_states, token)
            If init=False: (outputs, orig, hidden_states)
        """
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()

        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]

            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

            # Generate draft tokens using MTP
            mtp_logits = self.mtp_layer.topK_generate(
                hidden_states, input_ids, self.base_model.lm_head, logits_processor
            )

            if output_orig:
                return mtp_logits, outputs, orig, hidden_states, token
            return mtp_logits, hidden_states, token
        else:
            if output_orig:
                return outputs, orig, hidden_states

    @torch.no_grad()
    def mtp_generate(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        max_new_tokens: int = 512,
        max_length: int = 2048,
        tree_choices=None,
    ):
        """Generate text using MTP speculative decoding."""
        if tree_choices is None:
            tree_choices = mc_sim_7b_63

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        input_ids = input_ids.clone()
        self.mtp_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices,
                device=self.base_model.model.layers[-1].self_attn.qkv_proj.weight.device,
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device
            )
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_length):
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor,
            )

            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob,
                tree_logits[2], tree_buffers["p_indices"], tree_candidates,
                tree_buffers["b_indices"],
            )

            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length,
                tree_buffers["retrieve_indices"], logits_processor, logits,
                tree_logits, new_token, past_key_values_data, current_length_data,
                self, hidden_state, hidden_state_new, sample_p,
            )

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                return input_ids
            if new_token > max_new_tokens:
                return input_ids
            if input_ids.shape[1] > max_length:
                return input_ids

        return input_ids

    @torch.no_grad()
    def naive_generate(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        max_steps: int = 512,
        tree_choices=None,
    ):
        """Generate text using standard autoregressive decoding (baseline). Yields input_ids."""
        if tree_choices is None:
            tree_choices = mc_sim_7b_63

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        input_ids = input_ids.clone()
        self.mtp_layer.reset_kv()

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_steps):
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
