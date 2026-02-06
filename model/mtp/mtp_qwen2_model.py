"""
MTP (Multi-Token Prediction) Model for Qwen2 Speculative Decoding

This module provides the MTPQwen2Model class that wraps a base Qwen2 model
and 7 MTP draft layers for speculative decoding. Each MTP layer predicts
a different future token position using a tree-based candidate structure.

Key differences from EAGLE:
1. Separate RMSNorm for embeddings (enorm) and hidden states (hnorm)
2. 7 MTP layers instead of 1 EAGLE layer
3. Native Qwen2 architecture support

For other model architectures (e.g., Llama), see corresponding mtp_*_model.py files.
"""

import copy
import json
import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from .modeling_qwen2_kv import KVQwen2ForCausalLM
from .cnets import MTPDraftModel
from .configs import MTPConfig
from .kv_cache import initialize_past_key_values
from .utils import (
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


class MTPQwen2Model(nn.Module):
    """
    MTP Model for speculative decoding with Qwen2.

    This class combines a base Qwen2 model with 7 MTP draft layers.
    Each MTP layer predicts tokens at a different future position,
    creating a tree of candidate sequences for verification.

    Args:
        base_model: The base Qwen2 model (KVQwen2ForCausalLM)
        base_model_name_or_path: Path to base model for tokenizer
        mtp_model_path: Path to MTP weights config
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
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)

        # Create MTP config from base config
        mtp_config = self._create_mtp_config(mtp_model_path)

        # Create MTP draft model
        self.mtp_layer = MTPDraftModel(mtp_config, load_emb=True, path=base_model_name_or_path)

        # Handle multi-device scenarios
        low_memory = False
        device = base_model.model.layers[-1].self_attn.q_proj.weight.device

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

    def _create_mtp_config(self, mtp_model_path: str) -> MTPConfig:
        """Create MTPConfig from config file or base model config."""
        # Try to load from MTP config file
        if os.path.isdir(mtp_model_path):
            config_path = os.path.join(mtp_model_path, "config.json")
        else:
            config_path = mtp_model_path

        try:
            with open(config_path, "r") as f:
                config_dict = json.loads(f.read())

            # Check if this is a combined model with MTP layers
            num_mtp_layers = config_dict.get("num_nextn_predict_layers", 7)

            return MTPConfig(
                vocab_size=config_dict.get("vocab_size", self.vocab_size),
                hidden_size=config_dict.get("hidden_size", self.hidden_size),
                intermediate_size=config_dict.get("intermediate_size", self.config.intermediate_size),
                num_hidden_layers=1,
                num_mtp_layers=num_mtp_layers,
                num_attention_heads=config_dict.get("num_attention_heads", self.config.num_attention_heads),
                num_key_value_heads=config_dict.get("num_key_value_heads", self.config.num_key_value_heads),
                hidden_act=config_dict.get("hidden_act", "silu"),
                max_position_embeddings=config_dict.get("max_position_embeddings", self.config.max_position_embeddings),
                rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
                rope_theta=config_dict.get("rope_theta", 1000000.0),
                attention_bias=True,
            )
        except Exception as e:
            # Fall back to creating from base config
            return MTPConfig.from_qwen2_config(self.config, num_mtp_layers=7)

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
        Load MTPQwen2Model from pretrained weights.

        Args:
            base_model_path: Path to base Qwen2 model
            mtp_model_path: Path to MTP draft head weights
            **kwargs: Additional arguments for model loading

        Returns:
            MTPQwen2Model: Loaded model
        """
        # Verify it's a Qwen2 model
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        arch = config.architectures[0] if config.architectures else ""

        if "Qwen2" not in arch:
            raise ValueError(f"MTP model requires Qwen2 architecture, got {arch}")

        # Load base model
        base_model = KVQwen2ForCausalLM.from_pretrained(base_model_path, **kwargs)

        # Get config path
        if os.path.isdir(mtp_model_path):
            config_path = os.path.join(mtp_model_path, "config.json")
        else:
            config_path = mtp_model_path

        # Create model
        model = cls(base_model, base_model_path, config_path)

        # Load MTP weights
        model._load_mtp_weights(mtp_model_path)

        return model

    def _load_mtp_weights(self, mtp_model_path: str):
        """Load MTP layer weights from checkpoint."""
        from safetensors import safe_open

        if os.path.isdir(mtp_model_path):
            # Load from safetensors shards
            index_path = os.path.join(mtp_model_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    index = json.load(f)

                weight_map = index["weight_map"]
                mtp_weights = {}

                # Find which files contain MTP weights
                mtp_files = set()
                for name, filename in weight_map.items():
                    if name.startswith("model.mtp."):
                        mtp_files.add(filename)

                # Load MTP weights from each file
                for filename in mtp_files:
                    filepath = os.path.join(mtp_model_path, filename)
                    with safe_open(filepath, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.startswith("model.mtp."):
                                mtp_weights[key] = f.get_tensor(key)

                # Map weights to MTP layers
                self._map_mtp_weights(mtp_weights)

            elif os.path.exists(os.path.join(mtp_model_path, "pytorch_model.bin")):
                # Load from pytorch_model.bin
                weights = torch.load(
                    os.path.join(mtp_model_path, "pytorch_model.bin"),
                    map_location="cpu"
                )
                mtp_weights = {k: v for k, v in weights.items() if k.startswith("model.mtp.")}
                self._map_mtp_weights(mtp_weights)

    def _map_mtp_weights(self, mtp_weights: dict):
        """Map loaded MTP weights to the model structure."""
        state_dict = {}

        for name, tensor in mtp_weights.items():
            # Convert from model.mtp.{i}.* to mtp_layers.{i}.*
            # Example: model.mtp.0.enorm.weight -> mtp_layers.0.enorm.weight
            if name.startswith("model.mtp."):
                parts = name.split(".")
                layer_idx = int(parts[2])
                rest = ".".join(parts[3:])

                # Map to MTPLayer structure
                new_name = f"mtp_layers.{layer_idx}.{rest}"

                # Handle decoder sub-layer weights
                if rest.startswith("self_attn.") or rest.startswith("mlp.") or \
                   rest.startswith("input_layernorm.") or rest.startswith("post_attention_layernorm."):
                    new_name = f"mtp_layers.{layer_idx}.decoder.{rest}"

                state_dict[new_name] = tensor

        # Load the mapped weights
        missing, unexpected = self.mtp_layer.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"Warning: Missing MTP weights: {missing[:5]}..." if len(missing) > 5 else f"Missing: {missing}")
        if unexpected:
            print(f"Warning: Unexpected weights: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected: {unexpected}")

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

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for loss computation
            past_key_values: Cached key-value states
            output_orig: Whether to output original logits
            position_ids: Position IDs
            init: Whether this is the initial forward pass
            logits_processor: Processor for logits

        Returns:
            Tuple of (mtp_logits, hidden_states, token) or (outputs, orig_logits, hidden_states)
        """
        with torch.inference_mode():
            # Pass input through the base model
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
        """
        Generate text using MTP speculative decoding.

        Args:
            input_ids: Input token IDs
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum new tokens to generate
            max_length: Maximum total sequence length
            tree_choices: Tree structure for speculation

        Returns:
            Generated token IDs
        """
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

        # Set up tree buffers
        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices,
                device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device
            )
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize KV cache
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
                logits_processor
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
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2],
                tree_buffers["p_indices"], tree_candidates, tree_buffers["b_indices"]
            )

            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                return input_ids
            if new_token > max_new_tokens:
                return input_ids
            if input_ids.shape[1] > max_length:
                return input_ids

        return input_ids

    @torch.no_grad()
    def mtp_generate_yield(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        max_steps: int = 512,
        tree_choices=None,
    ):
        """
        Generate text using MTP speculative decoding with yield for streaming.

        Yields input_ids after each step.
        """
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
                device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
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

        for idx in range(max_steps):
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
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
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2],
                tree_buffers["p_indices"], tree_candidates, tree_buffers["b_indices"]
            )

            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break

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
        """
        Generate text using standard autoregressive decoding (baseline).

        Yields input_ids after each step.
        """
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
                device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
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
