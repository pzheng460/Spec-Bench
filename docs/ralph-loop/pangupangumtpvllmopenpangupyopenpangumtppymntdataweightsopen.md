# Spec: OpenPanGu & OpenPanGu MTP Adaptation for Spec-Bench

## Overview
Adapt OpenPanGu (PanguProMoEV2ForCausalLM, 72B MoE) model and its MTP speculative decoding draft model to work within the Spec-Bench framework. This includes implementing a full custom KV-cache-optimized base model, an MTP draft model with complete Sink Attention + MoE architecture, and both MTP speculative decoding and baseline autoregressive inference scripts.

## Architecture Context

### OpenPanGu Model Architecture
- **Architecture**: PanguProMoEV2ForCausalLM
- **hidden_size**: 4608
- **vocab_size**: 153600
- **num_hidden_layers**: 50 (base) + 1 (MTP layer at index 50)
- **num_attention_heads**: 64, **num_key_value_heads**: 4
- **Attention**: Sink Attention with qk_nope_dim=128, qk_rope_dim=64, v_channels=128, param_sink_number=128, param_sink_with_value=true
- **MLP**: Dense for layers 0-3 (intermediate_size=10240), MoE for layers 4-49 (80 routed experts, 2 shared experts, moe_intermediate_size=1280, num_experts_per_tok=8)
- **Norms**: sandwich_norm=true (input_layernorm, post_attention_layernorm, pre_mlp_layernorm, post_mlp_layernorm per layer)
- **Other**: routed_scaling_factor=2.5, router_enable_expert_bias=true, rope_theta=10000, tie_word_embeddings=false

### MTP Layer (layer 50) Structure
The MTP layer is stored as `model.layers.50.*` in the safetensors weights:
- `enorm.weight`, `hnorm.weight` — separate RMSNorm for embeddings and hidden states
- `eh_proj.weight` — linear projection from 2*hidden_size to hidden_size
- `embed_tokens.weight` — shared embedding for MTP
- `shared_head.head.weight`, `shared_head.norm.weight` — MTP-specific LM head
- Full decoder layer: self_attn (Sink Attention with qkv_proj, o_proj, k_layernorm, param_sink_key, param_sink_value), mlp (MoE with gate, shared_experts, 80 experts), sandwich norm layers
- `num_nextn_predict_layers`: 1 (single MTP draft layer)

### Key Differences from Qwen2 MTP
1. **Sink Attention** vs standard GQA: fused qkv_proj, k_layernorm, parametric sink keys/values, asymmetric K/V dims
2. **MoE in MTP layer**: Full MoE (80 experts + 2 shared) vs dense MLP
3. **Sandwich norm**: 4 norm layers per decoder layer vs 2
4. **shared_head**: MTP has its own LM head (norm + linear) vs using base model's lm_head
5. **Single MTP layer**: 1 vs 7 in Qwen2
6. **Asymmetric KV**: K head_dim=192 (qk_nope_dim+qk_rope_dim), V dim=128 (v_channels)

## User Stories

### US-1: Implement KV-cache-optimized PanGu base model
**Description**: Create `model/pangu/modeling_pangu_kv.py` with `KVPanguForCausalLM` that implements the full PanGu architecture with KV-cache support for efficient autoregressive decoding.

**Components**:
- PanguRMSNorm, PanguRotaryEmbedding
- PanguSinkAttention: fused qkv_proj splitting into Q(num_heads*(qk_nope_dim+qk_rope_dim)), K(num_kv_heads*(qk_nope_dim+qk_rope_dim)), V(num_kv_heads*v_channels); k_layernorm; partial rotary on qk_rope_dim portion; parametric sink keys/values prepended to KV cache
- PanguMLP (dense, for layers 0-3)
- PanguMoE (for layers 4-49): gate router, 80 experts, 2 shared experts with scoring_func="sigmoid"
- PanguDecoderLayer: sandwich_norm support (input_layernorm, post_attention_layernorm, pre_mlp_layernorm, post_mlp_layernorm)
- PanguModel: embedding + 50 decoder layers + final norm
- KVPanguForCausalLM: PanguModel + lm_head, with KV-cache-compatible forward()
- Asymmetric KV cache: K uses head_dim=192, V uses v_channels=128

**Acceptance Criteria**:
- Model loads from safetensors weights at `/mnt/data/weights/openPangu-R-72B-2512/` (layers 0-49 only, skipping layer 50)
- Forward pass produces logits of shape (batch, seq_len, 153600)
- KV cache correctly stores/retrieves asymmetric K(192)/V(128) dimensions
- Works with device_map="auto" across multiple GPUs/NPUs
- Loads with `trust_remote_code=True` for tokenizer from the weight directory

### US-2: Implement PanGu MTP draft model
**Description**: Create `model/pangu/cnets.py` with `PanguMTPLayer` and `PanguMTPDraftModel` implementing the full MTP draft architecture with Sink Attention + MoE.

**Components**:
- PanguMTPLayer: enorm, hnorm (separate RMSNorm), eh_proj (2*hidden→hidden), full PanguDecoderLayer (Sink Attn + MoE + sandwich norm), shared_head (norm + linear for logit generation)
- PanguMTPDraftModel: embedding layer, single MTPLayer, tree buffer management, topK_generate method

**Key Design**:
- MTP layer reuses same PanguSinkAttention and PanguMoE classes from US-1
- shared_head replaces base model's lm_head for draft logit generation
- Tree-based speculative decoding with EAGLE-style tree buffers (reuse model/mtp/utils.py, utils_c.py)

**Acceptance Criteria**:
- MTP draft model loads layer 50 weights correctly from safetensors
- Weight mapping: `model.layers.50.{enorm,hnorm,eh_proj,embed_tokens}.*` → MTP layer top-level; `model.layers.50.{self_attn,mlp,input_layernorm,...}.*` → decoder sub-layer; `model.layers.50.shared_head.*` → shared_head
- topK_generate produces tree of candidate tokens
- KV cache for draft layer handles asymmetric K/V dims

### US-3: Implement MTPPanguModel wrapper
**Description**: Create `model/pangu/mtp_pangu_model.py` with `MTPPanguModel` class that wraps KVPanguForCausalLM + PanguMTPDraftModel.

**Components**:
- MTPPanguModel: loads base model + MTP draft weights, manages combined forward pass
- from_pretrained classmethod: dual-path loading (--base-model-path, --mtp-model-path, both can point to same directory)
- Config creation: read config.json, create PanguMTPConfig from base config
- Multi-device handling: detect if MTP layer is on different device than lm_head

**Acceptance Criteria**:
- `MTPPanguModel.from_pretrained(base_model_path, mtp_model_path)` loads successfully with dual paths pointing to same directory
- Base model forward + MTP draft forward produces correct tree logits
- Works on multi-GPU with device_map="auto"
- Works on NPU with max_memory specification

### US-4: Implement PanGu MTP config
**Description**: Create `model/pangu/configs.py` with `PanguMTPConfig` extending PretrainedConfig.

**Config Parameters**:
- All base PanGu config fields: hidden_size, vocab_size, num_attention_heads, num_key_value_heads, intermediate_size, qk_nope_dim, qk_rope_dim, v_channels, param_sink_number, param_sink_with_value, sandwich_norm, etc.
- MoE fields: n_routed_experts, n_shared_experts, moe_intermediate_size, num_experts_per_tok, first_k_dense_replace, routed_scaling_factor, router_enable_expert_bias
- MTP-specific: num_mtp_layers=1, num_nextn_predict_layers=1
- Factory method: `from_pangu_config(config)` to create from base config

**Acceptance Criteria**:
- Config correctly parses config.json from weight directory
- num_nextn_predict_layers=1 is respected
- All architecture parameters are accessible

### US-5: Implement PanGu KV cache
**Description**: Extend or create KV cache initialization for PanGu's asymmetric K/V dimensions in `model/pangu/kv_cache.py`.

**Design Decision**: Use separate K and V cache tensors with different last dimensions:
- K cache: shape (num_layers*2_for_kv, batch, num_kv_heads, max_length, qk_nope_dim+qk_rope_dim=192)
- V cache: shape (num_layers*2_for_kv, batch, num_kv_heads, max_length, v_channels=128)
- Or: unified approach with max(head_dim, v_channels) and slicing

The concrete approach should be determined during implementation based on compatibility with the existing KVCache class from model/mtp/kv_cache.py.

**Acceptance Criteria**:
- KV cache correctly handles K dim=192, V dim=128
- initialize_past_key_values works across multiple devices
- Cache reset (zero_()) works correctly

### US-6: Create inference_pangu_mtp.py
**Description**: Create `evaluation/inference_pangu_mtp.py` for MTP speculative decoding inference with PanGu.

**Components**:
- pangu_mtp_forward function: same interface as mtp_forward in inference_mtp.py
- Argument parser: --base-model-path, --mtp-model-path, --model-id, --bench-name, --dtype, --temperature, etc.
- Device detection via device_utils.py (CUDA + NPU)
- Reuse model/mtp/utils.py for tree operations (generate_tree_buffers, initialize_tree, generate_candidates, tree_decoding, evaluate_posterior, update_inference_inputs)

**Acceptance Criteria**:
- `python -m evaluation.inference_pangu_mtp --base-model-path /mnt/data/weights/openPangu-R-72B-2512 --mtp-model-path /mnt/data/weights/openPangu-R-72B-2512 --model-id openpangu-72b-mtp --bench-name spec_bench --dtype bfloat16 --question-begin 0 --question-end 1` runs successfully
- Output JSONL file contains valid model answers
- Returns (input_ids, new_token, step, accept_length_list) tuple

### US-7: Create inference_pangu_baseline.py
**Description**: Create `evaluation/inference_pangu_baseline.py` for baseline autoregressive inference with PanGu.

**Components**:
- pangu_baseline_forward function: standard autoregressive generation
- Loads KVPanguForCausalLM directly (no MTP draft)
- Same argument interface as inference_baseline.py

**Acceptance Criteria**:
- `python -m evaluation.inference_pangu_baseline --model-path /mnt/data/weights/openPangu-R-72B-2512 --model-id openpangu-72b-vanilla --bench-name spec_bench --dtype bfloat16 --question-begin 0 --question-end 1` runs successfully
- Output JSONL file contains valid model answers
- accept_length_list is all 1s (no speculation)

### US-8: Update run.sh for PanGu
**Description**: Add PanGu MTP and baseline commands to run.sh.

**Acceptance Criteria**:
- run.sh contains commented-out PanGu commands for both baseline and MTP
- Commands use correct paths and default parameters

## File Structure

```
model/pangu/
├── __init__.py                      # Module exports
├── configs.py                       # PanguMTPConfig
├── modeling_pangu_kv.py             # KVPanguForCausalLM (base model with KV cache)
├── cnets.py                         # PanguMTPLayer, PanguMTPDraftModel (draft model)
├── mtp_pangu_model.py               # MTPPanguModel (wrapper combining base + draft)
└── kv_cache.py                      # PanGu-specific KV cache (asymmetric K/V)

evaluation/
├── inference_pangu_mtp.py           # MTP speculative decoding inference
└── inference_pangu_baseline.py      # Baseline autoregressive inference
```

## Weight Mapping

### Base Model (layers 0-49)
```
model.embed_tokens.weight                    → self.model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight      → self.model.layers[i].input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight → ...
model.layers.{i}.pre_mlp_layernorm.weight    → ... (sandwich_norm)
model.layers.{i}.post_mlp_layernorm.weight   → ... (sandwich_norm)
model.layers.{i}.self_attn.qkv_proj.weight   → split into Q, K, V
model.layers.{i}.self_attn.o_proj.weight      → ...
model.layers.{i}.self_attn.k_layernorm.weight → ...
model.layers.{i}.self_attn.param_sink_key    → ...
model.layers.{i}.self_attn.param_sink_value  → ...
model.layers.{i}.mlp.*                        → dense MLP (i<4) or MoE (i>=4)
model.norm.weight                             → self.model.norm.weight
lm_head.weight                                → self.lm_head.weight
```

### MTP Layer (layer 50)
```
model.layers.50.enorm.weight                 → mtp_layer.enorm.weight
model.layers.50.hnorm.weight                 → mtp_layer.hnorm.weight
model.layers.50.eh_proj.weight               → mtp_layer.eh_proj.weight
model.layers.50.embed_tokens.weight          → mtp_layer.embed_tokens.weight
model.layers.50.shared_head.head.weight      → mtp_layer.shared_head.head.weight
model.layers.50.shared_head.norm.weight      → mtp_layer.shared_head.norm.weight
model.layers.50.self_attn.*                  → mtp_layer.decoder.self_attn.*
model.layers.50.mlp.*                        → mtp_layer.decoder.mlp.*
model.layers.50.input_layernorm.*            → mtp_layer.decoder.input_layernorm.*
model.layers.50.post_attention_layernorm.*   → mtp_layer.decoder.post_attention_layernorm.*
model.layers.50.pre_mlp_layernorm.*          → mtp_layer.decoder.pre_mlp_layernorm.*
model.layers.50.post_mlp_layernorm.*         → mtp_layer.decoder.post_mlp_layernorm.*
```

## Implementation Phases

### Phase 1: Foundation (US-4, US-5)
- PanguMTPConfig
- PanGu KV cache with asymmetric K/V support
- **Verification**: Config loads correctly from weight directory

### Phase 2: Base Model (US-1)
- KVPanguForCausalLM with full Sink Attention + MoE
- **Verification**: Base model loads and produces logits for a single forward pass

### Phase 3: MTP Draft (US-2, US-3)
- PanguMTPDraftModel with Sink Attention + MoE
- MTPPanguModel wrapper
- **Verification**: Combined model loads and runs topK_generate

### Phase 4: Inference Scripts (US-6, US-7, US-8)
- inference_pangu_mtp.py
- inference_pangu_baseline.py
- run.sh updates
- **Verification**: Both scripts produce valid JSONL output on spec_bench question 0

## Dependencies
- Reuse from model/mtp/: utils.py, utils_c.py (tree operations)
- Reuse from model/eagle/: choices.py (mc_sim_7b_63 tree structure)
- Reuse from evaluation/: eval.py, device_utils.py
- External: torch, transformers, safetensors, accelerate

## Verification Commands
```bash
# Phase 2: Test base model loading
python -c "
from model.pangu.modeling_pangu_kv import KVPanguForCausalLM
import torch
model = KVPanguForCausalLM.from_pretrained('/mnt/data/weights/openPangu-R-72B-2512', torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
print('Base model loaded successfully')
"

# Phase 4: Test MTP inference
CUDA_VISIBLE_DEVICES=0,1 python -m evaluation.inference_pangu_mtp \
    --base-model-path /mnt/data/weights/openPangu-R-72B-2512 \
    --mtp-model-path /mnt/data/weights/openPangu-R-72B-2512 \
    --model-id openpangu-72b-mtp \
    --bench-name spec_bench \
    --dtype bfloat16 \
    --temperature 0.0 \
    --question-begin 0 \
    --question-end 1

# Phase 4: Test baseline inference
CUDA_VISIBLE_DEVICES=0,1 python -m evaluation.inference_pangu_baseline \
    --model-path /mnt/data/weights/openPangu-R-72B-2512 \
    --model-id openpangu-72b-vanilla \
    --bench-name spec_bench \
    --dtype bfloat16 \
    --temperature 0.0 \
    --question-begin 0 \
    --question-end 1
```
