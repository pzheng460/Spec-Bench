# MTP Model Adaptation Specification

## Overview

Adapt the QWQ-32B-MTP model for speculative decoding in Spec-Bench. MTP (Multi-Token Prediction) is architecturally similar to EAGLE but requires separate RMSNorm layers for embeddings and hidden states before concatenation. The implementation will be a standalone module under `model/mtp/`, independent of the EAGLE codebase.

## Background

### What is MTP?
MTP is a speculative decoding approach where multiple draft layers predict future tokens in parallel. Unlike EAGLE which typically uses 1 draft layer, MTP uses 7 layers where each layer predicts the token at a different future position (layer i predicts token at position t+i+1).

### Key Difference from EAGLE
```
EAGLE:
  embeddings ─────────────────┐
                              ├─→ concat → FC → decoder layers
  hidden_states ──────────────┘

MTP:
  embeddings → RMSNorm (enorm) ────┐
                                   ├─→ concat → eh_proj → decoder layer → final_layernorm
  hidden_states → RMSNorm (hnorm) ─┘
```

## Model Configuration

### Base Model (QwQ-32B)
- **Path**: `/mnt/data/weights/QwQ-32B/`
- **Architecture**: Qwen2ForCausalLM
- **Hidden size**: 5120
- **Intermediate size**: 27648
- **Attention heads**: 40 (8 KV heads, GQA)
- **Layers**: 64
- **Vocab size**: 152064
- **RMS norm epsilon**: 1e-05
- **Dtype**: bfloat16

### MTP Draft Head
- **Path**: `/mnt/data/weights/qwq-32b-mtp/`
- **Number of MTP layers**: 7 (`num_nextn_predict_layers: 7`)
- **Layer naming**: `model.mtp.0` through `model.mtp.6`

### MTP Layer Weight Structure
Each `model.mtp.{i}` layer contains:
| Weight | Shape | Description |
|--------|-------|-------------|
| `enorm.weight` | (5120,) | RMSNorm for embeddings |
| `hnorm.weight` | (5120,) | RMSNorm for hidden states |
| `eh_proj.weight` | (5120, 10240) | Linear projection after concat |
| `input_layernorm.weight` | (5120,) | Pre-attention norm |
| `self_attn.q_proj.weight` | (5120, 5120) | Query projection |
| `self_attn.k_proj.weight` | (1024, 5120) | Key projection |
| `self_attn.v_proj.weight` | (1024, 5120) | Value projection |
| `self_attn.o_proj.weight` | (5120, 5120) | Output projection |
| `self_attn.q_proj.bias` | (5120,) | Query bias |
| `self_attn.k_proj.bias` | (1024,) | Key bias |
| `self_attn.v_proj.bias` | (1024,) | Value bias |
| `post_attention_layernorm.weight` | (5120,) | Post-attention norm |
| `mlp.gate_proj.weight` | (27648, 5120) | Gate projection |
| `mlp.up_proj.weight` | (27648, 5120) | Up projection |
| `mlp.down_proj.weight` | (5120, 27648) | Down projection |
| `final_layernorm.weight` | (5120,) | Output normalization |

## Architecture

### MTP Forward Pass
```
For MTP layer i (predicting token at position t+i+1):

Input:
  - input_ids: token IDs to embed
  - hidden_states: output from base model's final layer

Process:
  1. inputs_embeds = embed_tokens(input_ids)
  2. inputs_embeds = enorm(inputs_embeds)          # RMSNorm on embeddings
  3. hidden_states_norm = hnorm(hidden_states)     # RMSNorm on hidden states
  4. combined = concat([inputs_embeds, hidden_states_norm], dim=-1)
  5. hidden = eh_proj(combined)                    # Project back to hidden_size
  6. hidden = Qwen2DecoderLayer(hidden)            # Self-attention + MLP
  7. output = final_layernorm(hidden)

Output:
  - output → lm_head → logits for next token prediction
```

### Tree Structure for Speculation
- **Design**: k MTP layers → k-level deep tree
- **Branching**: Each level branches into top-k candidates (configurable)
- **Example**: 7 layers with top-3 = up to 3^7 = 2187 candidate paths

```
Generation Flow:
Base model forward pass → hidden_states H_0
    ↓
MTP Layer 0: (embed(T_0), H_0) → logits → sample top-k → T_1 candidates
    ↓
MTP Layer 1: (embed(T_1), H_0) → logits → sample top-k → T_2 candidates
    ↓
... (for each branch)
    ↓
MTP Layer 6: → T_7 candidates (leaf nodes)
```

## Implementation Plan

### Directory Structure
```
model/mtp/
├── __init__.py
├── mtp_model.py          # MTPModel class (main wrapper)
├── cnets.py              # MTP layer architecture
├── configs.py            # MTPConfig class
├── modeling_qwen2_kv.py  # KVQwen2ForCausalLM (KV-optimized base model)
├── kv_cache.py           # KV-cache management (7 independent caches)
└── utils.py              # Tree generation and utility functions
```

### Inference Script
- **File**: `mtp_inference.py` (new file at repo root)
- **Benchmarks**: MT-bench, HumanEval, GSM8K, Alpaca (same as EAGLE)

## User Stories

### US-1: Create KVQwen2ForCausalLM Base Model
**Description**: Implement a KV-cache optimized version of Qwen2ForCausalLM following the pattern of existing KVLlamaForCausalLM.

**Acceptance Criteria**:
- [ ] Can load weights from `/mnt/data/weights/QwQ-32B/`
- [ ] Forward pass returns hidden_states and past_key_values
- [ ] Supports both CUDA and NPU devices
- [ ] KV-cache properly managed across forward passes

### US-2: Implement MTP Layer Architecture
**Description**: Create the MTPDecoderLayer class with enorm, hnorm, eh_proj, and Qwen2-style decoder layer.

**Acceptance Criteria**:
- [ ] MTPDecoderLayer correctly applies enorm to embeddings
- [ ] MTPDecoderLayer correctly applies hnorm to hidden states
- [ ] Concatenation and projection work correctly
- [ ] final_layernorm applied to output
- [ ] Weights load correctly from checkpoint

### US-3: Implement MTP Model Wrapper
**Description**: Create MTPModel class that wraps base model and 7 MTP layers, handles loading from both separate and combined paths.

**Acceptance Criteria**:
- [ ] Loads base model from separate path correctly
- [ ] Loads MTP layers from separate path correctly
- [ ] Auto-detects combined path mode via `num_nextn_predict_layers` in config
- [ ] All 7 MTP layers initialized with independent KV-caches
- [ ] Supports both NPU and GPU devices

### US-4: Implement Tree-based Candidate Generation
**Description**: Implement top-k tree generation across 7 MTP layers with configurable branching factor.

**Acceptance Criteria**:
- [ ] Tree depth = number of MTP layers (7)
- [ ] Top-k configurable via parameter
- [ ] Correct attention masking for tree structure
- [ ] Each MTP layer uses correct layer index for prediction

### US-5: Implement Verification and Acceptance
**Description**: Implement token verification using base model, following EAGLE's verification strategy.

**Acceptance Criteria**:
- [ ] Base model verifies draft tokens
- [ ] Accepts longest matching prefix
- [ ] Correctly handles rejection (stops at first mismatch)
- [ ] Returns accepted tokens + one new token from base model

### US-6: Create MTP Inference Script
**Description**: Create mtp_inference.py with CLI interface supporting all EAGLE benchmarks.

**Acceptance Criteria**:
- [ ] CLI arguments for model paths, device, top-k, temperature, etc.
- [ ] Supports MT-bench evaluation
- [ ] Supports HumanEval evaluation
- [ ] Supports GSM8K evaluation
- [ ] Supports Alpaca evaluation
- [ ] Reports speedup metrics

### US-7: Multi-Device Support
**Description**: Ensure MTP works correctly on multi-NPU and multi-GPU setups.

**Acceptance Criteria**:
- [ ] All MTP layers placed on same device as base model's last layer
- [ ] KV-cache device placement correct
- [ ] No cross-device tensor operations during generation
- [ ] Works with existing device_utils auto-detection

## Technical Specifications

### Loading Strategy
1. **Separate paths mode** (`base_model_path != mtp_model_path`):
   - Load base model from `base_model_path`
   - Load MTP weights from `mtp_model_path`

2. **Combined path mode** (`base_model_path == mtp_model_path`):
   - Check config.json for `num_nextn_predict_layers`
   - If present, filter weights: base model weights vs `model.mtp.*` weights
   - Load accordingly

### Multi-Device Strategy
- All 7 MTP layers on same device as base model's layer 63
- Each MTP layer has independent KV-cache
- Reduces cross-device communication

### Sampling Support
- Temperature scaling
- Top-p (nucleus) sampling
- Top-k sampling
- Greedy decoding (temperature=0)

### Error Handling
- Clear error if `num_nextn_predict_layers` missing in combined mode
- Validate model architecture matches expected Qwen2
- Standard validation for paths and device availability

## Verification Commands

```bash
# Run MT-bench evaluation
python mtp_inference.py \
    --base-model-path /mnt/data/weights/QwQ-32B/ \
    --mtp-model-path /mnt/data/weights/qwq-32b-mtp/ \
    --benchmark mt-bench \
    --top-k 5 \
    --temperature 0.0

# Test on single prompt
python mtp_inference.py \
    --base-model-path /mnt/data/weights/QwQ-32B/ \
    --mtp-model-path /mnt/data/weights/qwq-32b-mtp/ \
    --prompt "Write a Python function to calculate fibonacci numbers"
```

## Out of Scope

- Training/fine-tuning MTP layers (inference only)
- Modifying existing EAGLE code
- Supporting other model architectures (Qwen2 only)
- Quantization support (future work)

## Dependencies

- PyTorch
- Transformers (for Qwen2 config/tokenizer)
- safetensors (for weight loading)
- Existing Spec-Bench utilities (device_utils, etc.)

## References

- Existing EAGLE implementation: `model/eagle/`
- Reference MTP implementation: `/mnt/data/weights/qwq-32b-mtp/qwen2_mtp.py`
- Qwen2 model: HuggingFace transformers
