# MTP (Multi-Token Prediction) Model for Speculative Decoding
# Similar to EAGLE but with separate RMSNorm for embeddings and hidden states
#
# Supported models:
# - MTPQwen2Model: For Qwen2 architecture (QwQ-32B, etc.)
# - (Future) MTPLlamaModel: For Llama architecture

from .mtp_qwen2_model import MTPQwen2Model
from .configs import MTPConfig

# Backward compatibility alias
MTPModel = MTPQwen2Model

__all__ = ["MTPQwen2Model", "MTPModel", "MTPConfig"]
