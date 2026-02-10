# OpenPanGu MTP Model for Speculative Decoding
#
# Supported models:
# - MTPPanguModel: For OpenPanGu MoE architecture (PanguProMoEV2ForCausalLM)

from .mtp_pangu_model import MTPPanguModel
from .cnets import PanguMTPDraftModel
from .configs import PanguMTPConfig
from .modeling_pangu_kv import KVPanguForCausalLM

__all__ = ["MTPPanguModel", "PanguMTPDraftModel", "PanguMTPConfig", "KVPanguForCausalLM"]
