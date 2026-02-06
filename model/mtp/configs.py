from transformers.configuration_utils import PretrainedConfig


class MTPConfig(PretrainedConfig):
    """
    Configuration class for MTP (Multi-Token Prediction) model.

    This extends the Qwen2 configuration with MTP-specific parameters.
    The key difference from EAGLE is that MTP has separate RMSNorm layers
    for embeddings and hidden states before concatenation.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the Qwen2 model.
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 27648):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 1):
            Number of hidden layers in each MTP layer (decoder layers per MTP layer).
        num_mtp_layers (`int`, *optional*, defaults to 7):
            Number of MTP layers (each predicts a different future token).
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for GQA.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 40960):
            The maximum sequence length.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to True):
            Whether to use bias in attention q/k/v projections.
    """

    model_type = "mtp"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=1,  # decoder layers per MTP layer
        num_mtp_layers=7,  # number of MTP layers
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151645,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_mtp_layers = num_mtp_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_qwen2_config(cls, qwen2_config, num_mtp_layers=7):
        """Create MTPConfig from a Qwen2 config."""
        return cls(
            vocab_size=qwen2_config.vocab_size,
            hidden_size=qwen2_config.hidden_size,
            intermediate_size=qwen2_config.intermediate_size,
            num_hidden_layers=1,
            num_mtp_layers=num_mtp_layers,
            num_attention_heads=qwen2_config.num_attention_heads,
            num_key_value_heads=qwen2_config.num_key_value_heads,
            hidden_act=qwen2_config.hidden_act,
            max_position_embeddings=qwen2_config.max_position_embeddings,
            rms_norm_eps=qwen2_config.rms_norm_eps,
            rope_theta=getattr(qwen2_config, 'rope_theta', 1000000.0),
            attention_bias=True,
        )
