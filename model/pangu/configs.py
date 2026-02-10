from transformers.configuration_utils import PretrainedConfig


class PanguMTPConfig(PretrainedConfig):
    """
    Configuration class for OpenPanGu MTP (Multi-Token Prediction) model.

    This extends the PanguProMoE configuration with MTP-specific parameters.
    Key architecture features:
    - Sink Attention with asymmetric K/V dims (qk_nope_dim + qk_rope_dim != v_channels)
    - MoE layers (80 routed experts + 2 shared experts)
    - Sandwich norm (4 layer norms per decoder layer)
    - Parametric sink key/value tokens
    """

    model_type = "pangu_mtp"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=153600,
        hidden_size=4608,
        intermediate_size=10240,
        num_hidden_layers=1,  # decoder layers per MTP layer
        num_mtp_layers=1,  # number of MTP layers
        num_attention_heads=64,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=45892,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        # Sink Attention parameters
        qk_nope_dim=128,
        qk_rope_dim=64,
        v_channels=128,
        param_sink_number=128,
        param_sink_with_value=True,
        # MoE parameters
        n_routed_experts=80,
        n_shared_experts=2,
        moe_intermediate_size=1280,
        num_experts_per_tok=8,
        first_k_dense_replace=4,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        router_enable_expert_bias=True,
        # Sandwich norm
        sandwich_norm=True,
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
        # Sink Attention
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim
        self.v_channels = v_channels
        self.param_sink_number = param_sink_number
        self.param_sink_with_value = param_sink_with_value
        # MoE
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.router_enable_expert_bias = router_enable_expert_bias
        # Sandwich norm
        self.sandwich_norm = sandwich_norm

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self):
        """K head dim = qk_nope_dim + qk_rope_dim."""
        return self.qk_nope_dim + self.qk_rope_dim

    @classmethod
    def from_pangu_config(cls, pangu_config, num_mtp_layers=1):
        """Create PanguMTPConfig from a PanguProMoE config."""
        return cls(
            vocab_size=pangu_config.vocab_size,
            hidden_size=pangu_config.hidden_size,
            intermediate_size=pangu_config.intermediate_size,
            num_hidden_layers=1,
            num_mtp_layers=num_mtp_layers,
            num_attention_heads=pangu_config.num_attention_heads,
            num_key_value_heads=pangu_config.num_key_value_heads,
            hidden_act=pangu_config.hidden_act,
            max_position_embeddings=pangu_config.max_position_embeddings,
            rms_norm_eps=pangu_config.rms_norm_eps,
            rope_theta=getattr(pangu_config, 'rope_theta', 10000.0),
            qk_nope_dim=getattr(pangu_config, 'qk_nope_dim', 128),
            qk_rope_dim=getattr(pangu_config, 'qk_rope_dim', 64),
            v_channels=getattr(pangu_config, 'v_channels', 128),
            param_sink_number=getattr(pangu_config, 'param_sink_number', 128),
            param_sink_with_value=getattr(pangu_config, 'param_sink_with_value', True),
            n_routed_experts=getattr(pangu_config, 'n_routed_experts', 80),
            n_shared_experts=getattr(pangu_config, 'n_shared_experts', 2),
            moe_intermediate_size=getattr(pangu_config, 'moe_intermediate_size', 1280),
            num_experts_per_tok=getattr(pangu_config, 'num_experts_per_tok', 8),
            first_k_dense_replace=getattr(pangu_config, 'first_k_dense_replace', 4),
            routed_scaling_factor=getattr(pangu_config, 'routed_scaling_factor', 2.5),
            norm_topk_prob=getattr(pangu_config, 'norm_topk_prob', True),
            router_enable_expert_bias=getattr(pangu_config, 'router_enable_expert_bias', True),
            sandwich_norm=getattr(pangu_config, 'sandwich_norm', True),
        )
