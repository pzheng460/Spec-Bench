"""KV-cache management for OpenPanGu model with asymmetric K/V dimensions.

OpenPanGu uses Sink Attention where K head_dim = qk_nope_dim + qk_rope_dim = 192
and V dim = v_channels = 128. This requires separate K and V cache tensors.
"""

import torch


class KVCache:
    """
    A key-value cache for a single tensor (either key or value).

    This class provides a mechanism to maintain a growing cache,
    particularly useful for autoregressive decoding.
    """

    def __init__(self, data, current_length):
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)


def initialize_past_key_values(model, max_length=2048):
    """
    Initialize past key and value states for the OpenPanGu base model.

    Handles asymmetric K/V dimensions:
    - K cache dim: qk_nope_dim + qk_rope_dim = 192
    - V cache dim: v_channels = 128

    Args:
        model: The transformer model (KVPanguForCausalLM)
        max_length: Maximum sequence length for KV cache

    Returns:
        tuple:
            - past_key_values (list): List of [KVCache_k, KVCache_v] per layer
            - past_key_values_data_list (list): List of (k_data, v_data) tensors
            - current_length_data (torch.Tensor): Tensor tracking current lengths
    """
    config = model.config
    batch_size = 1

    k_head_dim = config.qk_nope_dim + config.qk_rope_dim  # 192
    v_head_dim = config.v_channels  # 128

    # Get devices for each layer
    devices = []
    for i in range(config.num_hidden_layers):
        try:
            device = model.model.layers[i].self_attn.qkv_proj.weight.device
        except AttributeError:
            try:
                device = model.layers[i].self_attn.qkv_proj.weight.device
            except AttributeError:
                device = next(model.parameters()).device
        devices.append(device)

    # Group layers by device and create cache tensors
    past_key_values_data_list = []
    startnum = 0
    startdevice = devices[0]

    for id, i in enumerate(devices):
        if startdevice != i:
            # Create K cache
            k_data = torch.zeros(
                startnum * 2,  # 2 entries per layer (but we alternate k,v)
                batch_size,
                config.num_key_value_heads,
                max_length,
                k_head_dim,
                device=startdevice,
                dtype=model.dtype,
            )
            # Create V cache
            v_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                max_length,
                v_head_dim,
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append((k_data, v_data))
            startdevice = i
            startnum = 0
        startnum += 1

    # Handle the last group
    k_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        k_head_dim,
        device=startdevice,
        dtype=model.dtype,
    )
    v_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        v_head_dim,
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append((k_data, v_data))

    # Current length tracker
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )

    # Create KVCache for each layer
    past_key_values = []
    bias = 0
    start_data_m = devices[0].index if hasattr(devices[0], 'index') else 0

    for i in range(config.num_hidden_layers):
        data_m = devices[i].index if hasattr(devices[i], 'index') else 0
        if data_m != start_data_m:
            bias = 0
            start_data_m = data_m
        try:
            data_idx = data_m - (devices[0].index if hasattr(devices[0], 'index') else 0)
            k_data, v_data = past_key_values_data_list[data_idx]
            past_key_values.append([
                KVCache(k_data[2 * bias], current_length_data[i * 2]),      # K cache
                KVCache(v_data[2 * bias + 1], current_length_data[i * 2 + 1]),  # V cache
            ])
        except (IndexError, TypeError):
            k_data, v_data = past_key_values_data_list[0]
            past_key_values.append([
                KVCache(k_data[2 * bias], current_length_data[i * 2]),
                KVCache(v_data[2 * bias + 1], current_length_data[i * 2 + 1]),
            ])
        bias += 1

    return past_key_values, past_key_values_data_list, current_length_data


def initialize_mtp_past_key_values(mtp_model, device, dtype, max_length=4096):
    """
    Initialize past key and value states for MTP draft layers.

    Args:
        mtp_model: The MTP draft model
        device: Device to place the cache on
        dtype: Data type for the cache
        max_length: Maximum sequence length

    Returns:
        tuple of (past_key_values, past_key_values_data, current_length_data)
    """
    config = mtp_model.config
    batch_size = 1
    num_mtp_layers = config.num_mtp_layers

    k_head_dim = config.qk_nope_dim + config.qk_rope_dim
    v_head_dim = config.v_channels

    k_data = torch.zeros(
        num_mtp_layers * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        k_head_dim,
        device=device,
        dtype=dtype,
    )
    v_data = torch.zeros(
        num_mtp_layers * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        v_head_dim,
        device=device,
        dtype=dtype,
    )

    current_length_data = torch.zeros(
        num_mtp_layers * 2, dtype=torch.long, device="cpu"
    )

    past_key_values = []
    for i in range(num_mtp_layers):
        past_key_values.append([
            KVCache(k_data[2 * i], current_length_data[i * 2]),
            KVCache(v_data[2 * i + 1], current_length_data[i * 2 + 1]),
        ])

    return past_key_values, (k_data, v_data), current_length_data


def reset_past_key_values(passed_key_values):
    """Reset the current lengths in the passed key-values to zero."""
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values
