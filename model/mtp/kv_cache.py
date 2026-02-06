"""KV-cache management for MTP model."""

import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)


def initialize_past_key_values(model, max_length=2048):
    """
    Initialize past key and value states for the base model.

    This function prepares key-value cache structures for the model, allowing it to store
    and reuse past key and value states during autoregressive decoding.

    Args:
        model: The transformer model for which past key-value states need to be initialized.
        max_length: Maximum sequence length for KV cache (default: 2048)

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer.
            - past_key_values_data_list (list): List of tensors storing all keys and values.
            - current_length_data (torch.Tensor): Tensor tracking current lengths.
    """
    config = model.config
    batch_size = 1

    # Get devices for each layer
    devices = []
    for i in range(config.num_hidden_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except AttributeError:
            device = model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)

    # Group layers by device
    past_key_values_data_list = []
    startnum = 0
    startdevice = devices[0]

    for id, i in enumerate(devices):
        if startdevice != i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                max_length,
                config.hidden_size // config.num_attention_heads,
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum = 0
        startnum += 1

    # Handle the last group
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        config.hidden_size // config.num_attention_heads,
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(past_key_values_data)

    # Current length tracker (on CPU for fast access)
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
            past_key_values.append([
                KVCache(past_key_values_data_list[data_idx][2 * bias + j], current_length_data[i * 2 + j])
                for j in range(2)
            ])
        except (IndexError, TypeError):
            past_key_values.append([
                KVCache(past_key_values_data_list[0][2 * bias + j], current_length_data[i * 2 + j])
                for j in range(2)
            ])
        bias += 1

    return past_key_values, past_key_values_data_list, current_length_data


def initialize_mtp_past_key_values(mtp_model, device, dtype, max_length=4096):
    """
    Initialize past key and value states for MTP layers.

    Each MTP layer has its own independent KV-cache.

    Args:
        mtp_model: The MTP draft model
        device: Device to place the cache on
        dtype: Data type for the cache
        max_length: Maximum sequence length

    Returns:
        tuple:
            - past_key_values (list): List of KVCache objects for each MTP layer
            - past_key_values_data (torch.Tensor): Tensor storing all keys and values
            - current_length_data (torch.Tensor): Tensor tracking current lengths
    """
    config = mtp_model.config
    batch_size = 1
    num_mtp_layers = config.num_mtp_layers

    # Create storage for all MTP layer KV caches
    # Each MTP layer has 1 decoder layer, so 2 entries (key and value) per MTP layer
    past_key_values_data = torch.zeros(
        num_mtp_layers * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        config.hidden_size // config.num_attention_heads,
        device=device,
        dtype=dtype,
    )

    current_length_data = torch.zeros(
        num_mtp_layers * 2, dtype=torch.long, device="cpu"
    )

    past_key_values = []
    for i in range(num_mtp_layers):
        past_key_values.append([
            KVCache(past_key_values_data[2 * i + j], current_length_data[i * 2 + j])
            for j in range(2)
        ])

    return past_key_values, past_key_values_data, current_length_data


def reset_past_key_values(passed_key_values):
    """
    Reset the current lengths in the passed key-values to zero.

    Args:
        passed_key_values (list): List of KVCache objects.

    Returns:
        list: Updated key-value states with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values
