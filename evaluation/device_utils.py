"""
Device utilities for NPU/CUDA compatibility.

This module provides device detection and synchronization functions
that work across both CUDA and Ascend NPU environments.
"""

import os
import torch

# Check if torch_npu is available
_NPU_AVAILABLE = False
try:
    import torch_npu
    _NPU_AVAILABLE = True
except ImportError:
    pass


def is_npu_available() -> bool:
    """Check if NPU is available."""
    if not _NPU_AVAILABLE:
        return False
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def get_device() -> str:
    """
    Get the best available device.

    Priority: NPU > CUDA > CPU

    Returns:
        str: Device string ("npu", "cuda", or "cpu")

    Raises:
        RuntimeError: If ASCEND_RT_VISIBLE_DEVICES is set but NPU is not available,
                      or if CUDA_VISIBLE_DEVICES is set but CUDA is not available.
    """
    # Check environment variables for explicit device preference
    ascend_devices = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

    # If ASCEND_RT_VISIBLE_DEVICES is set, expect NPU to be available
    if ascend_devices is not None and ascend_devices != '':
        if is_npu_available():
            return "npu"
        else:
            raise RuntimeError(
                f"ASCEND_RT_VISIBLE_DEVICES is set to '{ascend_devices}' but NPU is not available. "
                "Please check your torch_npu installation and CANN environment."
            )

    # If only CUDA_VISIBLE_DEVICES is set (and not ASCEND), use CUDA
    if cuda_devices is not None and cuda_devices != '':
        if is_cuda_available():
            return "cuda"
        else:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES is set to '{cuda_devices}' but CUDA is not available. "
                "Please check your CUDA installation."
            )

    # Auto-detect: prefer NPU over CUDA
    if is_npu_available():
        return "npu"
    elif is_cuda_available():
        return "cuda"
    else:
        return "cpu"


def get_visible_devices() -> str:
    """
    Get the visible devices environment variable value.

    Returns:
        str: The value of ASCEND_RT_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES
    """
    ascend_devices = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if ascend_devices is not None:
        return ascend_devices
    return os.environ.get('CUDA_VISIBLE_DEVICES', '')


def device_synchronize(device: str = None) -> None:
    """
    Synchronize the device.

    Args:
        device: Device type ("npu", "cuda", or None for auto-detect)
    """
    if device is None:
        device = get_device()

    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    # CPU doesn't need synchronization


def empty_cache(device: str = None) -> None:
    """
    Empty the device cache.

    Args:
        device: Device type ("npu", "cuda", or None for auto-detect)
    """
    if device is None:
        device = get_device()

    if device == "npu":
        torch.npu.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    # CPU doesn't need cache clearing


def set_device(device_id: int = 0, device: str = None) -> None:
    """
    Set the current device.

    Args:
        device_id: Device index
        device: Device type ("npu", "cuda", or None for auto-detect)
    """
    if device is None:
        device = get_device()

    if device == "npu":
        torch.npu.set_device(device_id)
    elif device == "cuda":
        torch.cuda.set_device(device_id)


def get_device_count(device: str = None) -> int:
    """
    Get the number of available devices.

    Args:
        device: Device type ("npu", "cuda", or None for auto-detect)

    Returns:
        int: Number of available devices
    """
    if device is None:
        device = get_device()

    if device == "npu":
        return torch.npu.device_count()
    elif device == "cuda":
        return torch.cuda.device_count()
    else:
        return 1  # CPU


def get_device_name(device_id: int = 0, device: str = None) -> str:
    """
    Get the device name.

    Args:
        device_id: Device index
        device: Device type ("npu", "cuda", or None for auto-detect)

    Returns:
        str: Device name
    """
    if device is None:
        device = get_device()

    if device == "npu":
        return torch.npu.get_device_name(device_id)
    elif device == "cuda":
        return torch.cuda.get_device_name(device_id)
    else:
        return "CPU"


def get_npu_device_map(max_memory_per_device: str = "60GiB") -> dict:
    """
    Generate a device map for multi-NPU model loading.

    Uses the actual visible device count (respects ASCEND_RT_VISIBLE_DEVICES).

    Args:
        max_memory_per_device: Maximum memory per NPU device (e.g., "60GiB")

    Returns:
        dict: Device map with max_memory configuration
    """
    if is_npu_available():
        import torch
        # Get actual visible device count (respects ASCEND_RT_VISIBLE_DEVICES)
        device_count = torch.npu.device_count()
    else:
        device_count = 0
    max_memory = {i: max_memory_per_device for i in range(device_count)}
    max_memory["cpu"] = "100GiB"
    return max_memory


# For convenience, export a default device at module load time
# This can be used for quick checks without repeated detection
DEFAULT_DEVICE = None

def init_device() -> str:
    """
    Initialize and cache the default device.

    Returns:
        str: The detected device type
    """
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = get_device()
    return DEFAULT_DEVICE
