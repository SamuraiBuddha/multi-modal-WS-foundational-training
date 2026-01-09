"""
Device management utilities.
"""

import torch
from typing import Optional, Dict


def get_device(preference: str = 'auto') -> torch.device:
    """
    Get the best available device.

    Args:
        preference: 'auto', 'cuda', 'mps', or 'cpu'

    Returns:
        PyTorch device
    """
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(preference)


def device_info() -> Dict[str, any]:
    """
    Get information about available devices.

    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }

    if info['cuda_available']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)

    return info


def print_device_info():
    """Print device information to console."""
    info = device_info()

    print("Device Information")
    print("=" * 40)
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"MPS Available: {info['mps_available']}")

    if info['cuda_available']:
        print(f"CUDA Devices: {info['cuda_device_count']}")
        print(f"Device Name: {info['cuda_device_name']}")
        print(f"Total Memory: {info['cuda_memory_total'] / 1e9:.2f} GB")

    print(f"Recommended Device: {get_device()}")
