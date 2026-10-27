"""Diagnostics for denoising, reconstruction, and latent storage."""
import math

import torch
from torch import Tensor

from backend.model import IMAGE_SIZE


def mse(reference: Tensor, result: Tensor) -> float:
    return float(torch.mean((reference - result) ** 2).item())


def psnr(reference: Tensor, result: Tensor) -> float:
    error = mse(reference, result)
    return float('inf') if error == 0 else 10.0 * math.log10(1.0 / error)


def compression_stats(active_dims: int) -> dict[str, float | int]:
    """Raw pixel bytes vs a float16 latent (excluding container metadata).

    This is a representation ratio, NOT a JPEG/PNG file-compression ratio.
    """
    input_bytes = IMAGE_SIZE * IMAGE_SIZE * 3
    latent_bytes = active_dims * 2  # one float16 per retained coefficient
    return {
        'raw_image_bytes': input_bytes,
        'latent_bytes': latent_bytes,
        'ratio': input_bytes / latent_bytes,
        'retained_features': active_dims,
    }
