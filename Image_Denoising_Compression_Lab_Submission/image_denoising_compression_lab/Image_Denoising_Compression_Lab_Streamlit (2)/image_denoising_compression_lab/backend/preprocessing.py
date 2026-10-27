"""Preprocessing and reproducible noise injection functions."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from backend.model import IMAGE_SIZE


def prepare_image(image: Image.Image) -> tuple[Tensor, Image.Image]:
    """Center-crop to a square, resize to the trained 64x64 input shape."""
    if image is None:
        raise ValueError('Please upload an image.')
    if not isinstance(image, Image.Image):
        raise TypeError('Expected a PIL image.')
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    edge = min(rgb.size)
    left = (rgb.width - edge) // 2
    top = (rgb.height - edge) // 2
    processed = rgb.crop((left, top, left + edge, top + edge)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    arr = np.asarray(processed, dtype=np.float32).copy() / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x, processed


def add_gaussian_noise(x: Tensor, noise_std: float, seed: int = 42) -> Tensor:
    """Gaussian noise applied BEFORE the model encoder, not after decoding."""
    if not 0.0 <= noise_std <= 0.5:
        raise ValueError('Noise standard deviation must be between 0 and 0.5')
    if noise_std == 0.0:
        return x.clone()
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    noise = torch.randn(x.shape, dtype=x.dtype, device='cpu', generator=generator)
    return (x.cpu() + noise_std * noise).clamp(0.0, 1.0)


def tensor_to_pil(x: Tensor) -> Image.Image:
    arr = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(np.clip(arr * 255.0, 0, 255).round()), 'RGB')
