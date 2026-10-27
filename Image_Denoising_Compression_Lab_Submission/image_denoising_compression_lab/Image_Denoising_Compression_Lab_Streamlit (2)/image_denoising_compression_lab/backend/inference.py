"""Actual model loading, VAE encoding, latent serialization and decoding."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backend.metrics import compression_stats, mse, psnr
from backend.model import ConvVAE, LATENT_OPTIONS, MAX_LATENT_DIM
from backend.preprocessing import add_gaussian_noise, prepare_image, tensor_to_pil

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / 'weights' / 'vae_denoiser.pt'


@dataclass
class LabResult:
    original: Image.Image
    noisy: Image.Image
    reconstructed: Image.Image
    residual: Image.Image
    mse_noisy: float
    mse_reconstructed: float
    psnr_noisy: float
    psnr_reconstructed: float
    compression: dict
    latent_array: np.ndarray
    latent_payload: bytes
    checkpoint_metadata: dict
    compression_curve: list[tuple[int, float]]


@lru_cache(maxsize=2)
def load_model(checkpoint_path: str = str(DEFAULT_CHECKPOINT)) -> tuple[ConvVAE, dict]:
    """Refuse inference without fitted weights; no untrained fallback."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Trained VAE weights missing: {path}. Run python train.py.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not checkpoint.get('trained', False):
        raise ValueError('Checkpoint has not been trained.')
    model = ConvVAE(latent_dim=MAX_LATENT_DIM)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint.get('metadata', {})


def reconstruct_from_latent(model: ConvVAE, latent: np.ndarray) -> torch.Tensor:
    """Decode ONLY the stored reduced-dimensional representation."""
    if latent.ndim != 1 or len(latent) not in LATENT_OPTIONS:
        raise ValueError('Latent array must contain 32, 64, 96, or 128 coefficients.')
    z = torch.zeros(1, MAX_LATENT_DIM, dtype=torch.float32)
    z[0, :len(latent)] = torch.from_numpy(latent.astype(np.float32))
    with torch.inference_mode():
        return model.decode(z).clamp(0.0, 1.0)


def process_image(
    image: Image.Image,
    noise_std: float = 0.10,
    active_dims: int = 64,
    noise_seed: int = 42,
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
) -> LabResult:
    if active_dims not in LATENT_OPTIONS:
        raise ValueError(f'Choose an active latent dimension from {LATENT_OPTIONS}.')
    clean_tensor, clean_pil = prepare_image(image)
    noisy_tensor = add_gaussian_noise(clean_tensor, noise_std, seed=noise_seed)
    model, metadata = load_model(checkpoint_path)
    with torch.inference_mode():
        # This is the actual learned convolutional encoder's output.
        mu, _logvar = model.encode(noisy_tensor)
        # Explicitly drop ALL unused latent dimensions BEFORE decoding.
        reduced = mu[0, :active_dims].cpu().numpy().astype(np.float16)
        reconstruction = reconstruct_from_latent(model, reduced)
        curve = [(d, mse(clean_tensor, reconstruct_from_latent(
            model, mu[0, :d].cpu().numpy().astype(np.float16))))
            for d in LATENT_OPTIONS]

    # Store the exact 16-bit latent coefficients and required decoding metadata.
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        latent=reduced,
        image_size=np.array([64], dtype=np.int32),
        model_version=np.array([1], dtype=np.int32),
    )
    residual = (clean_tensor - reconstruction).abs()
    residual_display = (residual * 3.0).clamp(0, 1)
    return LabResult(
        original=clean_pil,
        noisy=tensor_to_pil(noisy_tensor),
        reconstructed=tensor_to_pil(reconstruction),
        residual=tensor_to_pil(residual_display),
        mse_noisy=mse(clean_tensor, noisy_tensor),
        mse_reconstructed=mse(clean_tensor, reconstruction),
        psnr_noisy=psnr(clean_tensor, noisy_tensor),
        psnr_reconstructed=psnr(clean_tensor, reconstruction),
        compression=compression_stats(active_dims),
        latent_array=reduced,
        latent_payload=buffer.getvalue(),
        checkpoint_metadata=metadata,
        compression_curve=curve,
    )
