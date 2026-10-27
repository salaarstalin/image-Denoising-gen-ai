"""Executable checks for actual latent compression, loss, and denoising."""
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from backend.inference import load_model, process_image, reconstruct_from_latent
from backend.model import ConvVAE, LATENT_OPTIONS
from backend.preprocessing import add_gaussian_noise, prepare_image


def demo_image():
    return Image.open('assets/sample_astronaut.png').convert('RGB')


def test_trained_weights_are_loaded():
    model, meta = load_model()
    assert isinstance(model, ConvVAE)
    assert meta['epochs'] >= 1
    assert meta['source_image_count'] >= 1


def test_encoder_to_decoder_is_real():
    model, _ = load_model()
    tensor, _ = prepare_image(demo_image())
    with torch.inference_mode():
        mu, logvar = model.encode(tensor)
        decoded = model.decode(mu)
    assert mu.shape == (1, 128)
    assert logvar.shape == mu.shape
    assert decoded.shape == (1, 3, 64, 64)
    assert not torch.allclose(decoded, tensor)


def test_noise_injection_is_upstream_and_reproducible():
    x, _ = prepare_image(demo_image())
    assert torch.equal(x, add_gaussian_noise(x, 0.0))
    noisy1 = add_gaussian_noise(x, 0.25, seed=17)
    noisy2 = add_gaussian_noise(x, 0.25, seed=17)
    assert torch.equal(noisy1, noisy2)
    assert torch.mean((noisy1 - x).square()).item() > 0.01


@pytest.mark.parametrize('dims', LATENT_OPTIONS)
def test_actual_stored_latent_dim_and_output(dims):
    r = process_image(demo_image(), noise_std=0.18, active_dims=dims)
    z = np.load(BytesIO(r.latent_payload))['latent']
    assert z.dtype == np.float16
    assert z.shape == (dims,)
    assert r.reconstructed.size == (64, 64)
    assert np.asarray(r.reconstructed).std() > 3
    assert r.compression['latent_bytes'] == 2 * dims
    model, _ = load_model()
    reproducible = reconstruct_from_latent(model, z)
    actual_pixels = (reproducible.squeeze(0).permute(1, 2, 0).numpy() * 255).round().astype('uint8')
    assert np.max(np.abs(actual_pixels.astype(int) - np.asarray(r.reconstructed).astype(int))) <= 0


def test_compression_changes_reconstruction():
    a = process_image(demo_image(), noise_std=0.15, active_dims=32)
    b = process_image(demo_image(), noise_std=0.15, active_dims=128)
    assert a.compression['ratio'] > b.compression['ratio']
    assert np.mean(np.abs(np.asarray(a.reconstructed).astype(float) -
                          np.asarray(b.reconstructed).astype(float))) > 0.1
    assert a.mse_reconstructed >= 0
    assert len(a.compression_curve) == 4


def test_denoising_reduces_error_at_high_noise():
    r = process_image(demo_image(), noise_std=0.40, active_dims=128)
    # Valid for this bundled demo image & trained, included checkpoint.
    assert r.mse_reconstructed < r.mse_noisy
    assert r.psnr_reconstructed > r.psnr_noisy


def test_decode_saved_latent_without_original_image(tmp_path):
    from decode_latent import decode_latent_file
    r = process_image(demo_image(), noise_std=0.35, active_dims=64)
    latent_path = tmp_path / 'latent.npz'
    latent_path.write_bytes(r.latent_payload)
    output = tmp_path / 'from_latent.png'
    decode_latent_file(latent_path, output)
    restored = Image.open(output)
    assert np.array_equal(np.asarray(restored), np.asarray(r.reconstructed))


def test_more_features_reduce_information_loss_on_sample_photo():
    r = process_image(Image.open('assets/sample_coffee.png'), noise_std=0.4, active_dims=128)
    error_by_dims = dict(r.compression_curve)
    assert error_by_dims[128] < error_by_dims[32]
