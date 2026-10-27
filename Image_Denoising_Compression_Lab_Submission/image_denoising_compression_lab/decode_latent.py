"""Reconstruct a saved compressed latent file WITHOUT the source image.

Usage: python decode_latent.py saved_latent.npz --output decoded.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from backend.inference import load_model, reconstruct_from_latent
from backend.preprocessing import tensor_to_pil


def decode_latent_file(path: str | Path, output: str | Path) -> Path:
    # The only image information present is the saved 16-bit latent vector.
    with np.load(str(path), allow_pickle=False) as archive:
        latent = archive['latent']
        size = int(archive['image_size'][0])
        version = int(archive['model_version'][0])
    if size != 64 or version != 1 or latent.dtype != np.float16:
        raise ValueError('Invalid latent archive for this trained VAE.')
    model, _ = load_model()
    reconstruction = reconstruct_from_latent(model, latent)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(reconstruction).save(output)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Decode a saved VAE latent archive')
    p.add_argument('latent', help='Compressed .npz from Gradio frontend')
    p.add_argument('--output', default='decoded_from_latent.png')
    args = p.parse_args()
    print(f'Decoded to {decode_latent_file(args.latent, args.output)}')
