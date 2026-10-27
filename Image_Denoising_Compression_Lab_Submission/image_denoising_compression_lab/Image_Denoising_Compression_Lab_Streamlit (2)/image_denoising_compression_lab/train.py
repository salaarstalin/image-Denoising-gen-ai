"""Train a REAL denoising variational autoencoder on RGB image patches.

Default training uses real, freely bundled scikit-image photographs and works
without a downloaded ML dataset. Supply --data-dir to train on your own photos.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import data as skimage_data
from torch.utils.data import DataLoader, Dataset

from backend.model import ConvVAE, LATENT_OPTIONS, MAX_LATENT_DIM, IMAGE_SIZE

ROOT = Path(__file__).resolve().parent


def get_training_images(data_dir: str | None) -> list[Image.Image]:
    if data_dir:
        paths = [p for p in Path(data_dir).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
        images = []
        for path in paths:
            try:
                with Image.open(path) as src:
                    if min(src.size) >= 64:
                        images.append(src.convert('RGB').copy())
            except (OSError, ValueError):
                pass
        if not images:
            raise ValueError(f'No readable photos of at least 64 pixels found in {data_dir}')
        return images

    # These photographs are shipped with the scikit-image package; no internet.
    funcs = [
        skimage_data.astronaut, skimage_data.coffee,
        skimage_data.chelsea, skimage_data.rocket,
        skimage_data.immunohistochemistry,
        skimage_data.hubble_deep_field,
        skimage_data.colorwheel,
    ]
    return [Image.fromarray(f()).convert('RGB') for f in funcs]


class ImagePatchDataset(Dataset):
    def __init__(self, images: list[Image.Image], count: int, seed: int = 42):
        self.images = images
        self.count = count
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = random.Random(self.seed + index * 1777)
        img = self.images[index % len(self.images)]
        width, height = img.size
        edge = min(width, height)
        # Zoom in/out over photographs; keeps varied textures and object scales.
        crop_size = int(rng.uniform(max(64, edge * 0.15), edge))
        crop_size = min(crop_size, edge)
        x = rng.randint(0, width - crop_size)
        y = rng.randint(0, height - crop_size)
        patch = img.crop((x, y, x + crop_size, y + crop_size))
        if rng.random() < 0.5:
            patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            patch = patch.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        patch = patch.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        arr = np.array(patch, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    images = get_training_images(args.data_dir)
    dataset = ImagePatchDataset(images, args.samples_per_epoch, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = ConvVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f'Training on {device}: {len(images)} photos, {args.epochs} epochs, '
          f'{len(dataset)} patches/epoch, dims={LATENT_OPTIONS}', flush=True)
    last_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, clean in enumerate(loader):
            clean = clean.to(device)
            # A denoising-VAE: the INPUT is corrupted; the LABEL stays clean.
            noise_std = float(torch.rand(1).item() * args.max_train_noise)
            noisy = (clean + noise_std * torch.randn_like(clean)).clamp(0, 1)
            active_dims = LATENT_OPTIONS[(step + epoch) % len(LATENT_OPTIONS)]
            predicted, mu, logvar = model(noisy, active_dims=active_dims, sample=True)
            reconstruction_loss = torch.mean((predicted - clean) ** 2)
            kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean()
            loss = reconstruction_loss + args.beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(clean)
        last_loss = running / len(dataset)
        print(f'Epoch {epoch + 1:02d}/{args.epochs} | loss {last_loss:.6f}', flush=True)

    model.eval()
    ckpt = {
        'trained': True,
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'metadata': {
            'architecture': 'Convolutional Denoising Variational Autoencoder (VAE)',
            'training_dataset': ('User-provided photos' if args.data_dir else '7 bundled scikit-image natural/illustrative photographs'),
            'source_image_count': len(images),
            'patches_per_epoch': args.samples_per_epoch,
            'epochs': args.epochs,
            'max_train_noise_std': args.max_train_noise,
            'beta': args.beta,
            'last_training_loss': round(last_loss, 7),
            'image_size': IMAGE_SIZE,
            'latent_dims': list(LATENT_OPTIONS),
            'seed': args.seed,
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f'Saved trained checkpoint to: {path} ({path.stat().st_size:,} bytes)', flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train the denoising convolutional VAE')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--samples-per-epoch', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--max-train-noise', type=float, default=0.20)
    p.add_argument('--beta', type=float, default=0.0005)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--data-dir', type=str, default=None)
    p.add_argument('--output', type=str, default=str(ROOT / 'weights' / 'vae_denoiser.pt'))
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
