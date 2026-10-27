"""Generate a demonstration of REAL VAE results; no UI screenshot is faked."""
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from backend.inference import process_image

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'assets'
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def font(sz, bold=False):
    try:
        return ImageFont.truetype(BOLD_PATH if bold else FONT_PATH, sz)
    except OSError:
        return ImageFont.load_default()

F14, F18, F22, F27, F36 = [font(i) for i in (14, 18, 22, 27, 36)]
FB16, FB20, FB26 = [font(i, True) for i in (16, 20, 26)]
INK = (17, 61, 70)
SLATE = (92, 113, 123)
TEAL = (20, 131, 129)

def draw_rounded(draw, xy, fill, radius=14, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def frame_for(dims, result):
    canvas = Image.new('RGB', (1280, 750), (239, 246, 247))
    dr = ImageDraw.Draw(canvas)
    dr.rectangle((0, 0, 1280, 93), fill=(10, 66, 77))
    dr.text((36, 17), 'Image Denoising & Compression Lab', font=F36, fill='white')
    dr.text((38, 64), 'REAL TRAINED PyTorch VAE  |  noise = 0.40  |  64 x 64 RGB input', font=F18, fill=(190, 230, 228))
    for i, (lab, pic) in enumerate([
        ('01   Original input', result.original),
        ('02   Gaussian-noisy input', result.noisy),
        ('03   Learned VAE decoder output', result.reconstructed),
    ]):
        x0 = 29 + i*422
        draw_rounded(dr, (x0, 116, x0+394, 480), 'white', outline=(214, 228, 231))
        dr.text((x0+17, 129), lab, font=FB20, fill=INK)
        # Render image enlarged but preserve true 64x64 pixel content.
        img = pic.resize((315, 290), Image.Resampling.NEAREST)
        canvas.paste(img, (x0+40, 170))
    draw_rounded(dr, (29, 499, 1251, 718), 'white', outline=(214, 228, 231))
    dr.text((48, 515), 'ACTUAL LATENT COMPRESSION', font=FB20, fill=INK)
    dr.text((48, 551), 'Retained float16 features', font=F18, fill=SLATE)
    for i, d in enumerate((32,64,96,128)):
        x0 = 48 + i*148
        active = (d == dims)
        draw_rounded(dr, (x0, 584, x0+134, 630), TEAL if active else (228, 237, 239), radius=10)
        dr.text((x0+39, 593), str(d), font=FB26, fill='white' if active else INK)
    comp = result.compression
    dr.text((48, 653), f"Estimated feature ratio: {comp['ratio']:.0f}:1 vs raw RGB", font=F18, fill=INK)
    dr.text((679, 515), 'MEASURED RECONSTRUCTION', font=FB20, fill=INK)
    dr.text((679, 560), f"MSE noisy:          {result.mse_noisy:.5f}", font=F22, fill=SLATE)
    dr.text((679, 593), f"MSE reconstructed:  {result.mse_reconstructed:.5f}", font=F22, fill=TEAL)
    dr.text((679, 626), f"PSNR reconstructed: {result.psnr_reconstructed:.2f} dB", font=F22, fill=INK)
    dr.text((680, 674), 'Lower MSE / higher PSNR is better', font=F14, fill=SLATE)
    dr.text((38, 727), 'Inference: noise -> encoder -> stored reduced latent -> decoder. No resize-only/filter shortcut.', font=F14, fill=SLATE)
    return canvas


def main():
    source = Image.open(DEST / 'sample_coffee.png').convert('RGB')
    results = {d: process_image(source, noise_std=0.40, active_dims=d) for d in (32,64,96,128)}
    frame_for(128, results[128]).save(DEST / 'demo_results.png', optimize=True)
    # An 8-second demonstration with the compressor set to each actual latent size.
    frames_path = DEST / '_frames_tmp'
    frames_path.mkdir(exist_ok=True)
    count = 0
    for dims in (32, 64, 96, 128):
        img = frame_for(dims, results[dims])
        for _ in range(8):
            img.save(frames_path / f'{count:04d}.png')
            count += 1
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-framerate', '4', '-i', str(frames_path / '%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
        str(DEST / 'model_demonstration.mp4'),
    ]
    subprocess.run(cmd, check=True)
    for path in frames_path.iterdir():
        path.unlink()
    frames_path.rmdir()
    for d, result in results.items():
        print(d, 'MSE',round(result.mse_reconstructed,5),'latent_bytes',result.compression['latent_bytes'])
    print('Saved actual-output demonstration image/video.')

if __name__ == '__main__':
    main()
