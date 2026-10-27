"""Run the Image Denoising & Compression Lab with: python app.py"""
from __future__ import annotations

import html
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backend.inference import load_model, process_image
from backend.model import LATENT_OPTIONS

BASE = Path(__file__).resolve().parent
SAMPLE_PATH = BASE / 'assets' / 'sample_astronaut.png'


CSS = """
.gradio-container {max-width: 1240px !important; margin: auto !important;}
#hero {background: linear-gradient(125deg, #093d49, #0b6571); color: #fff;
  border-radius: 20px; padding: 25px 30px; margin-bottom: 10px;}
#hero h1 {color: white !important; font-size: 2.25em; margin-bottom: 4px;}
#hero p {color: #dbf4ef !important; margin-bottom: 0;}
#metrics-card {margin-top: 6px;}
.metric-grid {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 11px;}
.metric {background: white; border: 1px solid #d7e7e7; border-radius: 12px; padding: 15px;}
.metric .t {color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.035em;}
.metric .v {color:#0e4952; font-size:25px; font-weight:750; margin-top:5px;}
.metric .sub {color:#64748b; font-size:11px;}
@media (max-width:700px) {.metric-grid {grid-template-columns:repeat(2,1fr);}}
"""


def make_plot(values, xlabel, ylabel, title, kind='bar'):
    fig, ax = plt.subplots(figsize=(7.8, 3.1), dpi=120)
    if kind == 'bar':
        bars = ax.bar(range(len(values)), [v[1] for v in values], color=['#e89d70', '#137f7a'])
        ax.set_xticks(range(len(values)), [v[0] for v in values])
        for bar in bars:
            ax.annotate(f'{bar.get_height():.4f}',
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(v[1] for v in values) * 1.30 + 1e-8)
    else:
        x = [v[0] for v in values]
        y = [v[1] for v in values]
        ax.plot(x, y, marker='o', lw=2.5, markersize=7, color='#137f7a')
        ax.set_xticks(list(LATENT_OPTIONS))
        for xx, yy in zip(x, y):
            ax.annotate(f'{yy:.4f}', (xx, yy), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=8)
        ax.set_ylim(0, max(y) * 1.25 + 1e-8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis='y', alpha=.20)
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    return fig


def metric_panel(result) -> str:
    c = result.compression
    improvement = result.mse_noisy - result.mse_reconstructed
    trend = ('Improved vs noisy image' if improvement > 0 else 'Higher error than noisy image')
    tiles = [
        ('Reconstruction MSE', f'{result.mse_reconstructed:.5f}', 'Lower = better'),
        ('Reconstruction PSNR', f'{result.psnr_reconstructed:.2f} dB', 'Higher = better'),
        ('Latent dimensions', str(c['retained_features']), 'of 12,288 pixel values'),
        ('Representation ratio', f"{c['ratio']:.1f}:1", f"{c['latent_bytes']}B latent / {c['raw_image_bytes']:,}B raw"),
    ]
    inner = ''.join(
        f'<div class="metric"><div class="t">{html.escape(t)}</div>'
        f'<div class="v">{html.escape(v)}</div><div class="sub">{html.escape(s)}</div></div>'
        for t, v, s in tiles
    )
    return f'''<div class="metric-grid">{inner}</div>
    <p style="font-size:12px;color:#58707a;margin-top:10px;">
       Noisy MSE: <b>{result.mse_noisy:.5f}</b> · 
       Noisy PSNR: <b>{result.psnr_noisy:.2f} dB</b> · 
       <b>{html.escape(trend)}</b>.
       The ratio compares raw RGB bytes to 16-bit latent values, not JPEG/PNG file sizes.
    </p>'''


def run_lab(image, noise_level, compression, seed):
    if image is None:
        raise gr.Error('Upload a photo to begin.')
    dims = int(compression.split('—')[0].strip())
    if dims not in LATENT_OPTIONS:
        raise gr.Error('Invalid compression setting.')
    try:
        result = process_image(image, float(noise_level), dims, int(seed))
    except (OSError, FileNotFoundError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    chart_mse = make_plot(
        [('Noisy input', result.mse_noisy), ('VAE output', result.mse_reconstructed)],
        '', 'Mean squared error (0–1 scale)', 'Reconstruction error comparison', 'bar'
    )
    chart_curve = make_plot(
        result.compression_curve, 'Retained latent dimensions',
        'Reconstruction MSE (lower is better)', 'Latent bottleneck vs. information loss', 'line'
    )
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.npz',
                                     prefix=f'vae_latent_{dims}d_', delete=False) as file:
        file.write(result.latent_payload)
        latent_path = file.name
    checkpoint = result.checkpoint_metadata
    info = (f"**Real model:** {checkpoint.get('architecture', 'Convolutional VAE')}  "
            f"  \n**Model training:** {checkpoint.get('epochs', '?')} epochs on "
            f"{checkpoint.get('training_dataset', 'images')}  "
            f"  \n**Actual inference path:** noise injection → convolutional encoder → "
            f"{dims}-value float16 latent → zero padding → learned decoder → 64 × 64 RGB.  "
            f"  \n**Reconstruction caveat:** the image is center-cropped and rescaled to "
            f"64 × 64 because that is the trained model resolution. The result is not a filter or resized copy.")
    return (result.original, result.noisy, result.reconstructed,
            result.residual, metric_panel(result), chart_mse, chart_curve,
            latent_path, info)


def create_app():
    with gr.Blocks(title='Image Denoising & Compression Lab') as demo:
        gr.HTML('''<section id="hero"><h1>Image Denoising &amp; Compression Lab</h1>
          <p>Explore what a trained PyTorch variational autoencoder retains, loses, and reconstructs.</p></section>''')
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=290):
                gr.Markdown('### 1 · Set up the experiment')
                image = gr.Image(label='Upload an RGB image', type='pil',
                                 sources=['upload'], value=str(SAMPLE_PATH), height=244)
                noise = gr.Slider(0.0, 0.50, value=0.35, step=0.01,
                                  label='Gaussian noise level (standard deviation)')
                compression = gr.Radio(
                    ['32 — Highest compression', '64 — High compression',
                     '96 — Balanced', '128 — Best reconstruction'],
                    value='128 — Best reconstruction', label='Compression setting',
                )
                seed = gr.Number(value=42, precision=0, label='Noise seed (reproducible)')
                button = gr.Button('Run VAE experiment', variant='primary', size='lg')
                gr.Markdown('**What changes?** Less latent capacity saves more values but can lose visual detail. '
                            'The encoder receives the noisy input; the decoder reconstructs from stored latent data.')
            with gr.Column(scale=2, min_width=450):
                gr.Markdown('### 2 · View the model output')
                with gr.Row():
                    original = gr.Image(label='Original (64 × 64)', height=228, interactive=False)
                    noisy = gr.Image(label='Noise injected', height=228, interactive=False)
                    reconstructed = gr.Image(label='VAE reconstruction', height=228, interactive=False)
                metrics = gr.HTML(label='Quantitative metrics', elem_id='metrics-card')
                gr.Markdown('### 3 · Quantify the trade-offs')
                with gr.Row():
                    mse_chart = gr.Plot(label='Noise vs reconstruction error')
                    latent_chart = gr.Plot(label='Latent capacity vs information loss')
                with gr.Accordion('Inspect reconstruction residual and saved latent', open=False):
                    residual = gr.Image(label='Absolute reconstruction error × 3 (visualized)', height=200)
                    latent_file = gr.File(label='Download actual compressed latent (.npz)')
                    model_info = gr.Markdown()
        outs = [original, noisy, reconstructed, residual, metrics,
                mse_chart, latent_chart, latent_file, model_info]
        button.click(fn=run_lab, inputs=[image, noise, compression, seed], outputs=outs)
        gr.Markdown('**Method:** A convolutional **VAE is trained to map noisy image patches to clean targets**. '
                    'At inference the learned encoder produces a Gaussian posterior; its mean is stored in float16 '
                    'at the selected latent size, then decoded by the trained decoder. '
                    'MSE/PSNR are calculated against the same preprocessed original. '
                    'Denoising is **not guaranteed** at every noise and compression setting.')
    return demo


demo = create_app()

if __name__ == '__main__':
    # No internet, API key or external model service is needed.
    demo.launch(server_name='127.0.0.1', server_port=7861, share=False,
                theme=gr.themes.Soft(primary_hue='teal'), css=CSS)
