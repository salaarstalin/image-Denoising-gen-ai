"""Streamlit frontend for the trained VAE image denoising and compression lab."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from backend.inference import process_image

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Image Denoising & Compression Lab', layout='wide')
st.title('Image Denoising & Compression Lab')
st.caption('Real PyTorch Variational Autoencoder — noise injection, latent compression, and reconstruction.')

with st.sidebar:
    st.header('1. Set up the experiment')
    upload = st.file_uploader('Upload an RGB image', type=['png', 'jpg', 'jpeg', 'webp'])
    noise_level = st.slider('Gaussian noise level', min_value=0.0, max_value=0.50, value=0.35, step=0.01)
    active_dims = st.select_slider('Compression setting (latent dimensions)', options=[32, 64, 96, 128], value=128)
    seed = st.number_input('Noise seed', min_value=0, max_value=100000, value=42, step=1)
    run = st.button('Run VAE experiment', type='primary', use_container_width=True)

if upload is not None:
    img = Image.open(upload).convert('RGB')
else:
    img = Image.open(ROOT / 'assets' / 'sample_astronaut.png').convert('RGB')
    st.info('Using included astronaut sample image. Upload your own image in the sidebar.')

if run:
    try:
        with st.spinner('Running trained encoder and decoder...'):
            st.session_state['result'] = process_image(img, noise_level, active_dims, int(seed))
            st.session_state['settings'] = (noise_level, active_dims, int(seed))
    except (RuntimeError, OSError, ValueError, FileNotFoundError) as exc:
        st.error(f'Inference failed: {exc}')

if 'result' in st.session_state:
    r = st.session_state['result']
    st.subheader('2. View the model output')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(r.original, caption='Original (64 × 64)', use_container_width=True)
    with c2:
        st.image(r.noisy, caption='Noise injected', use_container_width=True)
    with c3:
        st.image(r.reconstructed, caption='VAE reconstruction', use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Reconstruction MSE ↓', f'{r.mse_reconstructed:.5f}')
    m2.metric('Reconstruction PSNR ↑', f'{r.psnr_reconstructed:.2f} dB')
    m3.metric('Latent dimensions', str(r.compression['retained_features']))
    m4.metric('Raw representation ratio', f"{r.compression['ratio']:.1f}:1")
    st.caption(f'Noisy MSE: {r.mse_noisy:.5f} | Noisy PSNR: {r.psnr_noisy:.2f} dB. '
               'Compression ratio compares raw 8-bit RGB values with 16-bit latent values, not JPEG/PNG sizes.')

    st.subheader('3. Quantify the trade-offs')
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.bar(['Noisy input', 'VAE output'], [r.mse_noisy, r.mse_reconstructed])
        ax.set_ylabel('MSE (lower is better)')
        ax.set_title('Noise vs reconstruction error')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.plot([x for x, y in r.compression_curve], [y for x, y in r.compression_curve], marker='o')
        ax.set_xticks([32, 64, 96, 128])
        ax.set_xlabel('Retained latent dimensions')
        ax.set_ylabel('Reconstruction MSE')
        ax.set_title('Latent capacity vs information loss')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander('Inspect reconstruction residual and saved latent'):
        st.image(r.residual, caption='Absolute reconstruction error × 3')
        st.download_button('Download actual compressed latent (.npz)', r.latent_payload,
                           file_name=f"vae_latent_{r.compression['retained_features']}d.npz",
                           mime='application/octet-stream')
        st.write('Model metadata:', r.checkpoint_metadata)
        st.caption('Inference: noisy image → learned VAE encoder → truncated float16 latent → learned decoder → reconstruction.')
else:
    st.subheader('Input preview')
    st.image(img, width=300)
    st.write('Choose your settings in the sidebar and click **Run VAE experiment**.')

st.caption('Method: a trained convolutional VAE reconstructs from encoded latent coefficients. Denoising is not guaranteed under every setting.')
