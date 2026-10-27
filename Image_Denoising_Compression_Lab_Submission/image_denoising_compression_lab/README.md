# Image Denoising & Compression Lab

A complete local GenAI / deep-learning class-test project using a **trained PyTorch convolutional variational autoencoder (VAE)** and a **Gradio UI**. The model actually encodes images into learned latent representations and decodes the reduced-size latents. No API call, standard denoising filter, or resize-only substitute is used.

## Requirements

- Python **3.10–3.13** (tested on Python 3.13)
- Windows / Linux / macOS with around 2 GB free RAM
- Internet **only during first-time `pip install`** (package download). A trained model checkpoint and sample image are **included**.

## Run locally

Extract the ZIP, open a terminal in the extracted `image_denoising_compression_lab` folder, then:

**Windows**
```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:7860** in your browser. The supplied example photo is preloaded: select noise and compression, then click **Run VAE experiment**. You can upload your own photos.

> On computers with no `py` launcher, use `python` instead. PyTorch CPU inference is supported and requires no GPU or API key. If port 7860 is already in use, stop the other local server or update `server_port` in `app.py`.

## Controls and visible evidence

1. **Image upload** — real RGB image; center-cropped and resized to the trained 64 × 64 resolution.
2. **Noise slider** — standard deviation of Gaussian pixel noise, 0.00–0.50.
3. **Compression setting** — only 32, 64, 96 or 128 latent values are stored/decoded.
4. **Noise seed** — make comparisons repeatable.

Results: **original**, **noisy**, and **VAE reconstruction** plus **error heatmap**, **MSE**, **PSNR**, latent size, estimated representation ratio, a **noise-vs-reconstruction bar chart**, a **latent capacity-vs-error curve**, and an actual downloadable **`.npz` encoded latent** file.

Metrics are computed over a normalized range of [0, 1] using the **same 64 × 64 clean target**:

- `MSE = mean((clean - result)²)` — smaller is better.
- `PSNR = 10 log10(1 / MSE)` dB — larger is better.
- `Representation ratio = 64×64×3 raw uint8 bytes / (latent_dimensions×2 float16 bytes)` — an *estimated in-memory representation ratio*, **not** a JPEG/PNG file compression ratio. Actual downloadable `.npz` includes additional zip/container overhead.

## Backend — actual learned encoder and decoder

```text
Input image ── center crop / 64×64 / normalize ── clean reference
                                          │
                                          └── add Gaussian noise (σ)
                                                    │
                                                    ▼
                                          CONVOLUTIONAL VAE ENCODER
                                          3→32→64→96 channels
                                          → posterior (μ, log σ²)
                                                    │
                                          take posterior mean μ
                                                    │
                                          retain 32/64/96/128 values
                                          quantize to float16 and save .npz
                                                    │
                                          pad discarded values with zeros
                                                    │
                                                    ▼
                                          TRAINED VAE DECODER
                                          FC→upsampling transposed conv
                                                    │
                                          64×64 RGB reconstruction
                                                    │
                                          compare clean using MSE & PSNR
```

This checkpoint is trained by minimizing pixel reconstruction error between **noisy inputs and clean image patches**, with a small VAE KL-divergence regularizer. During training the active latent prefix alternates to support variable compression. The network gets the noisy input, *not* the clean target, for encoding at inference. Deterministic inference uses the learned posterior mean (rather than randomly sampling it), so repeated tests are comparable.

## Retrain with PyTorch (optional)

A **genuine trained checkpoint** is already inside `weights/vae_denoiser.pt`.

To retrain (does not need an internet dataset download):

```bash
python train.py --epochs 14 --samples-per-epoch 2048
```

The default training dataset draws randomized, augmented 64×64 patches from seven color photographs/illustrations bundled with `scikit-image`. Train using **your own images** to improve performance on your topic:

```bash
python train.py --data-dir path/to/your/photos --epochs 20 --samples-per-epoch 4096
```

Training will replace the checkpoint at `weights/vae_denoiser.pt`. Accepted local formats: PNG/JPG/WEBP/BMP. Training is CPU-compatible; more photos / training epochs may help transfer to unfamiliar image types. This is a compact **educational VAE**, not a state-of-the-art restoration model.

## Decode the actual stored latent without the original image

After clicking **Run VAE experiment**, download the `.npz` file. The original photo is **not included** in this file. Reconstruct independently using:

```bash
python decode_latent.py path/to/vae_latent_128d_XXXX.npz --output decoded.png
```

This loads only the latent coefficients and the provided trained VAE decoder, demonstrating actual compression/reconstruction. It does *not* load the original photo.

## Visual demonstration / evidence

- `assets/demo_results.png` — captured outputs and measured reconstruction statistics from a real model pass.
- `assets/model_demonstration.mp4` — 8-second demonstration comparing all four real compression choices, showing outputs and metrics.
- **For a UI screenshot:** Run `python app.py`, click the experiment button, and capture your computer screen with the three image outputs and metric cards visible.

## Tests

From the project folder:

```bash
python -m pytest tests/ -v
```

Tests confirm: a **trained model is loaded**, actual **encode→decode execution**, **noise injection & determinism**, each of four **stored latent sizes**, reconstruction changes with latent capacity (information loss), and **denoising improvement at high noise** for the included astronaut example. Additional experimental settings may yield worse MSE after reconstruction: the application reports both errors transparently rather than implying universal improvement.

## Project structure

```text
app.py                     Gradio frontend; charts and results
backend/model.py           PyTorch VAE encoder, reparameterization, decoder
backend/preprocessing.py   Image crop/resize, noise injection
backend/inference.py       Load trained weights, true latent coding, decode
backend/metrics.py         MSE, PSNR, representation ratio
train.py                   Real offline VAE training pipeline
weights/vae_denoiser.pt    Included trained model checkpoint
assets/sample_*.png        Sample images for demo
assets/demo_results.png    Actual model output demonstration
assets/model_demonstration.mp4   Short generated demo using real VAE outputs
decode_latent.py           Decode saved latents without original image
tests/test_pipeline.py     Runnable test cases
requirements.txt           Dependencies
```

## Suggested class demonstration

Upload/sample photo → set **noise = 0.40**, **latent = 128** → click **Run VAE experiment** → point to three images, MSE and PSNR. Then set **latent = 32** and run again: the retained latent payload becomes smaller and reconstruction typically loses more information. Expand the latent section to show the actual `.npz` file containing only 32 float16 coefficients. Finally, show `backend/inference.py`'s `model.encode(...)` and `model.decode(...)` calls and execute `python -m pytest tests/ -v`.

**Caveats:** All models operate on 64×64 image patches, so fine details are lost; visually imperfect reconstruction is expected. The representation ratio compares stored float16 features against raw, uncompressed RGB image bytes only. The model weights are not counted in per-image storage comparisons and must be available at decoding time.
