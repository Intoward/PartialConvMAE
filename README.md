# PartialConvMAE: Spatio-Temporal Spectrum Cartography via Partial Convolutional Masked Autoencoders

An asymmetric 3D Masked Autoencoder architecture designed for high-fidelity radio frequency (RF) power map reconstruction and dynamic spectrum cartography from sparse, irregular spatio-temporal observations.

Developed as a part of my 2026 Senior Design Project for the University of Central Florida. This was designed to be used on Google Colab, but this project's .ipynb file can be partitioned into discrete .py files for similar functionality.

This model architecture was derived from the ConvMAE model featured in: https://arxiv.org/pdf/2505.15571

---

## Overview

Spectrum cartography estimates continuous spatial, temporal, and spectral distributions of RF signal power from sparse spatial sensor observations. Traditional methods (such as Kriging interpolation, Thin-Plate Splines, and classical autoencoders) degrade significantly under extreme sparsity, non-uniform spatial sampling, and irregular sensor trajectories.

**PartialConvMAE** addresses these challenges by uniting **3D Partial Convolutions (PConv)** with an **Asymmetric Masked Autoencoder (MAE)** pipeline:

1. **Partial Convolutional Stem**: Normalizes activations over irregular receptive fields and dynamically propagates observation masks across 3D volumes (frequency/time slices x spatial grid).
2. **Asymmetric Transformer Encoder**: Discards unmeasured patches entirely, processing only visible patch tokens with 3D sinusoidal positional embeddings to minimize compute and memory.
3. **Cross-Attention Reconstructive Decoder**: Uses learned mask tokens to query encoded visible context via multi-head cross-attention and self-attention, reconstructing the complete continuous RF power field.
4. **Hole-Focused Loss Formulation**: Employs a masked Mean Squared Error (MSE) criterion evaluated primarily over unmeasured voxels alongside an auxiliary global reconstruction term.

---

## Model Architecture

The architecture operates across three synchronized phases:

```
Raw Sparse RF Tensor (B, C, D, H, W) & Binary Mask (B, 1, D, H, W)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: 3D Partial Convolutional Stem                     │
│ - PartialConv3d (1 -> stem_dim, k=3, s=1, p=1) + BN + ReLU   │
│ - PartialConv3d (stem_dim -> embed_dim, k=3, s=ps, p=1)     │
│ - Normalizes convolution over valid observations            │
│ - Downsamples spatio-temporal dimensions and updates mask   │
└──────────────────────────┬──────────────────────────────────┘
                           │ Dense features & downsampled mask
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Asymmetric Transformer Encoder                     │
│ - 3D Sinusoidal Positional Encoding                         │
│ - Patch masking: Discards patches where mask sum == 0       │
│ - Processes only visible tokens (N_vis << N) through Le     │
│   Pre-LN Transformer blocks (MHA + FFN)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ Encoded visible tokens (B, N_vis, C_enc)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Cross-Attention Reconstructive Decoder             │
│ - Full sequence allocation (N tokens) with mask tokens      │
│ - In-place insertion of visible projected tokens            │
│ - Ld Cross-Attention blocks:                                │
│     Q = Full sequence (mask + visible)                      │
│     K, V = Encoded visible tokens                           │
│ - Patch projection head & 3D un-folding / rearrange         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
      Reconstructed RF Volume (B, C, D, H, W)
                           │
                           ▼
              Masked Reconstruction Loss
       L = MSE(pred[unmeasured], target[unmeasured]) + λ * MSE(pred, target)

```

### Key Architectural Specifications

* **Input Representation**: Spatio-temporal RF tensors of shape `(B, 1, D, H, W)` where $D$ represents temporal/frequency depth slices and $H, W$ represent 2D spatial coordinate bins (dBm values).
* **Masking Mechanism**: Pointwise measurement mask $M \in \{0, 1\}^{B \times 1 \times D \times H \times W}$ indicating spatial sensor coverage ($1 = \text{observed}, 0 = \text{unmeasured}$).
* **Target Platform**: Optimized for real-time edge deployment on NVIDIA Jetson Orin Nano systems (<500 ms target latency).

---

## Tech Stack

* **Language**: Python 3.10+
* **Deep Learning Framework**: PyTorch 2.0+
* **Numerical Computation**: NumPy, SciPy
* **Hardware Acceleration**: CUDA, cuDNN, TensorRT (target for edge deployment)
* **Target Edge Hardware**: NVIDIA Jetson Orin Nano / RTX-series GPUs

---

## Repository Structure

```
├── partial_conv.py      # 3D Partial Convolution layer & PartialConvStem
├── encoder.py           # 3D Sinusoidal positional encodings & Asymmetric Transformer Encoder
├── decoder.py           # Cross-Attention Transformer blocks & Reconstructive Decoder
├── loss.py              # MaskedReconstructionLoss (unmeasured hole MSE + auxiliary full loss)
├── model.py             # PartialConvMAE top-level model wrapper
├── main.py              # Synthetic RF evaluation script, benchmark pass, and sanity checks
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation

```

---

## Installation

### Prerequisites

* Python 3.10 or higher
* NVIDIA CUDA Toolkit 11.8+ or 12.0+ (for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/PartialConvMAE.git
cd PartialConvMAE

```


2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate

```


3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt

```



---

## Usage

### Forward Pass Benchmark / Sanity Check

Run a synthetic forward pass to verify installation, parameter allocation, and tensor reconstruction across simulated measurement sparsity:

```bash
python main.py --device cpu --sparsity 0.85 --batch 2 --depth 8 --height 32 --width 32

```

To run on GPU:

```bash
python main.py --device cuda --sparsity 0.85

```

### Python API Example

```python
import torch
from model import PartialConvMAE

# Instantiate model matching Jetson-oriented default parameters
model = PartialConvMAE(
    in_channels=1,
    stem_dim=64,
    embed_dim=256,
    dec_dim=128,
    enc_depth=6,
    dec_depth=4,
    num_heads_enc=8,
    num_heads_dec=8,
    mlp_ratio=4.0,
    patch_stride=4,
    dropout=0.1,
    lambda_full=0.1
)

# Input dimensions: (Batch, Channels, Depth/Time, Height, Width)
batch_size, c, d, h, w = 2, 1, 8, 32, 32

# Simulated sparse signal and observation mask (85% missing)
target = torch.randn(batch_size, c, d, h, w) * 10.0 - 80.0
mask = torch.bernoulli(torch.full((batch_size, 1, d, h, w), 0.15))
x_sparse = target * mask

# Training forward pass (calculates masked loss)
model.train()
recon, loss = model(x_sparse, mask, target=target)
loss.backward()

# Inference pass (reconstruction only)
model.eval()
with torch.no_grad():
    recon = model(x_sparse, mask)

print(f"Reconstruction shape: {recon.shape}")

```

---

## Configuration & Hyperparameters

| Parameter | Default | Description |
| --- | --- | --- |
| `in_channels` | `1` | Input RF power map channel count |
| `stem_dim` | `64` | Intermediate feature channels in PartialConvStem |
| `embed_dim` | `256` | Token embedding dimension ($C_{enc}$) |
| `dec_dim` | `128` | Decoder hidden dimension ($C_{dec}$) |
| `enc_depth` | `6` | Number of encoder Transformer blocks ($L_e$) |
| `dec_depth` | `4` | Number of decoder cross-attention blocks ($L_d$) |
| `num_heads_enc` | `8` | Multi-head attention heads in encoder |
| `num_heads_dec` | `8` | Multi-head attention heads in decoder |
| `mlp_ratio` | `4.0` | Expansion ratio for Feed-Forward Networks |
| `patch_stride` | `4` | Downsampling stride in 3D stem (patch side length) |
| `dropout` | `0.1` | Dropout rate for attention and FFN layers |
| `lambda_full` | `0.1` | Weighting coefficient for auxiliary full-map loss |

---

## Loss Formulation

Standard autoencoders compute mean squared error over the entire tensor, often rewarding trivial identity mapping of existing measurements. PartialConvMAE instead optimizes signal completion over unobserved coordinates:

$$\mathcal{L}_{\text{hole}} = \frac{1}{\sum (1 - M)} \sum (1 - M) \odot (\hat{P} - P)^2$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{hole}} + \lambda_{\text{full}} \cdot \text{MSE}(\hat{P}, P)$$

where:

* $\hat{P}$ is the reconstructed spatio-temporal power map tensor.
* $P$ is the ground-truth power map.
* $M \in \{0, 1\}$ is the binary measurement mask ($1 = \text{measured}$, $0 = \text{unmeasured}$).
* $\lambda_{\text{full}}$ balances hole completion against global feature consistency during initial training epochs.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
