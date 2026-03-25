# Brain-like Representations with Self-Supervised Vision Models

> **Deep model interpretability by analyzing brain-network correlations**  
> Jibril El Hassani · Thomas Prados · Hamza Berqoq El Alami — CentraleSupélec

---

## Overview

This repository contains the code associated with our NeurIPS 2025 paper. We extend the analysis of [Raugel et al. (2025)](https://arxiv.org/abs/2508.18226) along two complementary axes:

1. **Reproduction** of their key findings (temporal score on MEG, spatial score on fMRI) using DINO features, with a cross-subject signal averaging strategy that significantly boosts temporal encoding scores.
2. **Neural-guided learning** — a novel framework where MEG and fMRI brain signals act as biological regularizers during the training of a convolutional autoencoder, under two architectural setups: adversarial training and concatenation-based conditioning.

### Key results

| Metric | This work | Raugel et al. |
|---|---|---|
| Spatial score (fMRI) | **r = 0.910** (p < 0.001) | r = 0.38 |
| Peak temporal correlation (MEG) | **0.185** | 0.09 |
| Model size vs. DINOv3-Giant | **85× smaller** | baseline |

---

## Repository Structure

```
.
├── meg_temporal_score.ipynb              # MEG temporal encoding analysis (Section 3)
├── fMRI_spatial_score.ipynb             # fMRI spatial score analysis (Section 4)
├── fmri_spatial_score.png               # Spatial score figure
├── image_reconstruction_with_fMRI.ipynb # Neural-guided learning with fMRI (Section 5)
├── image_reconstruction_with_meg.ipynb  # Neural-guided learning with MEG (Section 5)
├── nsd_extractor.py                     # Utility to extract NSD fMRI data
├── setup_data.txt                       # Data setup instructions
└── 2508.18226v1.pdf                     # Reference paper (Raugel et al., 2025)
```

---

## Methods

### Temporal Score (MEG)

We reproduce the dynamic MEG encoding analysis of Raugel et al. with one key methodological change: instead of averaging encoding scores across subjects, we first compute a **grand average of MEG signals** across all subjects. This suppresses idiosyncratic noise and maximises SNR, allowing us to reach a peak Pearson correlation of **0.185** — more than double the original 0.09 — with a model 85× smaller than DINOv3-Giant.

Intermediate DINO layers (L5–L6) show the highest alignment with MEG signals, consistent with the ventral stream's rich object representations. Brain–model correlation closely tracks the MEG SNR, confirming that alignment is noise-limited at the single-trial level.

### Spatial Score (fMRI)

For each fMRI region of interest (ROI) from the Natural Scenes Dataset, we train a ridge regression mapping DINO layer activations to BOLD responses. The spatial score is computed as the Pearson correlation between each ROI's optimal encoding layer and its Euclidean distance from V1 in MNI space.

- Early visual areas (V1, V2) → best predicted by intermediate layers (L5)
- Higher-order regions (OFA, FFA, OPA, PPA, EBA, IPS) → best predicted by deep layers (L11–L12)
- DINO features outperform a PCA pixel baseline across all ROIs, with the gap widening in higher-order areas

### Neural-Guided Learning

We introduce two training setups that use brain signals as biological regularizers for a convolutional autoencoder:

**Setup 1 — Adversarial Training**  
A two-layer MLP probe maps the latent code `z` to a predicted brain signal. The combined loss is:
```
L = L_recon + λ · L_MSE(brain)
```
Brain gradients back-propagate through the probe into the encoder, steering the latent space toward brain-predictive features.

**Setup 2 — Concatenation Training**  
The brain signal embedding `c` (MEG or fMRI) is concatenated with the latent code to form `[z; c]`, which is passed to the decoder. Only the decoder is trained with brain information; the encoder receives no brain gradient.

**Finding:** Biological guidance consistently **fails to improve** visual reconstruction quality in both setups. The adversarial setup creates a representational competition between semantic and pixel-level features; the concatenation setup suffers from the high variance of single-trial neural data acting as a noisy prior.

---

## Datasets

| Dataset | Modality | Description |
|---|---|---|
| [THINGS-MEG](https://elifesciences.org/articles/82580) | MEG | 4 subjects, 19,848 natural images from the THINGS database |
| [NSD](https://www.nature.com/articles/s41593-021-00962-x) | 7T fMRI | 8 subjects, 10,000 natural scenes |

Both datasets are publicly available. See `setup_data.txt` for download and preprocessing instructions.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- `timm` (for DINO features)
- `scikit-learn` (ridge regression)
- `numpy`, `matplotlib`, `scipy`

### Running the notebooks

**Temporal score (MEG):**
```bash
jupyter notebook meg_temporal_score.ipynb
```

**Spatial score (fMRI):**
```bash
jupyter notebook fMRI_spatial_score.ipynb
```

**Image reconstruction with neural guidance:**
```bash
jupyter notebook image_reconstruction_with_fMRI.ipynb
jupyter notebook image_reconstruction_with_meg.ipynb
```

Data paths and hyperparameters are configured at the top of each notebook. All experiments were run on an **NVIDIA P100 GPU** (Kaggle), with approximately **6 minutes per epoch**.

---

## Citation

If you use this code, please cite our paper and the original Raugel et al. work:

```bibtex
@article{elhassani2025brainlike,
  title     = {Deep model interpretability by analyzing brain-network correlations},
  author    = {El Hassani, Jibril and Prados, Thomas and Berqoq El Alami, Hamza},
  year      = {2025},
  institution = {CentraleSupélec}
}

@article{raugel2025disentangling,
  title   = {Disentangling the factors of convergence between brains and computer vision models},
  author  = {Raugel, J. and Szafraniec, M. and Vo, H.V. and Couprie, C. and Labatut, P. and Bojanowski, P. and Wyart, V. and King, J.R.},
  journal = {arXiv preprint arXiv:2508.18226},
  year    = {2025}
}
```

---

## License

This project is released for research purposes. The datasets used (THINGS-MEG, NSD) are subject to their respective licenses and terms of use as defined by their original creators.
