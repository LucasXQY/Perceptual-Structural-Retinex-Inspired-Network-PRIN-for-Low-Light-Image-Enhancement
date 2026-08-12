# PRIN: A Perceptual-Structural Retinex-Inspired Network for Low-Light Image Enhancement

Official repository for the paper:

> **PRIN: A Perceptual-Structural Retinex-Inspired Network for Low-Light Image Enhancement**
> *Submitted to Information Sciences (under review).*

PRIN is a restoration-role-aware Retinex-inspired network. A shared encoder feeds two
asymmetric decoders that estimate reflectance and illumination; each branch is refined in the
Haar wavelet domain by a role-specific module — **WDGRR** (Wavelet Detail-Guided Reflectance
Refiner) for the reflectance branch and **WSAIR** (Wavelet Structure-Aware Illumination
Refiner) for the illumination branch — and the recomposed image is post-enhanced by a
lightweight Zero-DCE-style curve head (**DCECurveEnhancer**, T = 8 curve iterations).

## Release status

| Component | Status |
|---|---|
| Pretrained checkpoints (main model + all ablations) | ✅ Available via Google Drive (links below) |
| Enhanced result images for every experiment | ✅ Available via Google Drive (links below) |
| Evaluation protocol code (`evaluate.py`, `run_evaluate.py`) | ✅ In this repository |
| Inference / training / experiment harness scripts | ✅ In this repository |
| Network source code (`models/`) and loss source code (`losses/`) | 🔒 **Released upon paper acceptance** |

Until the network source is released, every quantitative number in the paper can still be
verified: download the enhanced result images from Google Drive and score them with
`evaluate.py` / `run_evaluate.py` — the exact scripts that produced the tables in the paper.

## Downloads (Google Drive)

### Checkpoints (≈ 1.8 GB total — place under `checkpoints/`, see [checkpoints/README.md](checkpoints/README.md))

| Item | Contents | Link |
|---|---|---|
| `prin_lolv1.pth` | PRIN trained on LOL-v1 | [Google Drive](https://drive.google.com/file/d/1aDnSX0CCXSPgGwsV03JI9gGZpehvatPd/view?usp=sharing) |
| `prin_lolv2real.pth` | PRIN trained on LOL-v2-Real | [Google Drive](https://drive.google.com/file/d/1WX2giiRfjcgHqXcwv2kKHJC8nUdupsgu/view?usp=sharing) |
| `prin_lolv2syn.pth` | PRIN trained on LOL-v2-Syn | [Google Drive](https://drive.google.com/file/d/1P0JGLlUYk_kkmWgrfjoph2FjOC_tgmOe/view?usp=sharing) |
| `module_ablation/` | 4 module-ablation checkpoints (LOL-v2-Real) | [Google Drive](https://drive.google.com/drive/folders/13rK3wy9BMPeARP5M0uIR_dQi-Ll1uFSO?usp=sharing) |
| `scale_ablation/` | 7 scale-factor-ablation checkpoints, s0–s6 (LOL-v1) | [Google Drive](https://drive.google.com/drive/folders/19ijwjQCOiwD4r_5Q_p_GdOUYFt2aIAi6?usp=sharing) |
| `loss_ablation/` | 5 loss-group-ablation checkpoints (LOL-v1) | [Google Drive](https://drive.google.com/drive/folders/12mqOD5WIB255teTqoLMvkSfB7R0rh9bz?usp=sharing) |

### Result images (≈ 1.0 GB total — enhanced outputs of every experiment)

| Folder | Contents | Link |
|---|---|---|
| `main_lolv1/` | LOL-v1 test set: enhanced / reflectance / illumination | [Google Drive](https://drive.google.com/drive/folders/1vEGBHZJ2MkZ2ofszVCdKgHNl0D27kSU_?usp=sharing) |
| `main_lolv2real/` | LOL-v2-Real test set: enhanced / reflectance / illumination | [Google Drive](https://drive.google.com/drive/folders/19Hbq5HhgRRdq4e9JKhsaOq0jt9pUwBN5?usp=sharing) |
| `main_lolv2syn/` | LOL-v2-Syn test set: enhanced / reflectance / illumination | [Google Drive](https://drive.google.com/drive/folders/15ed7aXu4QBQTI_mtdjK8NsyB0WCeW5zh?usp=sharing) |
| `module_ablation/` | module-ablation enhanced outputs (4 variants) | [Google Drive](https://drive.google.com/drive/folders/1rLJRuwJNzJE5YYYoszZVhDXGQQKVEiJx?usp=drive_link) |
| `scale_ablation/` | scale-ablation enhanced outputs (s0–s6) | [Google Drive](https://drive.google.com/drive/folders/1TxEImsguKTmx9WOOufIrGbsxGcA55EuR?usp=drive_link) |
| `loss_ablation/` | loss-group-ablation outputs (full / wo_g1 … wo_g4) | [Google Drive](https://drive.google.com/drive/folders/1A3dzN29HQkkCqFRbCfQm7RlsOREbKMFp?usp=sharing) |
| `variant_study/` | variant-study outputs (V1a / V1b / V3) | [Google Drive](https://drive.google.com/drive/folders/1q95SVIESrHaHuPKwaE5JdHQMx9IlyMyk?usp=drive_link) |
| `nonref_zeroshot/` | zero-shot outputs on DICM / LIME / MEF / NPE / VV | [Google Drive](https://drive.google.com/drive/folders/1ccV5pG6mKvhJa6qyCW0ficBbkuiZlujD?usp=drive_link) |

### Checkpoint inventory

| File | Model class | Trained on | Experiment |
|---|---|---|---|
| `checkpoints/prin_lolv1.pth` | `PRIN` | LOL-v1 | Main results |
| `checkpoints/prin_lolv2real.pth` | `PRIN` | LOL-v2-Real | Main results |
| `checkpoints/prin_lolv2syn.pth` | `PRIN` | LOL-v2-Syn | Main results |
| `checkpoints/module_ablation/dce_only.pth` | `PRIN_DCEOnly` | LOL-v2-Real | Module ablation (+DCE row) |
| `checkpoints/module_ablation/wdgrr_only.pth` | `PRIN_WDGRROnly` | LOL-v2-Real | Module ablation (+WDGRR row) |
| `checkpoints/module_ablation/wsair_only.pth` | `PRIN_WSAIROnly` | LOL-v2-Real | Module ablation (+WSAIR row) |
| `checkpoints/module_ablation/wdgrr_wsair.pth` | `PRIN_WDGRR_WSAIR` | LOL-v2-Real | Module ablation (+WDGRR+WSAIR row) |
| `checkpoints/scale_ablation/s0_default.pth` … `s6_sobel_high.pth` | `PRIN` (variant scales) | LOL-v1 | Scale-factor ablation (7 variants) |
| `checkpoints/loss_ablation/full.pth`, `wo_g1.pth` … `wo_g4.pth` | `PRIN` | LOL-v1 | Loss-group ablation (5 variants) |

Notes:
- The module-ablation *full* row is simply the main checkpoint `prin_lolv2real.pth`.
- The *Baseline* row of the module ablation (no WDGRR / WSAIR / DCE) is not archived.
- Scale-ablation variants share the `PRIN` architecture; the scale factors are **not** part of
  the state dict, so each checkpoint must be loaded through its matching variant config
  (handled automatically by `experiments/scale_ablation/test_scale_ablation.py`).

## Repository layout

```
PRIN/
├── train.py                    # Training entry (Adam, lr 1e-4, batch 2, 2500 epochs)
├── test.py                     # Inference: low-light input → enhanced / reflectance / illumination
├── evaluate.py                 # Canonical full-reference metric protocol (PSNR/SSIM/MAE/LPIPS/NIQE)
├── run_evaluate.py             # CLI driver around evaluate.py (no file editing needed)
├── requirements.txt
├── datasets/                   # PyTorch dataset loaders for LOL-v1 / LOL-v2-Real / LOL-v2-Syn
├── models/                     # 🔒 network code — released upon acceptance (see models/README.md)
├── losses/                     # 🔒 loss code — released upon acceptance (see losses/README.md)
├── checkpoints/                # ← place the Google Drive checkpoint download here
└── experiments/
    ├── main_results/           # per-image metric CSVs + training-loss logs of the main model
    ├── variant_study/          # V1a / V1b / V2 / V3 metric CSVs + complexity script
    ├── module_ablation/        # WDGRR / WSAIR / DCE ablation harness + metric CSVs
    ├── scale_ablation/         # wavelet-scale ablation harness + metric CSVs (s0–s6)
    ├── loss_ablation/          # loss-group ablation (G1–G4): metric CSVs + training logs
    ├── nonref_lol/             # MUSIQ / LIQE / Q-Align on the LOL test sets
    ├── nonref_zeroshot/        # zero-shot DICM / LIME / MEF / NPE / VV study
    ├── gtmean_study/           # GT-mean correction study
    ├── blur_study/             # Gaussian-blur PSNR analysis
    └── efficiency/             # Params / FLOPs / latency measurement
```

## Installation

```bash
conda create -n prin python=3.10 -y
conda activate prin
pip install -r requirements.txt
```

Tested with PyTorch ≥ 2.0 + CUDA on a single NVIDIA RTX 4090 (training) — inference fits on
much smaller GPUs. `pyiqa` downloads NIQE/MUSIQ/LIQE/Q-Align weights automatically on first use.

## Data preparation

Download the benchmarks and arrange them as follows (folder names matter):

```
data/
├── lolv1/        {train,test}/{low,high}        # LOL   (485 train / 15 test pairs)
├── lolv2real/    {train,test}/{low,high}        # LOL-v2-Real (689 / 100)
├── lolv2syn/     {train,test}/{low,high}        # LOL-v2-Syn  (900 / 100)
└── non-ref/      DICM/  LIME/  MEF/  NPE/  VV/  # unpaired sets for the zero-shot study
```

- LOL-v1: Wei et al., *Deep Retinex Decomposition for Low-Light Enhancement*, BMVC 2018.
- LOL-v2 (Real & Synthetic): Yang et al., *Sparse Gradient Regularized Deep Retinex Network*, TIP 2021.
- DICM / LIME / MEF / NPE / VV: standard unpaired LLIE evaluation sets.

## Quick start

### 1. Inference

Download the checkpoints, then:

```bash
python test.py
```

Defaults are set in the `__main__` block (checkpoint `checkpoints/prin_lolv1.pth`, inputs
`data/lolv1/test/low`, outputs `results/lolv1/{enhanced,reflectance,illumination}`).
Edit the three variables to switch dataset/checkpoint. Inputs are reflection-padded to a
multiple of 32 and cropped back — arbitrary resolutions are supported, no resizing occurs.

### 2. Evaluation (the protocol behind every table)

```bash
python run_evaluate.py --results_dir results/lolv1/enhanced --gt_dir data/lolv1/test/high --output_csv results/metrics_lolv1.csv
```

Protocol details (identical for all reported numbers):
- PSNR / SSIM: scikit-image on RGB in [0, 1] (`data_range=1.0`).
- LPIPS: AlexNet backbone, inputs linearly mapped to [-1, 1].
- MAE and NIQE (pyiqa) are logged in the same CSV.
- GT matching: exact filename → same stem → same trailing numeric id (`low00xxx` ↔ `normal00xxx`).
- The CSV ends with an `Average` row — those averages are the paper numbers.

To verify the paper without running the model: download the result images from Google Drive
and point `--results_dir` at them; the produced averages match
`experiments/main_results/metrics_prin_*.csv`.

### 3. Training (after code release)

```bash
python train.py
```

Protocol: Adam (lr 1e-4), batch size 2, 2500 epochs, full-resolution training (no crops or
resizing), seed 42, loss = 8-term PRIN loss with weights
λ_rec 1.0 / λ_ssim 0.15 / λ_edge 0.05 / λ_wav 0.08 / λ_base 0.20 / λ_illum 0.08 / λ_col 0.02 / λ_oe 0.01.
The dataset is selected by the `dataset_name` variable in the config block
(`lolv1` / `lolv2real` / `lolv2syn`). `train.py` is already in the repo; it becomes runnable
once `models/` and `losses/` are published upon acceptance.

## Main results

Raw (no GT-mean correction) averages over the official test splits, produced by the exact
pipeline above:

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Checkpoint |
|---|---|---|---|---|
| LOL-v1 | 23.390 | 0.853 | 0.119 | `prin_lolv1.pth` |
| LOL-v2-Real | 22.428 | 0.852 | 0.181 | `prin_lolv2real.pth` |
| LOL-v2-Syn | 24.596 | 0.931 | 0.056 | `prin_lolv2syn.pth` |

Per-image CSVs: `experiments/main_results/metrics_prin_*.csv`.
Training-loss logs: `experiments/main_results/train_loss_*.csv`.

---

## Experiments guide

Every experiment in the paper (and the revised manuscript) maps to one folder under
`experiments/`. Each entry below lists: purpose → how to reproduce → shipped artifacts.

### A. Main benchmark results

- **Purpose:** quantitative comparison on LOL-v1 / LOL-v2-Real / LOL-v2-Syn.
- **Reproduce:** `test.py` (three checkpoints) + `run_evaluate.py` per dataset (see Quick start).
- **Artifacts:** `experiments/main_results/` CSVs; enhanced/reflectance/illumination images on
  Google Drive — [LOL-v1](https://drive.google.com/drive/folders/1vEGBHZJ2MkZ2ofszVCdKgHNl0D27kSU_?usp=sharing),
  [LOL-v2-Real](https://drive.google.com/drive/folders/19Hbq5HhgRRdq4e9JKhsaOq0jt9pUwBN5?usp=sharing),
  [LOL-v2-Syn](https://drive.google.com/drive/folders/15ed7aXu4QBQTI_mtdjK8NsyB0WCeW5zh?usp=sharing).

### B. Variant study (V1a / V1b / V2 / V3)

- **Purpose:** architecture evolution study; **V2 is the final PRIN** (24.41 M params).
- **Results:** enhanced images for all variants are available at
  [Google Drive](https://drive.google.com/drive/folders/1q95SVIESrHaHuPKwaE5JdHQMx9IlyMyk?usp=drive_link)
  (`v1a_lolv1`, `v1b_lolv1`, `v1b_lolv2real`, `v3_lolv1`, `v3_lolv2real`, `v3_lolv2syn`;
  V2 = the main-results folders). Scoring any of these folders with `run_evaluate.py` reproduces
  the table numbers.
- **Artifacts:** `experiments/variant_study/metrics_v{1a,1b,2,3}_*.csv` — per-image metrics
  for each variant (v2 = final PRIN, identical to `main_results`).

### C. Module ablation (WDGRR / WSAIR / DCE)

- **Purpose:** contribution of each proposed module, all trained on LOL-v2-Real with the
  identical protocol: Baseline, +WDGRR, +WSAIR, +DCE, +WDGRR+WSAIR, full model.
- **Reproduce:** set `VARIANT` in `experiments/module_ablation/test_module_ablation.py` to one of
  `dce_only | wdgrr_only | wsair_only | wdgrr_wsair`, run it, then score with `run_evaluate.py`
  (GT: `data/lolv2real/test/high`). The *full* row is the main `prin_lolv2real.pth` checkpoint.
- **Artifacts:** `experiments/module_ablation/metrics_*.csv`; enhanced images on
  [Google Drive](https://drive.google.com/drive/folders/1rLJRuwJNzJE5YYYoszZVhDXGQQKVEiJx?usp=drive_link).
  The Baseline-row weights are not archived; its numbers are in the paper table.

### D. Wavelet-scale-factor ablation (s0–s6)

- **Purpose:** sensitivity/role analysis of the three wavelet refinement scales — reflectance
  high-frequency residual (0.10), illumination high-frequency residual (0.05), Sobel edge
  injection (0.10). Variants: s0 default (0.10/0.05/0.10), s1 symmetric-high (0.10/0.10),
  s2 inverted (0.05/0.10), s3 half (0.05/0.025), s4 double (0.20/0.10), s5 Sobel-low (0.05),
  s6 Sobel-high (0.20). Trained/evaluated on LOL-v1.
- **Reproduce:** `python experiments/scale_ablation/test_scale_ablation.py` — runs all seven
  checkpoints and writes per-variant CSVs plus a delta-vs-s0 summary automatically.
- **Artifacts:** `experiments/scale_ablation/metrics_params_s*.csv`,
  `summary_params_ablation.{csv,md}`; enhanced images on
  [Google Drive](https://drive.google.com/drive/folders/1TxEImsguKTmx9WOOufIrGbsxGcA55EuR?usp=drive_link).

### E. Loss-group ablation (G1–G4)

- **Purpose:** leave-one-group-out ablation of the 8-term loss —
  G1 structure/edge (L_ssim + L_edge), G2 frequency (L_wav), G3 decomposition
  (L_base + L_illum), G4 color/exposure (L_col + L_oe); L_rec always on.
- **Checkpoints:** `full.pth`, `wo_g1.pth` … `wo_g4.pth` via the
  [Google Drive folder](https://drive.google.com/drive/folders/12mqOD5WIB255teTqoLMvkSfB7R0rh9bz?usp=sharing)
  (download into `checkpoints/loss_ablation/`); all five are plain `PRIN` state dicts, so
  `test.py` + `evaluate.py` reproduce the numbers directly.
- **Data:** `experiments/loss_ablation/` — `metrics_summary.csv`, per-image
  `metrics_*.csv`, and per-epoch `train_log_*.csv` (the removed group's loss columns are
  identically zero in each `w/o` log). See `experiments/loss_ablation/README.md` for the
  grouping and the results table.
- **Result images:** enhanced outputs of all five variants on the
  [Google Drive results folder](https://drive.google.com/drive/folders/1A3dzN29HQkkCqFRbCfQm7RlsOREbKMFp?usp=sharing).

### F. No-reference perceptual quality on the LOL test sets

- **Purpose:** MUSIQ / LIQE / Q-Align (pyiqa: `musiq`, `liqe_mix`, `qalign`) on the enhanced
  LOL test outputs at saved resolution.
- **Reproduce:** edit the config block of `experiments/nonref_lol/evaluate_nr_iqa.py`
  (INPUT_DIR = an enhanced-results folder), run it; `evaluate_nr_iqa_subfolders.py` scores a
  folder of method subfolders in one pass.
- **Artifacts:** `experiments/nonref_lol/nr_iqa_prin_*.csv` (ours) and `nr_iqa_gt_*.csv`
  (ground-truth reference scores).

### G. Zero-shot generalization (DICM / LIME / MEF / NPE / VV)

- **Purpose:** apply the LOL-trained checkpoints unchanged to five unpaired benchmarks;
  no-reference scoring with NIQE / MUSIQ / LIQE (+ Q-Align separately).
- **Reproduce:**
  ```bash
  python experiments/nonref_zeroshot/test_nonref.py                  # inference + NIQE/MUSIQ/LIQE
  python experiments/nonref_zeroshot/run_qalign.py                   # Q-Align pass
  python experiments/nonref_zeroshot/run_nonref_gentle.py            # optional: throttled driver
  ```
  Useful flags for `test_nonref.py`: `--datasets NPE --ckpts prin_lolv1` (smoke test),
  `--skip-infer` (re-score only), `--sources <dir ...>` (score any result folders with the
  same protocol).
- **Artifacts:** `experiments/nonref_zeroshot/{metrics_summary,qalign_summary}.{csv,md}`;
  our enhanced outputs on
  [Google Drive](https://drive.google.com/drive/folders/1ccV5pG6mKvhJa6qyCW0ficBbkuiZlujD?usp=drive_link)
  (`<dataset>/prin_*` subfolders).

### H. GT-mean correction study

- **Purpose:** quantify how much of the PSNR gap stems from global brightness (exposure)
  mismatch: rescale each prediction so its mean matches the paired GT mean, then re-score.
- **Reproduce:** edit the config block of `experiments/gtmean_study/gt_mean.py` per dataset,
  run it, then score `results/gtmean/<dataset>` with `run_evaluate.py`.
- **Artifacts:** `experiments/gtmean_study/metrics_*_withgtmean.csv` (all three datasets).

### I. Gaussian-blur PSNR analysis

- **Purpose:** demonstrate that blending Gaussian-blurred variants of the enhanced output
  *raises* PSNR while SSIM/LPIPS degrade — motivating the perceptual-structural evaluation
  focus (radius/alpha grid includes the paper's r ∈ {0.5, 0.8, 1.0}, α = 0.2 settings).
- **Reproduce:** `python experiments/blur_study/blur_test.py` (edit config block for other
  datasets).
- **Artifacts:** `experiments/blur_study/blur_summary_by_variant.csv`,
  `blur_per_image_metrics.csv`.

### J. Efficiency analysis

- **Purpose:** Params / FLOPs (256×256 and 600×400) / single-image latency / FPS.
- **Reproduce (PRIN):** `python experiments/efficiency/measure.py --model PRIN`
  (works standalone once `models/` is released).
- **Reproduce (baselines):** clone each official baseline repo into
  `experiments/efficiency/repos/` first — third-party code is not redistributed here; the
  builders in `measure.py` document the exact configs used.
- **Artifacts:** `experiments/efficiency/results/efficiency.csv` (measured table).

---

## Code release policy

To protect the work prior to publication, the network definition (`models/`) and loss
implementation (`losses/`) are withheld until the paper is accepted; all evaluation,
inference-harness, and training scripts, every pretrained checkpoint, and every enhanced
output image are already public. Upon acceptance the two packages will be pushed and the
scripts in this repository will run end-to-end without modification.

## Citation

```bibtex
% Citation will be added upon publication.
```

## Acknowledgements

The curve-based post-composition head follows the curve-estimation formulation of Zero-DCE.
Evaluation relies on scikit-image, lpips, and the PyIQA toolbox.
