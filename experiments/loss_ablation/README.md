# Loss-group ablation (G1–G4)

Leave-one-group-out ablation of the 8-term training loss, trained and evaluated on LOL-v1
with the identical protocol as the main model. L_rec (the primary supervision) is always kept.

| Group | Members (weight) | Function |
|---|---|---|
| G1 — structure/edge | L_ssim (0.15) + L_edge (0.05) | structural & edge fidelity of the output |
| G2 — frequency | L_wav (0.08) | Haar sub-band supervision |
| G3 — decomposition | L_base (0.20) + L_illum (0.08) | recomposition fidelity + illumination smoothness |
| G4 — color/exposure | L_col (0.02) + L_oe (0.01) | color consistency + over-exposure suppression |

Five runs: `full`, `w/o G1`, `w/o G2`, `w/o G3`, `w/o G4`. Each variant is the standard
PRIN architecture trained for up to 1,000 epochs on LOL-v1 with identical optimization and
data-processing settings; the released checkpoint of each run is the one with the lowest
validation loss. The 1,000-epoch budget provides a preliminary assessment of each group's
contribution rather than fully converged asymptotic performance (see the paper for the
discussion).

## Results (LOL-v1 test, no GT-mean correction)

| Variant | PSNR ↑ | SSIM ↑ | MAE ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Full objective | 23.36 | 0.8439 | 0.0712 | 0.1293 |
| w/o G1 (structure–edge) | 22.68 | 0.8065 | 0.0673 | 0.1865 |
| w/o G2 (frequency-domain) | 23.37 | 0.8507 | 0.0684 | 0.1326 |
| w/o G3 (decomposition regularization) | 23.23 | 0.8446 | 0.0707 | 0.1315 |
| w/o G4 (color–exposure) | 23.26 | 0.8519 | 0.0688 | 0.1259 |

Removing G3 leaves the final-image metrics roughly unchanged but corrupts the intermediate
Retinex decomposition (texture leaks into the illumination map while the reflectance is
washed out) — see the qualitative comparison in the paper.

## Files in this directory

- `metrics_summary.csv` — the table above (plus NIQE), one row per variant.
- `metrics_full.csv`, `metrics_wo_g{1,2,3,4}.csv` — per-image PSNR / SSIM / MAE /
  LPIPS(AlexNet) / NIQE for the 15 LOL-v1 test images.
- `train_log_full.csv`, `train_log_wo_g{1,2,3,4}.csv` — per-epoch training logs with the
  individual loss terms; in each `w/o` log the columns of the removed group are identically
  zero, documenting the ablation configuration.

## Checkpoints & reproduction

**Checkpoints:** released via the
[Google Drive folder](https://drive.google.com/drive/folders/12mqOD5WIB255teTqoLMvkSfB7R0rh9bz?usp=sharing)
— download `full.pth`, `wo_g1.pth`, `wo_g2.pth`, `wo_g3.pth`, `wo_g4.pth` into
`checkpoints/loss_ablation/`.

All five checkpoints are plain `state_dict`s of the standard `PRIN` class (they load with
`strict=True`), so inference needs no dedicated harness — run the root `test.py` with
`model_path="checkpoints/loss_ablation/<variant>.pth"` and
`low_dir="data/lolv1/test/low"`, then score the outputs with the root `evaluate.py`.

**Result images:** the enhanced outputs of all five variants are shared via the
[Google Drive results folder](https://drive.google.com/drive/folders/1A3dzN29HQkkCqFRbCfQm7RlsOREbKMFp?usp=sharing).

The loss-variant implementations (`losses/loss_ablation/`) are released together with the
`losses/` package upon paper acceptance; each variant plugs into `train.py` by swapping the
loss import, with every other training hyperparameter unchanged.
