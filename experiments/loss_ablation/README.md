# Loss-group ablation (G1–G4)

Leave-one-group-out ablation of the 8-term training loss, trained and evaluated on LOL-v1
with the identical protocol as the main model. L_rec (the primary supervision) is always kept.

| Group | Members (weight) | Function |
|---|---|---|
| G1 — structure/edge | L_ssim (0.15) + L_edge (0.05) | structural & edge fidelity of the output |
| G2 — frequency | L_wav (0.08) | Haar sub-band supervision |
| G3 — decomposition | L_base (0.20) + L_illum (0.08) | recomposition fidelity + illumination smoothness |
| G4 — color/exposure | L_col (0.02) + L_oe (0.01) | color consistency + over-exposure suppression |

Five runs: `full`, `w/o G1`, `w/o G2`, `w/o G3`, `w/o G4`.

**Status:** the checkpoints and per-image metric CSVs for these five runs will be added to the
Google Drive checkpoints folder (`checkpoints/loss_ablation/`) and to this directory.

<!-- TODO: add Google Drive link for loss-ablation checkpoints when available -->

The loss-variant implementations (`losses/loss_ablation/`) are released together with the
`losses/` package upon paper acceptance; each variant plugs into `train.py` by swapping the
loss import, with every other training hyperparameter unchanged.
