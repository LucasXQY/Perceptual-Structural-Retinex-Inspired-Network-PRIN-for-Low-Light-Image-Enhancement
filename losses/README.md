# losses/ — loss source code

**The loss implementation will be published here upon paper acceptance** (together with
`models/`).

The training objective is fully specified in the paper (Sec. 3.7): an 8-term weighted sum

| Term | Meaning | Weight |
|---|---|---|
| L_rec  | Charbonnier reconstruction on the final output | 1.00 |
| L_ssim | 1 − SSIM | 0.15 |
| L_edge | L1 between Sobel gradient magnitudes | 0.05 |
| L_wav  | Haar wavelet-domain fidelity (LL ×0.5, LH/HL/HH ×1.0) | 0.08 |
| L_base | Charbonnier on the pre-curve recomposition R̂ ⊙ L̂ | 0.20 |
| L_illum| Edge-aware illumination smoothness (α = 10) | 0.08 |
| L_col  | Hue consistency + 0.5 × mean-RGB alignment | 0.02 |
| L_oe   | Over-exposure suppression (τ = 0.95) | 0.01 |

Files that will appear in this package:

```
losses/
├── prin_loss.py          # prin_loss(low, gt, r_out, l_out, enhanced) → (total, logs)
├── rgb_to_hsv.py         # differentiable RGB→HSV used by the color term
└── loss_ablation/        # leave-one-group-out (G1–G4) loss variants for the ablation study
```
