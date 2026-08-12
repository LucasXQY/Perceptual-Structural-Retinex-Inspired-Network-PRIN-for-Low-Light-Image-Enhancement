# Pretrained checkpoints

All checkpoints are hosted on Google Drive (≈ 1.8 GB total):

| Item | Link |
|---|---|
| `prin_lolv1.pth` | [Google Drive](https://drive.google.com/file/d/1aDnSX0CCXSPgGwsV03JI9gGZpehvatPd/view?usp=sharing) |
| `prin_lolv2real.pth` | [Google Drive](https://drive.google.com/file/d/1WX2giiRfjcgHqXcwv2kKHJC8nUdupsgu/view?usp=sharing) |
| `prin_lolv2syn.pth` | [Google Drive](https://drive.google.com/file/d/1P0JGLlUYk_kkmWgrfjoph2FjOC_tgmOe/view?usp=sharing) |
| `module_ablation/` folder | [Google Drive](https://drive.google.com/drive/folders/13rK3wy9BMPeARP5M0uIR_dQi-Ll1uFSO?usp=sharing) |
| `scale_ablation/` folder | [Google Drive](https://drive.google.com/drive/folders/19ijwjQCOiwD4r_5Q_p_GdOUYFt2aIAi6?usp=sharing) |
| `loss_ablation/` folder | [Google Drive](https://drive.google.com/drive/folders/12mqOD5WIB255teTqoLMvkSfB7R0rh9bz?usp=sharing) |

After downloading, place the contents directly in this folder so the layout is:

```
checkpoints/
├── prin_lolv1.pth                    # PRIN trained on LOL-v1          (main results)
├── prin_lolv2real.pth                # PRIN trained on LOL-v2-Real     (main results)
├── prin_lolv2syn.pth                 # PRIN trained on LOL-v2-Syn      (main results)
├── module_ablation/
│   ├── dce_only.pth                  # Baseline + DCE head only        → class PRIN_DCEOnly
│   ├── wdgrr_only.pth                # Baseline + WDGRR only           → class PRIN_WDGRROnly
│   ├── wsair_only.pth                # Baseline + WSAIR only           → class PRIN_WSAIROnly
│   └── wdgrr_wsair.pth               # Baseline + WDGRR + WSAIR        → class PRIN_WDGRR_WSAIR
├── scale_ablation/
│   ├── s0_default.pth   ├── s1_sym_high.pth   ├── s2_inverted.pth   ├── s3_half.pth
│   ├── s4_double.pth    ├── s5_sobel_low.pth  └── s6_sobel_high.pth  # all class PRIN (variant scales)
└── loss_ablation/                    # loss-group ablation, all class PRIN (LOL-v1)
    ├── full.pth                      # full 8-term objective
    ├── wo_g1.pth                     # w/o G1 (L_ssim + L_edge)
    ├── wo_g2.pth                     # w/o G2 (L_wav)
    ├── wo_g3.pth                     # w/o G3 (L_base + L_illum)
    └── wo_g4.pth                     # w/o G4 (L_col + L_oe)
```

Notes:

- Every file is a plain PyTorch `state_dict`; the scripts in this repository load them with
  `strict=True` into the classes listed above (class definitions ship with the `models/`
  package upon paper acceptance).
- Scale-ablation checkpoints share the `PRIN` architecture — the scale factors are constructor
  arguments, not weights, so each file must be loaded through its matching variant config.
  `experiments/scale_ablation/test_scale_ablation.py` handles this automatically.
- The module-ablation *full* row corresponds to `prin_lolv2real.pth`; the *Baseline* row
  (no WDGRR / WSAIR / DCE) is not archived.
