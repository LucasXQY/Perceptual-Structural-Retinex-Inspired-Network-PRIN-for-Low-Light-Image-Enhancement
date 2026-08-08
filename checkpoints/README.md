# Pretrained checkpoints

All checkpoints are hosted on Google Drive:

<!-- TODO: replace with your Google Drive share link -->
**[Download the full `checkpoints/` folder (≈ 1.3 GB) — Google Drive](https://TODO-GDRIVE-CHECKPOINTS-LINK)**

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
└── loss_ablation/                    # (to be added)
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
