# models/ — network source code

**The network source code will be published here upon paper acceptance.**

To protect the work while the paper is under review, the model definitions are withheld;
everything else needed to verify the paper is already public — pretrained checkpoints,
enhanced result images for every experiment, and the exact evaluation scripts
(`evaluate.py` / `run_evaluate.py`).

Files that will appear in this package:

```
models/
├── prin.py                        # class PRIN — the full model (paper variant V2, 24.41 M params)
├── wavelet_refiners.py            # class WDGRR (Sec 3.4), class WSAIR (Sec 3.5)
├── haar_wavelet.py                # HaarDWT / HaarIDWT (Sec 3.3)
├── dce_curve_enhancer.py          # DCECurveEnhancer — curve-based post-composition head (Sec 3.6)
├── coordinate_attention.py        # coordinate attention block
├── triplet_attention.py           # triplet attention block
├── gated_channel_attention.py     # gated channel attention block
├── sobel_edge.py                  # Sobel edge map for explicit edge injection
├── scale_ablation/                # s0–s6 scale-factor variant configs (subclass PRIN)
└── ablation/                      # module-ablation variants (PRIN_DCEOnly, PRIN_WDGRROnly, …)
```
