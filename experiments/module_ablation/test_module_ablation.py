"""Module ablation inference (paper Table 8).

Runs one of the four partial-model variants on the LOL-v2-real test set:
  dce_only    - Baseline + DCE curve head only
  wdgrr_only  - Baseline + WDGRR reflectance refiner only
  wsair_only  - Baseline + WSAIR illumination refiner only
  wdgrr_wsair - Baseline + WDGRR + WSAIR (no DCE head)
Set VARIANT below, drop the matching checkpoint into
checkpoints/module_ablation/<variant>.pth, then score the outputs with
run_evaluate.py. The full-model row of Table 8 is produced with test.py and
checkpoints/prin_lolv2real.pth; the Baseline row's weights are not archived.

Usage (from the repo root):
    python experiments/module_ablation/test_module_ablation.py
"""

import importlib
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from datasets.lolv2real_dataset import LOLv2RealDataset

# ====== Config ======
VARIANT = "dce_only"  # one of: "dce_only", "wdgrr_only", "wsair_only", "wdgrr_wsair"

# variant -> (module path in models.ablation, class name, checkpoint path)
VARIANTS = {
    "dce_only":    ("models.ablation.dce_only",    "PRIN_DCEOnly",
                    "checkpoints/module_ablation/dce_only.pth"),
    "wdgrr_only":  ("models.ablation.wdgrr_only",  "PRIN_WDGRROnly",
                    "checkpoints/module_ablation/wdgrr_only.pth"),
    "wsair_only":  ("models.ablation.wsair_only",  "PRIN_WSAIROnly",
                    "checkpoints/module_ablation/wsair_only.pth"),
    "wdgrr_wsair": ("models.ablation.wdgrr_wsair", "PRIN_WDGRR_WSAIR",
                    "checkpoints/module_ablation/wdgrr_wsair.pth"),
}


def get_model_class(variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {sorted(VARIANTS)}")
    module_path, class_name, _ = VARIANTS[variant]
    return getattr(importlib.import_module(module_path), class_name)


def save_image(tensor, path):
    tensor = torch.clamp(tensor, 0, 1)
    vutils.save_image(tensor, path)


def load_model_weights(model, model_path, device, strict=True):
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=strict)


@torch.no_grad()
def test(
        variant=VARIANT,
        model_path=None,
        low_dir=os.path.join(str(_ROOT), "data", "lolv2real", "test", "low"),
        output_dir=None,
    batch_size=1,
    num_workers=0,
    illum_one_channel=False,
    save_reflectance=True,
    save_illumination=True,
    strict=True,
):
    if model_path is None:
        model_path = os.path.join(str(_ROOT), *VARIANTS[variant][2].split("/"))
    if output_dir is None:
        output_dir = os.path.join(str(_ROOT), "results", "module_ablation", variant)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)

    if save_reflectance:
        os.makedirs(os.path.join(output_dir, "reflectance"), exist_ok=True)
    if save_illumination:
        os.makedirs(os.path.join(output_dir, "illumination"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LOLv2RealDataset(low_dir=low_dir, mode="test")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_class = get_model_class(variant)
    model = model_class(illum_one_channel=illum_one_channel).to(device)
    load_model_weights(model, model_path, device, strict=strict)
    model.eval()

    print(f"[INFO] Variant: {variant} ({model_class.__name__})")
    print("[INFO] Generating enhanced images...")

    for idx, (low, filename) in enumerate(dataloader):
        low = low.to(device, non_blocking=True)

        r_out, l_out, enhanced = model(low)

        for b in range(low.size(0)):
            name = os.path.splitext(filename[b])[0]

            save_image(
                enhanced[b:b+1],
                os.path.join(output_dir, "enhanced", f"{name}.png")
            )

            if save_reflectance:
                save_image(
                    r_out[b:b+1],
                    os.path.join(output_dir, "reflectance", f"{name}.png")
                )

            if save_illumination:
                l_vis = l_out[b:b+1]
                if l_vis.shape[1] == 1:
                    l_vis = l_vis.repeat(1, 3, 1, 1)
                save_image(
                    l_vis,
                    os.path.join(output_dir, "illumination", f"{name}.png")
                )

        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataloader):
            print(f"[INFO] Done {min((idx + 1) * batch_size, len(dataset))}/{len(dataset)}")

    print(f"[INFO] Results saved to: {output_dir}")


if __name__ == "__main__":
    test()
