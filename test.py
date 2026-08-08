"""Inference script for PRIN (produces the enhanced images scored in the paper).

Runs the final PRIN model (paper Table 5 variant V2) on a low-light test set
and saves enhanced/, reflectance/, illumination/ outputs. These outputs feed
evaluate.py (the canonical full-reference protocol behind paper Tables 2-4).
Inputs are reflect-padded to a multiple of 32 for the model's downsampling /
DWT stages, then cropped back to the original size.

NOTE: the `models/` package is released upon paper acceptance; until then
this script's model import will not resolve. Checkpoints (Google Drive) go
under checkpoints/ (see README).

Usage:
    python test.py
(Edit the defaults in the __main__ block for other datasets, e.g.
 model_path="checkpoints/prin_lolv2real.pth", low_dir="data/lolv2real/test/low",
 output_dir="results/lolv2real".)
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from datasets.lol_dataset import LOLDataset
from models.prin import PRIN


def pad_to_multiple(x, multiple=32, mode="reflect"):
    """
    Pad H/W up to a multiple of `multiple`; return the padded tensor and the
    original (H, W).
    """
    B, C, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (H, W)


def crop_back(x, hw):
    H, W = hw
    return x[:, :, :H, :W]


def save_image(tensor, path):
    tensor = torch.clamp(tensor, 0, 1)
    vutils.save_image(tensor, path)


@torch.no_grad()
def test(
    model_path="checkpoints/prin_lolv1.pth",
    low_dir="data/lolv1/test/low",
    output_dir="results/lolv1",
    multiple=32,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reflectance"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "illumination"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LOLDataset(low_dir=low_dir, mode="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = PRIN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print("[INFO] Generating enhanced images...")

    for idx, (low, filename) in enumerate(dataloader):
        low = low.to(device)
        name = os.path.splitext(filename[0])[0]

        # Record the original size and pad to a multiple of 32
        # (keeps PixelUnshuffle / DWT stages valid).
        low_pad, hw0 = pad_to_multiple(low, multiple=multiple, mode="reflect")

        # PRIN returns (r_out, l_out, enhanced)
        r_out, l_out, enhanced = model(low_pad)

        # Crop back so output size == original input size
        r_out = crop_back(r_out, hw0)
        l_out = crop_back(l_out, hw0)
        enhanced = crop_back(enhanced, hw0)

        # If illumination is single-channel, repeat to 3 channels for viewing
        if l_out.shape[1] == 1:
            l_vis = l_out.repeat(1, 3, 1, 1)
        else:
            l_vis = l_out

        # Save enhanced / reflectance / illumination
        save_image(enhanced, os.path.join(output_dir, "enhanced", f"{name}.png"))
        save_image(r_out,     os.path.join(output_dir, "reflectance", f"{name}.png"))
        save_image(l_vis,     os.path.join(output_dir, "illumination", f"{name}.png"))

        if (idx + 1) % 10 == 0:
            print(f"[INFO] Done {idx+1}/{len(dataset)}")

    print(f"[INFO] Results saved to: {output_dir}")


if __name__ == "__main__":
    test(
        model_path="checkpoints/prin_lolv1.pth",
        low_dir="data/lolv1/test/low",
        output_dir="results/lolv1",
        multiple=32,
    )
