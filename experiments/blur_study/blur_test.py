"""Gaussian-blur PSNR analysis (paper Fig. 2).

Demonstrates that PSNR can be gamed by smoothing: blending Gaussian-blurred
variants of the network outputs into the originals raises PSNR while SSIM
and LPIPS degrade. For each (radius, alpha) setting in PARAM_GRID the script
generates blended images, evaluates PSNR/SSIM/LPIPS against GT, and writes
per-variant and per-image CSV summaries.

Usage (from the repo root):
    python experiments/blur_study/blur_test.py
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
from torchvision import transforms
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


warnings.filterwarnings("ignore")


# =========================================================
# 1. Config paths (relative to the repo root)
# =========================================================

# Folder of original model outputs to perturb.
INPUT_DIR = "results/lolv1/enhanced"

# Matching GT folder.
GT_DIR = "data/lolv1/test/high"

# Where all blur variants and CSVs are saved.
OUTPUT_ROOT = "results/blur_study_lolv1"

# Save outputs as PNG (recommended: avoids JPEG compression affecting metrics).
SAVE_AS_PNG = True

# Set to an integer (e.g. 10) to test only the first few images;
# keep None to run the full test set.
MAX_IMAGES = None


# =========================================================
# 2. Blur parameter grid
# =========================================================
# radius: Gaussian blur strength
# alpha : blend ratio between the blurred image and the original
#
# output = (1 - alpha) * original + alpha * blurred
#
# alpha = 0.0 is the original-image baseline
# alpha = 1.0 uses the fully blurred image

PARAM_GRID = [
    (0.0, 0.0),   # original baseline

    (0.5, 0.2),
    (0.8, 0.2),
    (1.0, 0.2),
    (1.2, 0.2),

    (0.8, 0.35),
    (1.0, 0.35),
    (1.2, 0.35),
    (1.5, 0.35),

    (1.0, 0.5),
    (1.2, 0.5),
    (1.5, 0.5),
    (2.0, 0.5),

    (1.5, 0.65),
    (2.0, 0.65),

    (1.0, 1.0),  # pure blur
    (1.5, 1.0),
    (2.0, 1.0),
]


VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# =========================================================
# 3. File-matching utilities
# =========================================================

def is_image_file(path):
    return path.lower().endswith(VALID_EXTS)


def extract_id(fname: str):
    """
    Extract the last run of digits in the filename.
    Suitable for LOL-v1 / LOL-v2-Real:
    low00123.png -> 00123
    normal00123.png -> 00123
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    nums = re.findall(r"\d+", base)
    return nums[-1] if nums else None


def build_gt_index(gt_dir):
    gt_files = sorted([
        p for p in glob.glob(os.path.join(gt_dir, "*"))
        if os.path.isfile(p) and is_image_file(p)
    ])

    gt_by_filename = {}
    gt_by_stem = {}
    gt_by_id = {}

    for p in gt_files:
        bn = os.path.basename(p)
        stem = os.path.splitext(bn)[0].lower()

        gt_by_filename[bn.lower()] = p

        if stem not in gt_by_stem:
            gt_by_stem[stem] = p

        img_id = extract_id(bn)
        if img_id is not None:
            gt_by_id.setdefault(img_id, []).append(p)

    return gt_files, gt_by_filename, gt_by_stem, gt_by_id


def pick_best_gt(paths):
    """
    When several GT files share the same id, prefer names containing "normal".
    """
    paths = sorted(
        paths,
        key=lambda p: (
            "normal" not in os.path.basename(p).lower(),
            os.path.basename(p).lower()
        )
    )
    return paths[0]


def find_gt_for_image(img_path, gt_by_filename, gt_by_stem, gt_by_id):
    fname = os.path.basename(img_path)
    fname_lower = fname.lower()
    stem = os.path.splitext(fname)[0].lower()

    # 1. Exact filename match
    if fname_lower in gt_by_filename:
        return gt_by_filename[fname_lower]

    # 2. Stem match (suits LOL-v2-Syn)
    if stem in gt_by_stem:
        return gt_by_stem[stem]

    # 3. Numeric-id match (suits LOL-v1 / LOL-v2-Real)
    img_id = extract_id(fname)
    if img_id is not None and img_id in gt_by_id:
        return pick_best_gt(gt_by_id[img_id])

    return None


def param_to_name(radius, alpha):
    """
    Folder names must avoid dots, so "." is replaced with "p".
    """
    r = str(radius).replace(".", "p")
    a = str(alpha).replace(".", "p")

    if radius == 0.0 and alpha == 0.0:
        return "original"

    return f"blur_r{r}_a{a}"


# =========================================================
# 4. Blur generation
# =========================================================

def apply_blur_variant(img: Image.Image, radius: float, alpha: float):
    img = img.convert("RGB")

    if radius == 0.0 or alpha == 0.0:
        return img.copy()

    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    out = Image.blend(img, blurred, alpha=alpha)
    return out


def generate_blur_images(input_dir, output_root, param_grid):
    input_files = sorted([
        p for p in glob.glob(os.path.join(input_dir, "*"))
        if os.path.isfile(p) and is_image_file(p)
    ])

    if MAX_IMAGES is not None:
        input_files = input_files[:MAX_IMAGES]

    if len(input_files) == 0:
        raise RuntimeError(f"No images found in INPUT_DIR: {input_dir}")

    print(f"[INFO] Found {len(input_files)} input images.")

    variant_dirs = []

    for radius, alpha in param_grid:
        variant_name = param_to_name(radius, alpha)
        variant_dir = os.path.join(output_root, variant_name)
        os.makedirs(variant_dir, exist_ok=True)
        variant_dirs.append((variant_name, variant_dir, radius, alpha))

        print(f"[INFO] Generating variant: {variant_name}")

        for i, img_path in enumerate(input_files):
            img = Image.open(img_path).convert("RGB")
            out = apply_blur_variant(img, radius, alpha)

            stem = os.path.splitext(os.path.basename(img_path))[0]

            if SAVE_AS_PNG:
                save_name = f"{stem}.png"
            else:
                save_name = os.path.basename(img_path)

            save_path = os.path.join(variant_dir, save_name)
            out.save(save_path)

        print(f"[INFO] Saved to: {variant_dir}")

    return variant_dirs


# =========================================================
# 5. Metric computation
# =========================================================

def calculate_ssim(gt_np, img_np):
    try:
        return structural_similarity(
            gt_np,
            img_np,
            channel_axis=-1,
            data_range=1.0
        )
    except TypeError:
        return structural_similarity(
            gt_np,
            img_np,
            multichannel=True,
            data_range=1.0
        )


def evaluate_variant(
    variant_name,
    variant_dir,
    radius,
    alpha,
    gt_by_filename,
    gt_by_stem,
    gt_by_id,
    lpips_fn,
    device
):
    to_tensor = transforms.ToTensor()

    image_files = sorted([
        p for p in glob.glob(os.path.join(variant_dir, "*"))
        if os.path.isfile(p) and is_image_file(p)
    ])

    if len(image_files) == 0:
        print(f"[WARN] No images found in {variant_dir}")
        return None, []

    per_image_records = []

    for idx, enh_path in enumerate(image_files):
        gt_path = find_gt_for_image(
            enh_path,
            gt_by_filename,
            gt_by_stem,
            gt_by_id
        )

        if gt_path is None:
            print(f"[WARN] Missing GT for: {os.path.basename(enh_path)}")
            continue

        enh_img = Image.open(enh_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if enh_img.size != gt_img.size:
            enh_img = enh_img.resize(gt_img.size, Image.BICUBIC)

        enh_np = np.array(enh_img).astype(np.float32) / 255.0
        gt_np = np.array(gt_img).astype(np.float32) / 255.0

        psnr_val = peak_signal_noise_ratio(gt_np, enh_np, data_range=1.0)
        ssim_val = calculate_ssim(gt_np, enh_np)

        enh_tensor = to_tensor(enh_img).unsqueeze(0).to(device)
        gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device)

        enh_tensor_norm = (enh_tensor - 0.5) * 2.0
        gt_tensor_norm = (gt_tensor - 0.5) * 2.0

        with torch.no_grad():
            lpips_val = lpips_fn(enh_tensor_norm, gt_tensor_norm).item()

        per_image_records.append({
            "variant": variant_name,
            "radius": radius,
            "alpha": alpha,
            "image": os.path.basename(enh_path),
            "GT": os.path.basename(gt_path),
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "LPIPS": lpips_val,
        })

    if len(per_image_records) == 0:
        return None, []

    df = pd.DataFrame(per_image_records)

    summary = {
        "variant": variant_name,
        "radius": radius,
        "alpha": alpha,
        "num_images": len(df),
        "PSNR": df["PSNR"].mean(),
        "SSIM": df["SSIM"].mean(),
        "LPIPS": df["LPIPS"].mean(),
    }

    return summary, per_image_records


# =========================================================
# 6. Main
# =========================================================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("=" * 60)
    print("[STEP 1] Building GT index...")
    gt_files, gt_by_filename, gt_by_stem, gt_by_id = build_gt_index(GT_DIR)

    if len(gt_files) == 0:
        raise RuntimeError(f"No GT images found in GT_DIR: {GT_DIR}")

    print(f"[INFO] Found {len(gt_files)} GT images.")
    print(f"[INFO] GT stem keys: {len(gt_by_stem)}")
    print(f"[INFO] GT numeric ID keys: {len(gt_by_id)}")

    print("=" * 60)
    print("[STEP 2] Generating blur variants...")
    variant_dirs = generate_blur_images(INPUT_DIR, OUTPUT_ROOT, PARAM_GRID)

    print("=" * 60)
    print("[STEP 3] Loading LPIPS model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    print("=" * 60)
    print("[STEP 4] Evaluating each blur variant...")

    all_summary = []
    all_per_image = []

    for variant_name, variant_dir, radius, alpha in variant_dirs:
        print(f"\n[INFO] Evaluating: {variant_name}")

        summary, per_image_records = evaluate_variant(
            variant_name=variant_name,
            variant_dir=variant_dir,
            radius=radius,
            alpha=alpha,
            gt_by_filename=gt_by_filename,
            gt_by_stem=gt_by_stem,
            gt_by_id=gt_by_id,
            lpips_fn=lpips_fn,
            device=device
        )

        if summary is None:
            print(f"[WARN] No valid metrics for {variant_name}")
            continue

        all_summary.append(summary)
        all_per_image.extend(per_image_records)

        print(
            f"[RESULT] {variant_name} | "
            f"PSNR: {summary['PSNR']:.4f} | "
            f"SSIM: {summary['SSIM']:.4f} | "
            f"LPIPS: {summary['LPIPS']:.4f}"
        )

    if len(all_summary) == 0:
        print("[ERROR] No valid evaluation results.")
        return

    summary_df = pd.DataFrame(all_summary)
    per_image_df = pd.DataFrame(all_per_image)

    # Sort by PSNR descending to surface the highest-PSNR blur setting.
    summary_df = summary_df.sort_values(
        by="PSNR",
        ascending=False
    ).reset_index(drop=True)

    summary_csv = os.path.join(OUTPUT_ROOT, "blur_summary_by_variant.csv")
    per_image_csv = os.path.join(OUTPUT_ROOT, "blur_per_image_metrics.csv")

    summary_df.to_csv(summary_csv, index=False)
    per_image_df.to_csv(per_image_csv, index=False)

    print("\n" + "=" * 60)
    print("[DONE] Blur study finished.")
    print(f"[SAVE] Summary CSV: {summary_csv}")
    print(f"[SAVE] Per-image CSV: {per_image_csv}")

    print("\n[TOP 5 by PSNR]")
    print(summary_df.head(5).to_string(index=False))

    best = summary_df.iloc[0]
    print("\n[BEST PSNR SETTING]")
    print(
        f"variant={best['variant']} | "
        f"radius={best['radius']} | "
        f"alpha={best['alpha']} | "
        f"PSNR={best['PSNR']:.4f} | "
        f"SSIM={best['SSIM']:.4f} | "
        f"LPIPS={best['LPIPS']:.4f}"
    )


if __name__ == "__main__":
    main()
