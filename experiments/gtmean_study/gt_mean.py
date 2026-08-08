"""GT-mean correction study (paper Sec. 4.6, Table 9).

Applies GT-mean brightness correction to existing network outputs: each
prediction is rescaled so that its mean intensity matches the mean of the
paired ground-truth image (scalar or per-RGB-channel gain). This isolates
how much of the full-reference metric gap is explained by global brightness
alone. Run once per dataset by editing the config block below, then score
the corrected outputs with run_evaluate.py.

Usage (from the repo root):
    python experiments/gtmean_study/gt_mean.py
"""

import os
import re
import glob
import csv
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image


# =========================
# Config: edit paths here (relative to the repo root)
# =========================
PRED_DIR = r"results/lolv2real/enhanced"   # directory of original network outputs
GT_DIR = r"data/lolv2real/test/high"       # matching GT directory
SAVE_DIR = r"results/gtmean/lolv2real"     # output directory for GT-mean corrected images

# GT mean method:
#   "match_scalar" : scale pred by the whole-image mean ratio (recommended, most common)
#   "match_rgb"    : scale each RGB channel by its own mean ratio
METHOD = "match_scalar"

# Save side-by-side pred | adjusted comparison images.
SAVE_COMPARE = True

# Save per-image mean / gain info to a CSV.
SAVE_METADATA_CSV = True

# Clamp the gain to avoid extreme scaling.
GAIN_MIN = 0.0
GAIN_MAX = 3.0
EPS = 1e-6

# Force-resize pred to the GT size when they differ.
RESIZE_PRED_TO_GT = True

# Supported image extensions.
VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# =========================
# Utilities
# =========================
def is_img(path: str) -> bool:
    return path.lower().endswith(VALID_EXTS)


def list_imgs(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if is_img(f)
    )


def extract_id(fname: str) -> Optional[str]:
    """
    Extract the last run of digits in the filename, so that pairs like
    low00123 <-> normal00123 can be matched.
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    nums = re.findall(r"\d+", base)
    return nums[-1] if nums else None


def build_gt_index(gt_dir: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    Build a 3-tier index:
      1) exact filename
      2) same stem
      3) same numeric id
    """
    gt_files = [p for p in glob.glob(os.path.join(gt_dir, "*")) if os.path.isfile(p) and is_img(p)]

    by_filename: Dict[str, str] = {}
    by_stem: Dict[str, str] = {}
    by_id: Dict[str, List[str]] = {}

    def ext_rank(x: str) -> int:
        ext = os.path.splitext(x)[1].lower()
        return {".png": 0, ".jpg": 1, ".jpeg": 2}.get(ext, 3)

    for p in gt_files:
        bn = os.path.basename(p)
        stem = os.path.splitext(bn)[0]
        by_filename[bn] = p

        if stem not in by_stem or ext_rank(p) < ext_rank(by_stem[stem]):
            by_stem[stem] = p

        k = extract_id(bn)
        if k is not None:
            by_id.setdefault(k, []).append(p)

    return by_filename, by_stem, by_id


def pick_best_gt(paths: List[str]) -> str:
    """
    When several GT files share the same numeric id, prefer names
    containing "normal", then lexicographic order.
    """
    return sorted(
        paths,
        key=lambda p: (("normal" not in os.path.basename(p).lower()), os.path.basename(p).lower())
    )[0]


def find_gt_for_pred(pred_path: str, gt_index) -> Optional[str]:
    by_filename, by_stem, by_id = gt_index
    filename = os.path.basename(pred_path)
    stem = os.path.splitext(filename)[0]

    gt_path = by_filename.get(filename)
    if gt_path is not None:
        return gt_path

    gt_path = by_stem.get(stem)
    if gt_path is not None:
        return gt_path

    k = extract_id(filename)
    if k is not None and k in by_id:
        return pick_best_gt(by_id[k])

    return None


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def save_rgb(arr: np.ndarray, path: str) -> None:
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))
    img.save(path)


def resize_like(src: np.ndarray, ref_shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = ref_shape_hw
    if src.shape[0] == h and src.shape[1] == w:
        return src
    img = Image.fromarray((np.clip(src, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    img = img.resize((w, h), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def safe_mean(x, eps: float = 1e-6):
    return np.maximum(x, eps)


# =========================
# GT-mean correction
# =========================
def apply_gt_mean(pred: np.ndarray, gt: np.ndarray, method: str = "match_scalar"):
    """
    pred, gt: HWC RGB in [0,1]

    match_scalar:
        gain = mean(gt) / mean(pred)
        out = pred * gain

    match_rgb:
        gain_c = mean(gt_c) / mean(pred_c)
        out = pred * gain_rgb
    """
    mu_pred_scalar = float(pred.mean())
    mu_gt_scalar = float(gt.mean())

    mu_pred_rgb = pred.mean(axis=(0, 1))
    mu_gt_rgb = gt.mean(axis=(0, 1))

    info = {
        "mu_pred": mu_pred_scalar,
        "mu_gt": mu_gt_scalar,
    }

    if method == "match_scalar":
        gain = mu_gt_scalar / safe_mean(mu_pred_scalar, EPS)
        gain = float(np.clip(gain, GAIN_MIN, GAIN_MAX))
        out = pred * gain
        info["gain"] = gain

    elif method == "match_rgb":
        gain = mu_gt_rgb / safe_mean(mu_pred_rgb, EPS)
        gain = np.clip(gain, GAIN_MIN, GAIN_MAX)
        out = pred * gain.reshape(1, 1, 3)
        info["gain_r"] = float(gain[0])
        info["gain_g"] = float(gain[1])
        info["gain_b"] = float(gain[2])

    else:
        raise ValueError(f"Unknown METHOD: {method}")

    out = np.clip(out, 0.0, 1.0)
    info["mu_out"] = float(out.mean())
    return out, info


def make_compare(pred: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.concatenate([pred, out], axis=1)


# =========================
# Main
# =========================
def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    compare_dir = None
    if SAVE_COMPARE:
        compare_dir = os.path.join(SAVE_DIR, "compare_pred_vs_gtmean")
        os.makedirs(compare_dir, exist_ok=True)

    pred_paths = list_imgs(PRED_DIR)
    if len(pred_paths) == 0:
        raise RuntimeError(f"No images found in PRED_DIR: {PRED_DIR}")

    gt_index = build_gt_index(GT_DIR)

    csv_rows = []
    processed = 0
    skipped = 0

    print("=" * 70)
    print("GT Mean from Existing Network Outputs")
    print(f"PRED_DIR : {PRED_DIR}")
    print(f"GT_DIR   : {GT_DIR}")
    print(f"SAVE_DIR : {SAVE_DIR}")
    print(f"METHOD   : {METHOD}")
    print(f"FILES    : {len(pred_paths)}")
    print("=" * 70)

    for idx, pred_path in enumerate(pred_paths, start=1):
        name = os.path.basename(pred_path)
        gt_path = find_gt_for_pred(pred_path, gt_index)

        if gt_path is None:
            print(f"[Skip {idx}/{len(pred_paths)}] {name}: no matched GT found")
            skipped += 1
            continue

        pred = load_rgb(pred_path)
        gt = load_rgb(gt_path)

        if pred.shape[:2] != gt.shape[:2]:
            if RESIZE_PRED_TO_GT:
                pred = resize_like(pred, gt.shape[:2])
            else:
                print(f"[Skip {idx}/{len(pred_paths)}] {name}: size mismatch pred={pred.shape[:2]} gt={gt.shape[:2]}")
                skipped += 1
                continue

        out, info = apply_gt_mean(pred=pred, gt=gt, method=METHOD)

        save_rgb(out, os.path.join(SAVE_DIR, name))

        if SAVE_COMPARE and compare_dir is not None:
            compare = make_compare(pred, out)
            save_rgb(compare, os.path.join(compare_dir, name))

        row = {
            "image": name,
            "gt": os.path.basename(gt_path),
            "mu_pred": info["mu_pred"],
            "mu_gt": info["mu_gt"],
            "mu_out": info["mu_out"],
        }
        if "gain" in info:
            row["gain"] = info["gain"]
        if "gain_r" in info:
            row["gain_r"] = info["gain_r"]
            row["gain_g"] = info["gain_g"]
            row["gain_b"] = info["gain_b"]
        csv_rows.append(row)

        processed += 1
        msg = (
            f"[Done {idx}/{len(pred_paths)}] {name} | "
            f"pred={info['mu_pred']:.4f} gt={info['mu_gt']:.4f} out={info['mu_out']:.4f}"
        )
        if "gain" in info:
            msg += f" | gain={info['gain']:.4f}"
        elif "gain_r" in info:
            msg += (
                f" | gain_rgb=({info['gain_r']:.4f}, {info['gain_g']:.4f}, {info['gain_b']:.4f})"
            )
        print(msg)

    if SAVE_METADATA_CSV and len(csv_rows) > 0:
        csv_path = os.path.join(SAVE_DIR, f"gtmean_metadata_{METHOD}.csv")
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Metadata CSV saved to: {csv_path}")

    print("=" * 70)
    print(f"Finished. processed={processed}, skipped={skipped}")
    print(f"Results saved to: {SAVE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
