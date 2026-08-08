"""No-reference IQA (MUSIQ / LIQE / Q-Align via pyiqa) on LOL test-set outputs (paper Sec. 4.4).

Evaluates MUSIQ, LIQE and Q-Align for every enhanced image in a folder.
All three metrics are no-reference IQA metrics: no GT folder is needed.
Higher scores indicate better predicted perceptual quality.

Usage (from the repo root):
    1. Edit USER CONFIGURATION below.
    2. Run: python experiments/nonref_lol/evaluate_nr_iqa.py
"""

from pathlib import Path
import gc
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor


# ============================================================
# USER CONFIGURATION: only modify this section
# ============================================================

# Hugging Face cache folder for Q-Align / OneAlign model weights.
# If you need a custom cache location, set it BEFORE importing pyiqa, e.g.:
# os.environ["HF_HOME"] = r"D:\huggingface_cache"

# Optional: suppress the Windows symlink-cache warning.
# It does not change metric values.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pyiqa

# Folder containing your final enhanced images (relative to the repo root).
INPUT_DIR = r"results/lolv1/enhanced"

# Folder for saving evaluation CSV.
OUTPUT_DIR = r"experiments/nonref_lol/metrics"

# Output CSV filename.
OUTPUT_CSV_NAME = "nr_iqa_results.csv"

# LIQE model variant: "liqe_mix" or "liqe".
# liqe_mix is recommended for general perceptual evaluation.
LIQE_VARIANT = "liqe_mix"

# Computing device: "auto", "cuda", or "cpu".
DEVICE = "auto"

# Search images in all subfolders under INPUT_DIR when True.
RECURSIVE = False

# Select metrics to run. Q-Align consumes more GPU memory.
RUN_MUSIQ = True
RUN_LIQE = True
RUN_QALIGN = True

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ============================================================
# END OF USER CONFIGURATION
# ============================================================


def get_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda' is selected, but CUDA is not available.")
    if DEVICE not in {"cuda", "cpu"}:
        raise ValueError("DEVICE must be 'auto', 'cuda', or 'cpu'.")
    return torch.device(DEVICE)


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {folder}")
    candidates = folder.rglob("*") if RECURSIVE else folder.glob("*")
    paths = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in VALID_EXTS)
    if not paths:
        raise RuntimeError(f"No supported images found in: {folder}")
    return paths


def load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Read an RGB image as [1, 3, H, W] in [0, 1], without resizing."""
    with Image.open(path) as img:
        image = img.convert("RGB")
        return to_tensor(image).unsqueeze(0).to(device, non_blocking=True)


def release_model(model, device: torch.device) -> None:
    if model is not None:
        del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_metric(label: str, model_name: str, paths: list[Path], device: torch.device) -> list[float]:
    """Load one IQA model, evaluate all images, and release it afterward."""
    model = None
    scores = []
    print(f"\n[INFO] Loading {label}: pyiqa.create_metric('{model_name}')")
    try:
        model = pyiqa.create_metric(model_name, device=device, as_loss=False)
        model.eval()
    except Exception as exc:
        print(f"[ERROR] Cannot load {label}: {exc}")
        return [np.nan] * len(paths)

    try:
        for i, image_path in enumerate(paths, start=1):
            tensor = None
            try:
                tensor = load_tensor(image_path, device)
                with torch.inference_mode():
                    if label == "Q-Align":
                        value = model(tensor, task_="quality")
                    else:
                        value = model(tensor)
                score = float(value.detach().cpu().reshape(-1)[0])
                scores.append(score)
                print(f"[{label:<7}] {i:>4}/{len(paths)} | {image_path.name:<38} | {score:.6f}")
            except torch.cuda.OutOfMemoryError:
                print(f"[ERROR] GPU out of memory for {label} on {image_path.name}; recorded as NaN.")
                scores.append(np.nan)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[WARN] {label} failed on {image_path.name}: {exc}")
                scores.append(np.nan)
            finally:
                if tensor is not None:
                    del tensor
    finally:
        release_model(model, device)
    return scores


def add_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    mean_row = {"image": "Average", "relative_path": "-", "width": np.nan, "height": np.nan}
    std_row = {"image": "Std", "relative_path": "-", "width": np.nan, "height": np.nan}
    for metric in ["MUSIQ", "LIQE", "Q-Align"]:
        valid = df[metric].dropna()
        mean_row[metric] = float(valid.mean()) if len(valid) else np.nan
        std_row[metric] = float(valid.std(ddof=0)) if len(valid) else np.nan
    return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def main() -> None:
    if LIQE_VARIANT not in {"liqe", "liqe_mix"}:
        raise ValueError("LIQE_VARIANT must be 'liqe' or 'liqe_mix'.")

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_csv = output_dir / OUTPUT_CSV_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    image_paths = collect_images(input_dir)

    print("=" * 72)
    print("[INFO] No-reference IQA evaluation: MUSIQ / LIQE / Q-Align")
    print(f"[INFO] Input images : {input_dir}")
    print(f"[INFO] Output CSV   : {output_csv}")
    print(f"[INFO] Image count  : {len(image_paths)}")
    print(f"[INFO] Device       : {device}")
    print(f"[INFO] HF cache     : {os.environ.get('HF_HOME', '(default)')}")
    print("[INFO] Original saved resolution is used; no GT and no resize.")
    print("=" * 72)

    rows = []
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
        rows.append({
            "image": path.name,
            "relative_path": str(path.relative_to(input_dir)),
            "width": width,
            "height": height,
        })
    df = pd.DataFrame(rows)
    df["MUSIQ"] = np.nan
    df["LIQE"] = np.nan
    df["Q-Align"] = np.nan

    if RUN_MUSIQ:
        df["MUSIQ"] = run_metric("MUSIQ", "musiq", image_paths, device)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if RUN_LIQE:
        df["LIQE"] = run_metric("LIQE", LIQE_VARIANT, image_paths, device)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if RUN_QALIGN:
        df["Q-Align"] = run_metric("Q-Align", "qalign", image_paths, device)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Calculate the Average / Std rows only after all selected metric values are collected.
    final_df = add_summary_rows(df)
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    avg = final_df.loc[final_df["image"] == "Average"].iloc[0]

    print("\n" + "=" * 72)
    print("[RESULT] Average scores (higher is better)")
    for metric in ["MUSIQ", "LIQE", "Q-Align"]:
        value = avg[metric]
        text = "not calculated" if pd.isna(value) else f"{value:.6f}"
        print(f"[RESULT] {metric:<7}: {text}")
    print(f"[RESULT] CSV saved to: {output_csv}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        raise SystemExit(1)
