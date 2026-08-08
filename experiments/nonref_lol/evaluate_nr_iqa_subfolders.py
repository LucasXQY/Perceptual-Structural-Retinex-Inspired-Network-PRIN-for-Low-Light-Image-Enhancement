"""Batch no-reference IQA (MUSIQ / LIQE / Q-Align via pyiqa) on LOL test-set outputs (paper Sec. 4.4).

Evaluates all image-containing subfolders under ROOT_INPUT_DIR.
A separate CSV named after each subfolder is saved to OUTPUT_DIR.
Each CSV contains per-image scores plus Average and Std rows.

Efficiency design:
Each IQA metric model is loaded only once, then evaluates every selected
subfolder. This avoids repeatedly loading the large Q-Align/OneAlign model.

Usage (from the repo root):
    1. Edit USER CONFIGURATION below.
    2. Run: python experiments/nonref_lol/evaluate_nr_iqa_subfolders.py
"""

from __future__ import annotations

import gc
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torchvision.transforms.functional import to_tensor

# ============================================================
# USER CONFIGURATION: only modify this section
# ============================================================

# Hugging Face cache folder for Q-Align / OneAlign model weights.
# Keep HF_HOME consistent with the location used for `hf auth login`.
# If you need a custom cache location, set it BEFORE importing pyiqa, e.g.:
# os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

# Root directory that contains image subfolders to be evaluated
# (relative to the repo root).
# Example structure (test.py output for one dataset):
# ROOT_INPUT_DIR/
#   enhanced/       -> enhanced.csv
#   reflectance/    -> reflectance.csv
#   illumination/   -> illumination.csv
ROOT_INPUT_DIR = r"results/lolv1"

# All per-folder CSV files will be saved here.
OUTPUT_DIR = r"experiments/nonref_lol/metrics"

# Folder scanning mode:
#   "direct"    -> test only first-level folders immediately under ROOT_INPUT_DIR.
#   "recursive" -> test every nested folder that directly contains images.
SCAN_MODE = "direct"

# LIQE variant: "liqe_mix" or "liqe".
LIQE_VARIANT = "liqe_mix"

# Device: "auto", "cuda", or "cpu".
DEVICE = "auto"

# Select metrics to run. Q-Align consumes substantially more GPU memory.
RUN_MUSIQ = True
RUN_LIQE = True
RUN_QALIGN = True

# Also save a table containing the Average result of every subfolder.
SAVE_FOLDER_SUMMARY = True
SUMMARY_CSV_NAME = "_all_folders_average_summary.csv"

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ============================================================
# END OF USER CONFIGURATION
# ============================================================

# pyiqa must be imported after setting Hugging Face environment variables.
import pyiqa


def get_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE='cuda' is selected, but CUDA is not available.")
        return torch.device("cuda")
    if DEVICE == "cpu":
        return torch.device("cpu")
    raise ValueError("DEVICE must be 'auto', 'cuda', or 'cpu'.")


def list_images_directly(folder: Path) -> list[Path]:
    """List supported images directly inside one folder, without descending further."""
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    )


def collect_image_folders(root: Path) -> dict[Path, list[Path]]:
    """Locate selected subfolders and collect their directly contained images."""
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"ROOT_INPUT_DIR does not exist or is not a directory: {root}")

    if SCAN_MODE == "direct":
        candidate_folders = sorted(path for path in root.iterdir() if path.is_dir())
    elif SCAN_MODE == "recursive":
        candidate_folders = sorted(path for path in root.rglob("*") if path.is_dir())
    else:
        raise ValueError("SCAN_MODE must be 'direct' or 'recursive'.")

    folders: dict[Path, list[Path]] = {}
    for folder in candidate_folders:
        images = list_images_directly(folder)
        if images:
            folders[folder] = images

    if not folders:
        raise RuntimeError(
            f"No image-containing subfolders were found under: {root}\n"
            f"Current SCAN_MODE='{SCAN_MODE}'."
        )

    # CSV files are named after the folders. Prevent accidental overwrite.
    by_name: dict[str, list[Path]] = {}
    for folder in folders:
        by_name.setdefault(folder.name, []).append(folder)
    duplicate_names = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if duplicate_names:
        details = "\n".join(
            f"  {name}: " + ", ".join(str(path) for path in paths)
            for name, paths in duplicate_names.items()
        )
        raise RuntimeError(
            "Duplicate image-folder names would generate conflicting CSV filenames:\n"
            f"{details}\nRename the duplicated folders or use SCAN_MODE='direct'."
        )

    return folders


def load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load an RGB image as [1, 3, H, W] in [0, 1], without resizing."""
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            return to_tensor(image).unsqueeze(0).to(device, non_blocking=True)
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Cannot read image {path}: {exc}") from exc


def release_model(model: Optional[torch.nn.Module], device: torch.device) -> None:
    if model is not None:
        del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def safe_csv_stem(folder_name: str) -> str:
    """Preserve folder name while replacing characters illegal in Windows filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", folder_name).strip().rstrip(".")
    return sanitized or "unnamed_folder"


def csv_path_for_folder(folder: Path, output_dir: Path) -> Path:
    return output_dir / f"{safe_csv_stem(folder.name)}.csv"


def build_table(image_paths: list[Path], folder: Path) -> pd.DataFrame:
    rows = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            width, height = np.nan, np.nan
        rows.append({
            "image": path.name,
            "relative_path": str(path.relative_to(folder)),
            "width": width,
            "height": height,
            "MUSIQ": np.nan,
            "LIQE": np.nan,
            "Q-Align": np.nan,
        })
    return pd.DataFrame(rows)


def run_metric_across_folders(
    label: str,
    model_name: str,
    folder_images: dict[Path, list[Path]],
    tables: dict[Path, pd.DataFrame],
    device: torch.device,
    output_dir: Path,
) -> None:
    """Load one metric once; evaluate every folder; preserve partial CSV outputs."""
    model = None
    print(f"\n[INFO] Loading {label}: pyiqa.create_metric('{model_name}')")
    try:
        model = pyiqa.create_metric(model_name, device=device, as_loss=False)
        model.eval()
    except Exception as exc:
        print(f"[ERROR] Cannot load {label}: {exc}")
        return

    total_images = sum(len(images) for images in folder_images.values())
    completed = 0
    try:
        for folder, image_paths in folder_images.items():
            scores: list[float] = []
            print(f"\n[INFO] {label}: folder '{folder.name}' ({len(image_paths)} images)")
            for image_path in image_paths:
                tensor = None
                try:
                    tensor = load_tensor(image_path, device)
                    with torch.inference_mode():
                        value = (
                            model(tensor, task_="quality")
                            if label == "Q-Align"
                            else model(tensor)
                        )
                    score = float(value.detach().cpu().reshape(-1)[0].item())
                    scores.append(score)
                    completed += 1
                    print(
                        f"[{label:<7}] {completed:>4}/{total_images} | "
                        f"{folder.name}/{image_path.name:<38} | {score:.6f}"
                    )
                    del value
                except torch.cuda.OutOfMemoryError:
                    scores.append(np.nan)
                    completed += 1
                    print(
                        f"[ERROR] GPU out of memory for {label} on "
                        f"{folder.name}/{image_path.name}; recorded as NaN."
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:
                    scores.append(np.nan)
                    completed += 1
                    print(f"[WARN] {label} failed on {folder.name}/{image_path.name}: {exc}")
                finally:
                    if tensor is not None:
                        del tensor

            tables[folder][label] = scores
            tables[folder].to_csv(
                csv_path_for_folder(folder, output_dir),
                index=False,
                encoding="utf-8-sig",
            )
    finally:
        release_model(model, device)


def add_summary_rows(table: pd.DataFrame) -> pd.DataFrame:
    mean_row = {"image": "Average", "relative_path": "-", "width": np.nan, "height": np.nan}
    std_row = {"image": "Std", "relative_path": "-", "width": np.nan, "height": np.nan}
    for metric in ["MUSIQ", "LIQE", "Q-Align"]:
        valid = table[metric].dropna()
        mean_row[metric] = float(valid.mean()) if not valid.empty else np.nan
        std_row[metric] = float(valid.std(ddof=0)) if not valid.empty else np.nan
    return pd.concat([table, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def save_final_csvs(tables: dict[Path, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    summary_rows = []
    for folder, table in tables.items():
        final_table = add_summary_rows(table)
        output_csv = csv_path_for_folder(folder, output_dir)
        final_table.to_csv(output_csv, index=False, encoding="utf-8-sig")
        avg = final_table.loc[final_table["image"] == "Average"].iloc[0]

        summary_rows.append({
            "folder": folder.name,
            "image_count": len(table),
            "MUSIQ": avg["MUSIQ"],
            "LIQE": avg["LIQE"],
            "Q-Align": avg["Q-Align"],
            "csv_path": str(output_csv.resolve()),
        })

        print("\n" + "-" * 72)
        print(f"[RESULT] Folder: {folder.name}")
        for metric in ["MUSIQ", "LIQE", "Q-Align"]:
            score = avg[metric]
            result = "not calculated" if pd.isna(score) else f"{score:.6f}"
            print(f"[RESULT] {metric:<7}: {result}")
        print(f"[RESULT] CSV saved to: {output_csv.resolve()}")
    return pd.DataFrame(summary_rows)


def main() -> None:
    if LIQE_VARIANT not in {"liqe", "liqe_mix"}:
        raise ValueError("LIQE_VARIANT must be 'liqe' or 'liqe_mix'.")

    root_dir = Path(ROOT_INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    folder_images = collect_image_folders(root_dir)
    tables = {folder: build_table(paths, folder) for folder, paths in folder_images.items()}

    print("=" * 72)
    print("[INFO] Batch no-reference IQA evaluation: MUSIQ / LIQE / Q-Align")
    print(f"[INFO] Root input folder : {root_dir.resolve()}")
    print(f"[INFO] Output folder     : {output_dir.resolve()}")
    print(f"[INFO] Folder count      : {len(folder_images)}")
    print(f"[INFO] Total image count : {sum(len(paths) for paths in folder_images.values())}")
    print(f"[INFO] Device            : {device}")
    print(f"[INFO] HF cache          : {os.environ.get('HF_HOME', '(default)')}")
    print("[INFO] Original saved resolution is used; no GT and no resize.")
    print("[INFO] Image-containing folders found:")
    for folder, images in folder_images.items():
        print(f"       - {folder.name}: {len(images)} image(s)")
    print("=" * 72)

    if RUN_MUSIQ:
        run_metric_across_folders("MUSIQ", "musiq", folder_images, tables, device, output_dir)
    if RUN_LIQE:
        run_metric_across_folders("LIQE", LIQE_VARIANT, folder_images, tables, device, output_dir)
    if RUN_QALIGN:
        run_metric_across_folders("Q-Align", "qalign", folder_images, tables, device, output_dir)

    summary_table = save_final_csvs(tables, output_dir)
    if SAVE_FOLDER_SUMMARY:
        summary_csv = output_dir / SUMMARY_CSV_NAME
        summary_table.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 72)
        print(f"[RESULT] All-folder average summary saved to: {summary_csv.resolve()}")
        print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        raise SystemExit(1)
