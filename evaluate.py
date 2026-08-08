"""Full-reference evaluation protocol (canonical).

This is the exact metric protocol used for EVERY full-reference number reported
in the paper (main comparison tables and all ablation tables):
  - PSNR / SSIM: scikit-image, data_range=1.0
  - LPIPS: AlexNet backbone, inputs scaled to [-1, 1]
  - MAE: mean absolute error on [0, 1] arrays
  - NIQE: pyiqa, [0, 1] input
Predictions whose size differs from the GT are bicubic-resized to the GT size.
GT matching uses a 3-tier priority: exact filename -> same stem -> same numeric
id (e.g. low00123.png <-> normal00123.png), preferring GT names containing
"normal" when several share an id.

Edit the config block below (or use run_evaluate.py, which substitutes the
three config variables without touching this file), then run from the repo root:

    python evaluate.py
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import lpips
import pyiqa
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_absolute_error
import warnings

# Silence version warnings to keep the output readable
warnings.filterwarnings("ignore")

# ====== 1. Config ======
results_dir = "results/lolv1/enhanced"   # folder with enhanced images (e.g. low00123.png)
gt_dir = "data/lolv1/test/high"          # folder with ground-truth images (e.g. normal00123.png)
output_csv = "results/metrics_lolv1.csv"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Running on device: {device}")

# ====== 2. Metric models ======
print("⏳ Loading metrics models...")

# LPIPS: Expects [-1, 1] input
lpips_fn = lpips.LPIPS(net='alex').to(device)

# NIQE: Expects [0, 1] input (PyIQA default)
try:
    niqe_metric = pyiqa.create_metric('niqe', device=device)
except Exception as e:
    print(f"⚠️ Warning: Could not load NIQE metric. Error: {e}")
    niqe_metric = None

to_tensor = transforms.ToTensor()

valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ====== 3. Helper: SSIM across scikit-image versions ======
def calculate_ssim(gt, img):
    """
    SSIM compatible with both old and new scikit-image APIs.
    Input format: HWC numpy array, range [0, 1].
    """
    try:
        # New scikit-image (>0.19)
        return structural_similarity(gt, img, channel_axis=-1, data_range=1.0)
    except TypeError:
        # Old scikit-image
        return structural_similarity(gt, img, multichannel=True, data_range=1.0)


def extract_id(fname: str):
    """
    Extract the last digit group from a filename (leading zeros kept).
    Examples:
      low00123.png -> "00123"
      normal00123.png -> "00123"
      xxx_00123_any.png -> "00123"
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    nums = re.findall(r"\d+", base)
    return nums[-1] if nums else None


def pick_best_gt(paths):
    """
    When several GT files share the same id, prefer the one whose name
    contains "normal", then lexicographic order.
    """
    paths = sorted(paths, key=lambda p: (("normal" not in os.path.basename(p).lower()), os.path.basename(p).lower()))
    return paths[0]


# ====== 4. Main loop ======
metrics = []
file_list = sorted(glob.glob(os.path.join(results_dir, "*.png")))

if not file_list:
    print(f"❌ Error: No images found in '{results_dir}'. Please check the path.")
    exit()

# --- Index GT files by filename / stem / numeric id ---
gt_files = glob.glob(os.path.join(gt_dir, "*"))
gt_files = [p for p in gt_files if os.path.isfile(p) and p.lower().endswith(valid_exts)]

gt_by_filename = {}  # key: exact filename e.g. normal00123.png
gt_by_stem = {}      # key: stem e.g. normal00123
gt_by_id = {}        # key: "00123" -> [paths...]

for p in gt_files:
    bn = os.path.basename(p)
    stem = os.path.splitext(bn)[0]
    gt_by_filename[bn] = p

    # A stem may exist with several extensions: prefer png, then jpg/jpeg
    if stem not in gt_by_stem:
        gt_by_stem[stem] = p
    else:
        old = gt_by_stem[stem].lower()
        new = p.lower()

        def ext_rank(x):
            e = os.path.splitext(x)[1]
            return {".png": 0, ".jpg": 1, ".jpeg": 2}.get(e, 3)

        if ext_rank(new) < ext_rank(old):
            gt_by_stem[stem] = p

    k = extract_id(bn)
    if k is not None:
        gt_by_id.setdefault(k, []).append(p)

print(f"📂 Found {len(file_list)} enhanced images.")
print(f"🧾 GT indexed: {len(gt_files)} files | by_id keys: {len(gt_by_id)}")
print("🚦 Matching priority: same filename -> same stem -> same numeric id (low00123 <-> normal00123)")
print("📂 Starting evaluation...")

for i, enhanced_path in enumerate(file_list):
    filename = os.path.basename(enhanced_path)
    stem = os.path.splitext(filename)[0]

    # --- 1) Exact filename match (including extension) ---
    gt_path = gt_by_filename.get(filename, None)

    # --- 2) Same stem (any extension) ---
    if gt_path is None:
        gt_path = gt_by_stem.get(stem, None)

    # --- 3) Numeric-id match: low00123.png <-> normal00123.png ---
    if gt_path is None:
        k = extract_id(filename)
        if k is not None and k in gt_by_id:
            gt_path = pick_best_gt(gt_by_id[k])

    if gt_path is None or (not os.path.exists(gt_path)):
        print(f"⚠️ [Skip] Missing GT for {filename}")
        continue

    # --- A. Load images ---
    enh_img = Image.open(enhanced_path).convert("RGB")
    gt_img = Image.open(gt_path).convert("RGB")

    # Make sure sizes match
    if enh_img.size != gt_img.size:
        enh_img = enh_img.resize(gt_img.size, Image.BICUBIC)

    # --- B. Preprocessing ---
    # Numpy [0, 1] for PSNR, SSIM, MAE
    enh_np = np.array(enh_img).astype(np.float32) / 255.0
    gt_np = np.array(gt_img).astype(np.float32) / 255.0

    # Tensor [0, 1]
    enh_tensor = to_tensor(enh_img).unsqueeze(0).to(device)
    gt_tensor = to_tensor(gt_img).unsqueeze(0).to(device)

    # Tensor [-1, 1] for LPIPS
    enh_tensor_norm = (enh_tensor - 0.5) * 2
    gt_tensor_norm = (gt_tensor - 0.5) * 2

    # --- C. Metrics ---
    psnr = peak_signal_noise_ratio(gt_np, enh_np, data_range=1.0)
    ssim = calculate_ssim(gt_np, enh_np)
    mae = mean_absolute_error(gt_np.flatten(), enh_np.flatten())

    with torch.no_grad():
        lpips_val = lpips_fn(enh_tensor_norm, gt_tensor_norm).item()
        niqe_val = niqe_metric(enh_tensor).item() if niqe_metric else 0.0

    print(f"[{i + 1}/{len(file_list)}] {filename} | GT: {os.path.basename(gt_path)} | PSNR: {psnr:.2f} | LPIPS: {lpips_val:.4f}")

    metrics.append({
        "image": filename,
        "GT": os.path.basename(gt_path),
        "PSNR": psnr,
        "SSIM": ssim,
        "MAE": mae,
        "LPIPS": lpips_val,
        "NIQE": niqe_val
    })

# ====== 5. Save results ======
if metrics:
    df = pd.DataFrame(metrics)

    # Average over numeric columns only
    mean_values = df.select_dtypes(include=[np.number]).mean()
    mean_row = {"image": "Average", "GT": "-"}
    mean_row.update(mean_values.to_dict())

    # Append the average row
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Save
    df.to_csv(output_csv, index=False)
    print("\n" + "=" * 40)
    print(f"✅ Evaluation Finished!")
    print(f"📊 Average Results:\n{mean_values}")
    print(f"💾 Saved to: {output_csv}")
    print("=" * 40)
else:
    print("❌ No metrics calculated.")
