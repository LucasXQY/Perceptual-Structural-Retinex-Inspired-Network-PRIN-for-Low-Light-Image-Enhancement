# -*- coding: utf-8 -*-
"""Scale-factor ablation evaluation (revision experiments R1-2 / R2-6).

For the seven s0-s6 checkpoints in checkpoints/scale_ablation/:
  1) build the model with the MATCHING variant scales (the scales are plain
     floats that never enter the state_dict, so the model must be constructed
     with the same configuration used at training time);
  2) run inference on the LOL-v1 test set, saving results to
     results/scale_ablation/<sX>/enhanced/;
  3) score with the paper protocol (PSNR/SSIM: skimage, data_range=1.0;
     LPIPS: alex, [-1,1]; MAE; NIQE) and write per-variant CSVs plus a
     summary table (with deltas relative to the s0 baseline) to
     experiments/scale_ablation/.

The INFER_SLEEP / METRIC_SLEEP / COOL_* constants below are OPTIONAL thermal
safeguards (per-image sleeps and a between-variant GPU cool-down) for machines
that are unstable under sustained full load; set the sleeps to 0 to run at full
speed. Already-enhanced images are skipped automatically, so an interrupted run
can simply be restarted and will resume where it left off.

Usage (from the repo root):
    python experiments/scale_ablation/test_scale_ablation.py
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.scale_ablation.variants import VARIANTS, make_model_class

_HERE = Path(__file__).resolve().parent

CKPT_DIR = os.path.join(str(_ROOT), "checkpoints", "scale_ablation")
LOW_DIR = os.path.join(str(_ROOT), "data", "lolv1", "test", "low")
GT_DIR = os.path.join(str(_ROOT), "data", "lolv1", "test", "high")
RESULTS_ROOT = os.path.join(str(_ROOT), "results", "scale_ablation")
CSV_DIR = str(_HERE)  # per-variant CSVs + summary tables

# short checkpoint tag -> variant key (must match models/scale_ablation/variants.py)
VARIANT_MAP = {
    "s0": "s0_default",
    "s1": "s1_sym_high",
    "s2": "s2_inverted",
    "s3": "s3_half",
    "s4": "s4_double",
    "s5": "s5_sobel_low",
    "s6": "s6_sobel_high",
}

# Optional thermal safeguards (set sleeps to 0 to disable)
INFER_SLEEP = 2.0    # sleep after each inferred image (seconds)
METRIC_SLEEP = 1.0   # sleep after each scored image (seconds)
COOL_TARGET = 48     # between variants, wait until the GPU drops below this temperature
COOL_MAX_WAIT = 180


def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def cool_down():
    t0 = time.time()
    while time.time() - t0 < COOL_MAX_WAIT:
        t = gpu_temp()
        if t is None or t <= COOL_TARGET:
            return
        time.sleep(10)


def calculate_ssim(gt, img):
    try:
        return structural_similarity(gt, img, channel_axis=-1, data_range=1.0)
    except TypeError:
        return structural_similarity(gt, img, multichannel=True, data_range=1.0)


@torch.no_grad()
def run_inference(device):
    names = sorted(f for f in os.listdir(LOW_DIR) if f.lower().endswith((".png", ".jpg", ".bmp")))
    for sx, vkey in VARIANT_MAP.items():
        ckpt = os.path.join(CKPT_DIR, f"{vkey}.pth")
        if not os.path.exists(ckpt):
            print(f"[WARN] missing {ckpt}, skip {sx}", flush=True)
            continue
        dst = os.path.join(RESULTS_ROOT, sx, "enhanced")
        os.makedirs(dst, exist_ok=True)
        todo = [n for n in names
                if not os.path.exists(os.path.join(dst, os.path.splitext(n)[0] + ".png"))]
        if not todo:
            print(f"[SKIP] {sx}: all {len(names)} images already done", flush=True)
            continue

        model = make_model_class(vkey)().to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=True)  # strict: fail fast on any architecture mismatch
        model.eval()
        print(f"[INFO] {sx} ({vkey}) scales={VARIANTS[vkey]} <- {ckpt}", flush=True)

        t0 = time.time()
        for n in todo:
            img = Image.open(os.path.join(LOW_DIR, n)).convert("RGB")
            x = to_tensor(img).unsqueeze(0).to(device)
            _, _, enhanced = model(x)
            out = enhanced.clamp(0, 1).squeeze(0).cpu()
            to_pil_image(out).save(os.path.join(dst, os.path.splitext(n)[0] + ".png"))
            time.sleep(INFER_SLEEP)
        print(f"  [{sx}] {len(todo)} images, {time.time()-t0:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()
        cool_down()


@torch.no_grad()
def run_metrics(device):
    import lpips
    import pyiqa
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)

    summary = []
    for sx, vkey in VARIANT_MAP.items():
        res_dir = os.path.join(RESULTS_ROOT, sx, "enhanced")
        if not os.path.isdir(res_dir):
            continue
        rows = []
        files = sorted(f for f in os.listdir(res_dir) if f.endswith(".png"))
        for fn in files:
            gt_path = os.path.join(GT_DIR, fn)
            if not os.path.exists(gt_path):
                stem = os.path.splitext(fn)[0]
                cands = [g for g in os.listdir(GT_DIR) if os.path.splitext(g)[0] == stem]
                if not cands:
                    print(f"[WARN] no GT for {sx}/{fn}", flush=True)
                    continue
                gt_path = os.path.join(GT_DIR, cands[0])
            enh = Image.open(os.path.join(res_dir, fn)).convert("RGB")
            gt = Image.open(gt_path).convert("RGB")
            if enh.size != gt.size:
                enh = enh.resize(gt.size, Image.BICUBIC)
            enh_np = np.array(enh).astype(np.float32) / 255.0
            gt_np = np.array(gt).astype(np.float32) / 255.0
            enh_t = to_tensor(enh).unsqueeze(0).to(device)
            gt_t = to_tensor(gt).unsqueeze(0).to(device)

            rows.append({
                "image": fn,
                "PSNR": peak_signal_noise_ratio(gt_np, enh_np, data_range=1.0),
                "SSIM": calculate_ssim(gt_np, enh_np),
                "MAE": float(np.abs(gt_np - enh_np).mean()),
                "LPIPS": lpips_fn((enh_t - 0.5) * 2, (gt_t - 0.5) * 2).item(),
                "NIQE": niqe_metric(enh_t).item(),
            })
            time.sleep(METRIC_SLEEP)

        if not rows:
            continue
        avg = {k: sum(r[k] for r in rows) / len(rows) for k in ("PSNR", "SSIM", "MAE", "LPIPS", "NIQE")}
        out_csv = os.path.join(CSV_DIR, f"metrics_params_{sx}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image", "PSNR", "SSIM", "MAE", "LPIPS", "NIQE"])
            w.writeheader()
            w.writerows(rows)
            w.writerow({"image": "Average", **{k: f"{v:.6f}" for k, v in avg.items()}})
        cfg = VARIANTS[vkey]
        summary.append({"variant": sx, "key": vkey,
                        "refl": cfg["refl_high_scale"], "illum": cfg["illum_high_scale"],
                        "edge": cfg["refl_edge_scale"], "n": len(rows), **avg})
        print(f"[{sx}] n={len(rows)} PSNR={avg['PSNR']:.3f} SSIM={avg['SSIM']:.4f} "
              f"LPIPS={avg['LPIPS']:.4f}", flush=True)

    # Summary table (with deltas relative to s0)
    base = next((s for s in summary if s["variant"] == "s0"), None)
    sum_csv = os.path.join(CSV_DIR, "summary_params_ablation.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["variant", "key", "refl", "illum", "edge", "n",
                  "PSNR", "SSIM", "MAE", "LPIPS", "NIQE", "dPSNR", "dSSIM", "dLPIPS"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summary:
            row = {**s, "PSNR": f"{s['PSNR']:.4f}", "SSIM": f"{s['SSIM']:.5f}",
                   "MAE": f"{s['MAE']:.5f}", "LPIPS": f"{s['LPIPS']:.5f}", "NIQE": f"{s['NIQE']:.4f}"}
            if base and s["variant"] != "s0":
                row["dPSNR"] = f"{s['PSNR']-base['PSNR']:+.4f}"
                row["dSSIM"] = f"{s['SSIM']-base['SSIM']:+.5f}"
                row["dLPIPS"] = f"{s['LPIPS']-base['LPIPS']:+.5f}"
            w.writerow(row)

    md = os.path.join(CSV_DIR, "summary_params_ablation.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Scale-factor ablation (LOL-v1 test set, full 1000-epoch schedule)\n\n")
        f.write("| Variant | (refl, illum, edge) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MAE ↓ | NIQE ↓ | ΔPSNR | ΔLPIPS |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for s in summary:
            d_p = f"{s['PSNR']-base['PSNR']:+.3f}" if base and s["variant"] != "s0" else "—"
            d_l = f"{s['LPIPS']-base['LPIPS']:+.4f}" if base and s["variant"] != "s0" else "—"
            f.write(f"| {s['variant']} | ({s['refl']}, {s['illum']}, {s['edge']}) | "
                    f"{s['PSNR']:.3f} | {s['SSIM']:.4f} | {s['LPIPS']:.4f} | "
                    f"{s['MAE']:.4f} | {s['NIQE']:.3f} | {d_p} | {d_l} |\n")
    print(f"[DONE] {sum_csv}\n[DONE] {md}", flush=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    run_inference(device)
    cool_down()
    run_metrics(device)
