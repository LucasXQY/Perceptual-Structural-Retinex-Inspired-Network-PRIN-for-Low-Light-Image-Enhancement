# -*- coding: utf-8 -*-
"""Non-reference zero-shot evaluation on DICM / LIME / MEF / NPE / VV.

Zero-shot generalization study (revised manuscript): the three PRIN
checkpoints (one per training set, no fine-tuning) enhance the five standard
no-reference benchmarks, and NIQE / MUSIQ / LIQE are computed per image via
pyiqa (official metric weights auto-download on first use; Q-Align is scored
separately by run_qalign.py in this folder). The unprocessed input images
are scored as a reference row.

Enhanced images are saved to results/nonref/<dataset>/<checkpoint-tag>/ and
the CSV outputs (metrics_per_image.csv, metrics_summary.csv/.md) are written
next to this script under experiments/nonref_zeroshot/. Metric rows merge
incrementally: a re-run only overwrites the (dataset, source) combinations
it recomputed and keeps all other existing rows.

Usage (from the repo root):
    python experiments/nonref_zeroshot/test_nonref.py                # everything
    python experiments/nonref_zeroshot/test_nonref.py --datasets NPE --ckpts prin_lolv1   # smoke test
    python experiments/nonref_zeroshot/test_nonref.py --skip-infer   # recompute metrics only
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from models.prin import PRIN

REPO = str(_ROOT)
DATASETS = ["DICM", "LIME", "MEF", "NPE", "VV"]
CKPTS = {
    "prin_lolv1":     os.path.join(REPO, "checkpoints", "prin_lolv1.pth"),
    "prin_lolv2real": os.path.join(REPO, "checkpoints", "prin_lolv2real.pth"),
    "prin_lolv2syn":  os.path.join(REPO, "checkpoints", "prin_lolv2syn.pth"),
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
NONREF_DIR = os.path.join(REPO, "data", "non-ref")
OUT_DIR = os.path.join(REPO, "results", "nonref")
CSV_DIR = os.path.dirname(os.path.abspath(__file__))  # experiments/nonref_zeroshot
MAX_SIDE_FALLBACK = 1536  # longest side to downscale to after a CUDA OOM


def list_images(folder):
    names = [f for f in sorted(os.listdir(folder))
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    return names


@torch.no_grad()
def infer_one(model, img, device):
    x = to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W], 0-1
    _, _, enhanced = model(x)                   # PRIN forward pads/crops internally
    return enhanced.clamp(0, 1).squeeze(0).cpu()


@torch.no_grad()
def run_inference(datasets, ckpt_tags, device, overwrite=False, sleep=0.0):
    oom_log = []
    for tag in ckpt_tags:
        path = CKPTS[tag]
        model = PRIN().to(device)
        model.load_state_dict(torch.load(path, map_location=device), strict=True)
        model.eval()
        print(f"[INFO] loaded {tag}: {path}", flush=True)

        for ds in datasets:
            src = os.path.join(NONREF_DIR, ds)
            dst = os.path.join(OUT_DIR, ds, tag)
            os.makedirs(dst, exist_ok=True)
            names = list_images(src)
            t0 = time.time()
            skipped = 0
            for name in names:
                stem = os.path.splitext(name)[0]
                out_path = os.path.join(dst, stem + ".png")
                if not overwrite and os.path.exists(out_path):
                    skipped += 1
                    continue
                img = Image.open(os.path.join(src, name)).convert("RGB")
                try:
                    out = infer_one(model, img, device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    w, h = img.size
                    s = MAX_SIDE_FALLBACK / max(w, h)
                    img_small = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
                    out = infer_one(model, img_small, device)
                    oom_log.append(f"{ds}/{tag}/{name}: {w}x{h} -> {img_small.size[0]}x{img_small.size[1]}")
                    print(f"  [OOM->downscale] {name}", flush=True)
                to_pil_image(out).save(out_path)
                if sleep > 0:
                    time.sleep(sleep)  # optional thermal safeguard: limits sustained GPU load
            print(f"  [{tag}] {ds}: {len(names)} images ({skipped} skipped), "
                  f"{time.time()-t0:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    if oom_log:
        log_path = os.path.join(OUT_DIR, "downscaled.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(oom_log) + "\n")
        print(f"[WARN] {len(oom_log)} images downscaled due to OOM (see {log_path})")


@torch.no_grad()
def run_metrics(datasets, ckpt_tags, device, sleep=0.0, only_sources=None):
    import pyiqa
    metrics = {}
    for mname in ("niqe", "musiq", "liqe"):
        print(f"[INFO] creating metric {mname} (official weights auto-download on first run)...", flush=True)
        metrics[mname] = pyiqa.create_metric(mname, device=device)

    rows = []  # dataset, source, filename, niqe, musiq, liqe
    recomputed = set()  # (dataset, source) combinations recomputed this run
    for ds in datasets:
        sources = {"input": os.path.join(NONREF_DIR, ds)}
        for tag in ckpt_tags:
            sources[tag] = os.path.join(OUT_DIR, ds, tag)
        if only_sources is not None:
            # any source name is allowed: it maps to results/nonref/<ds>/<name>/
            for s in only_sources:
                if s not in sources:
                    sources[s] = os.path.join(OUT_DIR, ds, s)
            sources = {s: p for s, p in sources.items() if s in only_sources}
        for source, folder in sources.items():
            if not os.path.isdir(folder):
                print(f"[WARN] missing {folder}, skip")
                continue
            names = list_images(folder)
            t0 = time.time()
            for name in names:
                p = os.path.join(folder, name)
                row = {"dataset": ds, "source": source, "filename": name}
                for mname, metric in metrics.items():
                    row[mname] = float(metric(p).item())
                rows.append(row)
                if sleep > 0:
                    time.sleep(sleep)  # optional thermal safeguard: limits sustained GPU load
            recomputed.add((ds, source))
            print(f"  [{ds}] {source}: {len(names)} images, {time.time()-t0:.1f}s", flush=True)

    # Merge with the existing per-image CSV: recomputed (dataset, source)
    # combinations overwrite their old rows, all other rows are kept.
    per_image_csv = os.path.join(CSV_DIR, "metrics_per_image.csv")
    if os.path.exists(per_image_csv):
        with open(per_image_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["dataset"], r["source"]) in recomputed:
                    continue
                try:
                    for m in ("niqe", "musiq", "liqe"):
                        r[m] = float(r[m])
                except (ValueError, KeyError):
                    continue
                rows.append(r)
    rows.sort(key=lambda r: (r["dataset"], r["source"], r["filename"]))
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "source", "filename", "niqe", "musiq", "liqe"])
        w.writeheader()
        w.writerows(rows)

    # per-(dataset, source) means
    agg = {}
    for r in rows:
        key = (r["dataset"], r["source"])
        agg.setdefault(key, []).append(r)
    summary = []
    for (ds, source), rs in sorted(agg.items()):
        summary.append({
            "dataset": ds, "source": source, "n": len(rs),
            "niqe": sum(x["niqe"] for x in rs) / len(rs),
            "musiq": sum(x["musiq"] for x in rs) / len(rs),
            "liqe": sum(x["liqe"] for x in rs) / len(rs),
        })
    summary_csv = os.path.join(CSV_DIR, "metrics_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "source", "n", "niqe", "musiq", "liqe"])
        w.writeheader()
        for s in summary:
            w.writerow({**s, "niqe": f"{s['niqe']:.4f}",
                        "musiq": f"{s['musiq']:.4f}", "liqe": f"{s['liqe']:.4f}"})

    md = os.path.join(CSV_DIR, "metrics_summary.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Non-reference zero-shot results (NIQE lower better; MUSIQ/LIQE higher better)\n\n")
        f.write("Checkpoint mapping: prin_lolv1=checkpoints/prin_lolv1.pth, "
                "prin_lolv2real=checkpoints/prin_lolv2real.pth, "
                "prin_lolv2syn=checkpoints/prin_lolv2syn.pth\n\n")
        f.write("| Dataset | Source | N | NIQE ↓ | MUSIQ ↑ | LIQE ↑ |\n|---|---|---|---|---|---|\n")
        for s in summary:
            f.write(f"| {s['dataset']} | {s['source']} | {s['n']} | "
                    f"{s['niqe']:.3f} | {s['musiq']:.3f} | {s['liqe']:.3f} |\n")
    print(f"[DONE] per-image: {per_image_csv}\n[DONE] summary: {summary_csv} / {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--ckpts", nargs="+", default=list(CKPTS), choices=list(CKPTS))
    ap.add_argument("--skip-infer", action="store_true")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run inference for existing outputs "
                         "(default: skip them, so interrupted runs can resume)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to sleep between images "
                         "(optional thermal safeguard: limits sustained GPU load)")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="compute metrics only for these sources (input / a "
                         "checkpoint tag / any subfolder name under "
                         "results/nonref/<ds>/)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.skip_infer:
        run_inference(args.datasets, args.ckpts, device,
                      overwrite=args.overwrite, sleep=args.sleep)
    if not args.skip_metrics:
        run_metrics(args.datasets, args.ckpts, device,
                    sleep=args.sleep, only_sources=args.sources)


if __name__ == "__main__":
    main()
