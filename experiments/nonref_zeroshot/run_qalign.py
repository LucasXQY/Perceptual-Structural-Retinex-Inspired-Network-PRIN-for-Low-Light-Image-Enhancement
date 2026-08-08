# -*- coding: utf-8 -*-
"""Q-Align (8-bit) scoring of the non-reference zero-shot results.

Fourth no-reference metric of the zero-shot generalization study
(complements the NIQE / MUSIQ / LIQE computed by test_nonref.py). The 8B
LMM is expensive to load, so a single process loads it once, scores image
by image and appends each row to qalign_per_image.csv immediately --
re-running after an interruption skips images that are already scored.
A short sleep between images limits sustained GPU load (optional thermal
safeguard). Summaries are written to qalign_summary.csv / .md next to this
script.

Usage (from the repo root):
    python experiments/nonref_zeroshot/run_qalign.py
"""
import csv
import os
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
DATASETS = ["DICM", "LIME", "MEF", "NPE", "VV"]
SOURCES = ["input", "prin_lolv1", "prin_lolv2real", "prin_lolv2syn"]
NONREF_DIR = os.path.join(ROOT, "data", "non-ref")
OUT_DIR = os.path.join(ROOT, "results", "nonref")
CSV_DIR = os.path.dirname(os.path.abspath(__file__))  # experiments/nonref_zeroshot
PER_IMG_CSV = os.path.join(CSV_DIR, "qalign_per_image.csv")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SLEEP = 2.0  # seconds between images (optional thermal safeguard)


def list_images(folder):
    return [f for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in IMG_EXTS]


def src_folder(ds, source):
    return (os.path.join(NONREF_DIR, ds) if source == "input"
            else os.path.join(OUT_DIR, ds, source))


def main():
    done = set()
    if os.path.exists(PER_IMG_CSV):
        with open(PER_IMG_CSV, newline="", encoding="utf-8") as f:
            done = {(r["dataset"], r["source"], r["filename"])
                    for r in csv.DictReader(f)}
        print(f"[resume] {len(done)} images already scored", flush=True)

    todo = []
    for ds in DATASETS:
        for source in SOURCES:
            folder = src_folder(ds, source)
            if not os.path.isdir(folder):
                print(f"[WARN] missing {folder}, skip", flush=True)
                continue
            for name in list_images(folder):
                if (ds, source, name) not in done:
                    todo.append((ds, source, name))
    print(f"[plan] {len(todo)} images to score", flush=True)
    if not todo:
        summarize()
        return

    import pyiqa
    t0 = time.time()
    print("[load] creating qalign_8bit ...", flush=True)
    metric = pyiqa.create_metric("qalign_8bit", device="cuda")
    print(f"[load] done in {time.time()-t0:.0f}s", flush=True)

    new_file = not os.path.exists(PER_IMG_CSV)
    with open(PER_IMG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "source", "filename", "qalign"])
        if new_file:
            w.writeheader()
        cur_key, t_unit = None, time.time()
        for i, (ds, source, name) in enumerate(todo):
            key = (ds, source)
            if key != cur_key:
                if cur_key is not None:
                    print(f"  [{cur_key[0]}] {cur_key[1]}: {time.time()-t_unit:.0f}s",
                          flush=True)
                cur_key, t_unit = key, time.time()
            score = float(metric(os.path.join(src_folder(ds, source), name)).item())
            w.writerow({"dataset": ds, "source": source, "filename": name,
                        "qalign": f"{score:.6f}"})
            f.flush()
            time.sleep(SLEEP)
        print(f"  [{cur_key[0]}] {cur_key[1]}: {time.time()-t_unit:.0f}s", flush=True)

    summarize()


def summarize():
    agg = {}
    with open(PER_IMG_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            agg.setdefault((r["dataset"], r["source"]), []).append(float(r["qalign"]))
    rows = [{"dataset": ds, "source": s, "n": len(v),
             "qalign": sum(v) / len(v)} for (ds, s), v in sorted(agg.items())]
    with open(os.path.join(CSV_DIR, "qalign_summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "source", "n", "qalign"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "qalign": f"{r['qalign']:.4f}"})
    with open(os.path.join(CSV_DIR, "qalign_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Q-Align (qalign_8bit, 1-5 higher better) on non-ref zero-shot\n\n")
        f.write("| Dataset | Source | N | Q-Align |\n|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['dataset']} | {r['source']} | {r['n']} "
                    f"| {r['qalign']:.3f} |\n")
    print("[DONE] qalign_summary.csv / .md written", flush=True)


if __name__ == "__main__":
    main()
