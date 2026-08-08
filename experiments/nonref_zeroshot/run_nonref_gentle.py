# -*- coding: utf-8 -*-
"""Paced driver for the non-reference zero-shot study (test_nonref.py).

Runs one dataset x one source per independent subprocess; between units it
waits for the GPU to cool down, and test_nonref.py itself sleeps between
images (--sleep). The pacing constants below are optional thermal
safeguards for thermally limited machines (e.g. laptop GPUs) -- reduce or
zero them on a workstation GPU. Interrupted runs resume automatically:
finished units are skipped (inference outputs already on disk, metric rows
already in the per-image CSV).

Usage (from the repo root):
    python experiments/nonref_zeroshot/run_nonref_gentle.py
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
PY = sys.executable
SCRIPT = os.path.join(ROOT, "experiments", "nonref_zeroshot", "test_nonref.py")
DATASETS = ["DICM", "LIME", "MEF", "NPE", "VV"]
SOURCES = ["input", "prin_lolv1", "prin_lolv2real", "prin_lolv2syn"]
CKPT_TAGS = ["prin_lolv1", "prin_lolv2real", "prin_lolv2syn"]
NONREF_DIR = os.path.join(ROOT, "data", "non-ref")
OUT_DIR = os.path.join(ROOT, "results", "nonref")
CSV_PATH = os.path.join(ROOT, "experiments", "nonref_zeroshot",
                        "metrics_per_image.csv")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

INFER_SLEEP = "3"      # seconds to sleep after each inferred image
METRIC_SLEEP = "1.5"   # seconds to sleep after each scored image
COOL_TARGET = 48       # between units, wait until GPU temp (C) drops below this
COOL_MAX_WAIT = 180    # but never wait longer than this many seconds
UNIT_GAP = 15          # fixed gap between units, seconds


def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def cooldown(tag):
    time.sleep(UNIT_GAP)
    t0 = time.time()
    while time.time() - t0 < COOL_MAX_WAIT:
        t = gpu_temp()
        if t < 0 or t <= COOL_TARGET:
            print(f"[cool] after {tag}: GPU {t}C, continue", flush=True)
            return
        print(f"[cool] after {tag}: GPU {t}C, waiting...", flush=True)
        time.sleep(15)


def run_unit(args, tag):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===", flush=True)
    r = subprocess.run([PY, "-u", SCRIPT] + args, cwd=ROOT)
    if r.returncode != 0:
        print(f"[WARN] {tag} exited with code {r.returncode}", flush=True)
    cooldown(tag)


def n_images(folder):
    if not os.path.isdir(folder):
        return -1
    return len([f for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in IMG_EXTS])


def csv_counts():
    counts = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                k = (r["dataset"], r["source"])
                counts[k] = counts.get(k, 0) + 1
    return counts


def main():
    # 1) fill in missing inference outputs (existing outputs are skipped)
    for ck in CKPT_TAGS:
        for ds in DATASETS:
            need = n_images(os.path.join(NONREF_DIR, ds))
            have = n_images(os.path.join(OUT_DIR, ds, ck))
            if have >= need > 0:
                print(f"[skip] infer {ck}/{ds}: {have}/{need} done", flush=True)
                continue
            run_unit(["--datasets", ds, "--ckpts", ck,
                      "--skip-metrics", "--sleep", INFER_SLEEP],
                     f"infer {ck}/{ds}")

    # 2) metrics: one dataset x one source per process; rows merge
    #    incrementally into the per-image CSV
    done = csv_counts()
    for ds in DATASETS:
        for src in SOURCES:
            folder = (os.path.join(NONREF_DIR, ds) if src == "input"
                      else os.path.join(OUT_DIR, ds, src))
            need = n_images(folder)
            if done.get((ds, src), 0) >= need > 0:
                print(f"[skip] metrics {ds}/{src}: already in csv", flush=True)
                continue
            run_unit(["--datasets", ds, "--ckpts"] + CKPT_TAGS +
                     ["--skip-infer", "--sources", src, "--sleep", METRIC_SLEEP],
                     f"metrics {ds}/{src}")

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
