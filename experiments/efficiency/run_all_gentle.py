"""
Paced driver for the efficiency comparison (paper Table 6): measures every
model in measure.py's BUILDERS one subprocess at a time, sleeping between
models so the GPU never sees long sustained load (optional thermal safeguard
for thermally limited machines, e.g. laptop GPUs).

Resume-friendly: measure.py skips models already present in the results CSV
(experiments/efficiency/results/efficiency.csv).

Usage (from the repo root):
  python experiments/efficiency/run_all_gentle.py                # all models, skip existing
  python experiments/efficiency/run_all_gentle.py --sleep 10     # longer cooldown
  python experiments/efficiency/run_all_gentle.py --models PRIN Retinexformer
"""

import argparse
import subprocess
import sys
import time
import os

ROOT = os.path.dirname(os.path.abspath(__file__))  # experiments/efficiency


def list_models():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "measure.py"), "--list"],
                         capture_output=True, text=True, cwd=ROOT)
    line = out.stdout.strip()
    return [m.strip() for m in line.split(":", 1)[1].split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--sleep", type=float, default=8.0,
                    help="cooldown seconds between models")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    models = args.models or list_models()
    print(f"[DRIVER] measuring {len(models)} models: {models}")

    for i, m in enumerate(models):
        cmd = [sys.executable, os.path.join(ROOT, "measure.py"), "--model", m]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"[DRIVER] ({i+1}/{len(models)}) {m}")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[DRIVER] {m} FAILED (rc={r.returncode}), continuing")
        if i < len(models) - 1:
            time.sleep(args.sleep)

    print(f"[DRIVER] done. Results in {os.path.join(ROOT, 'results', 'efficiency.csv')}")


if __name__ == "__main__":
    main()
