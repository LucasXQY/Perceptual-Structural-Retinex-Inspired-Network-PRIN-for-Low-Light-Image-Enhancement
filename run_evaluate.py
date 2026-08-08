"""Drive the ORIGINAL evaluate.py with different paths, without editing it.

Reads evaluate.py, substitutes ONLY the three config assignments
(results_dir / gt_dir / output_csv) via regex, and executes the rest verbatim,
so the metric protocol is exactly the canonical one used for every
full-reference number in the paper (see evaluate.py).

Usage (from the repo root, or anywhere — paths are resolved against the repo root):
  python run_evaluate.py --results_dir results/lolv2real/enhanced \
      --gt_dir data/lolv2real/test/high --output_csv results/metrics_lolv2real.csv
"""

import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))  # repo root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    os.chdir(HERE)  # evaluate.py uses repo-root-relative paths

    src = open(os.path.join(HERE, "evaluate.py"), encoding="utf-8").read()
    subs = {
        "results_dir": args.results_dir.replace("\\", "/"),
        "gt_dir": args.gt_dir.replace("\\", "/"),
        "output_csv": args.output_csv.replace("\\", "/"),
    }
    for var, val in subs.items():
        src, n = re.subn(rf'^{var} = ".*?"', f'{var} = "{val}"', src,
                         count=1, flags=re.M)
        assert n == 1, f"could not substitute {var} in evaluate.py"

    exec(compile(src, "evaluate.py", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
