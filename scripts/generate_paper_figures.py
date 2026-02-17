#!/usr/bin/env python3
"""
Run all paper-figure scripts sequentially.
This uses placeholders unless you replace the data loading blocks.
"""

import subprocess
import sys


SCRIPTS = [
    "scripts/fig01_pipeline_diagram.py",
    "scripts/fig02_qualitative_maps.py",
    "scripts/fig03_spatial_error_maps.py",
    "scripts/fig04_metrics_bar.py",
    "scripts/fig05_timeseries_stations.py",
    "scripts/fig06_seq_len_ablation.py",
]


def main():
    ok = True
    for script in SCRIPTS:
        print(f"▶ Running {script}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            ok = False
            print(f"❌ Failed: {script}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
