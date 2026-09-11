from __future__ import annotations

import argparse
from pathlib import Path

from experiments.sc_acil_v1.smoke import run_smoke


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(run_smoke(args.output_root, device_name=args.device), flush=True)

