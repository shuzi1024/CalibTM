from __future__ import annotations

import argparse
from pathlib import Path

from experiments.sc_acil_v1.adjudication import adjudicate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(adjudicate(args.output_root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
