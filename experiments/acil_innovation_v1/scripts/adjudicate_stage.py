"""Publish one immutable adjudication for a completed active-manifest stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..adjudication import write_stage_adjudication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and publish one frozen ACIL-Innovation stage decision"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("stage0_acil_tune", "stage_h", "stage_i", "full_tune"),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    path = write_stage_adjudication(arguments.stage, arguments.output_root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
