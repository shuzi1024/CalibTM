"""CLI for exactly one manifest-resolved SC-ACIL GPU job."""

from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import canonical_json_bytes
from .run_job import run_planned_job


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen SC-ACIL job")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    path = run_planned_job(args.stage, args.job_id, args.output_root)
    print(
        canonical_json_bytes(
            {"job_id": args.job_id, "result_directory": str(path)}
        ).decode("ascii"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

