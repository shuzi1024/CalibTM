"""CLI for one manifest-resolved GPU worker."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import canonical_json_bytes
from .result_io import load_job_result
from .run_job import run_planned_job


_ALGORITHMIC_FAILURE_EXIT_CODE = 20


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen ACIL-Innovation job")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> int:
    arguments = _build_parser().parse_args()
    result = run_planned_job(
        arguments.stage, arguments.job_id, arguments.output_root
    )
    payload = load_job_result(
        stage=arguments.stage,
        job_id=arguments.job_id,
        output_root=arguments.output_root,
    )
    status = str(payload["status"])
    if status not in {"succeeded", "algorithmic_failure"}:
        raise RuntimeError("worker finalized an unknown result status")
    print(
        canonical_json_bytes(
            {
                "job_id": arguments.job_id,
                "result_directory": str(result),
                "status": status,
            }
        ).decode("ascii"),
        flush=True,
    )
    return 0 if status == "succeeded" else _ALGORITHMIC_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
