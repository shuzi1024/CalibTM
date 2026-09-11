"""Run one authority-bound job from the fixed six-job AnchorCV final gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .gate_identity import verify_gate_authority


def build_parser() -> argparse.ArgumentParser:
    """Expose only an opaque job ID and exact authority/artifact roots."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--authority-record", required=True, type=Path)
    parser.add_argument("--extension-report", required=True, type=Path)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--prototype-output-root", required=True, type=Path)
    parser.add_argument("--extension-output-root", required=True, type=Path)
    return parser


def build_gate_specs(output_root: Path, authority):
    # One implementation owns the fixed grid for both the runner and launcher.
    from .gate_launcher import build_gate_specs as build

    return build(output_root, authority)


def execute_gate_job(*args, **kwargs):
    # Delayed import keeps CLI inspection independent of gate runtime/data code.
    from .gate_runtime import execute_gate_job as execute

    return execute(*args, **kwargs)


def main() -> int:
    args = build_parser().parse_args()
    authority = verify_gate_authority(
        authority_record=args.authority_record,
        extension_report=args.extension_report,
        freeze_record=args.freeze_record,
        prototype_output_root=args.prototype_output_root,
        extension_output_root=args.extension_output_root,
    )
    specs = build_gate_specs(args.output_root, authority)
    selected = [spec for spec in specs if spec.job_id == args.job_id]
    if len(selected) != 1:
        raise ValueError(
            "job-id is not in the authority-bound registered six-job grid"
        )
    spec = selected[0]
    result = execute_gate_job(
        spec,
        prototype_output_root=args.prototype_output_root,
        extension_output_root=args.extension_output_root,
    )
    if not isinstance(result, Mapping):
        raise TypeError("gate runtime result must be a mapping")
    status = result.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("gate runtime result status is missing")
    print(
        json.dumps(
            {"job_id": spec.job_id, "status": status},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
