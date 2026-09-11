"""Run one branch-sealed AnchorCV job from an independently verified freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .job import JobSpec
from .job_runtime import execute_job
from .protocol import fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("abilene", "geant"))
    parser.add_argument("--seed-bundle", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--stage", required=True, choices=("prototype", "extension"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--freeze-record", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # Delayed import keeps parser inspection independent from filesystem work.
    from .freeze import verify_freeze_record

    freeze = verify_freeze_record(args.freeze_record)
    if freeze.get("config_sha256") != fingerprint():
        raise RuntimeError("verified freeze config differs from the active protocol")
    spec = JobSpec(
        dataset=args.dataset,
        seed_bundle=args.seed_bundle,
        stage=args.stage,
        output_root=args.output_root,
        device="cuda",
        source_tree_sha256=str(freeze["source_tree_sha256"]),
        config_sha256=str(freeze["config_sha256"]),
    )
    result = execute_job(spec)
    print(
        json.dumps(
            {"job_id": spec.job_id, "status": result["status"]},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
