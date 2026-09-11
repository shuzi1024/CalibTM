"""Manifest-only public job entrypoint; data paths are intentionally absent."""

from __future__ import annotations

from pathlib import Path


def run_planned_job(stage: str, job_id: str, output_root: str | Path):
    """Resolve and execute one frozen job.

    The implementation is installed only after the training/evaluation capsules
    pass their focused tests. Keeping this exact signature is an integrity
    boundary: callers cannot inject data paths, split names, or test overrides.
    """

    from .execution import execute_job

    return execute_job(stage=stage, job_id=job_id, output_root=Path(output_root))


__all__ = ["run_planned_job"]
