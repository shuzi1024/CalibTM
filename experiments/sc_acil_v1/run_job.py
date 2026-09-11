"""Manifest-only public entrypoint; no data or split override exists."""

from __future__ import annotations

from pathlib import Path


def run_planned_job(stage: str, job_id: str, output_root: str | Path):
    from .execution import execute_job

    return execute_job(stage=stage, job_id=job_id, output_root=Path(output_root))


__all__ = ["run_planned_job"]

