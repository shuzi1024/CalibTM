"""Outcome-independent Stage-A job identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True, slots=True)
class PlannedJob:
    stage: str
    method: str
    dataset: str
    seed_bundle: int
    job_id: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


_STAGE_METHODS = {
    "fit_acil": ("acil_only",),
    "formal_gate": ("sc_acil_u0", "sc_acil"),
}


def _job_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"sc-acil-v1:job:v1\x00" + encoded).hexdigest()


def planned_jobs(stage: str) -> tuple[PlannedJob, ...]:
    try:
        methods = _STAGE_METHODS[stage]
    except KeyError:
        raise ValueError(f"unknown frozen stage {stage!r}") from None
    rows = []
    for method in methods:
        for dataset in ("abilene", "geant"):
            for seed_bundle in (4, 5, 6):
                payload = {
                    "dataset": dataset,
                    "method": method,
                    "protocol": "sc-acil-v1",
                    "seed_bundle": seed_bundle,
                    "stage": stage,
                }
                rows.append(
                    PlannedJob(
                        stage=stage,
                        method=method,
                        dataset=dataset,
                        seed_bundle=seed_bundle,
                        job_id=_job_id(payload),
                    )
                )
    return tuple(rows)


def resolve_job(stage: str, job_id: str) -> PlannedJob:
    if not isinstance(stage, str) or not isinstance(job_id, str):
        raise TypeError("stage and job_id must be strings")
    matches = tuple(job for job in planned_jobs(stage) if job.job_id == job_id)
    if len(matches) != 1:
        raise ValueError("job_id is not a planned job in the requested stage")
    return matches[0]


__all__ = ["PlannedJob", "planned_jobs", "resolve_job"]

