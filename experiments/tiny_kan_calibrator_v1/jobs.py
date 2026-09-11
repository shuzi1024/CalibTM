"""Closed development grid for the one-shot tiny-KAN collision test."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


PROTOCOL_ID = "tiny-kan-calibrator-v1"
DATASETS = ("abilene", "geant")
SEED_BUNDLES = (1, 2, 3)
METHODS = ("mlp_value", "kan_value", "kan_static")
MASK_FAMILIES = ("random", "internal_block", "two_burst")
STRUCTURED_FAMILIES = ("internal_block", "two_burst")
_JOB_DOMAIN = b"tiny-kan-calibrator-v1:job:v1\x00"


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class Job:
    dataset: str
    method: str
    protocol: str
    seed_bundle: int
    stage: str
    job_id: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def registered_job(dataset: str, seed_bundle: int, method: str) -> Job:
    if dataset not in DATASETS:
        raise ValueError("dataset is outside the closed grid")
    if type(seed_bundle) is not int or seed_bundle not in SEED_BUNDLES:
        raise ValueError("seed bundle is outside the closed grid")
    if method not in METHODS:
        raise ValueError("method is outside the closed grid")
    identity = {
        "dataset": dataset,
        "method": method,
        "protocol": PROTOCOL_ID,
        "seed_bundle": seed_bundle,
        "stage": "development_collision",
    }
    job_id = hashlib.sha256(
        _JOB_DOMAIN + canonical_json(identity)
    ).hexdigest()
    return Job(job_id=job_id, **identity)


def expected_jobs() -> tuple[Job, ...]:
    return tuple(
        registered_job(dataset, seed_bundle, method)
        for dataset in DATASETS
        for seed_bundle in SEED_BUNDLES
        for method in METHODS
    )


__all__ = [
    "DATASETS",
    "Job",
    "MASK_FAMILIES",
    "METHODS",
    "PROTOCOL_ID",
    "SEED_BUNDLES",
    "STRUCTURED_FAMILIES",
    "canonical_json",
    "expected_jobs",
    "registered_job",
]
