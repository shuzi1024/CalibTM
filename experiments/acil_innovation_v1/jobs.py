"""Outcome-independent job identities and mixed-mask schedule."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


FAMILIES = ("random", "internal_block", "two_burst")


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
    "stage0_acil_tune": ("acil",),
    "stage_h": ("truth_q_deepsets",),
    "stage_i": ("local_loo", "global_loo"),
    "full_tune": ("full_u0", "full_scratch", "full_gpt2"),
    "formal_acil": ("acil",),
    "formal_gate": (
        "full_u0",
        "loo_deepsets",
        "full_scratch",
        "full_gpt2",
        "ari_llm",
        "imputeformer",
    ),
}


def _job_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"acil-innovation-v1:job:v1\x00" + encoded).hexdigest()


def planned_jobs(stage: str) -> tuple[PlannedJob, ...]:
    if stage not in _STAGE_METHODS:
        raise ValueError(f"unknown frozen stage {stage!r}")
    seeds = (4, 5, 6) if stage.startswith("formal") else (1, 2, 3)
    jobs = []
    for method in _STAGE_METHODS[stage]:
        for dataset in ("abilene", "geant"):
            for seed in seeds:
                payload = {
                    "dataset": dataset,
                    "method": method,
                    "protocol": "acil-innovation-v1",
                    "seed_bundle": seed,
                    "stage": stage,
                }
                jobs.append(
                    PlannedJob(
                        stage=stage,
                        method=method,
                        dataset=dataset,
                        seed_bundle=seed,
                        job_id=_job_id(payload),
                    )
                )
    return tuple(jobs)


def training_family(*, epoch: int, epoch_order_position: int, seed_bundle: int) -> str:
    for name, value, upper in (
        ("epoch", epoch, 19),
        ("epoch_order_position", epoch_order_position, 511),
        ("seed_bundle", seed_bundle, 6),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        lower = 1 if name == "seed_bundle" else 0
        if not lower <= value <= upper:
            raise ValueError(f"{name} is outside the frozen range")
    global_index = epoch * 512 + epoch_order_position
    return FAMILIES[(global_index + seed_bundle - 1) % len(FAMILIES)]


__all__ = ["FAMILIES", "PlannedJob", "planned_jobs", "training_family"]
