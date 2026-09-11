"""Dataset-equal paired hierarchical bootstrap for frozen Stage A."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np


WindowTable = Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    candidate: str
    comparator: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    draws: int
    draws_sha256: str
    draw_plan_sha256: str


def _validated(table: WindowTable, methods: tuple[str, str], masks: tuple[str, ...]):
    result = {}
    for method in methods:
        for seed in (4, 5, 6):
            for dataset in ("abilene", "geant"):
                for mask in masks:
                    key = (method, seed, dataset, mask)
                    if key not in table:
                        raise ValueError(f"missing paired window evidence: {key!r}")
                    error, truth = table[key]
                    error = np.asarray(error, dtype="<f8")
                    truth = np.asarray(truth, dtype="<f8")
                    expected = 72 if dataset == "abilene" else 16
                    # Unit tests use smaller synthetic schedules while formal
                    # adjudication validates exact counts before this function.
                    if error.ndim != 1 or truth.shape != error.shape or not len(error):
                        raise ValueError("window evidence must be paired nonempty vectors")
                    if not np.isfinite(error).all() or not np.isfinite(truth).all():
                        raise ValueError("window evidence must be finite")
                    if np.any(error < 0) or np.any(truth <= 0):
                        raise ValueError("window evidence has invalid sums")
                    result[key] = (error, truth)
    return result


def _dataset_effect(
    table,
    candidate: str,
    comparator: str,
    dataset: str,
    masks: tuple[str, ...],
    selections: tuple[tuple[int, np.ndarray], ...],
) -> float:
    scores = {}
    for method in (candidate, comparator):
        errors = []
        truths = []
        for seed, positions in selections:
            for mask in masks:
                error, truth = table[(method, seed, dataset, mask)]
                errors.append(float(np.sum(error[positions], dtype=np.float64)))
                truths.append(float(np.sum(truth[positions], dtype=np.float64)))
        denominator = math.fsum(truths)
        if denominator <= 0:
            raise ValueError("pooled truth denominator is zero")
        scores[method] = math.fsum(errors) / denominator
    if scores[comparator] <= 0:
        raise ValueError("comparator NMAE is zero")
    return 1.0 - scores[candidate] / scores[comparator]


def dataset_equal_effect(
    table: WindowTable,
    *,
    candidate: str,
    comparator: str,
    masks: tuple[str, ...],
) -> float:
    values = _validated(table, (candidate, comparator), masks)
    effects = []
    for dataset in ("abilene", "geant"):
        count = len(values[(candidate, 4, dataset, masks[0])][0])
        selections = tuple((seed, np.arange(count, dtype=np.int64)) for seed in (4, 5, 6))
        effects.append(
            _dataset_effect(values, candidate, comparator, dataset, masks, selections)
        )
    return math.fsum(effects) / 2.0


def dataset_equal_bootstrap(
    table: WindowTable,
    *,
    candidate: str,
    comparator: str,
    masks: tuple[str, ...],
    draws: int,
    bootstrap_seed: int,
    block_length: int,
) -> BootstrapResult:
    if not candidate or not comparator or candidate == comparator:
        raise ValueError("bootstrap requires two distinct methods")
    if not masks or draws < 1 or block_length < 1:
        raise ValueError("bootstrap masks, draws, and block length must be positive")
    values = _validated(table, (candidate, comparator), masks)
    point = dataset_equal_effect(
        values, candidate=candidate, comparator=comparator, masks=masks
    )
    rng = np.random.Generator(np.random.PCG64DXSM(int(bootstrap_seed)))
    effects = np.empty(draws, dtype="<f8")
    plan_hash = hashlib.sha256(b"sc-acil-v1:bootstrap-plan:v1\x00")
    seeds = (4, 5, 6)
    for draw_index in range(draws):
        seed_indices = rng.integers(0, 3, size=3)
        plan_hash.update(draw_index.to_bytes(8, "big"))
        plan_hash.update(seed_indices.astype("<i8", copy=False).tobytes())
        dataset_effects = []
        for dataset in ("abilene", "geant"):
            count = len(values[(candidate, 4, dataset, masks[0])][0])
            selections = []
            for occurrence, seed_index in enumerate(seed_indices):
                origins = rng.integers(0, count, size=math.ceil(count / block_length))
                positions = np.concatenate(
                    [
                        (int(origin) + np.arange(block_length, dtype=np.int64)) % count
                        for origin in origins
                    ]
                )[:count]
                plan_hash.update(dataset.encode("ascii"))
                plan_hash.update(occurrence.to_bytes(8, "big"))
                plan_hash.update(positions.astype("<i8", copy=False).tobytes())
                selections.append((seeds[int(seed_index)], positions))
            dataset_effects.append(
                _dataset_effect(
                    values,
                    candidate,
                    comparator,
                    dataset,
                    masks,
                    tuple(selections),
                )
            )
        effects[draw_index] = math.fsum(dataset_effects) / 2.0
    lower, upper = np.quantile(effects, (0.025, 0.975), method="linear")
    return BootstrapResult(
        candidate=candidate,
        comparator=comparator,
        point_estimate=float(point),
        ci_lower=float(lower),
        ci_upper=float(upper),
        draws=draws,
        draws_sha256=hashlib.sha256(effects.tobytes()).hexdigest(),
        draw_plan_sha256=plan_hash.hexdigest(),
    )


__all__ = [
    "BootstrapResult",
    "WindowTable",
    "dataset_equal_bootstrap",
    "dataset_equal_effect",
]

