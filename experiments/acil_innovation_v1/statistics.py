"""Paired hierarchical bootstrap over registered seed/window clusters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


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


def _window_table(records, methods):
    grouped: dict[tuple[str, int, str, str, int], list[tuple[float, float]]] = {}
    for record in records:
        identity = record["identity"] if isinstance(record, Mapping) else record.identity
        method = str(identity["method"])
        if method not in methods:
            continue
        key = (
            method,
            int(identity["seed_bundle"]),
            str(identity["dataset"]),
            str(identity["mask_family"]),
            int(identity["window_start"]),
        )
        getter = record.__getitem__ if isinstance(record, Mapping) else lambda name: getattr(record, name)
        grouped.setdefault(key, []).append(
            (float(getter("absolute_error_sum")), float(getter("absolute_truth_sum")))
        )
    return {
        key: (math.fsum(value[0] for value in rows), math.fsum(value[1] for value in rows))
        for key, rows in grouped.items()
    }


def _effect(table, candidate, comparator, selections):
    sums = {}
    for method in (candidate, comparator):
        errors = []
        truths = []
        for seed, dataset, mask, start in selections:
            error, truth = table[(method, seed, dataset, mask, start)]
            errors.append(error)
            truths.append(truth)
        error_sum = math.fsum(errors)
        truth_sum = math.fsum(truths)
        if truth_sum == 0.0:
            raise ValueError("bootstrap pooled truth denominator is zero")
        sums[method] = error_sum / truth_sum
    if sums[comparator] == 0.0:
        raise ValueError("bootstrap comparator NMAE is zero")
    return 1.0 - sums[candidate] / sums[comparator]


def paired_bootstrap(
    *,
    records: Sequence[object],
    candidate: str,
    comparator: str,
    seed_bundles: tuple[int, ...],
    datasets: Mapping[str, tuple[int, ...]],
    mask_families: tuple[str, ...],
    draws: int,
    bootstrap_seed: int,
    block_length: int,
) -> BootstrapResult:
    if not records or not candidate or not comparator or candidate == comparator:
        raise ValueError("paired bootstrap requires records and two distinct methods")
    if draws < 1 or block_length < 1:
        raise ValueError("draws and block length must be positive")
    table = _window_table(records, {candidate, comparator})
    expected = [
        (method, seed, dataset, mask, start)
        for method in (candidate, comparator)
        for seed in seed_bundles
        for dataset, starts in datasets.items()
        for mask in mask_families
        for start in starts
    ]
    missing = [key for key in expected if key not in table]
    if missing:
        raise ValueError(f"missing paired window record: {missing[0]!r}")

    canonical = [
        (seed, dataset, mask, start)
        for seed in seed_bundles
        for dataset, starts in datasets.items()
        for mask in mask_families
        for start in starts
    ]
    point = _effect(table, candidate, comparator, canonical)
    rng = np.random.Generator(np.random.PCG64DXSM(int(bootstrap_seed)))
    effects = np.empty(draws, dtype="<f8")
    plan_hash = hashlib.sha256(b"acil-innovation-v1:bootstrap-plan:v1\x00")
    for draw_index in range(draws):
        seed_indices = rng.integers(0, len(seed_bundles), size=len(seed_bundles))
        plan_hash.update(draw_index.to_bytes(8, "big"))
        plan_hash.update(seed_indices.astype("<i8", copy=False).tobytes())
        selections = []
        for occurrence, seed_index in enumerate(seed_indices):
            seed = seed_bundles[int(seed_index)]
            for dataset, starts in datasets.items():
                count = len(starts)
                origins = rng.integers(0, count, size=math.ceil(count / block_length))
                positions = np.concatenate(
                    [
                        (int(origin) + np.arange(block_length, dtype=np.int64)) % count
                        for origin in origins
                    ]
                )[:count]
                encoded = positions.astype("<i8", copy=False)
                plan_hash.update(occurrence.to_bytes(8, "big"))
                plan_hash.update(dataset.encode("ascii"))
                plan_hash.update(encoded.tobytes())
                for mask in mask_families:
                    for position in positions:
                        selections.append((seed, dataset, mask, starts[int(position)]))
        effects[draw_index] = _effect(table, candidate, comparator, selections)
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


__all__ = ["BootstrapResult", "paired_bootstrap"]
