"""Frozen extension-stage uncertainty analysis for AnchorCV v1.

The sampling unit is deliberately never a flow.  Seed bundles are sampled
with replacement at the outer level.  At the inner level, whole windows are
sampled by a circular block bootstrap, carrying every flow in the window.
One inner draw is reused for every mask and contrast belonging to the sampled
bundle occurrence and dataset.

Only already-validated :class:`~experiments.anchorcv_v1.evaluation.CaseEvidence`
objects are accepted.  This module has no data-loading or path authority.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .evaluation import CaseEvidence


_SCHEMA_VERSION = "anchorcv-v1:extension-uncertainty:v1"
_BUNDLES = (1, 2, 3)
_DATASETS = ("abilene", "geant")
_MASKS = ("random", "internal_block", "two_burst")
_STRUCTURED_MASKS = ("internal_block", "two_burst")
_BIT_GENERATOR = "PCG64DXSM"
_RNG_SEED = 81_001
_BOOTSTRAP_DRAWS = 10_000
_BUNDLE_DRAWS = 3
_BLOCK_LENGTH = 4
_CI_QUANTILES = (0.025, 0.975)
_CI_METHOD = "linear"
_CHUNK_SIZE = 512
_ERROR_FIELDS = (
    "p_error_sum",
    "n_error_sum",
    "hard_error_sum",
    "oracle_error_sum",
    "truth_sum",
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
GridKey = tuple[int, str, str]


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _little_i8(values: object) -> IntArray:
    return np.ascontiguousarray(values, dtype=np.dtype("<i8"))


def _little_f8(values: object) -> FloatArray:
    return np.ascontiguousarray(values, dtype=np.dtype("<f8"))


def _expected_keys() -> set[GridKey]:
    return {
        (bundle, dataset, mask)
        for bundle in _BUNDLES
        for dataset in _DATASETS
        for mask in _MASKS
    }


def _rectangular_layout(
    evidence: CaseEvidence,
    *,
    label: str,
) -> tuple[IntArray, IntArray]:
    windows = np.unique(evidence.window_index)
    flows = np.unique(evidence.flow_index)
    if windows.size < _BLOCK_LENGTH:
        raise ValueError(
            f"{label}: at least {_BLOCK_LENGTH} windows are required for the "
            "frozen circular block bootstrap"
        )
    if flows.size < 1:
        raise ValueError(f"{label}: at least one flow is required")
    if (
        not np.array_equal(flows, np.arange(flows.size, dtype=np.int64))
        or (
            windows.size > 1
            and not np.array_equal(
                np.diff(windows), np.ones(windows.size - 1, dtype=np.int64)
            )
        )
    ):
        raise ValueError(
            f"{label}: evidence must have a row-major rectangular window/flow layout"
        )
    expected_windows = np.repeat(windows, flows.size)
    expected_flows = np.tile(flows, windows.size)
    if not np.array_equal(
        evidence.window_index, expected_windows
    ) or not np.array_equal(evidence.flow_index, expected_flows):
        raise ValueError(
            f"{label}: evidence must have a row-major rectangular window/flow layout"
        )
    return _little_i8(windows), _little_i8(flows)


def _validate_grid(
    evidence_grid: Mapping[GridKey, CaseEvidence],
) -> tuple[
    dict[GridKey, CaseEvidence],
    dict[str, IntArray],
    dict[str, IntArray],
]:
    if not isinstance(evidence_grid, Mapping):
        raise TypeError("evidence_grid must be a mapping")
    actual_keys = set(evidence_grid)
    expected_keys = _expected_keys()
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "evidence_grid must contain the complete 3 x 2 x 3 grid; "
            f"missing={missing!r}, extra={extra!r}"
        )

    validated: dict[GridKey, CaseEvidence] = {}
    dataset_windows: dict[str, IntArray] = {}
    dataset_flows: dict[str, IntArray] = {}
    for bundle in _BUNDLES:
        for dataset in _DATASETS:
            for mask in _MASKS:
                key = (bundle, dataset, mask)
                evidence = evidence_grid[key]
                label = f"bundle={bundle}, dataset={dataset}, mask={mask}"
                if not isinstance(evidence, CaseEvidence):
                    raise TypeError(f"{label}: value must be CaseEvidence")
                windows, flows = _rectangular_layout(evidence, label=label)
                if dataset not in dataset_windows:
                    dataset_windows[dataset] = windows
                    dataset_flows[dataset] = flows
                elif not np.array_equal(
                    windows, dataset_windows[dataset]
                ) or not np.array_equal(flows, dataset_flows[dataset]):
                    raise ValueError(
                        f"{label}: every bundle/mask for a dataset must have "
                        "identical window/flow layout"
                    )
                validated[key] = evidence
    return validated, dataset_windows, dataset_flows


def _window_operands(
    evidence_grid: Mapping[GridKey, CaseEvidence],
    dataset_windows: Mapping[str, IntArray],
    dataset_flows: Mapping[str, IntArray],
) -> dict[str, FloatArray]:
    """Aggregate all flows once, leaving windows as the resampling axis."""

    operands: dict[str, FloatArray] = {}
    for dataset in _DATASETS:
        window_count = int(dataset_windows[dataset].size)
        flow_count = int(dataset_flows[dataset].size)
        values = np.empty(
            (
                len(_BUNDLES),
                len(_MASKS),
                len(_ERROR_FIELDS),
                window_count,
            ),
            dtype=np.float64,
        )
        for bundle_position, bundle in enumerate(_BUNDLES):
            for mask_position, mask in enumerate(_MASKS):
                evidence = evidence_grid[(bundle, dataset, mask)]
                for field_position, field in enumerate(_ERROR_FIELDS):
                    case_values = np.asarray(getattr(evidence, field))
                    values[bundle_position, mask_position, field_position] = (
                        case_values.reshape(window_count, flow_count).sum(
                            axis=1, dtype=np.float64
                        )
                    )
                truth_by_window = values[
                    bundle_position,
                    mask_position,
                    _ERROR_FIELDS.index("truth_sum"),
                ]
                if math.fsum(float(value) for value in truth_by_window) <= 0.0:
                    raise ValueError(
                        f"bundle={bundle}, dataset={dataset}, mask={mask}: "
                        "pooled truth denominator must be strictly positive"
                    )
                p_total = math.fsum(
                    float(value)
                    for value in values[bundle_position, mask_position, 0]
                )
                n_total = math.fsum(
                    float(value)
                    for value in values[bundle_position, mask_position, 1]
                )
                if min(p_total, n_total) <= 0.0:
                    raise ValueError(
                        f"bundle={bundle}, dataset={dataset}, mask={mask}: "
                        "best-single error denominator must be strictly positive"
                    )
        values.setflags(write=False)
        operands[dataset] = values
    return operands


def _relative_effect(
    p_error: FloatArray | float,
    n_error: FloatArray | float,
    candidate_error: FloatArray | float,
    truth_sum: FloatArray | float,
    *,
    label: str,
) -> FloatArray:
    p = np.asarray(p_error, dtype=np.float64)
    n = np.asarray(n_error, dtype=np.float64)
    candidate = np.asarray(candidate_error, dtype=np.float64)
    truth = np.asarray(truth_sum, dtype=np.float64)
    if (
        np.any(~np.isfinite(p))
        or np.any(~np.isfinite(n))
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(truth))
        or np.any(truth <= 0.0)
    ):
        raise ValueError(f"{label}: sampled ratio-of-sums operands are invalid")
    p_nmae = p / truth
    n_nmae = n / truth
    candidate_nmae = candidate / truth
    best = np.minimum(p_nmae, n_nmae)
    if np.any(best <= 0.0):
        raise ValueError(
            f"{label}: sampled best-single denominator must be strictly positive"
        )
    result = 1.0 - candidate_nmae / best
    return np.asarray(result, dtype=np.float64)


def _point_cell_effects(
    operands: Mapping[str, FloatArray],
    *,
    candidate_field: str,
) -> dict[int, dict[str, dict[str, float]]]:
    candidate_position = _ERROR_FIELDS.index(candidate_field)
    result: dict[int, dict[str, dict[str, float]]] = {}
    for bundle_position, bundle in enumerate(_BUNDLES):
        result[bundle] = {}
        for dataset in _DATASETS:
            result[bundle][dataset] = {}
            for mask_position, mask in enumerate(_MASKS):
                cell = operands[dataset][bundle_position, mask_position]
                effect = _relative_effect(
                    cell[0].sum(dtype=np.float64),
                    cell[1].sum(dtype=np.float64),
                    cell[candidate_position].sum(dtype=np.float64),
                    cell[4].sum(dtype=np.float64),
                    label=(
                        f"bundle={bundle}, dataset={dataset}, mask={mask}, "
                        f"candidate={candidate_field}"
                    ),
                )
                result[bundle][dataset][mask] = float(effect)
    return result


def _paired_bundle_effects(
    cell_effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[int, float]:
    result: dict[int, float] = {}
    for bundle in _BUNDLES:
        structured = [
            float(cell_effects[bundle][dataset][mask])
            for dataset in _DATASETS
            for mask in _STRUCTURED_MASKS
        ]
        result[bundle] = math.fsum(structured) / len(structured)
    return result


def _circular_block_sums(
    values: FloatArray,
    starts: IntArray,
) -> FloatArray:
    """Return sampled sums for every bundle/mask/operand and replicate.

    ``values`` has shape ``[bundle, mask, operand, window]``.  Dense block
    start counts make runtime depend on windows rather than flows, while still
    exactly carrying every flow associated with each sampled window.
    """

    draw_count, block_count = starts.shape
    window_count = values.shape[-1]
    full_blocks, tail_length = divmod(window_count, _BLOCK_LENGTH)
    if block_count != full_blocks + (1 if tail_length else 0):
        raise RuntimeError("circular bootstrap block count drifted")

    flattened_values = values.reshape(-1, window_count)
    full_block_values = np.zeros_like(flattened_values)
    for offset in range(_BLOCK_LENGTH):
        full_block_values += np.roll(flattened_values, -offset, axis=1)
    tail_values: FloatArray | None = None
    if tail_length:
        tail_values = np.zeros_like(flattened_values)
        for offset in range(tail_length):
            tail_values += np.roll(flattened_values, -offset, axis=1)

    sampled = np.empty(
        (draw_count, flattened_values.shape[0]), dtype=np.float64
    )
    for lower in range(0, draw_count, _CHUNK_SIZE):
        upper = min(lower + _CHUNK_SIZE, draw_count)
        chunk_starts = starts[lower:upper]
        rows = upper - lower
        if full_blocks:
            row_offset = (
                np.arange(rows, dtype=np.int64)[:, None] * window_count
            )
            encoded = (
                row_offset + chunk_starts[:, :full_blocks]
            ).reshape(-1)
            counts = np.bincount(
                encoded, minlength=rows * window_count
            ).reshape(rows, window_count)
            sampled_chunk = counts @ full_block_values.T
        else:  # window_count >= block length makes this unreachable
            sampled_chunk = np.zeros(
                (rows, flattened_values.shape[0]), dtype=np.float64
            )
        if tail_length:
            assert tail_values is not None
            sampled_chunk += tail_values[
                :, chunk_starts[:, full_blocks]
            ].T
        sampled[lower:upper] = sampled_chunk
    return sampled.reshape(
        draw_count, len(_BUNDLES), len(_MASKS), len(_ERROR_FIELDS)
    )


def _bootstrap(
    operands: Mapping[str, FloatArray],
    dataset_windows: Mapping[str, IntArray],
    draw_plan: Mapping[str, Any],
) -> tuple[FloatArray, FloatArray, str]:
    rng = np.random.Generator(np.random.PCG64DXSM(_RNG_SEED))
    bundle_positions = rng.integers(
        0,
        len(_BUNDLES),
        size=(_BOOTSTRAP_DRAWS, _BUNDLE_DRAWS),
        dtype=np.int64,
    )
    hard_distribution = np.zeros(_BOOTSTRAP_DRAWS, dtype=np.float64)
    oracle_distribution = np.zeros(_BOOTSTRAP_DRAWS, dtype=np.float64)

    plan_hash = hashlib.sha256()
    plan_hash.update(b"anchorcv-v1:two-level-bootstrap-plan:v1\0")
    plan_hash.update(_canonical_json(draw_plan))
    plan_hash.update(b"\0bundle_ids\0")
    plan_hash.update(
        _little_i8(np.asarray(_BUNDLES)[bundle_positions]).tobytes(order="C")
    )
    for dataset in _DATASETS:
        plan_hash.update(f"\0window_ids:{dataset}\0".encode("ascii"))
        plan_hash.update(
            _little_i8(dataset_windows[dataset]).tobytes(order="C")
        )

    row = np.arange(_BOOTSTRAP_DRAWS, dtype=np.int64)
    normalization = float(
        _BUNDLE_DRAWS * len(_DATASETS) * len(_STRUCTURED_MASKS)
    )
    for occurrence in range(_BUNDLE_DRAWS):
        selected_bundle = bundle_positions[:, occurrence]
        for dataset in _DATASETS:
            window_count = int(dataset_windows[dataset].size)
            block_count = math.ceil(window_count / _BLOCK_LENGTH)
            starts = rng.integers(
                0,
                window_count,
                size=(_BOOTSTRAP_DRAWS, block_count),
                dtype=np.int64,
            )
            plan_hash.update(
                (
                    f"\0block_starts:occurrence={occurrence}:"
                    f"dataset={dataset}\0"
                ).encode("ascii")
            )
            plan_hash.update(_little_i8(starts).tobytes(order="C"))
            sampled = _circular_block_sums(operands[dataset], starts)
            for mask in _STRUCTURED_MASKS:
                mask_position = _MASKS.index(mask)
                selected = sampled[row, selected_bundle, mask_position]
                hard_distribution += _relative_effect(
                    selected[:, 0],
                    selected[:, 1],
                    selected[:, 2],
                    selected[:, 4],
                    label=(
                        f"bootstrap occurrence={occurrence}, dataset={dataset}, "
                        f"mask={mask}, candidate=hard"
                    ),
                ) / normalization
                oracle_distribution += _relative_effect(
                    selected[:, 0],
                    selected[:, 1],
                    selected[:, 3],
                    selected[:, 4],
                    label=(
                        f"bootstrap occurrence={occurrence}, dataset={dataset}, "
                        f"mask={mask}, candidate=oracle"
                    ),
                ) / normalization

    return hard_distribution, oracle_distribution, plan_hash.hexdigest()


def _distribution_sha256(
    hard_distribution: FloatArray,
    oracle_distribution: FloatArray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"anchorcv-v1:extension-bootstrap-distributions:v1\0")
    digest.update(b"hard_vs_best_single\0")
    digest.update(_little_f8(hard_distribution).tobytes(order="C"))
    digest.update(b"\0oracle_vs_best_single\0")
    digest.update(_little_f8(oracle_distribution).tobytes(order="C"))
    return digest.hexdigest()


def _interval(
    *,
    contrast: str,
    point_estimate: float,
    distribution: FloatArray,
) -> dict[str, object]:
    lower, upper = np.quantile(
        distribution,
        _CI_QUANTILES,
        method=_CI_METHOD,
    )
    return {
        "contrast": contrast,
        "aggregation": "equal_seed_bundle_then_dataset_equal_structured_mask",
        "point_estimate": float(point_estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_method": "linear_percentile_2.5_97.5",
        "bootstrap_draws": _BOOTSTRAP_DRAWS,
    }


def analyze_extension_uncertainty(
    evidence_grid: Mapping[GridKey, CaseEvidence],
) -> dict[str, object]:
    """Audit the complete extension grid and run the frozen paired bootstrap.

    Parameters
    ----------
    evidence_grid:
        Exact mapping from ``(seed_bundle, dataset, mask_family)`` to verified
        ``CaseEvidence``.  Required keys are bundles 1--3, Abilene and GEANT,
        and all three registered masks.

    Returns
    -------
    dict
        JSON-serializable point estimates, 95% percentile intervals, paired
        bundle effects, positive-bundle count, and deterministic draw hashes.
    """

    grid, dataset_windows, dataset_flows = _validate_grid(evidence_grid)
    operands = _window_operands(grid, dataset_windows, dataset_flows)
    hard_cells = _point_cell_effects(
        operands, candidate_field="hard_error_sum"
    )
    oracle_cells = _point_cell_effects(
        operands, candidate_field="oracle_error_sum"
    )
    hard_bundles = _paired_bundle_effects(hard_cells)
    oracle_bundles = _paired_bundle_effects(oracle_cells)
    hard_point = math.fsum(hard_bundles.values()) / len(_BUNDLES)
    oracle_point = math.fsum(oracle_bundles.values()) / len(_BUNDLES)

    draw_plan: dict[str, object] = {
        "bit_generator": _BIT_GENERATOR,
        "seed": _RNG_SEED,
        "bootstrap_draws": _BOOTSTRAP_DRAWS,
        "bundle_draws_per_replicate": _BUNDLE_DRAWS,
        "window_resampling": "circular_block",
        "window_block_length": _BLOCK_LENGTH,
        "window_draw_unit": "whole_window_all_flows",
        "window_draw_scope": (
            "independent_per_sampled_bundle_occurrence_and_dataset"
        ),
        "window_draw_shared_across": "all_masks_and_contrasts",
        "ci_quantiles": list(_CI_QUANTILES),
        "ci_quantile_method": _CI_METHOD,
        "bundles": list(_BUNDLES),
        "datasets": list(_DATASETS),
        "masks": list(_MASKS),
        "structured_masks": list(_STRUCTURED_MASKS),
        "window_counts": {
            dataset: int(dataset_windows[dataset].size)
            for dataset in _DATASETS
        },
        "flow_counts": {
            dataset: int(dataset_flows[dataset].size)
            for dataset in _DATASETS
        },
    }
    hard_distribution, oracle_distribution, draw_plan_sha256 = _bootstrap(
        operands, dataset_windows, draw_plan
    )

    return {
        "schema_version": _SCHEMA_VERSION,
        "headline": _interval(
            contrast="hard_vs_best_single",
            point_estimate=hard_point,
            distribution=hard_distribution,
        ),
        "diagnostic_oracle_headline": _interval(
            contrast="oracle_vs_best_single",
            point_estimate=oracle_point,
            distribution=oracle_distribution,
        ),
        "per_bundle_paired_effect": {
            str(bundle): float(hard_bundles[bundle]) for bundle in _BUNDLES
        },
        "per_bundle_oracle_paired_effect": {
            str(bundle): float(oracle_bundles[bundle]) for bundle in _BUNDLES
        },
        "per_bundle_cell_effect": {
            str(bundle): hard_cells[bundle] for bundle in _BUNDLES
        },
        "positive_bundle_count": sum(
            hard_bundles[bundle] > 0.0 for bundle in _BUNDLES
        ),
        "bundle_count": len(_BUNDLES),
        "draw_plan": draw_plan,
        "draw_plan_sha256": draw_plan_sha256,
        "bootstrap_distribution_sha256": _distribution_sha256(
            hard_distribution, oracle_distribution
        ),
    }


__all__ = ["analyze_extension_uncertainty"]
