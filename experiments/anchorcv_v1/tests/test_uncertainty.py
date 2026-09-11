from __future__ import annotations

import copy
import re

import numpy as np
import pytest

from experiments.anchorcv_v1.evaluation import CaseEvidence
from experiments.anchorcv_v1.uncertainty import analyze_extension_uncertainty


_DATASETS = ("abilene", "geant")
_MASKS = ("random", "internal_block", "two_burst")


def _case_evidence(*, routes: list[str], scale: float = 1.0) -> CaseEvidence:
    """Build two-flow windows with P/N tied globally and controlled routing."""

    if not routes:
        raise ValueError("routes must be nonempty")
    window_count = len(routes)
    window_index = np.repeat(np.arange(window_count, dtype=np.int64), 2)
    flow_index = np.tile(np.arange(2, dtype=np.int64), window_count)
    p_error = np.tile(np.array([10.0, 5.0]), window_count) * scale
    n_error = np.tile(np.array([5.0, 10.0]), window_count) * scale
    truth_sum = np.full(window_count * 2, 100.0 * scale)
    target_count = np.ones(window_count * 2, dtype=np.int64)
    k3_scale = np.ones(window_count * 2)

    winners: list[bool] = []
    for route in routes:
        if route == "correct":
            winners.extend((True, False))
        elif route == "neutral":
            winners.extend((True, True))
        elif route == "wrong":
            winners.extend((False, True))
        else:
            raise ValueError(f"unknown route {route!r}")
    hard_winner = np.asarray(winners, dtype=np.bool_)
    oracle_winner = p_error > n_error
    hard_error = np.where(hard_winner, n_error, p_error)
    oracle_error = np.minimum(p_error, n_error)
    p_anchor_error = np.where(hard_winner, 2.0, 1.0)
    n_anchor_error = np.where(hard_winner, 1.0, 2.0)
    loo_score = p_anchor_error - n_anchor_error
    target_regret = p_error - n_error

    return CaseEvidence(
        window_index=window_index,
        flow_index=flow_index,
        middle_anchor_index=np.full(window_count * 2, 3, dtype=np.int64),
        p_error_sum=p_error,
        n_error_sum=n_error,
        hard_error_sum=hard_error,
        oracle_error_sum=oracle_error,
        truth_sum=truth_sum,
        target_count=target_count,
        k3_scale=k3_scale,
        p_anchor_absolute_error=p_anchor_error,
        n_anchor_absolute_error=n_anchor_error,
        p_anchor_normalized_error=p_anchor_error,
        n_anchor_normalized_error=n_anchor_error,
        loo_score=loo_score,
        target_regret=target_regret,
        hard_winner_is_n=hard_winner,
        oracle_winner_is_n=oracle_winner,
    )


def _grid(
    routes_by_bundle: dict[int, list[str]] | None = None,
) -> dict[tuple[int, str, str], CaseEvidence]:
    routes_by_bundle = routes_by_bundle or {
        1: ["correct"] * 4,
        2: ["neutral"] * 4,
        3: ["wrong"] * 4,
    }
    return {
        (bundle, dataset, mask): _case_evidence(
            routes=routes_by_bundle[bundle],
            scale=1.0 if dataset == "abilene" else 2.0,
        )
        for bundle in (1, 2, 3)
        for dataset in _DATASETS
        for mask in _MASKS
    }


def test_frozen_two_level_bootstrap_reports_paired_bundle_evidence() -> None:
    report = analyze_extension_uncertainty(_grid())

    assert report["schema_version"] == "anchorcv-v1:extension-uncertainty:v1"
    assert report["headline"]["contrast"] == "hard_vs_best_single"
    assert report["headline"]["aggregation"] == (
        "equal_seed_bundle_then_dataset_equal_structured_mask"
    )
    assert report["headline"]["point_estimate"] == pytest.approx(0.0)
    assert report["headline"]["ci_lower"] == pytest.approx(-1.0 / 3.0)
    assert report["headline"]["ci_upper"] == pytest.approx(1.0 / 3.0)
    assert report["headline"]["ci_method"] == "linear_percentile_2.5_97.5"
    assert report["per_bundle_paired_effect"] == pytest.approx(
        {"1": 1.0 / 3.0, "2": 0.0, "3": -1.0 / 3.0}
    )
    assert report["positive_bundle_count"] == 1
    assert report["bundle_count"] == 3

    assert report["draw_plan"] == {
        "bit_generator": "PCG64DXSM",
        "seed": 81001,
        "bootstrap_draws": 10_000,
        "bundle_draws_per_replicate": 3,
        "window_resampling": "circular_block",
        "window_block_length": 4,
        "window_draw_unit": "whole_window_all_flows",
        "window_draw_scope": "independent_per_sampled_bundle_occurrence_and_dataset",
        "window_draw_shared_across": "all_masks_and_contrasts",
        "ci_quantiles": [0.025, 0.975],
        "ci_quantile_method": "linear",
        "bundles": [1, 2, 3],
        "datasets": ["abilene", "geant"],
        "masks": ["random", "internal_block", "two_burst"],
        "structured_masks": ["internal_block", "two_burst"],
        "window_counts": {"abilene": 4, "geant": 4},
        "flow_counts": {"abilene": 2, "geant": 2},
    }
    assert re.fullmatch(r"[0-9a-f]{64}", report["draw_plan_sha256"])
    assert re.fullmatch(
        r"[0-9a-f]{64}", report["bootstrap_distribution_sha256"]
    )


def test_bootstrap_is_bitwise_reproducible_and_does_not_mutate_evidence() -> None:
    grid = _grid(
        {
            1: ["correct", "wrong", "correct", "neutral", "wrong"],
            2: ["neutral", "correct", "wrong", "neutral", "correct"],
            3: ["wrong", "neutral", "correct", "wrong", "neutral"],
        }
    )
    snapshots = {
        key: {
            name: np.array(getattr(evidence, name), copy=True)
            for name in evidence.__dataclass_fields__
        }
        for key, evidence in grid.items()
    }

    first = analyze_extension_uncertainty(grid)
    second = analyze_extension_uncertainty(grid)

    assert first == second
    for key, evidence in grid.items():
        for name, expected in snapshots[key].items():
            np.testing.assert_array_equal(getattr(evidence, name), expected)


def test_window_draw_is_shared_across_masks_and_never_samples_flows_iid() -> None:
    grid = _grid({bundle: ["neutral"] * 6 for bundle in (1, 2, 3)})
    alternating = ["correct", "wrong", "correct", "wrong", "correct", "wrong"]
    opposite = ["wrong", "correct", "wrong", "correct", "wrong", "correct"]
    for bundle in (1, 2, 3):
        for dataset in _DATASETS:
            grid[(bundle, dataset, "internal_block")] = _case_evidence(
                routes=alternating
            )
            grid[(bundle, dataset, "two_burst")] = _case_evidence(routes=opposite)

    report = analyze_extension_uncertainty(grid)

    # A shared whole-window draw makes the two complementary structured masks
    # cancel in every bootstrap replicate. Independent mask or flow draws do not.
    assert report["headline"]["point_estimate"] == pytest.approx(0.0)
    assert report["headline"]["ci_lower"] == pytest.approx(0.0, abs=1e-15)
    assert report["headline"]["ci_upper"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_cell", "complete 3 x 2 x 3 grid"),
        ("extra_cell", "complete 3 x 2 x 3 grid"),
        ("wrong_order", "row-major rectangular window/flow layout"),
        ("missing_flow", "row-major rectangular window/flow layout"),
        ("layout_drift", "identical window/flow layout"),
    ],
)
def test_grid_and_window_flow_layout_are_strictly_validated(
    mutation: str,
    match: str,
) -> None:
    grid = _grid()
    key = (3, "geant", "two_burst")
    if mutation == "missing_cell":
        del grid[key]
    elif mutation == "extra_cell":
        grid[(4, "geant", "two_burst")] = grid[key]
    elif mutation == "wrong_order":
        evidence = grid[key]
        order = np.arange(evidence.case_count)
        order[[0, 1]] = order[[1, 0]]
        grid[key] = CaseEvidence(
            **{
                name: np.asarray(getattr(evidence, name))[order]
                for name in evidence.__dataclass_fields__
            }
        )
    elif mutation == "missing_flow":
        evidence = grid[key]
        keep = np.arange(evidence.case_count) != 1
        grid[key] = CaseEvidence(
            **{
                name: np.asarray(getattr(evidence, name))[keep]
                for name in evidence.__dataclass_fields__
            }
        )
    elif mutation == "layout_drift":
        evidence = grid[key]
        changed = {
            name: np.array(getattr(evidence, name), copy=True)
            for name in evidence.__dataclass_fields__
        }
        changed["window_index"] += 1
        grid[key] = CaseEvidence(**changed)
    else:  # pragma: no cover - guards this test table
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=match):
        analyze_extension_uncertainty(grid)


def test_input_must_be_a_mapping_of_case_evidence() -> None:
    with pytest.raises(TypeError, match="mapping"):
        analyze_extension_uncertainty([])  # type: ignore[arg-type]

    grid = _grid()
    bad = copy.copy(grid)
    bad[(1, "abilene", "random")] = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="CaseEvidence"):
        analyze_extension_uncertainty(bad)
