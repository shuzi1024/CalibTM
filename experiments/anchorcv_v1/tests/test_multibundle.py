from __future__ import annotations

import copy

import pytest

from experiments.anchorcv_v1.multibundle import adjudicate_multibundle


BUNDLES = (1, 2, 3)
DATASETS = ("abilene", "geant")
MASKS = ("random", "internal_block", "two_burst")
STRUCTURED_MASKS = ("internal_block", "two_burst")


def _summary(
    *,
    oracle: float,
    hard: float,
    auroc: float = 0.65,
    spearman: float = 0.20,
) -> dict[str, object]:
    truth = 10_000.0
    best = 0.20
    p_nmae = best
    n_nmae = 0.21
    oracle_nmae = best * (1.0 - oracle)
    hard_nmae = best * (1.0 - hard)
    gap = best - oracle_nmae
    capture = (best - hard_nmae) / gap
    return {
        "case_count": 100,
        "target_count": 4700,
        "absolute_truth_sum": truth,
        "p_absolute_error_sum": p_nmae * truth,
        "n_absolute_error_sum": n_nmae * truth,
        "hard_absolute_error_sum": hard_nmae * truth,
        "oracle_absolute_error_sum": oracle_nmae * truth,
        "p_nmae": p_nmae,
        "n_nmae": n_nmae,
        "hard_nmae": hard_nmae,
        "oracle_nmae": oracle_nmae,
        "best_single_expert": "p",
        "best_single_nmae": best,
        "oracle_relative_improvement": oracle,
        "hard_relative_improvement": hard,
        "neural_selection_rate": 0.5,
        "oracle_neural_rate": 0.5,
        "loo_winner_auroc": auroc,
        "loo_target_regret_spearman": spearman,
        "oracle_gap_capture": capture,
    }


def _grid() -> dict[tuple[int, str, str], dict[str, object]]:
    effects = {1: 0.01, 2: 0.02, 3: 0.03}
    return {
        (bundle, dataset, mask): _summary(
            oracle=0.01 if mask == "random" else 0.05,
            hard=-0.01 if mask == "random" else effects[bundle],
        )
        for bundle in BUNDLES
        for dataset in DATASETS
        for mask in MASKS
    }


def _interval(
    contrast: str,
    point: float,
    lower: float,
    upper: float,
) -> dict[str, object]:
    return {
        "contrast": contrast,
        "aggregation": "equal_seed_bundle_then_dataset_equal_structured_mask",
        "point_estimate": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_method": "linear_percentile_2.5_97.5",
        "bootstrap_draws": 10_000,
    }


def _uncertainty(
    grid: dict[tuple[int, str, str], dict[str, object]],
) -> dict[str, object]:
    hard_cells = {
        str(bundle): {
            dataset: {
                mask: grid[(bundle, dataset, mask)][
                    "hard_relative_improvement"
                ]
                for mask in MASKS
            }
            for dataset in DATASETS
        }
        for bundle in BUNDLES
    }
    oracle_cells = {
        bundle: [
            float(
                grid[(bundle, dataset, mask)][
                    "oracle_relative_improvement"
                ]
            )
            for dataset in DATASETS
            for mask in STRUCTURED_MASKS
        ]
        for bundle in BUNDLES
    }
    hard_bundles = {
        str(bundle): sum(
            float(
                grid[(bundle, dataset, mask)][
                    "hard_relative_improvement"
                ]
            )
            for dataset in DATASETS
            for mask in STRUCTURED_MASKS
        )
        / 4.0
        for bundle in BUNDLES
    }
    oracle_bundles = {
        str(bundle): sum(oracle_cells[bundle]) / 4.0 for bundle in BUNDLES
    }
    hard_point = sum(hard_bundles.values()) / 3.0
    oracle_point = sum(oracle_bundles.values()) / 3.0
    return {
        "schema_version": "anchorcv-v1:extension-uncertainty:v1",
        "headline": _interval(
            "hard_vs_best_single", hard_point, 0.005, 0.035
        ),
        "diagnostic_oracle_headline": _interval(
            "oracle_vs_best_single", oracle_point, 0.035, 0.065
        ),
        "per_bundle_paired_effect": hard_bundles,
        "per_bundle_oracle_paired_effect": oracle_bundles,
        "per_bundle_cell_effect": hard_cells,
        "positive_bundle_count": sum(
            value > 0.0 for value in hard_bundles.values()
        ),
        "bundle_count": 3,
        "draw_plan": {
            "bit_generator": "PCG64DXSM",
            "seed": 81_001,
            "bootstrap_draws": 10_000,
            "bundle_draws_per_replicate": 3,
            "window_resampling": "circular_block",
            "window_block_length": 4,
            "window_draw_unit": "whole_window_all_flows",
            "window_draw_scope": (
                "independent_per_sampled_bundle_occurrence_and_dataset"
            ),
            "window_draw_shared_across": "all_masks_and_contrasts",
            "ci_quantiles": [0.025, 0.975],
            "ci_quantile_method": "linear",
            "bundles": [1, 2, 3],
            "datasets": ["abilene", "geant"],
            "masks": ["random", "internal_block", "two_burst"],
            "structured_masks": ["internal_block", "two_burst"],
            "window_counts": {"abilene": 10, "geant": 10},
            "flow_counts": {"abilene": 144, "geant": 462},
        },
        "draw_plan_sha256": "a" * 64,
        "bootstrap_distribution_sha256": "b" * 64,
    }


def _passing_inputs() -> tuple[
    dict[tuple[int, str, str], dict[str, object]],
    dict[str, object],
]:
    grid = _grid()
    return grid, _uncertainty(grid)


def test_complete_grid_passes_with_bundle_then_four_cell_equal_aggregation() -> None:
    grid, uncertainty = _passing_inputs()
    report = adjudicate_multibundle(grid, uncertainty)

    assert report["schema"] == "anchorcv-v1:multibundle-adjudication:v1"
    assert report["verdict"] == "proceed"
    assert report["reason"] == "all_scientific_gates_pass"
    assert report["grid"]["complete"] is True
    assert report["grid"]["expected_cell_count"] == 18
    assert report["headline"]["cell_count_after_bundle_average"] == 4
    assert report["headline"]["metrics"]["hard_relative_improvement"][
        "actual"
    ] == pytest.approx(0.02)
    assert report["headline"]["worst_actual_cell"]["actual"] == pytest.approx(
        0.02
    )
    assert report["per_bundle_paired_effect"] == pytest.approx(
        {"1": 0.01, "2": 0.02, "3": 0.03}
    )
    assert all(gate["pass"] for gate in report["gates"].values())
    assert len(report["non_headline_random"]) == 6


def test_random_is_report_only_even_when_it_is_poor() -> None:
    grid, _ = _passing_inputs()
    for bundle in BUNDLES:
        for dataset in DATASETS:
            grid[(bundle, dataset, "random")] = _summary(
                oracle=0.001,
                hard=-4.0,
                auroc=0.0,
                spearman=-1.0,
            )
    uncertainty = _uncertainty(grid)

    report = adjudicate_multibundle(grid, uncertainty)

    assert report["verdict"] == "proceed"
    assert all(
        row["hard_relative_improvement"] == -4.0
        for row in report["non_headline_random"].values()
    )


def test_schema_valid_undefined_random_diagnostic_remains_report_only() -> None:
    grid, _ = _passing_inputs()
    cell = grid[(1, "abilene", "random")]
    cell["loo_winner_auroc"] = None
    cell["loo_winner_auroc_undefined_reason"] = (
        "AUROC requires both classes"
    )
    uncertainty = _uncertainty(grid)

    report = adjudicate_multibundle(grid, uncertainty)

    assert report["integrity"] == {"valid": True, "issues": []}
    assert report["verdict"] == "proceed"
    assert (
        report["non_headline_random"]["bundle1/abilene"][
            "loo_winner_auroc"
        ]
        is None
    )


@pytest.mark.parametrize(
    ("gate_name", "metric", "value"),
    [
        (
            "structured_per_flow_oracle_dataset_equal_min",
            "oracle",
            0.029,
        ),
        ("loo_winner_auroc_min", "auroc", 0.59),
        ("loo_target_regret_spearman_min", "spearman", 0.14),
        (
            "hard_anchorcv_oracle_gap_capture_min",
            "oracle_and_hard",
            (0.05, 0.014),
        ),
        (
            "hard_anchorcv_actual_improvement_min",
            "hard",
            0.004,
        ),
        (
            "worst_structured_oracle_cell_relative_improvement_min",
            "one_oracle",
            0.014,
        ),
        (
            "worst_structured_cell_relative_improvement_min",
            "one_hard",
            -0.011,
        ),
    ],
)
def test_each_reused_protocol_gate_can_kill(
    gate_name: str,
    metric: str,
    value: object,
) -> None:
    grid, _ = _passing_inputs()
    for bundle in BUNDLES:
        for dataset in DATASETS:
            for mask in STRUCTURED_MASKS:
                old = grid[(bundle, dataset, mask)]
                oracle = float(old["oracle_relative_improvement"])
                hard = float(old["hard_relative_improvement"])
                auroc = float(old["loo_winner_auroc"])
                spearman = float(old["loo_target_regret_spearman"])
                if metric == "oracle":
                    oracle = float(value)
                    hard = min(hard, oracle)
                elif metric == "hard":
                    hard = float(value)
                elif metric == "auroc":
                    auroc = float(value)
                elif metric == "spearman":
                    spearman = float(value)
                elif metric == "oracle_and_hard":
                    oracle, hard = value  # type: ignore[misc]
                elif metric == "one_oracle" and (
                    dataset,
                    mask,
                ) == ("abilene", "internal_block"):
                    oracle = float(value)
                    hard = min(hard, oracle)
                elif metric == "one_hard" and (
                    dataset,
                    mask,
                ) == ("abilene", "internal_block"):
                    hard = float(value)
                grid[(bundle, dataset, mask)] = _summary(
                    oracle=oracle,
                    hard=hard,
                    auroc=auroc,
                    spearman=spearman,
                )
    uncertainty = _uncertainty(grid)
    report = adjudicate_multibundle(grid, uncertainty)

    assert report["verdict"] == "kill"
    assert report["reason"] == "scientific_gate_failure"
    assert report["gates"][gate_name]["pass"] is False
    assert report["scientific_failure_reasons"]


@pytest.mark.parametrize("uncertainty_gate", ["ci", "positive_bundles"])
def test_uncertainty_replication_gates_can_kill(uncertainty_gate: str) -> None:
    grid, uncertainty = _passing_inputs()
    if uncertainty_gate == "ci":
        uncertainty["headline"]["ci_lower"] = 0.0
    else:
        for bundle in (1, 2):
            for dataset in DATASETS:
                for mask in STRUCTURED_MASKS:
                    grid[(bundle, dataset, mask)] = _summary(
                        oracle=0.05, hard=-0.01
                    )
        uncertainty = _uncertainty(grid)

    report = adjudicate_multibundle(grid, uncertainty)

    assert report["verdict"] == "kill"
    expected_gate = (
        "uncertainty_headline_ci_lower_gt_zero"
        if uncertainty_gate == "ci"
        else "positive_bundle_count_min"
    )
    assert report["gates"][expected_gate]["pass"] is False


@pytest.mark.parametrize(
    ("metric", "reason_field", "reason"),
    [
        (
            "loo_winner_auroc",
            "loo_winner_auroc_undefined_reason",
            "AUROC requires both classes",
        ),
        (
            "loo_target_regret_spearman",
            "loo_target_regret_spearman_undefined_reason",
            "Spearman correlation is undefined for constant input",
        ),
        (
            "oracle_gap_capture",
            "oracle_gap_capture_undefined_reason",
            "oracle gap must be positive",
        ),
    ],
)
def test_schema_valid_undefined_diagnostic_is_scientific_kill(
    metric: str,
    reason_field: str,
    reason: str,
) -> None:
    grid, _ = _passing_inputs()
    key = (1, "abilene", "internal_block")
    if metric == "oracle_gap_capture":
        grid[key] = _summary(oracle=0.05, hard=0.0)
        grid[key]["oracle_nmae"] = 0.20
        grid[key]["oracle_absolute_error_sum"] = 2_000.0
        grid[key]["oracle_relative_improvement"] = 0.0
    grid[key][metric] = None
    grid[key][reason_field] = reason
    uncertainty = _uncertainty(grid)

    report = adjudicate_multibundle(grid, uncertainty)

    assert report["integrity"] == {"valid": True, "issues": []}
    assert report["verdict"] == "kill"
    assert report["reason"] == "scientific_gate_failure"
    metric_report = report["headline"]["metrics"][metric]
    assert metric_report["actual"] is None
    assert metric_report["undefined_cells"] == [
        "abilene/internal_block"
    ]
    gate_name = {
        "loo_winner_auroc": "loo_winner_auroc_min",
        "loo_target_regret_spearman": (
            "loo_target_regret_spearman_min"
        ),
        "oracle_gap_capture": "hard_anchorcv_oracle_gap_capture_min",
    }[metric]
    assert report["gates"][gate_name]["actual"] is None
    assert report["gates"][gate_name]["undefined_cells"] == [
        "abilene/internal_block"
    ]
    assert report["gates"][gate_name]["pass"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "missing_cell",
        "extra_summary_field",
        "undefined_metric",
        "undefined_reason_nonstring",
        "defined_metric_with_reason",
        "required_numeric_none",
        "required_numeric_nonfinite",
        "nonfinite_metric",
        "missing_uncertainty_field",
        "bad_hash",
        "uncertainty_schema",
        "effect_mismatch",
        "positive_count_mismatch",
    ],
)
def test_grid_schema_undefined_hash_and_report_faults_are_revise(
    fault: str,
) -> None:
    grid, uncertainty = _passing_inputs()
    if fault == "missing_cell":
        grid.pop((3, "geant", "two_burst"))
    elif fault == "extra_summary_field":
        grid[(1, "abilene", "internal_block")]["unexpected"] = 1
    elif fault == "undefined_metric":
        grid[(1, "abilene", "internal_block")]["loo_winner_auroc"] = None
    elif fault == "undefined_reason_nonstring":
        cell = grid[(1, "abilene", "internal_block")]
        cell["loo_winner_auroc"] = None
        cell["loo_winner_auroc_undefined_reason"] = 7
    elif fault == "defined_metric_with_reason":
        grid[(1, "abilene", "internal_block")][
            "loo_winner_auroc_undefined_reason"
        ] = "must not accompany a finite metric"
    elif fault == "required_numeric_none":
        grid[(1, "abilene", "internal_block")]["hard_nmae"] = None
    elif fault == "required_numeric_nonfinite":
        grid[(1, "abilene", "internal_block")]["hard_nmae"] = float(
            "inf"
        )
    elif fault == "nonfinite_metric":
        grid[(1, "abilene", "internal_block")][
            "loo_target_regret_spearman"
        ] = float("nan")
    elif fault == "missing_uncertainty_field":
        uncertainty.pop("draw_plan")
    elif fault == "bad_hash":
        uncertainty["draw_plan_sha256"] = "not-a-hash"
    elif fault == "uncertainty_schema":
        uncertainty["schema_version"] = "drifted"
    elif fault == "effect_mismatch":
        uncertainty["per_bundle_paired_effect"]["2"] = 0.123
    else:
        uncertainty["positive_bundle_count"] = 2

    report = adjudicate_multibundle(grid, uncertainty)

    assert report["verdict"] == "revise"
    assert report["reason"] == "incomplete_or_invalid_evidence"
    assert report["integrity"]["valid"] is False
    assert report["failure_reasons"] == report["integrity"]["issues"]


def test_legacy_geant_529_flow_count_is_rejected_in_favor_of_registry_462() -> None:
    grid, uncertainty = _passing_inputs()
    uncertainty["draw_plan"]["flow_counts"]["geant"] = 529

    report = adjudicate_multibundle(grid, uncertainty)

    assert report["verdict"] == "revise"
    assert any(
        "flow_counts drifted" in issue
        for issue in report["integrity"]["issues"]
    )


def test_worst_cell_is_taken_after_bundle_averaging() -> None:
    grid, _ = _passing_inputs()
    hard_by_bundle = {1: -0.04, 2: 0.02, 3: 0.05}
    for bundle, hard in hard_by_bundle.items():
        grid[(bundle, "abilene", "internal_block")] = _summary(
            oracle=0.05, hard=hard
        )
    uncertainty = _uncertainty(grid)

    report = adjudicate_multibundle(grid, uncertainty)

    assert report["headline"]["worst_actual_cell"] == {
        "actual": pytest.approx(0.01),
        "cell": "abilene/internal_block",
        "undefined_cells": [],
    }
    assert report["gates"][
        "worst_structured_cell_relative_improvement_min"
    ]["pass"] is True


def test_inputs_are_not_mutated() -> None:
    grid, uncertainty = _passing_inputs()
    original_grid = copy.deepcopy(grid)
    original_uncertainty = copy.deepcopy(uncertainty)

    adjudicate_multibundle(grid, uncertainty)

    assert grid == original_grid
    assert uncertainty == original_uncertainty
