from __future__ import annotations

import copy

import pytest

from experiments.anchorcv_v1.protocol import fingerprint


DATASETS = ("abilene", "geant")
MASKS = ("random", "internal_block", "two_burst")


def _sha(character: str) -> str:
    return character * 64


def _summary(
    *,
    oracle: float,
    auroc: float | None,
    spearman: float | None,
    capture: float | None,
    actual: float,
) -> dict[str, object]:
    truth = 10_000.0
    best = 0.20
    p_nmae = best
    n_nmae = 0.21
    oracle_nmae = best * (1.0 - oracle)
    hard_nmae = best * (1.0 - actual)
    result: dict[str, object] = {
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
        "hard_relative_improvement": actual,
        "neural_selection_rate": 0.5,
        "oracle_neural_rate": 0.5,
        "loo_winner_auroc": auroc,
        "loo_target_regret_spearman": spearman,
        "oracle_gap_capture": capture,
    }
    if auroc is None:
        result["loo_winner_auroc_undefined_reason"] = "one class"
    if spearman is None:
        result["loo_target_regret_spearman_undefined_reason"] = "constant rank"
    if capture is None:
        result["oracle_gap_capture_undefined_reason"] = "no oracle gap"
    return result


def _identity(
    dataset: str,
    *,
    source_character: str = "a",
    config_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "acil_checkpoint_file_sha256": _sha(
            "c" if dataset == "abilene" else "d"
        ),
        "config_sha256": fingerprint() if config_sha256 is None else config_sha256,
        "compute_lane": "cuda_bf16_math_sdp",
        "data_sha256": _sha("1" if dataset == "abilene" else "2"),
        "dataset": dataset,
        "evaluation_cohort": "tune",
        "flows": 144 if dataset == "abilene" else 462,
        "git_available": False,
        "git_commit": None,
        "mask_families": list(MASKS),
        "model_seed": 2021,
        "parsed_array_sha256": {
            "train": _sha("3" if dataset == "abilene" else "4"),
            "val": _sha("5" if dataset == "abilene" else "6"),
        },
        "protocol": "anchorcv-v1",
        "seed_bundle": 1,
        "source_tree_sha256": _sha(source_character),
        "stage": "prototype",
        "test_access": False,
        "training_cohort": "fit",
        "checkpoint_selection_cohort": "source_dev",
    }


def _artifact(
    dataset: str,
    *,
    status: str = "succeeded",
    structured: dict[str, tuple[float | None, ...]] | None = None,
    source_character: str = "a",
    config_sha256: str | None = None,
) -> dict[str, object]:
    identity = _identity(
        dataset,
        source_character=source_character,
        config_sha256=config_sha256,
    )
    job_id = _sha("7" if dataset == "abilene" else "8")
    neural_file = _sha("9" if dataset == "abilene" else "a")
    neural_tensor = _sha("b" if dataset == "abilene" else "c")
    defaults: dict[str, tuple[float | None, ...]] = {
        # Random is deliberately poor and is report-only.
        "random": (0.01, 0.40, -0.10, -1.0, -0.01),
        "internal_block": (0.04, 0.65, 0.20, 0.40, 0.016),
        "two_burst": (0.03, 0.63, 0.18, 0.35, 0.0105),
    }
    if dataset == "geant":
        defaults.update(
            {
                "internal_block": (0.05, 0.67, 0.21, 0.38, 0.019),
                "two_burst": (0.02, 0.61, 0.16, 0.32, 0.0064),
            }
        )
    if structured is not None:
        defaults.update(structured)
    cells: dict[str, object] = {}
    for index, mask in enumerate(MASKS):
        oracle, auroc, spearman, capture, actual = defaults[mask]
        assert oracle is not None and actual is not None
        cells[mask] = {
            "summary": _summary(
                oracle=float(oracle),
                auroc=None if auroc is None else float(auroc),
                spearman=None if spearman is None else float(spearman),
                capture=None if capture is None else float(capture),
                actual=float(actual),
            ),
            "evidence_file": f"evidence_{mask}.npz",
            "evidence_file_sha256": _sha(str(index + 1)),
            "evidence_content_sha256": _sha(str(index + 4)),
            "mask_sha256": _sha(str(index + 7)),
            "oracle_q_sha256": _sha(chr(ord("d") + index)),
            "oracle_e_sha256": _sha(chr(ord("7") + index)),
        }
    result = {
        "job_id": job_id,
        "status": status,
        "scientific_identity": copy.deepcopy(identity),
        "neural_parameter_count": 1_000_000,
        "neural_checkpoint": {
            "file": "neural_best.safetensors",
            "file_sha256": neural_file,
            "tensor_sha256": neural_tensor,
            "best_epoch": 19,
            "best_source_dev_nmae": 0.2,
        },
        "prior_checkpoint": {
            "file_sha256": identity["acil_checkpoint_file_sha256"],
        },
        "training": {
            "epochs_completed": 20,
            "optimizer_updates": 320,
            "best_epoch": 19,
            "best_source_dev_nmae": 0.2,
            "epoch_records": [{"epoch": epoch} for epoch in range(20)],
        },
        "cells": cells,
    }
    manifest = {
        **copy.deepcopy(identity),
        "job_id": job_id,
        "status": status,
        "result_file": "result.json",
        "result_sha256": _sha("e" if dataset == "abilene" else "f"),
        "neural_checkpoint_file_sha256": neural_file,
        "neural_checkpoint_tensor_sha256": neural_tensor,
    }
    return {"result": result, "manifest": manifest}


def _passing_artifacts() -> list[dict[str, object]]:
    return [_artifact("abilene"), _artifact("geant")]


def _replace_cell(
    artifacts: list[dict[str, object]],
    *,
    dataset: str,
    mask: str,
    values: tuple[float | None, ...],
) -> None:
    index = DATASETS.index(dataset)
    artifacts[index] = _artifact(dataset, structured={mask: values})


def test_complete_grid_passes_every_frozen_gate_and_ignores_random() -> None:
    from experiments.anchorcv_v1.adjudicate import adjudicate_prototype

    report = adjudicate_prototype(_passing_artifacts())

    assert report["schema"] == "anchorcv-v1:prototype-adjudication:v1"
    assert report["verdict"] == "proceed"
    assert report["reason"] == "all_scientific_gates_pass"
    assert report["grid"]["complete"] is True
    assert report["grid"]["received_successful_jobs"] == 2
    assert report["headline"]["masks"] == ["internal_block", "two_burst"]
    assert report["headline"]["cell_count"] == 4
    assert report["headline"]["metrics"]["oracle_relative_improvement"][
        "actual"
    ] == pytest.approx(0.035)
    assert report["headline"]["metrics"]["loo_winner_auroc"]["actual"] == (
        pytest.approx(0.64)
    )
    assert report["headline"]["metrics"]["loo_target_regret_spearman"][
        "actual"
    ] == pytest.approx(0.1875)
    assert report["headline"]["metrics"]["oracle_gap_capture"]["actual"] == (
        pytest.approx(0.3625)
    )
    assert report["headline"]["metrics"]["hard_relative_improvement"][
        "actual"
    ] == pytest.approx(0.012975)
    assert report["headline"]["worst_oracle_cell"]["actual"] == pytest.approx(0.02)
    assert report["headline"]["worst_actual_cell"]["actual"] == pytest.approx(
        0.0064
    )
    assert all(item["pass"] for item in report["gates"].values())
    assert set(report["non_headline_random"]) == set(DATASETS)
    assert report["non_headline_random"]["abilene"][
        "oracle_relative_improvement"
    ] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("gate_name", "replacements"),
    [
        (
            "structured_per_flow_oracle_dataset_equal_min",
            {
                "internal_block": (0.015, 0.65, 0.20, 0.40, 0.006),
                "two_burst": (0.015, 0.63, 0.18, 0.40, 0.006),
            },
        ),
        (
            "loo_winner_auroc_min",
            {
                "internal_block": (0.04, 0.20, 0.20, 0.40, 0.016),
                "two_burst": (0.03, 0.20, 0.18, 0.35, 0.0105),
            },
        ),
        (
            "loo_target_regret_spearman_min",
            {
                "internal_block": (0.04, 0.65, -0.20, 0.40, 0.016),
                "two_burst": (0.03, 0.63, -0.18, 0.35, 0.0105),
            },
        ),
        (
            "hard_anchorcv_oracle_gap_capture_min",
            {
                "internal_block": (0.04, 0.65, 0.20, 0.20, 0.008),
                "two_burst": (0.03, 0.63, 0.18, 0.20, 0.006),
            },
        ),
        (
            "hard_anchorcv_actual_improvement_min",
            {
                "internal_block": (0.04, 0.65, 0.20, -0.25, -0.010),
                "two_burst": (0.03, 0.63, 0.18, -0.25, -0.0075),
            },
        ),
        (
            "worst_structured_oracle_cell_relative_improvement_min",
            {
                "internal_block": (0.014999, 0.65, 0.20, 0.40, 0.0059996),
            },
        ),
        (
            "worst_structured_cell_relative_improvement_min",
            {
                "internal_block": (0.04, 0.65, 0.20, -0.250025, -0.010001),
            },
        ),
    ],
)
def test_each_frozen_scientific_gate_can_kill(
    gate_name: str,
    replacements: dict[str, tuple[float | None, ...]],
) -> None:
    from experiments.anchorcv_v1.adjudicate import adjudicate_prototype

    artifacts = _passing_artifacts()
    artifacts[0] = _artifact("abilene", structured=replacements)
    if gate_name in {
        "structured_per_flow_oracle_dataset_equal_min",
        "loo_winner_auroc_min",
        "loo_target_regret_spearman_min",
        "hard_anchorcv_oracle_gap_capture_min",
        "hard_anchorcv_actual_improvement_min",
    }:
        artifacts[1] = _artifact("geant", structured=replacements)

    report = adjudicate_prototype(artifacts)

    assert report["verdict"] == "kill"
    assert report["reason"] == "scientific_gate_failure"
    assert report["gates"][gate_name]["pass"] is False
    assert set(report["gates"][gate_name]) >= {
        "actual",
        "threshold",
        "pass",
    }


def test_undefined_structured_metric_is_a_scientific_gate_failure() -> None:
    from experiments.anchorcv_v1.adjudicate import adjudicate_prototype

    artifacts = _passing_artifacts()
    _replace_cell(
        artifacts,
        dataset="abilene",
        mask="internal_block",
        values=(0.04, None, 0.20, 0.40, 0.016),
    )
    report = adjudicate_prototype(artifacts)

    assert report["verdict"] == "kill"
    gate = report["gates"]["loo_winner_auroc_min"]
    assert gate["actual"] is None
    assert gate["pass"] is False
    assert gate["undefined_cells"] == ["abilene/internal_block"]


@pytest.mark.parametrize("problem", ["missing", "failed", "duplicate", "extra"])
def test_incomplete_or_failed_two_by_two_job_grid_is_revise(problem: str) -> None:
    from experiments.anchorcv_v1.adjudicate import adjudicate_prototype

    artifacts = _passing_artifacts()
    if problem == "missing":
        artifacts = artifacts[:1]
    elif problem == "failed":
        artifacts[1] = _artifact("geant", status="failed")
    elif problem == "duplicate":
        artifacts.append(copy.deepcopy(artifacts[0]))
    else:
        extra = _artifact("geant")
        extra["result"]["scientific_identity"]["seed_bundle"] = 2
        extra["manifest"]["seed_bundle"] = 2
        artifacts.append(extra)

    report = adjudicate_prototype(artifacts)

    assert report["verdict"] == "revise"
    assert report["reason"] == "incomplete_or_invalid_evidence"
    assert report["grid"]["complete"] is False
    assert report["grid"]["issues"]


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_source_hash",
        "config_not_frozen",
        "cross_job_source_drift",
        "result_manifest_drift",
        "checkpoint_drift",
        "cell_hash_bad",
        "missing_mask",
        "duplicate_mask",
        "summary_inconsistent",
    ],
)
def test_result_manifest_per_mask_and_summary_integrity_faults_are_revise(
    mutation: str,
) -> None:
    from experiments.anchorcv_v1.adjudicate import adjudicate_prototype

    artifacts = _passing_artifacts()
    if mutation == "bad_source_hash":
        artifacts[0]["manifest"]["source_tree_sha256"] = "not-a-sha"
    elif mutation == "config_not_frozen":
        artifacts[0] = _artifact("abilene", config_sha256=_sha("0"))
    elif mutation == "cross_job_source_drift":
        artifacts[1] = _artifact("geant", source_character="9")
    elif mutation == "result_manifest_drift":
        artifacts[0]["result"]["scientific_identity"]["data_sha256"] = _sha("9")
    elif mutation == "checkpoint_drift":
        artifacts[0]["result"]["neural_checkpoint"]["file_sha256"] = _sha("0")
    elif mutation == "cell_hash_bad":
        artifacts[0]["result"]["cells"]["internal_block"][
            "evidence_content_sha256"
        ] = "bad"
    elif mutation == "missing_mask":
        artifacts[0]["result"]["cells"].pop("two_burst")
    elif mutation == "duplicate_mask":
        artifacts[0]["result"]["cells"]["random_copy"] = copy.deepcopy(
            artifacts[0]["result"]["cells"]["random"]
        )
    else:
        artifacts[0]["result"]["cells"]["internal_block"]["summary"][
            "oracle_relative_improvement"
        ] = 0.5

    report = adjudicate_prototype(artifacts)

    assert report["verdict"] == "revise"
    assert report["integrity"]["valid"] is False
    assert report["integrity"]["issues"]


def test_structured_aggregation_is_dataset_and_cell_equal_not_count_weighted() -> None:
    from experiments.anchorcv_v1.adjudicate import adjudicate_prototype

    artifacts = _passing_artifacts()
    internal = artifacts[0]["result"]["cells"]["internal_block"]["summary"]
    burst = artifacts[0]["result"]["cells"]["two_burst"]["summary"]
    internal["case_count"] = 1
    internal["absolute_truth_sum"] = 1.0
    internal["p_absolute_error_sum"] = internal["p_nmae"]
    internal["n_absolute_error_sum"] = internal["n_nmae"]
    internal["hard_absolute_error_sum"] = internal["hard_nmae"]
    internal["oracle_absolute_error_sum"] = internal["oracle_nmae"]
    burst["case_count"] = 10_000_000
    burst["absolute_truth_sum"] = 1e20
    burst["p_absolute_error_sum"] = burst["p_nmae"] * 1e20
    burst["n_absolute_error_sum"] = burst["n_nmae"] * 1e20
    burst["hard_absolute_error_sum"] = burst["hard_nmae"] * 1e20
    burst["oracle_absolute_error_sum"] = burst["oracle_nmae"] * 1e20

    report = adjudicate_prototype(artifacts)

    assert report["verdict"] == "proceed"
    assert report["headline"]["metrics"]["oracle_relative_improvement"][
        "per_dataset"
    ]["abilene"] == pytest.approx(0.035)
    assert report["headline"]["metrics"]["oracle_relative_improvement"][
        "actual"
    ] == pytest.approx(0.035)


def test_threshold_equality_passes_and_report_exposes_every_actual_threshold() -> None:
    from experiments.anchorcv_v1.adjudicate import adjudicate_prototype

    equal_oracle = 0.03
    equal_capture = 0.30
    equal_actual = equal_oracle * equal_capture
    artifacts = [
        _artifact(
            dataset,
            structured={
                "internal_block": (
                    0.015,
                    0.60,
                    0.15,
                    0.30,
                    0.0045,
                ),
                "two_burst": (
                    0.045,
                    0.60,
                    0.15,
                    equal_capture,
                    0.0135,
                ),
            },
        )
        for dataset in DATASETS
    ]
    report = adjudicate_prototype(artifacts)

    assert equal_actual == pytest.approx(0.009)
    assert report["verdict"] == "proceed"
    assert report["gates"][
        "structured_per_flow_oracle_dataset_equal_min"
    ]["actual"] == pytest.approx(0.03)
    for gate in report["gates"].values():
        assert gate["actual"] is not None
        assert isinstance(gate["threshold"], float)
        assert gate["operator"] == ">="
        assert gate["pass"] is True
