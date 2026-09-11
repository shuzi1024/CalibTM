from __future__ import annotations

import copy
import json

import pytest


_FAMILIES = ("random", "internal_block", "two_burst")


def _result(
    dataset: str,
    method: str,
    *,
    static_error: float,
    blinear_error: float,
    value_error: float,
    linear_error: float,
) -> dict[str, object]:
    from experiments.minimal_calibrator_v1.run_job import registered_job

    selected = {
        "static_zero": static_error,
        "blinear_only": blinear_error,
        "value_only": value_error,
    }[method]
    return {
        "cells": {
            family: {
                "linear_error_sum_per_window": [linear_error] * 4,
                "mask_grid_sha256": family[0] * 64,
                "model_error_sum_per_window": [selected] * 4,
                "target_count_per_window": [100] * 4,
                "truth_sum_per_window": [100.0] * 4,
                "window_starts": [0, 50, 100, 150],
            }
            for family in _FAMILIES
        },
        "job": registered_job(dataset, method),
    }


def _grid(
    *,
    static_error: float,
    blinear_error: float,
    value_error: float,
    linear_error: float = 23.0,
) -> list[dict[str, object]]:
    return [
        _result(
            dataset,
            method,
            static_error=static_error,
            blinear_error=blinear_error,
            value_error=value_error,
            linear_error=linear_error,
        )
        for dataset in ("abilene", "geant")
        for method in ("static_zero", "blinear_only", "value_only")
    ]


def test_kill_when_static_calibration_explains_value_only_gain() -> None:
    from experiments.minimal_calibrator_v1.adjudicate import adjudicate_payloads

    report = adjudicate_payloads(
        _grid(static_error=19.0, blinear_error=19.0, value_error=18.95),
        bootstrap_draws=100,
    )
    assert report["verdict"] == "kill"
    assert report["reason"] == "input_conditioning_effect_too_small"
    assert report["headline"]["value_over_static"]["actual"] < 0.005


def test_revise_when_point_effect_passes_but_cross_dataset_fails() -> None:
    from experiments.minimal_calibrator_v1.adjudicate import adjudicate_payloads

    values = _grid(
        static_error=20.0,
        blinear_error=19.0,
        value_error=18.0,
    )
    for result in values:
        if result["job"]["dataset"] == "geant":
            selected = (
                20.1 if result["job"]["method"] == "value_only" else 20.0
            )
            for cell in result["cells"].values():
                cell["model_error_sum_per_window"] = [selected] * 4
    report = adjudicate_payloads(values, bootstrap_draws=100)
    assert report["verdict"] == "revise"
    assert report["gates"]["value_over_static_each_dataset_positive"][
        "pass"
    ] is False


def test_proceed_blinear_when_single_value_is_noninferior() -> None:
    from experiments.minimal_calibrator_v1.adjudicate import adjudicate_payloads

    report = adjudicate_payloads(
        _grid(
            static_error=20.0,
            blinear_error=17.99,
            value_error=18.0,
        ),
        bootstrap_draws=100,
    )
    assert report["verdict"] == "proceed"
    assert report["reason"] == "blinear_conditioning_supported"
    assert all(item["pass"] for item in report["primary_gates"].values())
    assert all(item["pass"] for item in report["blinear_gates"].values())


def test_proceed_value_when_primary_passes_but_blinear_is_inferior() -> None:
    from experiments.minimal_calibrator_v1.adjudicate import adjudicate_payloads

    report = adjudicate_payloads(
        _grid(
            static_error=20.0,
            blinear_error=19.0,
            value_error=18.0,
        ),
        bootstrap_draws=100,
    )
    assert report["verdict"] == "proceed"
    assert report["reason"] == "value_scale_conditioning_supported"
    assert report["blinear_gates"]["blinear_noninferior_to_value"]["pass"] is False


def test_adjudication_rejects_missing_duplicate_and_unpaired_evidence() -> None:
    from experiments.minimal_calibrator_v1.adjudicate import adjudicate_payloads

    values = _grid(static_error=20.0, blinear_error=19.0, value_error=18.0)
    with pytest.raises(ValueError, match="identity"):
        adjudicate_payloads(values[:-1], bootstrap_draws=10)
    duplicate = copy.deepcopy(values)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(ValueError, match="identity"):
        adjudicate_payloads(duplicate, bootstrap_draws=10)
    unpaired = copy.deepcopy(values)
    unpaired[-1]["cells"]["internal_block"]["truth_sum_per_window"][0] += 1
    with pytest.raises(ValueError, match="paired"):
        adjudicate_payloads(unpaired, bootstrap_draws=10)


def test_bootstrap_is_deterministic() -> None:
    from experiments.minimal_calibrator_v1.adjudicate import adjudicate_payloads

    values = _grid(static_error=20.0, blinear_error=19.0, value_error=18.0)
    first = adjudicate_payloads(values, bootstrap_draws=100)
    second = adjudicate_payloads(values, bootstrap_draws=100)
    assert first == second


def test_result_discovery_ignores_operational_and_adjudication_json(
    tmp_path,
) -> None:
    from experiments.minimal_calibrator_v1.adjudicate import _discover_results

    result = tmp_path / "geant_static_zero.json"
    result.write_text("{}\n", encoding="utf-8")
    result.with_name(result.name + ".manifest.json").write_text(
        json.dumps(
            {
                "protocol": "minimal-calibrator-v1",
                "result_file": result.name,
                "schema": "minimal-calibrator-v1:result-manifest:v1",
            }
        ),
        encoding="utf-8",
    )
    launch = tmp_path / "launch_manifest.json"
    launch.write_text("{}\n", encoding="utf-8")
    launch.with_name(launch.name + ".manifest.json").write_text(
        json.dumps(
            {
                "schema": "minimal-calibrator-v1:launch-manifest:v1"
            }
        ),
        encoding="utf-8",
    )
    adjudication = tmp_path / "adjudication.json"
    adjudication.write_text("{}\n", encoding="utf-8")
    adjudication.with_name(adjudication.name + ".manifest.json").write_text(
        json.dumps(
            {
                "schema": (
                    "minimal-calibrator-v1:adjudication-manifest:v1"
                )
            }
        ),
        encoding="utf-8",
    )
    assert _discover_results(tmp_path, "new-adjudication.json") == [result]


def test_training_selection_requires_earliest_source_dev_minimum() -> None:
    from experiments.minimal_calibrator_v1.adjudicate import _validate_training

    rows = [
        {
            "epoch": epoch,
            "optimizer_updates": 16,
            "physical_batches": 64,
            "selected_as_best": epoch == 3,
            "source_dev_absolute_error_sum": 20.0,
            "source_dev_absolute_truth_sum": 100.0,
            "source_dev_nmae": 0.2,
        }
        for epoch in range(20)
    ]
    training = {
        "best_epoch": 3,
        "best_source_dev_nmae": 0.2,
        "epochs": rows,
        "epochs_completed": 20,
        "optimizer_updates": 320,
        "parameter_count": 5475,
    }
    with pytest.raises(ValueError, match="earliest|minimum"):
        _validate_training(training)


def _external_result(root):
    from experiments.anchorcv_v1.data_access import canonical_data_identity
    from experiments.minimal_calibrator_v1.model import new_calibrator
    from experiments.minimal_calibrator_v1.run_job import (
        registered_job,
        write_result,
    )
    from experiments.sc_acil_v1.checkpoint import save_checkpoint

    freeze = {
        "freeze_file_sha256": "a" * 64,
        "research_card_sha256": "b" * 64,
        "source_tree_sha256": "c" * 64,
    }
    job = registered_job("abilene", "static_zero")
    rows = []
    for epoch in range(20):
        source_dev_nmae = 0.2 + abs(epoch - 3) * 0.001
        rows.append(
            {
                "epoch": epoch,
                "mean_loss": 1.0,
                "optimizer_updates": 16,
                "physical_batches": 64,
                "selected_as_best": epoch == 3,
                "source_dev_absolute_error_sum": source_dev_nmae * 100.0,
                "source_dev_absolute_truth_sum": 100.0,
                "source_dev_nmae": source_dev_nmae,
            }
        )
    identity = {
        "best_epoch": 3,
        "feature_set": "static_zero",
        "job": job,
        "probe_freeze": freeze,
        "source_dev_nmae": 0.2,
    }
    checkpoint = save_checkpoint(
        new_calibrator("static_zero", model_seed=41001),
        identity=identity,
        directory=root / "checkpoints",
    )
    result = {
        "checkpoint": {
            "file_sha256": checkpoint.file_sha256,
            "identity": identity,
            "identity_sha256": checkpoint.identity_sha256,
            "metadata": checkpoint.metadata_path.relative_to(root).as_posix(),
            "tensor_sha256": checkpoint.tensor_sha256,
            "weights": checkpoint.weights_path.relative_to(root).as_posix(),
        },
        "data_identity": canonical_data_identity("abilene"),
        "evaluation_cohort": "tune",
        "feature_set": "static_zero",
        "job": job,
        "probe_freeze": freeze,
        "protocol": "minimal-calibrator-v1",
        "schema": "minimal-calibrator-v1:result:v1",
        "selection_cohort": "source_dev",
        "status": "succeeded",
        "test_access": False,
        "training": {
            "best_epoch": 3,
            "best_source_dev_nmae": 0.2,
            "epochs": rows,
            "epochs_completed": 20,
            "optimizer_updates": 320,
            "parameter_count": 5475,
        },
    }
    output = root / "result.json"
    write_result(output, result)
    return output, freeze, checkpoint


def test_external_validation_replays_checkpoint_and_manifest_hashes(
    tmp_path,
) -> None:
    from experiments.minimal_calibrator_v1.adjudicate import _load_external

    root = tmp_path / "valid"
    root.mkdir()
    path, freeze, _ = _external_result(root)
    assert _load_external(path, freeze)["status"] == "succeeded"

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    path, freeze, checkpoint = _external_result(corrupt)
    checkpoint.weights_path.write_bytes(
        checkpoint.weights_path.read_bytes() + b"corrupt"
    )
    with pytest.raises(ValueError, match="checkpoint"):
        _load_external(path, freeze)

    wrong_manifest = tmp_path / "wrong-manifest"
    wrong_manifest.mkdir()
    path, freeze, _ = _external_result(wrong_manifest)
    companion = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(companion.read_text(encoding="utf-8"))
    manifest["schema"] = "wrong"
    companion.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        _load_external(path, freeze)
