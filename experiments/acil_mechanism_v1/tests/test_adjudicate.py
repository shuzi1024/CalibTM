from __future__ import annotations

import copy

import pytest


_FAMILIES = ("random", "internal_block", "two_burst")


def _result(
    dataset: str,
    seed_bundle: int,
    method: str,
    *,
    full_error: float,
    value_error: float,
    no_anchor_error: float,
    linear_error: float = 23.0,
) -> dict[str, object]:
    from experiments.acil_mechanism_v1.run_job import registered_job

    method_error = {
        "full": full_error,
        "value_only": value_error,
        "no_anchor": no_anchor_error,
    }[method]
    cells = {}
    for family in _FAMILIES:
        cells[family] = {
            "linear_error_sum_per_window": [linear_error] * 4,
            "mask_grid_sha256": family[0] * 64,
            "model_error_sum_per_window": [method_error] * 4,
            "target_count_per_window": [100] * 4,
            "truth_sum_per_window": [100.0] * 4,
            "window_starts": [0, 50, 100, 150],
        }
    return {
        "cells": cells,
        "job": registered_job(dataset, seed_bundle, method),
    }


def _prototype(
    *, full_error: float, value_error: float, no_anchor_error: float
) -> list[dict[str, object]]:
    return [
        _result(
            dataset,
            1,
            method,
            full_error=full_error,
            value_error=value_error,
            no_anchor_error=no_anchor_error,
        )
        for dataset in ("abilene", "geant")
        for method in ("full", "value_only", "no_anchor")
    ]


def test_prototype_proceeds_only_when_full_beats_both_matched_controls() -> None:
    from experiments.acil_mechanism_v1.adjudicate import adjudicate_payloads

    report = adjudicate_payloads(
        _prototype(full_error=18.0, value_error=20.0, no_anchor_error=19.0),
        stage="prototype",
        bootstrap_draws=100,
    )

    assert report["verdict"] == "proceed"
    assert report["headline"]["full_over_value_only"]["actual"] == pytest.approx(
        0.1
    )
    assert report["headline"]["full_over_no_anchor"]["actual"] == pytest.approx(
        1.0 - 18.0 / 19.0
    )
    assert all(item["pass"] for item in report["gates"].values())


def test_prototype_kills_when_geometry_control_is_not_beaten() -> None:
    from experiments.acil_mechanism_v1.adjudicate import adjudicate_payloads

    report = adjudicate_payloads(
        _prototype(full_error=20.0, value_error=20.0, no_anchor_error=19.0),
        stage="prototype",
        bootstrap_draws=100,
    )

    assert report["verdict"] == "kill"
    assert (
        report["gates"]["full_over_value_structured_min"]["pass"] is False
    )


def test_adjudication_rejects_missing_duplicate_or_unpaired_evidence() -> None:
    from experiments.acil_mechanism_v1.adjudicate import adjudicate_payloads

    values = _prototype(full_error=18.0, value_error=20.0, no_anchor_error=19.0)
    with pytest.raises(ValueError, match="identity"):
        adjudicate_payloads(values[:-1], stage="prototype", bootstrap_draws=10)

    duplicate = copy.deepcopy(values)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(ValueError, match="identity"):
        adjudicate_payloads(duplicate, stage="prototype", bootstrap_draws=10)

    unpaired = copy.deepcopy(values)
    unpaired[-1]["cells"]["internal_block"]["truth_sum_per_window"][0] += 1.0
    with pytest.raises(ValueError, match="paired"):
        adjudicate_payloads(unpaired, stage="prototype", bootstrap_draws=10)


def test_bootstrap_is_deterministic() -> None:
    from experiments.acil_mechanism_v1.adjudicate import adjudicate_payloads

    values = _prototype(full_error=18.0, value_error=20.0, no_anchor_error=19.0)
    first = adjudicate_payloads(values, stage="prototype", bootstrap_draws=100)
    second = adjudicate_payloads(values, stage="prototype", bootstrap_draws=100)
    assert first == second


def test_full_must_beat_linear_on_each_dataset() -> None:
    from experiments.acil_mechanism_v1.adjudicate import adjudicate_payloads

    values = _prototype(full_error=18.0, value_error=20.0, no_anchor_error=19.0)
    for result in values:
        if result["job"]["dataset"] == "geant":
            for cell in result["cells"].values():
                cell["linear_error_sum_per_window"] = [17.0] * 4

    report = adjudicate_payloads(
        values,
        stage="prototype",
        bootstrap_draws=100,
    )

    assert report["verdict"] == "kill"
    assert (
        report["gates"]["full_over_linear_each_dataset_positive"]["pass"]
        is False
    )


def _external_result(
    root,
    *,
    feature_set: str = "full",
    corrupt_data: bool = False,
):
    from experiments.acil_mechanism_v1.model import new_acil
    from experiments.acil_mechanism_v1.run_job import (
        registered_job,
        write_result,
    )
    from experiments.anchorcv_v1.data_access import canonical_data_identity
    from experiments.sc_acil_v1.checkpoint import save_checkpoint

    freeze = {
        "freeze_file_sha256": "a" * 64,
        "research_card_sha256": "b" * 64,
        "source_tree_sha256": "c" * 64,
    }
    job = registered_job("abilene", 1, "full")
    identity = {
        "best_epoch": 3,
        "feature_set": feature_set,
        "job": job,
        "probe_freeze": freeze,
        "source_dev_nmae": 0.2,
    }
    checkpoint = save_checkpoint(
        new_acil("full", model_seed=41001),
        identity=identity,
        directory=root / "checkpoints",
    )
    data_identity = canonical_data_identity("abilene")
    if corrupt_data:
        data_identity = {**data_identity, "data_sha256": "0" * 64}
    result = {
        "checkpoint": {
            "file_sha256": checkpoint.file_sha256,
            "identity": identity,
            "identity_sha256": checkpoint.identity_sha256,
            "metadata": checkpoint.metadata_path.relative_to(root).as_posix(),
            "tensor_sha256": checkpoint.tensor_sha256,
            "weights": checkpoint.weights_path.relative_to(root).as_posix(),
        },
        "data_identity": data_identity,
        "evaluation_cohort": "tune",
        "feature_set": feature_set,
        "job": job,
        "probe_freeze": freeze,
        "protocol": "acil-mechanism-v1",
        "selection_cohort": "source_dev",
        "status": "succeeded",
        "test_access": False,
        "training": {
            "best_epoch": 3,
            "best_source_dev_nmae": 0.2,
            "epochs_completed": 20,
            "optimizer_updates": 320,
            "parameter_count": 5475,
        },
    }
    output = root / "result.json"
    write_result(output, result)
    return output, freeze, checkpoint


def test_external_validation_rejects_method_feature_and_data_drift(
    tmp_path,
) -> None:
    from experiments.acil_mechanism_v1.adjudicate import _load_external

    wrong_feature = tmp_path / "wrong-feature"
    wrong_feature.mkdir()
    path, freeze, _ = _external_result(
        wrong_feature,
        feature_set="value_only",
    )
    with pytest.raises(ValueError, match="feature"):
        _load_external(path, freeze)

    wrong_data = tmp_path / "wrong-data"
    wrong_data.mkdir()
    path, freeze, _ = _external_result(wrong_data, corrupt_data=True)
    with pytest.raises(ValueError, match="data"):
        _load_external(path, freeze)


def test_external_validation_replays_checkpoint_hashes(tmp_path) -> None:
    from experiments.acil_mechanism_v1.adjudicate import _load_external

    root = tmp_path / "corrupt-checkpoint"
    root.mkdir()
    path, freeze, checkpoint = _external_result(root)
    checkpoint.weights_path.write_bytes(
        checkpoint.weights_path.read_bytes() + b"corrupt"
    )

    with pytest.raises(ValueError, match="checkpoint"):
        _load_external(path, freeze)
