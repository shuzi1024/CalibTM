from __future__ import annotations

import hashlib
import json

import pytest


def test_registered_prototype_grid_is_exactly_six_jobs() -> None:
    from experiments.minimal_calibrator_v1.run_job import (
        expected_job_identities,
        registered_job,
    )

    expected = expected_job_identities()
    assert len(expected) == 6
    assert registered_job("abilene", "static_zero") in expected
    assert registered_job("geant", "value_only") in expected
    assert {
        (job["dataset"], job["method"], job["seed_bundle"], job["stage"])
        for job in expected
    } == {
        (dataset, method, 1, "prototype")
        for dataset in ("abilene", "geant")
        for method in ("static_zero", "blinear_only", "value_only")
    }


@pytest.mark.parametrize(
    ("dataset", "method"),
    [
        ("wsdream", "static_zero"),
        ("geant", "full"),
        ("geant", "bias_only"),
    ],
)
def test_registered_job_rejects_every_unfrozen_choice(
    dataset: str, method: str
) -> None:
    from experiments.minimal_calibrator_v1.run_job import registered_job

    with pytest.raises(ValueError):
        registered_job(dataset, method)


def test_public_cli_has_no_seed_split_or_data_override() -> None:
    from experiments.minimal_calibrator_v1.run_job import build_parser

    parser = build_parser()
    assert {action.dest for action in parser._actions} == {
        "help",
        "dataset",
        "method",
        "output",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset",
                "geant",
                "--method",
                "static_zero",
                "--output",
                "result.json",
                "--split",
                "test",
            ]
        )


def test_loader_exposes_only_permitted_discovery_cohorts() -> None:
    from experiments.minimal_calibrator_v1.run_job import _load_windows

    for cohort in ("fit", "source_dev", "tune"):
        windows = _load_windows("geant", cohort)
        assert windows.dataset == "geant"
        assert windows.cohort == cohort
    for forbidden in ("gate", "test"):
        with pytest.raises(ValueError):
            _load_windows("geant", forbidden)


def test_result_manifest_is_write_once_and_hash_bound(tmp_path) -> None:
    from experiments.minimal_calibrator_v1.run_job import write_result

    output = tmp_path / "result.json"
    result = {
        "job": {
            "dataset": "abilene",
            "job_id": "a" * 64,
            "method": "static_zero",
            "seed_bundle": 1,
            "stage": "prototype",
        },
        "probe_freeze": {"freeze_file_sha256": "b" * 64},
        "protocol": "minimal-calibrator-v1",
        "status": "succeeded",
        "test_access": False,
    }
    manifest = write_result(output, result)
    assert manifest["result_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert json.loads(output.read_text(encoding="utf-8")) == result
    with pytest.raises(FileExistsError):
        write_result(output, result)

