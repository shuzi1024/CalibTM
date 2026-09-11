from __future__ import annotations

import hashlib
import json

import pytest


def test_registered_grid_is_fixed_and_outcome_independent() -> None:
    from experiments.acil_mechanism_v1.run_job import (
        expected_job_identities,
        registered_job,
    )

    prototype = registered_job("abilene", 1, "full")
    extension = registered_job("geant", 3, "value_only")

    assert prototype["stage"] == "prototype"
    assert extension["stage"] == "extension"
    assert len(prototype["job_id"]) == 64
    assert prototype in expected_job_identities("prototype")
    assert len(expected_job_identities("prototype")) == 6
    assert len(expected_job_identities("extension")) == 18


@pytest.mark.parametrize(
    ("dataset", "seed_bundle", "method"),
    [
        ("wsdream", 1, "full"),
        ("geant", 0, "full"),
        ("geant", 4, "full"),
        ("geant", 1, "shuffled"),
    ],
)
def test_registered_job_rejects_every_unfrozen_choice(
    dataset: str, seed_bundle: int, method: str
) -> None:
    from experiments.acil_mechanism_v1.run_job import registered_job

    with pytest.raises(ValueError):
        registered_job(dataset, seed_bundle, method)


def test_public_cli_exposes_no_data_or_split_override() -> None:
    from experiments.acil_mechanism_v1.run_job import build_parser

    parser = build_parser()
    assert {action.dest for action in parser._actions} == {
        "help",
        "dataset",
        "seed_bundle",
        "method",
        "output",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset",
                "geant",
                "--seed-bundle",
                "1",
                "--method",
                "full",
                "--output",
                "result.json",
                "--split",
                "test",
            ]
        )


def test_loader_exposes_only_permitted_discovery_cohorts() -> None:
    from experiments.acil_mechanism_v1.run_job import _load_windows

    for cohort in ("fit", "source_dev", "tune"):
        windows = _load_windows("geant", cohort)
        assert windows.dataset == "geant"
        assert windows.cohort == cohort
    for forbidden in ("gate", "test"):
        with pytest.raises(ValueError):
            _load_windows("geant", forbidden)


def test_result_manifest_is_write_once_and_hash_bound(tmp_path) -> None:
    from experiments.acil_mechanism_v1.run_job import write_result

    output = tmp_path / "result.json"
    result = {
        "protocol": "acil-mechanism-v1",
        "job": {
            "dataset": "abilene",
            "job_id": "a" * 64,
            "method": "full",
            "seed_bundle": 1,
            "stage": "prototype",
        },
        "probe_freeze": {"freeze_file_sha256": "b" * 64},
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
