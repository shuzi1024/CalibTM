from __future__ import annotations

import sys
from pathlib import Path

import pytest

from experiments.anchorcv_v1 import gate_identity, run_gate_job
from experiments.anchorcv_v1.gate_identity import VerifiedGateAuthority


def _authority() -> VerifiedGateAuthority:
    bindings = []
    for bundle in (1, 2, 3):
        for dataset in ("abilene", "geant"):
            marker = f"{bundle}{dataset[0]}" * 32
            bindings.append(
                {
                    "dataset": dataset,
                    "seed_bundle": bundle,
                    "source_training_stage": (
                        "prototype" if bundle == 1 else "extension"
                    ),
                    "source_training_job_id": marker,
                    "source_result_sha256": "1" * 64,
                    "source_manifest_sha256": "2" * 64,
                    "neural_checkpoint_file_sha256": "3" * 64,
                    "neural_checkpoint_tensor_sha256": "4" * 64,
                    "acil_checkpoint_file_sha256": "5" * 64,
                    "best_epoch": 3,
                    "best_source_dev_nmae": 0.1,
                }
            )
    payload = {
        "source_tree_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "checkpoint_bindings": bindings,
    }
    return VerifiedGateAuthority(
        payload,
        "c" * 64,
        _seal=gate_identity._AUTHORITY_SEAL,
    )


def _argv(tmp_path: Path, job_id: str) -> list[str]:
    return [
        "run_gate_job",
        "--job-id",
        job_id,
        "--output-root",
        str(tmp_path / "gate"),
        "--authority-record",
        str(tmp_path / "authority.json"),
        "--extension-report",
        str(tmp_path / "extension-report.json"),
        "--freeze-record",
        str(tmp_path / "freeze.json"),
        "--prototype-output-root",
        str(tmp_path / "prototype"),
        "--extension-output-root",
        str(tmp_path / "extension"),
    ]


def test_parser_exposes_only_job_id_and_fixed_authority_artifact_paths() -> None:
    parser = run_gate_job.build_parser()
    destinations = {action.dest for action in parser._actions}

    assert destinations == {
        "help",
        "job_id",
        "output_root",
        "authority_record",
        "extension_report",
        "freeze_record",
        "prototype_output_root",
        "extension_output_root",
    }
    forbidden = {
        "dataset",
        "seed_bundle",
        "bundle",
        "cohort",
        "mask",
        "split",
        "test",
        "data",
        "device",
    }
    assert destinations.isdisjoint(forbidden)


def test_runner_verifies_exact_roots_then_selects_only_registered_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authority = _authority()
    specs = run_gate_job.build_gate_specs(tmp_path / "gate", authority)
    selected = specs[4]
    verify_calls: list[dict[str, object]] = []
    execute_calls: list[tuple[object, Path, Path]] = []

    monkeypatch.setattr(
        run_gate_job,
        "verify_gate_authority",
        lambda **kwargs: verify_calls.append(kwargs) or authority,
    )
    monkeypatch.setattr(
        run_gate_job,
        "execute_gate_job",
        lambda spec, **kwargs: execute_calls.append(
            (
                spec,
                kwargs["prototype_output_root"],
                kwargs["extension_output_root"],
            )
        )
        or {"status": "succeeded"},
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, selected.job_id))

    assert run_gate_job.main() == 0
    assert verify_calls == [
        {
            "authority_record": tmp_path / "authority.json",
            "extension_report": tmp_path / "extension-report.json",
            "freeze_record": tmp_path / "freeze.json",
            "prototype_output_root": tmp_path / "prototype",
            "extension_output_root": tmp_path / "extension",
        }
    ]
    assert execute_calls == [
        (
            selected,
            tmp_path / "prototype",
            tmp_path / "extension",
        )
    ]
    assert f'"job_id": "{selected.job_id}"' in capsys.readouterr().out


def test_runner_rejects_unregistered_job_id_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()
    monkeypatch.setattr(
        run_gate_job,
        "verify_gate_authority",
        lambda **kwargs: authority,
    )
    monkeypatch.setattr(
        run_gate_job,
        "execute_gate_job",
        lambda *args, **kwargs: pytest.fail(
            "unregistered job must never execute"
        ),
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "0" * 64))

    with pytest.raises(ValueError, match="registered six-job grid"):
        run_gate_job.main()
