from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from experiments.anchorcv_v1 import gate_identity
from experiments.anchorcv_v1.gate_identity import (
    GateJobSpec,
    VerifiedGateAuthority,
    create_verified_gate_authority,
    verify_gate_authority,
    write_gate_authority,
)


_SHA = {
    name: character * 64
    for name, character in {
        "source": "a",
        "config": "b",
        "result": "c",
        "manifest": "d",
        "neural_file": "e",
        "neural_tensor": "f",
        "acil": "1",
        "window": "2",
    }.items()
}


def _freeze() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": "anchorcv-v1",
        "source_tree_sha256": _SHA["source"],
        "config_sha256": _SHA["config"],
        "git_available": False,
        "git_commit": None,
    }


def _binding(bundle: int, dataset: str) -> dict[str, object]:
    marker = hashlib.sha256(f"{bundle}/{dataset}".encode("ascii")).hexdigest()
    return {
        "dataset": dataset,
        "seed_bundle": bundle,
        "source_training_stage": (
            "prototype" if bundle == 1 else "extension"
        ),
        "source_training_job_id": marker,
        "source_result_sha256": _SHA["result"],
        "source_manifest_sha256": _SHA["manifest"],
        "neural_checkpoint_file_sha256": _SHA["neural_file"],
        "neural_checkpoint_tensor_sha256": _SHA["neural_tensor"],
        "acil_checkpoint_file_sha256": _SHA["acil"],
        "best_epoch": 3,
        "best_source_dev_nmae": 0.125,
    }


def _report() -> dict[str, object]:
    bindings = [
        _binding(bundle, dataset)
        for bundle in (1, 2, 3)
        for dataset in ("abilene", "geant")
    ]
    return {
        "schema": "anchorcv-v1:extension-review:v1",
        "status": "verified_complete",
        "verdict": "proceed",
        "confirmation_authorized": True,
        "job_count": 6,
        "checkpoint_bindings": bindings,
        "grid": [
            {
                "dataset": item["dataset"],
                "job_id": item["source_training_job_id"],
                "seed_bundle": item["seed_bundle"],
                "stage": item["source_training_stage"],
            }
            for item in bindings
        ],
        "integrity": {
            "valid": True,
            "source_tree_sha256": _SHA["source"],
            "config_sha256": _SHA["config"],
            "git_available": False,
            "git_commit": None,
            "test_access": False,
        },
        "prototype": {"verdict": "proceed"},
        "uncertainty": {"status": "complete"},
        "multibundle_adjudication": {
            "schema": "anchorcv-v1:multibundle-adjudication:v1",
            "verdict": "proceed",
            "integrity": {"valid": True, "issues": []},
        },
        "interpretation": "verified full-tune extension review",
    }


def _write_json(path: Path, value: object, *, canonical: bool = False) -> None:
    if canonical:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    else:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    path.write_text(encoded, encoding="utf-8")


def _create_authority(
    freeze_path: Path,
    report_path: Path,
) -> VerifiedGateAuthority:
    return create_verified_gate_authority(
        extension_report=report_path,
        freeze_record=freeze_path,
        prototype_output_root=freeze_path.parent / "prototype-results",
        extension_output_root=freeze_path.parent / "extension-results",
    )


def _verify_authority(
    authority_path: Path,
    freeze_path: Path,
    report_path: Path,
) -> VerifiedGateAuthority:
    return verify_gate_authority(
        authority_record=authority_path,
        extension_report=report_path,
        freeze_record=freeze_path,
        prototype_output_root=freeze_path.parent / "prototype-results",
        extension_output_root=freeze_path.parent / "extension-results",
    )


@pytest.fixture
def authority_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    freeze_path = tmp_path / "freeze.json"
    report_path = tmp_path / "extension-report.json"
    freeze = _freeze()
    _write_json(freeze_path, freeze, canonical=True)
    _write_json(report_path, _report())
    monkeypatch.setattr(
        gate_identity,
        "verify_freeze_record",
        lambda path: dict(freeze),
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.review_extension",
        lambda **kwargs: _report(),
    )
    return freeze_path, report_path


def test_authority_is_factory_sealed_deeply_immutable_and_content_bound(
    authority_inputs: tuple[Path, Path],
) -> None:
    freeze_path, report_path = authority_inputs

    with pytest.raises(TypeError, match="verified factory"):
        VerifiedGateAuthority({}, "0" * 64)
    authority = _create_authority(freeze_path, report_path)

    assert isinstance(authority.payload, MappingProxyType)
    assert authority.payload["source_tree_sha256"] == _SHA["source"]
    assert authority.payload["config_sha256"] == _SHA["config"]
    assert authority.payload["freeze_record_file_sha256"] == hashlib.sha256(
        freeze_path.read_bytes()
    ).hexdigest()
    canonical_report = json.dumps(
        _report(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert (
        authority.payload["extension_report_canonical_sha256"]
        == hashlib.sha256(canonical_report).hexdigest()
    )
    bindings = authority.payload["checkpoint_bindings"]
    assert isinstance(bindings, tuple) and len(bindings) == 6
    assert isinstance(bindings[0], MappingProxyType)
    with pytest.raises(TypeError):
        authority.payload["source_tree_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(TypeError):
        bindings[0]["dataset"] = "other"  # type: ignore[index]
    with pytest.raises(AttributeError, match="immutable"):
        authority._method_freeze_sha256 = "0" * 64
    assert len(authority.method_freeze_sha256) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("verdict", "verdict"),
        ("missing_binding", "six"),
        ("duplicate_binding", "grid"),
        ("unknown_binding_field", "fields"),
        ("bad_checkpoint_hash", "SHA-256"),
        ("grid_mismatch", "grid"),
        ("source_mismatch", "source"),
        ("test_access", "test_access"),
        ("missing_multibundle", "fields"),
        ("multibundle_kill", "multi-bundle"),
        ("multibundle_invalid", "multi-bundle"),
    ],
)
def test_factory_rejects_incomplete_or_drifted_extension_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    freeze = _freeze()
    report = _report()
    bindings = report["checkpoint_bindings"]
    assert isinstance(bindings, list)
    if mutation == "verdict":
        report["verdict"] = "revise"
    elif mutation == "missing_binding":
        bindings.pop()
    elif mutation == "duplicate_binding":
        bindings[-1] = dict(bindings[0])
    elif mutation == "unknown_binding_field":
        bindings[0]["invented"] = True
    elif mutation == "bad_checkpoint_hash":
        bindings[0]["neural_checkpoint_tensor_sha256"] = "bad"
    elif mutation == "grid_mismatch":
        report["grid"][0]["job_id"] = "0" * 64
    elif mutation == "source_mismatch":
        report["integrity"]["source_tree_sha256"] = "0" * 64
    elif mutation == "test_access":
        report["integrity"]["test_access"] = True
    elif mutation == "missing_multibundle":
        del report["multibundle_adjudication"]
    elif mutation == "multibundle_kill":
        report["multibundle_adjudication"]["verdict"] = "kill"
    elif mutation == "multibundle_invalid":
        report["multibundle_adjudication"]["integrity"]["valid"] = False
    freeze_path = tmp_path / "freeze.json"
    report_path = tmp_path / "report.json"
    _write_json(freeze_path, freeze, canonical=True)
    _write_json(report_path, report)
    monkeypatch.setattr(
        gate_identity,
        "verify_freeze_record",
        lambda path: dict(freeze),
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.review_extension",
        lambda **kwargs: _report(),
    )

    with pytest.raises(ValueError, match=message):
        _create_authority(freeze_path, report_path)


def test_factory_rejects_duplicate_report_keys(
    authority_inputs: tuple[Path, Path],
) -> None:
    freeze_path, report_path = authority_inputs
    encoded = report_path.read_text(encoding="utf-8")
    report_path.write_text(
        encoded.replace(
            '"schema":',
            '"schema":"duplicate","schema":',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        _create_authority(freeze_path, report_path)


@pytest.mark.parametrize("review_outcome", ["different", "kill"])
def test_factory_rejects_disk_proceed_not_rederived_from_exact_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    review_outcome: str,
) -> None:
    freeze = _freeze()
    disk_report = _report()
    reviewed_report = _report()
    if review_outcome == "different":
        reviewed_report["interpretation"] = "different artifact-derived report"
    else:
        reviewed_report["verdict"] = "kill"
        reviewed_report["confirmation_authorized"] = False
        reviewed_report["multibundle_adjudication"]["verdict"] = "kill"
    freeze_path = tmp_path / "freeze.json"
    report_path = tmp_path / "forged-proceed.json"
    _write_json(freeze_path, freeze, canonical=True)
    _write_json(report_path, disk_report)
    monkeypatch.setattr(
        gate_identity,
        "verify_freeze_record",
        lambda path: dict(freeze),
    )
    calls: list[dict[str, object]] = []

    def fake_review(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return reviewed_report

    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.review_extension",
        fake_review,
    )

    with pytest.raises(ValueError, match="artifact|re-review|rederived"):
        _create_authority(freeze_path, report_path)

    assert calls == [
        {
            "prototype_output_root": tmp_path / "prototype-results",
            "extension_output_root": tmp_path / "extension-results",
            "freeze_record": freeze_path,
        }
    ]


def test_authority_roots_are_required_but_not_part_of_scientific_identity(
    authority_inputs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    freeze_path, report_path = authority_inputs

    first = create_verified_gate_authority(
        extension_report=report_path,
        freeze_record=freeze_path,
        prototype_output_root=tmp_path / "prototype-a",
        extension_output_root=tmp_path / "extension-a",
    )
    second = create_verified_gate_authority(
        extension_report=report_path,
        freeze_record=freeze_path,
        prototype_output_root=tmp_path / "prototype-b",
        extension_output_root=tmp_path / "extension-b",
    )

    assert first.method_freeze_sha256 == second.method_freeze_sha256
    assert first.payload == second.payload
    assert "output_root" not in json.dumps(
        dict(first.payload),
        default=list,
        sort_keys=True,
    )


def test_verify_reruns_artifact_review_instead_of_trusting_signed_json(
    authority_inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_path, report_path = authority_inputs
    authority = _create_authority(freeze_path, report_path)
    authority_path = tmp_path / "gate-authority.json"
    write_gate_authority(authority, authority_path)
    killed = _report()
    killed["verdict"] = "kill"
    killed["confirmation_authorized"] = False
    killed["multibundle_adjudication"]["verdict"] = "kill"
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.review_extension",
        lambda **kwargs: killed,
    )

    with pytest.raises(ValueError, match="artifact|re-review|rederived"):
        _verify_authority(
            authority_path,
            freeze_path,
            report_path,
        )


def test_authority_exclusive_write_readonly_and_strict_current_verification(
    authority_inputs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    freeze_path, report_path = authority_inputs
    authority = _create_authority(freeze_path, report_path)
    path = tmp_path / "gate-authority.json"

    write_gate_authority(authority, path)

    assert path.stat().st_mode & 0o222 == 0
    assert (
        _verify_authority(
            path,
            freeze_path,
            report_path,
        ).method_freeze_sha256
        == authority.method_freeze_sha256
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        write_gate_authority(authority, path)

    report = _report()
    report["interpretation"] = "current report changed"
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="current .*report|authority"):
        _verify_authority(path, freeze_path, report_path)


def test_verify_rejects_authority_symlink_duplicate_key_and_unknown_field(
    authority_inputs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    freeze_path, report_path = authority_inputs
    authority = _create_authority(freeze_path, report_path)
    real = tmp_path / "real.json"
    write_gate_authority(authority, real)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        _verify_authority(link, freeze_path, report_path)

    record = json.loads(real.read_text(encoding="ascii"))
    record["unknown"] = True
    unknown = tmp_path / "unknown.json"
    _write_json(unknown, record, canonical=True)
    unknown.chmod(0o444)
    with pytest.raises(ValueError, match="fields"):
        _verify_authority(unknown, freeze_path, report_path)

    encoded = real.read_text(encoding="ascii")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        encoded.replace(
            '"schema":',
            '"schema":"duplicate","schema":',
            1,
        ),
        encoding="ascii",
    )
    duplicate.chmod(0o444)
    with pytest.raises(ValueError, match="duplicate"):
        _verify_authority(duplicate, freeze_path, report_path)


@pytest.mark.parametrize("bundle", [1, 2, 3])
@pytest.mark.parametrize("dataset", ["abilene", "geant"])
def test_gate_job_is_exact_fixed_six_job_checkpoint_reuse_identity(
    authority_inputs: tuple[Path, Path],
    tmp_path: Path,
    dataset: str,
    bundle: int,
) -> None:
    freeze_path, report_path = authority_inputs
    authority = _create_authority(freeze_path, report_path)
    first = GateJobSpec(
        dataset=dataset,
        seed_bundle=bundle,
        output_root=tmp_path / "a",
        authority=authority,
        window_schedule_sha256=_SHA["window"],
    )
    second = GateJobSpec(
        dataset=dataset,
        seed_bundle=bundle,
        output_root=tmp_path / "b",
        authority=authority,
        window_schedule_sha256=_SHA["window"],
    )

    identity = first.scientific_identity
    assert first.job_id == second.job_id
    assert first.job_directory == (tmp_path / "a").resolve() / first.job_id
    assert identity["stage"] == "final_gate"
    assert identity["evaluation_cohort"] == "gate"
    assert identity["training"] is False
    assert identity["checkpoint_reuse"] == "exact_verified_full_tune"
    assert identity["compute_lane"] == "cuda_bf16_math_sdp"
    assert identity["flows"] == {"abilene": 144, "geant": 462}[dataset]
    assert identity["mask_families"] == [
        "random",
        "internal_block",
        "two_burst",
    ]
    assert identity["method_freeze_sha256"] == authority.method_freeze_sha256
    assert identity["window_schedule_sha256"] == _SHA["window"]
    assert identity["test_access"] is False
    assert identity["source_training_job_id"] == _binding(
        bundle, dataset
    )["source_training_job_id"]
    assert identity["source_neural_checkpoint_file_sha256"] == _SHA[
        "neural_file"
    ]
    assert identity["source_neural_checkpoint_tensor_sha256"] == _SHA[
        "neural_tensor"
    ]
    assert identity["source_acil_checkpoint_file_sha256"] == _SHA["acil"]
    encoded = json.dumps(identity, sort_keys=True).lower()
    assert "output_root" not in encoded
    assert "data_path" not in encoded


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dataset": "wsdream"}, "dataset"),
        ({"seed_bundle": 0}, "bundle"),
        ({"seed_bundle": 4}, "bundle"),
        ({"seed_bundle": True}, "bundle"),
        ({"window_schedule_sha256": "bad"}, "window"),
        ({"output_root": Path("/")}, "output_root"),
        ({"authority": object()}, "authority"),
    ],
)
def test_gate_job_rejects_any_axis_or_unverified_authority(
    authority_inputs: tuple[Path, Path],
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    freeze_path, report_path = authority_inputs
    authority = _create_authority(freeze_path, report_path)
    values: dict[str, object] = {
        "dataset": "abilene",
        "seed_bundle": 1,
        "output_root": tmp_path,
        "authority": authority,
        "window_schedule_sha256": _SHA["window"],
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError), match=message):
        GateJobSpec(**values)  # type: ignore[arg-type]
