from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.anchorcv_v1 import gate_identity
from experiments.anchorcv_v1.gate_data import gate_window_schedule_sha256
from experiments.anchorcv_v1.gate_identity import (
    GateJobSpec,
    create_verified_gate_authority,
)
from experiments.anchorcv_v1.job import JobSpec
from experiments.anchorcv_v1.job_runtime import (
    evaluate_evaluation_batch as registered_evaluate_evaluation_batch,
    scientific_tensor_sha256,
)
from experiments.anchorcv_v1.protocol import fingerprint


_SOURCE_TREE = "a" * 64
_CONFIG = fingerprint()
_DATASETS = ("abilene", "geant")
_BUNDLES = (1, 2, 3)
_FAMILIES = ("random", "internal_block", "two_burst")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object, *, canonical: bool = False) -> None:
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
    path.write_text(encoded + ("" if canonical else "\n"), encoding="utf-8")


def _source_spec(
    prototype_root: Path,
    extension_root: Path,
    *,
    dataset: str,
    bundle: int,
) -> JobSpec:
    stage = "prototype" if bundle == 1 else "extension"
    return JobSpec(
        dataset=dataset,
        seed_bundle=bundle,
        stage=stage,
        output_root=prototype_root if stage == "prototype" else extension_root,
        device="cuda",
        source_tree_sha256=_SOURCE_TREE,
        config_sha256=_CONFIG,
    )


def _placeholder_binding(spec: JobSpec) -> dict[str, object]:
    return {
        "dataset": spec.dataset,
        "seed_bundle": spec.seed_bundle,
        "source_training_stage": spec.stage,
        "source_training_job_id": spec.job_id,
        "source_result_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "neural_checkpoint_file_sha256": "d" * 64,
        "neural_checkpoint_tensor_sha256": "e" * 64,
        "acil_checkpoint_file_sha256": spec.scientific_identity[
            "acil_checkpoint_file_sha256"
        ],
        "best_epoch": 3,
        "best_source_dev_nmae": 0.125,
    }


def _write_selected_source(
    spec: JobSpec,
    *,
    reported_checkpoint_name: str = "neural_best.safetensors",
) -> dict[str, object]:
    directory = spec.job_directory
    directory.mkdir(parents=True)
    checkpoint = directory / "neural_best.safetensors"
    checkpoint.write_bytes(b"test-only-neural-state")
    checkpoint_file_sha = _sha(checkpoint)
    state = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    checkpoint_tensor_sha = scientific_tensor_sha256(state)
    identity = spec.scientific_identity
    result = {
        "job_id": spec.job_id,
        "status": "succeeded",
        "scientific_identity": identity,
        "neural_parameter_count": 1,
        "neural_checkpoint": {
            "file": reported_checkpoint_name,
            "file_sha256": checkpoint_file_sha,
            "tensor_sha256": checkpoint_tensor_sha,
            "best_epoch": 3,
            "best_source_dev_nmae": 0.125,
        },
        "prior_checkpoint": {
            "file_sha256": identity["acil_checkpoint_file_sha256"],
        },
        "training": {
            "epochs_completed": 20,
            "optimizer_updates": 320,
            "best_epoch": 3,
            "best_source_dev_nmae": 0.125,
            "epoch_records": [],
        },
        "cells": {},
    }
    result_path = directory / "result.json"
    _json(result_path, result)
    manifest = {
        **identity,
        "job_id": spec.job_id,
        "status": "succeeded",
        "result_file": "result.json",
        "result_sha256": _sha(result_path),
        "neural_checkpoint_file_sha256": checkpoint_file_sha,
        "neural_checkpoint_tensor_sha256": checkpoint_tensor_sha,
    }
    manifest_path = directory / "manifest.json"
    _json(manifest_path, manifest)
    return {
        "dataset": spec.dataset,
        "seed_bundle": spec.seed_bundle,
        "source_training_stage": spec.stage,
        "source_training_job_id": spec.job_id,
        "source_result_sha256": _sha(result_path),
        "source_manifest_sha256": _sha(manifest_path),
        "neural_checkpoint_file_sha256": checkpoint_file_sha,
        "neural_checkpoint_tensor_sha256": checkpoint_tensor_sha,
        "acil_checkpoint_file_sha256": identity[
            "acil_checkpoint_file_sha256"
        ],
        "best_epoch": 3,
        "best_source_dev_nmae": 0.125,
    }


def _extension_report(
    prototype_root: Path,
    extension_root: Path,
    selected: dict[str, object],
) -> dict[str, object]:
    bindings = []
    for bundle in _BUNDLES:
        for dataset in _DATASETS:
            spec = _source_spec(
                prototype_root,
                extension_root,
                dataset=dataset,
                bundle=bundle,
            )
            item = _placeholder_binding(spec)
            if dataset == selected["dataset"] and bundle == selected["seed_bundle"]:
                item = dict(selected)
            bindings.append(item)
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
            "source_tree_sha256": _SOURCE_TREE,
            "config_sha256": _CONFIG,
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
        "interpretation": "test-only verified full-tune report",
    }


def _make_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reported_checkpoint_name: str = "neural_best.safetensors",
) -> tuple[GateJobSpec, Path, Path]:
    prototype_root = tmp_path / "prototype"
    extension_root = tmp_path / "extension"
    prototype_root.mkdir()
    extension_root.mkdir()
    source_spec = _source_spec(
        prototype_root,
        extension_root,
        dataset="abilene",
        bundle=1,
    )
    selected = _write_selected_source(
        source_spec,
        reported_checkpoint_name=reported_checkpoint_name,
    )
    report = _extension_report(prototype_root, extension_root, selected)
    freeze = {
        "schema_version": 1,
        "protocol": "anchorcv-v1",
        "source_tree_sha256": _SOURCE_TREE,
        "config_sha256": _CONFIG,
        "git_available": False,
        "git_commit": None,
    }
    report_path = tmp_path / "extension-report.json"
    freeze_path = tmp_path / "freeze.json"
    _json(report_path, report)
    _json(freeze_path, freeze, canonical=True)
    monkeypatch.setattr(
        gate_identity,
        "verify_freeze_record",
        lambda path: dict(freeze),
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.review_extension",
        lambda **kwargs: report,
    )
    authority = create_verified_gate_authority(
        extension_report=report_path,
        freeze_record=freeze_path,
        prototype_output_root=prototype_root,
        extension_output_root=extension_root,
    )
    spec = GateJobSpec(
        dataset="abilene",
        seed_bundle=1,
        output_root=tmp_path / "gate-output",
        authority=authority,
        window_schedule_sha256=gate_window_schedule_sha256("abilene"),
    )
    return spec, prototype_root, extension_root


class _FakeModel:
    def load_state_dict(self, state, *, strict: bool):
        assert strict is True
        self.state = state
        return SimpleNamespace()

    def requires_grad_(self, enabled: bool):
        assert enabled is False
        return self

    def eval(self):
        return self

    def to(self, device):
        assert device.type == "cuda"
        return self


class _FakePrior:
    def eval(self):
        return self


class _FakeEvidence:
    def __init__(self, marker: int) -> None:
        self.marker = marker

    def as_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "window_index": np.array([self.marker], dtype="<i8"),
            "hard_error": np.array([float(self.marker + 1)], dtype="<f8"),
        }


def _stub_success_lane(
    monkeypatch: pytest.MonkeyPatch,
    spec: GateJobSpec,
) -> tuple[list[str], list[object]]:
    from experiments.anchorcv_v1 import gate_runtime

    gate_loads: list[object] = []
    built_families: list[str] = []
    binding = spec.authority.checkpoint_binding(
        spec.dataset, spec.seed_bundle
    )
    monkeypatch.setattr(
        gate_runtime,
        "load_safetensors",
        lambda path, device="cpu": {
            "weight": torch.tensor([1.0], dtype=torch.float32)
        },
    )
    monkeypatch.setattr(
        gate_runtime,
        "verify_neural_state",
        lambda dataset, state, reported_parameter_count: None,
    )
    monkeypatch.setattr(
        gate_runtime,
        "_build_frozen_neural",
        lambda dataset: _FakeModel(),
    )
    monkeypatch.setattr(
        gate_runtime,
        "load_frozen_acil",
        lambda dataset, bundle, device: (
            _FakePrior(),
            SimpleNamespace(
                file_sha256=binding["acil_checkpoint_file_sha256"]
            ),
        ),
    )
    monkeypatch.setattr(
        gate_runtime,
        "_load_gate_windows",
        lambda authority, dataset: gate_loads.append(authority)
        or SimpleNamespace(dataset=dataset, cohort="gate"),
    )
    monkeypatch.setattr(
        gate_runtime,
        "permitted_fit_fallback",
        lambda dataset: SimpleNamespace(mean=0.0, std=1.0),
    )

    def build_batch(windows, *, seed_bundle: int, family: str):
        built_families.append(family)
        marker = hashlib.sha256(family.encode("ascii")).hexdigest()
        return SimpleNamespace(
            family=family,
            mask_sha256=((marker,),),
            oracle_q_sha256=(("a" * 64,),),
            oracle_e_sha256=(("b" * 64,),),
        )

    monkeypatch.setattr(gate_runtime, "build_evaluation_batch", build_batch)
    monkeypatch.setattr(
        gate_runtime,
        "evaluate_evaluation_batch",
        lambda **kwargs: _FakeEvidence(len(built_families)),
    )
    monkeypatch.setattr(
        gate_runtime,
        "summarize_case_evidence",
        lambda evidence: {"marker": evidence.marker},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "fake")
    return built_families, gate_loads


def test_gate_runtime_reuses_registered_truth_independent_evaluator() -> None:
    from experiments.anchorcv_v1 import gate_runtime

    assert (
        gate_runtime.evaluate_evaluation_batch
        is registered_evaluate_evaluation_batch
    )


def test_execute_gate_job_reuses_exact_checkpoint_without_training_and_writes_three_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.anchorcv_v1 import gate_runtime
    from experiments.anchorcv_v1 import training_runtime
    from experiments.acil_innovation_v1 import training as upstream_training

    spec, prototype_root, extension_root = _make_spec(tmp_path, monkeypatch)
    families, gate_loads = _stub_success_lane(monkeypatch, spec)

    def forbidden(*args, **kwargs):
        raise AssertionError("a final-gate job must never train")

    monkeypatch.setattr(training_runtime, "train_registered_expert", forbidden)
    monkeypatch.setattr(upstream_training, "run_protocol_training", forbidden)

    result = gate_runtime.execute_gate_job(
        spec,
        prototype_output_root=prototype_root,
        extension_output_root=extension_root,
    )

    assert result["training_performed"] is False
    assert result["source_binding"] == dict(
        spec.authority.checkpoint_binding("abilene", 1)
    )
    assert tuple(result["cells"]) == _FAMILIES
    assert families == list(_FAMILIES)
    assert gate_loads == [spec.authority]
    assert gate_runtime.verify_completed_gate_job(spec) is True
    names = {item.name for item in spec.job_directory.iterdir()}
    assert not any("checkpoint" in name or name.endswith(".safetensors") for name in names)
    assert names == {
        "attempt_manifest.json",
        "evidence_random.npz",
        "evidence_internal_block.npz",
        "evidence_two_burst.npz",
        "result.json",
        "runtime.json",
        "manifest.json",
    }

    second = gate_runtime.execute_gate_job(
        spec,
        prototype_output_root=tmp_path / "unused-prototype",
        extension_output_root=tmp_path / "unused-extension",
    )
    assert second == result


@pytest.mark.parametrize(
    "failure",
    (
        "result_hash",
        "source_symlink",
        "wrong_stage_root",
        "path_traversal",
        "forged_checkpoint",
    ),
)
def test_source_tampering_fails_before_gate_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from experiments.anchorcv_v1 import gate_runtime

    reported_name = (
        "../../neural_best.safetensors"
        if failure == "path_traversal"
        else "neural_best.safetensors"
    )
    spec, prototype_root, extension_root = _make_spec(
        tmp_path,
        monkeypatch,
        reported_checkpoint_name=reported_name,
    )
    _, gate_loads = _stub_success_lane(monkeypatch, spec)
    source_job = prototype_root / str(
        spec.scientific_identity["source_training_job_id"]
    )
    if failure == "result_hash":
        with (source_job / "result.json").open("ab") as handle:
            handle.write(b" ")
    elif failure == "source_symlink":
        checkpoint = source_job / "neural_best.safetensors"
        target = tmp_path / "checkpoint-target"
        target.write_bytes(checkpoint.read_bytes())
        checkpoint.unlink()
        checkpoint.symlink_to(target)
    elif failure == "wrong_stage_root":
        prototype_root, extension_root = extension_root, prototype_root
    elif failure == "forged_checkpoint":
        monkeypatch.setattr(
            gate_runtime,
            "verify_neural_state",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("neural checkpoint tensor schema drifted")
            ),
        )

    with pytest.raises((ValueError, FileNotFoundError), match="source|checkpoint|artifact|stage|hash|filename|directory"):
        gate_runtime.execute_gate_job(
            spec,
            prototype_output_root=prototype_root,
            extension_output_root=extension_root,
        )
    assert gate_loads == []
    assert (spec.job_directory / "attempt_manifest.json").is_file()
    assert (spec.job_directory / "failure.json").is_file()


def test_completed_gate_verifier_rejects_partial_symlink_and_result_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.anchorcv_v1 import gate_runtime

    spec, prototype_root, extension_root = _make_spec(tmp_path, monkeypatch)
    _stub_success_lane(monkeypatch, spec)
    gate_runtime.execute_gate_job(
        spec,
        prototype_output_root=prototype_root,
        extension_output_root=extension_root,
    )
    (spec.job_directory / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schema|hash|identity"):
        gate_runtime.verify_completed_gate_job(spec)

    partial_spec = GateJobSpec(
        dataset=spec.dataset,
        seed_bundle=spec.seed_bundle,
        output_root=tmp_path / "partial",
        authority=spec.authority,
        window_schedule_sha256=spec.window_schedule_sha256,
    )
    partial_spec.job_directory.mkdir(parents=True)
    (partial_spec.job_directory / "manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="partial"):
        gate_runtime.verify_completed_gate_job(partial_spec)

    symlink_spec = GateJobSpec(
        dataset=spec.dataset,
        seed_bundle=spec.seed_bundle,
        output_root=tmp_path / "symlink",
        authority=spec.authority,
        window_schedule_sha256=spec.window_schedule_sha256,
    )
    symlink_spec.output_root.mkdir(parents=True)
    symlink_spec.job_directory.symlink_to(spec.job_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|directory"):
        gate_runtime.verify_completed_gate_job(symlink_spec)
