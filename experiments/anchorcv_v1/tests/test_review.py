from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.acil_innovation_v1.batching import EvaluationBatch
from experiments.acil_innovation_v1.preprocessing import (
    FitFallback,
    observation_statistics,
)
from experiments.anchorcv_v1.evaluation import CaseEvidence
from experiments.anchorcv_v1.job_runtime import scientific_array_sha256
from experiments.anchorcv_v1.review import (
    AuditedTrainingJob,
    audit_training_job,
    build_extension_report,
    build_parser,
    file_sha256,
    load_evidence_npz,
    safe_job_artifact_path,
    verify_training_record,
    verify_neural_state,
    verify_registered_evidence,
)
from experiments.anchorcv_v1.launcher import build_stage_specs
from experiments.anchorcv_v1.protocol import fingerprint
from experiments.anchorcv_v1.training_runtime import (
    build_neural_expert,
    model_parameter_count,
)


def _evidence() -> CaseEvidence:
    return CaseEvidence(
        window_index=np.array([0]),
        flow_index=np.array([0]),
        middle_anchor_index=np.array([3], dtype=np.int64),
        p_error_sum=np.array([2.0]),
        n_error_sum=np.array([1.0]),
        hard_error_sum=np.array([1.0]),
        oracle_error_sum=np.array([1.0]),
        truth_sum=np.array([10.0]),
        target_count=np.array([4], dtype=np.int64),
        k3_scale=np.array([1.0]),
        p_anchor_absolute_error=np.array([2.0]),
        n_anchor_absolute_error=np.array([1.0]),
        p_anchor_normalized_error=np.array([2.0]),
        n_anchor_normalized_error=np.array([1.0]),
        loo_score=np.array([1.0]),
        target_regret=np.array([0.25]),
        hard_winner_is_n=np.array([True]),
        oracle_winner_is_n=np.array([True]),
    )


def test_review_parser_exposes_only_result_artifact_paths_not_data_authority() -> None:
    destinations = {action.dest for action in build_parser()._actions}

    assert destinations == {"help", "output_root", "freeze_record", "output"}
    assert not {"cohort", "split", "test_path", "data_path"} & destinations


def test_evidence_loader_recomputes_content_hash_and_case_invariants(tmp_path: Path) -> None:
    evidence = _evidence()
    path = tmp_path / "evidence.npz"
    np.savez_compressed(path, **evidence.as_npz_dict())

    loaded = load_evidence_npz(path)

    assert loaded.case_count == 1
    assert scientific_array_sha256(loaded.as_npz_dict()) == scientific_array_sha256(
        evidence.as_npz_dict()
    )
    assert len(file_sha256(path)) == 64


@pytest.mark.parametrize(
    "reported",
    (
        "../../sealed_test.npz",
        "/tmp/sealed_test.npz",
        "wrong-name.npz",
    ),
)
def test_review_artifact_path_is_fixed_and_cannot_escape_job_directory(
    tmp_path: Path, reported: str
) -> None:
    job = tmp_path / "job"
    job.mkdir()

    with pytest.raises(ValueError, match="filename"):
        safe_job_artifact_path(
            job,
            reported,
            expected_name="evidence_random.npz",
        )


def test_review_artifact_path_rejects_symlink_even_with_expected_name(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    target = tmp_path / "outside.npz"
    target.write_bytes(b"not evidence")
    (job / "evidence_random.npz").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        safe_job_artifact_path(
            job,
            "evidence_random.npz",
            expected_name="evidence_random.npz",
        )


def _registered_pair() -> tuple[CaseEvidence, EvaluationBatch, FitFallback]:
    truth = torch.tensor([[[0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 2.1213202]]])
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., (0, 3, 6)] = True
    model_input = torch.where(
        observed, truth, torch.full_like(truth, float("nan"))
    )
    fallback = FitFallback(mean=2.0, std=1.0)
    scale = float(
        observation_statistics(model_input, observed, fallback)
        .std.reshape(-1)[0]
        .item()
    )
    evidence = CaseEvidence(
        window_index=np.array([0]),
        flow_index=np.array([0]),
        middle_anchor_index=np.array([3], dtype=np.int64),
        p_error_sum=np.array([2.0]),
        n_error_sum=np.array([1.0]),
        hard_error_sum=np.array([1.0]),
        oracle_error_sum=np.array([1.0]),
        truth_sum=np.array([10.0]),
        target_count=np.array([4], dtype=np.int64),
        k3_scale=np.array([scale]),
        p_anchor_absolute_error=np.array([2.0]),
        n_anchor_absolute_error=np.array([1.0]),
        p_anchor_normalized_error=np.array([2.0 / scale]),
        n_anchor_normalized_error=np.array([1.0 / scale]),
        loo_score=np.array([1.0 / scale]),
        target_regret=np.array([1.0 / (4.0 * scale)]),
        hard_winner_is_n=np.array([True]),
        oracle_winner_is_n=np.array([True]),
    )
    batch = EvaluationBatch(
        dataset="abilene",
        cohort="tune",
        seed_bundle=1,
        family="random",
        window_indices=(0,),
        absolute_starts=(0,),
        mask_identities=tuple(),
        mask_sha256=(("a" * 64,),),
        oracle_q_sha256=(("b" * 64,),),
        oracle_e_sha256=(("c" * 64,),),
        truth=truth,
        model_input=model_input,
        observed=observed,
        target=~observed,
        controlled_gap=torch.zeros_like(observed),
        oracle_q=torch.zeros_like(observed),
        oracle_e=~observed,
        oracle_support_q=None,  # type: ignore[arg-type]
    )
    return evidence, batch, fallback


def test_review_binds_evidence_rows_middle_anchor_scale_and_truth_to_registry() -> None:
    evidence, batch, fallback = _registered_pair()

    verify_registered_evidence(
        evidence,
        registered=batch,
        fit_fallback=fallback,
    )
    payload = evidence.as_npz_dict()
    payload["middle_anchor_index"] = np.array([2], dtype=np.int64)
    tampered = CaseEvidence(**payload)

    with pytest.raises(ValueError, match="middle-anchor"):
        verify_registered_evidence(
            tampered,
            registered=batch,
            fit_fallback=fallback,
        )


def test_review_allows_only_measured_gpu_fp32_k3_scale_roundoff() -> None:
    evidence, batch, fallback = _registered_pair()
    payload = evidence.as_npz_dict()
    rounded_scale = np.asarray(evidence.k3_scale) * (1.0 + 1.8e-7)
    payload["k3_scale"] = rounded_scale
    payload["p_anchor_normalized_error"] = (
        payload["p_anchor_absolute_error"] / rounded_scale
    )
    payload["n_anchor_normalized_error"] = (
        payload["n_anchor_absolute_error"] / rounded_scale
    )
    payload["loo_score"] = (
        payload["p_anchor_normalized_error"]
        - payload["n_anchor_normalized_error"]
    )
    payload["target_regret"] = (
        payload["p_error_sum"] - payload["n_error_sum"]
    ) / (payload["target_count"] * rounded_scale)
    rounded = CaseEvidence(**payload)

    verify_registered_evidence(
        rounded,
        registered=batch,
        fit_fallback=fallback,
    )

    payload = rounded.as_npz_dict()
    material_scale = np.asarray(evidence.k3_scale) * (1.0 + 3.0e-7)
    payload["k3_scale"] = material_scale
    payload["p_anchor_normalized_error"] = (
        payload["p_anchor_absolute_error"] / material_scale
    )
    payload["n_anchor_normalized_error"] = (
        payload["n_anchor_absolute_error"] / material_scale
    )
    payload["loo_score"] = (
        payload["p_anchor_normalized_error"]
        - payload["n_anchor_normalized_error"]
    )
    payload["target_regret"] = (
        payload["p_error_sum"] - payload["n_error_sum"]
    ) / (payload["target_count"] * material_scale)
    material = CaseEvidence(**payload)
    with pytest.raises(ValueError, match="K3 scale"):
        verify_registered_evidence(
            material,
            registered=batch,
            fit_fallback=fallback,
        )


def test_review_strictly_binds_checkpoint_tensor_schema_and_parameter_count() -> None:
    model = build_neural_expert("abilene")
    state = model.state_dict()
    count = model_parameter_count(model)

    verify_neural_state("abilene", state, reported_parameter_count=count)
    tampered = dict(state)
    tampered.pop(next(iter(tampered)))
    with pytest.raises(ValueError, match="schema"):
        verify_neural_state(
            "abilene",
            tampered,
            reported_parameter_count=count,
        )


def _valid_training_record() -> tuple[dict[str, object], dict[str, object]]:
    records = []
    for epoch in range(20):
        truth = 100.0
        nmae = 0.25 - 0.005 * epoch
        records.append(
            {
                "epoch": epoch,
                "physical_batches": 64,
                "optimizer_updates": 16,
                "mean_loss": 0.5 + 0.01 * epoch,
                "source_dev_absolute_error_sum": nmae * truth,
                "source_dev_absolute_truth_sum": truth,
                "source_dev_nmae": nmae,
                "selected_as_best": epoch == 19,
            }
        )
    training = {
        "epochs_completed": 20,
        "optimizer_updates": 320,
        "best_epoch": 19,
        "best_source_dev_nmae": records[19]["source_dev_nmae"],
        "epoch_records": records,
    }
    checkpoint = {
        "file": "neural_best.safetensors",
        "file_sha256": "a" * 64,
        "tensor_sha256": "b" * 64,
        "best_epoch": 19,
        "best_source_dev_nmae": records[19]["source_dev_nmae"],
    }
    return training, checkpoint


@pytest.mark.parametrize(
    "fault",
    (
        "budget",
        "record_schema",
        "record_epoch",
        "record_ratio",
        "selected_count",
        "not_argmin",
        "checkpoint_epoch",
        "checkpoint_metric",
    ),
)
def test_review_binds_full_training_budget_selection_and_checkpoint(
    fault: str,
) -> None:
    training, checkpoint = _valid_training_record()
    records = training["epoch_records"]
    assert isinstance(records, list)
    if fault == "budget":
        training["optimizer_updates"] = 319
    elif fault == "record_schema":
        records[0]["unexpected"] = 1
    elif fault == "record_epoch":
        records[2]["epoch"] = 3
    elif fault == "record_ratio":
        records[2]["source_dev_nmae"] = 0.99
    elif fault == "selected_count":
        records[0]["selected_as_best"] = True
    elif fault == "not_argmin":
        records[0]["source_dev_absolute_error_sum"] = 1.0
        records[0]["source_dev_nmae"] = 0.01
    elif fault == "checkpoint_epoch":
        checkpoint["best_epoch"] = 18
    elif fault == "checkpoint_metric":
        checkpoint["best_source_dev_nmae"] = 0.99

    with pytest.raises(ValueError, match="training|epoch|checkpoint|source-dev"):
        verify_training_record(training, checkpoint=checkpoint)


def test_review_accepts_exact_full_training_record() -> None:
    training, checkpoint = _valid_training_record()

    verify_training_record(training, checkpoint=checkpoint)


def test_formal_training_audit_replays_all_tune_evidence_from_bound_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = {
        "source_tree_sha256": "a" * 64,
        "config_sha256": fingerprint(),
        "git_available": False,
        "git_commit": None,
    }
    spec = build_stage_specs("prototype", tmp_path, freeze)[0]
    data_sha = spec.scientific_identity["data_sha256"]
    freeze["data_identities"] = {
        spec.dataset: {"data_sha256": data_sha}
    }
    training, checkpoint = _valid_training_record()
    checkpoint["file_sha256"] = "b" * 64
    checkpoint["tensor_sha256"] = "c" * 64
    evidence = _evidence()
    cells = {
        family: {
            "summary": {"ok": True},
            "evidence_file": f"evidence_{family}.npz",
            "evidence_file_sha256": "e" * 64,
            "evidence_content_sha256": "f" * 64,
            "mask_sha256": "1" * 64,
            "oracle_q_sha256": "1" * 64,
            "oracle_e_sha256": "1" * 64,
        }
        for family in ("random", "internal_block", "two_burst")
    }
    result = {
        "job_id": spec.job_id,
        "status": "succeeded",
        "scientific_identity": spec.scientific_identity,
        "neural_parameter_count": 1,
        "neural_checkpoint": checkpoint,
        "prior_checkpoint": {
            "file_sha256": spec.scientific_identity[
                "acil_checkpoint_file_sha256"
            ],
        },
        "training": training,
        "cells": cells,
    }
    manifest = {
        **spec.scientific_identity,
        "job_id": spec.job_id,
        "result_sha256": "d" * 64,
        "neural_checkpoint_file_sha256": "b" * 64,
        "neural_checkpoint_tensor_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.verify_completed_job",
        lambda _spec: True,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.safe_job_artifact_path",
        lambda _directory, _reported, *, expected_name: tmp_path
        / expected_name,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review._json_object",
        lambda path: result if path.name == "result.json" else manifest,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.file_sha256",
        lambda path: {
            "result.json": "d" * 64,
            "manifest.json": "9" * 64,
            "neural_best.safetensors": "b" * 64,
        }.get(path.name, "e" * 64),
    )
    state = {"weight": torch.ones(1)}
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.load_safetensors",
        lambda *_args, **_kwargs: state,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.scientific_tensor_sha256",
        lambda _state: "c" * 64,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.scientific_array_sha256",
        lambda _arrays: "f" * 64,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.verify_neural_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.verify_training_record",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.load_permitted_windows",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.load_evidence_npz",
        lambda _path: evidence,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.summarize_case_evidence",
        lambda _evidence: {"ok": True},
    )
    registered = _registered_pair()[1]
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.build_evaluation_batch",
        lambda *_args, **_kwargs: registered,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.verify_registered_evidence",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.mask_identity_sha256",
        lambda _grid: "1" * 64,
    )
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.replay_and_verify_tune_evidence",
        lambda **kwargs: captured.append(kwargs),
    )

    audited = audit_training_job(spec, freeze=freeze)

    assert audited.spec == spec
    assert len(captured) == 1
    assert captured[0]["neural_state"] is state
    assert captured[0]["expected_neural_tensor_sha256"] == "c" * 64
    assert (
        captured[0]["expected_prior_file_sha256"]
        == spec.scientific_identity["acil_checkpoint_file_sha256"]
    )
    assert set(captured[0]["registered_batches"]) == {
        "random",
        "internal_block",
        "two_burst",
    }
    assert set(captured[0]["persisted"]) == {
        "random",
        "internal_block",
        "two_burst",
    }


def test_extension_report_requires_and_maps_exact_six_verified_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = {
        "source_tree_sha256": "a" * 64,
        "config_sha256": fingerprint(),
        "git_available": False,
        "git_commit": None,
    }
    specs = (
        *build_stage_specs("prototype", tmp_path / "prototype", freeze),
        *build_stage_specs("extension", tmp_path / "extension", freeze),
    )
    audited = [
        AuditedTrainingJob(
            spec=spec,
            result={
                "job_id": spec.job_id,
                "neural_checkpoint": {
                    "file_sha256": "b" * 64,
                    "tensor_sha256": "c" * 64,
                },
                "prior_checkpoint": {
                    "file_sha256": spec.scientific_identity[
                        "acil_checkpoint_file_sha256"
                    ],
                },
                "training": {
                    "best_epoch": 3,
                    "best_source_dev_nmae": 0.2,
                },
            },
            manifest={
                "job_id": spec.job_id,
                "result_sha256": "d" * 64,
            },
            evidence={
                family: f"{spec.seed_bundle}/{spec.dataset}/{family}"
                for family in ("random", "internal_block", "two_burst")
            },  # type: ignore[arg-type]
            result_file_sha256="d" * 64,
            manifest_file_sha256="e" * 64,
        )
        for spec in specs
    ]
    captured = []
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.analyze_extension_uncertainty",
        lambda grid: captured.append(grid)
        or {"schema_version": "frozen-uncertainty"},
    )
    summary_grid = []
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.summarize_case_evidence",
        lambda evidence: {"evidence": evidence},
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.adjudicate_multibundle",
        lambda summaries, uncertainty: summary_grid.append(
            (summaries, uncertainty)
        )
        or {
            "schema": "anchorcv-v1:multibundle-adjudication:v1",
            "verdict": "proceed",
            "integrity": {"valid": True},
        },
    )
    prototype = {
        "verdict": "proceed",
        "integrity": {
            "valid": True,
            "common_identity": {
                "source_tree_sha256": "a" * 64,
                "config_sha256": fingerprint(),
            },
        },
    }

    report = build_extension_report(
        prototype_report=prototype,
        audited_jobs=audited,
        freeze=freeze,
    )

    assert report["schema"] == "anchorcv-v1:extension-review:v1"
    assert report["status"] == "verified_complete"
    assert report["verdict"] == "proceed"
    assert report["confirmation_authorized"] is True
    assert report["multibundle_adjudication"]["verdict"] == "proceed"
    assert report["job_count"] == 6
    assert len(report["checkpoint_bindings"]) == 6
    assert report["checkpoint_bindings"][0][
        "neural_checkpoint_tensor_sha256"
    ] == "c" * 64
    assert set(captured[0]) == {
        (bundle, dataset, family)
        for bundle in (1, 2, 3)
        for dataset in ("abilene", "geant")
        for family in ("random", "internal_block", "two_burst")
    }
    assert set(summary_grid[0][0]) == set(captured[0])
    assert summary_grid[0][1] == {"schema_version": "frozen-uncertainty"}


def test_extension_report_scientific_failure_does_not_authorize_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = {
        "source_tree_sha256": "a" * 64,
        "config_sha256": fingerprint(),
        "git_available": False,
        "git_commit": None,
    }
    specs = (
        *build_stage_specs("prototype", tmp_path / "prototype", freeze),
        *build_stage_specs("extension", tmp_path / "extension", freeze),
    )
    audited = [
        AuditedTrainingJob(
            spec=spec,
            result={
                "job_id": spec.job_id,
                "neural_checkpoint": {
                    "file_sha256": "b" * 64,
                    "tensor_sha256": "c" * 64,
                },
                "prior_checkpoint": {
                    "file_sha256": spec.scientific_identity[
                        "acil_checkpoint_file_sha256"
                    ],
                },
                "training": {
                    "best_epoch": 3,
                    "best_source_dev_nmae": 0.2,
                },
            },
            manifest={"job_id": spec.job_id, "result_sha256": "d" * 64},
            evidence={
                family: f"{spec.seed_bundle}/{spec.dataset}/{family}"
                for family in ("random", "internal_block", "two_burst")
            },  # type: ignore[arg-type]
            result_file_sha256="d" * 64,
            manifest_file_sha256="e" * 64,
        )
        for spec in specs
    ]
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.analyze_extension_uncertainty",
        lambda _grid: {"schema_version": "frozen-uncertainty"},
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.summarize_case_evidence",
        lambda evidence: {"evidence": evidence},
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.adjudicate_multibundle",
        lambda _summaries, _uncertainty: {
            "schema": "anchorcv-v1:multibundle-adjudication:v1",
            "verdict": "kill",
            "integrity": {"valid": True},
        },
    )
    prototype = {
        "verdict": "proceed",
        "integrity": {
            "valid": True,
            "common_identity": {
                "source_tree_sha256": "a" * 64,
                "config_sha256": fingerprint(),
            },
        },
    }

    report = build_extension_report(
        prototype_report=prototype,
        audited_jobs=audited,
        freeze=freeze,
    )

    assert report["verdict"] == "kill"
    assert report["confirmation_authorized"] is False
