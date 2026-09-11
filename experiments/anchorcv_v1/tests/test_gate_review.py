from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.acil_innovation_v1.batching import EvaluationBatch
from experiments.acil_innovation_v1.preprocessing import (
    FitFallback,
    observation_statistics,
)
from experiments.anchorcv_v1 import gate_identity, gate_review
from experiments.anchorcv_v1.evaluation import (
    CaseEvidence,
    summarize_case_evidence,
)
from experiments.anchorcv_v1.gate_identity import VerifiedGateAuthority
from experiments.anchorcv_v1.job_runtime import (
    mask_identity_sha256,
    scientific_array_sha256,
)


_FAMILIES = ("random", "internal_block", "two_burst")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(*, p_error: float = 2.0) -> CaseEvidence:
    return CaseEvidence(
        window_index=np.array([0]),
        flow_index=np.array([0]),
        middle_anchor_index=np.array([3], dtype=np.int64),
        p_error_sum=np.array([p_error]),
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
        target_regret=np.array([(p_error - 1.0) / 4.0]),
        hard_winner_is_n=np.array([True]),
        oracle_winner_is_n=np.array([True]),
    )


def _registered() -> tuple[EvaluationBatch, FitFallback]:
    truth = torch.tensor([[[0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 3.0]]])
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., (0, 3, 6)] = True
    model_input = torch.where(
        observed, truth, torch.full_like(truth, float("nan"))
    )
    batch = EvaluationBatch(
        dataset="abilene",
        cohort="gate",
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
    return batch, FitFallback(mean=2.0, std=1.0)


def _registered_evidence(
    registered: EvaluationBatch,
    fallback: FitFallback,
) -> CaseEvidence:
    scale = float(
        observation_statistics(
            registered.model_input, registered.observed, fallback
        )
        .std.reshape(-1)[0]
        .item()
    )
    arrays = _evidence().as_npz_dict()
    arrays.update(
        {
            "k3_scale": np.array([scale]),
            "p_anchor_normalized_error": np.array([2.0 / scale]),
            "n_anchor_normalized_error": np.array([1.0 / scale]),
            "loo_score": np.array([1.0 / scale]),
            "target_regret": np.array([1.0 / (4.0 * scale)]),
        }
    )
    return CaseEvidence(**arrays)


def _write_cell(
    job: Path,
    *,
    evidence: CaseEvidence,
    registered: EvaluationBatch,
) -> dict[str, object]:
    path = job / "evidence_random.npz"
    np.savez_compressed(path, **evidence.as_npz_dict())
    return {
        "summary": summarize_case_evidence(evidence),
        "evidence_file": path.name,
        "evidence_file_sha256": _sha(path),
        "evidence_content_sha256": scientific_array_sha256(
            evidence.as_npz_dict()
        ),
        "mask_sha256": mask_identity_sha256(registered.mask_sha256),
        "oracle_q_sha256": mask_identity_sha256(
            registered.oracle_q_sha256
        ),
        "oracle_e_sha256": mask_identity_sha256(
            registered.oracle_e_sha256
        ),
    }


def _authority() -> VerifiedGateAuthority:
    bindings = []
    for bundle in (1, 2, 3):
        for dataset in ("abilene", "geant"):
            bindings.append(
                {
                    "dataset": dataset,
                    "seed_bundle": bundle,
                    "source_training_stage": (
                        "prototype" if bundle == 1 else "extension"
                    ),
                    "source_training_job_id": (
                        f"{bundle}{dataset[0]}" * 32
                    ),
                    "source_result_sha256": "1" * 64,
                    "source_manifest_sha256": "2" * 64,
                    "neural_checkpoint_file_sha256": "3" * 64,
                    "neural_checkpoint_tensor_sha256": "4" * 64,
                    "acil_checkpoint_file_sha256": "5" * 64,
                    "best_epoch": 3,
                    "best_source_dev_nmae": 0.1,
                }
            )
    return VerifiedGateAuthority(
        {
            "source_tree_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "checkpoint_bindings": bindings,
        },
        "c" * 64,
        _seal=gate_identity._AUTHORITY_SEAL,
    )


def _review_kwargs(tmp_path: Path) -> dict[str, Path]:
    return {
        "output_root": tmp_path / "gate",
        "authority_record": tmp_path / "authority.json",
        "extension_report": tmp_path / "extension.json",
        "freeze_record": tmp_path / "freeze.json",
        "prototype_output_root": tmp_path / "prototype",
        "extension_output_root": tmp_path / "extension",
    }


def test_missing_job_revises_before_gate_data_evidence_or_partial_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate_review, "verify_gate_authority", lambda **kwargs: _authority()
    )
    monkeypatch.setattr(
        gate_review,
        "_load_gate_windows",
        lambda *args, **kwargs: pytest.fail("must not open gate data"),
    )
    monkeypatch.setattr(
        gate_review,
        "load_evidence_npz",
        lambda *args, **kwargs: pytest.fail("must not read partial metrics"),
    )
    monkeypatch.setattr(
        gate_review,
        "analyze_extension_uncertainty",
        lambda *args, **kwargs: pytest.fail("must not analyze partial grid"),
    )

    report = gate_review.review_final_gate(**_review_kwargs(tmp_path))

    assert report["status"] == "invalid"
    assert report["verdict"] == "revise"
    assert report["uncertainty"] is None
    assert report["multibundle_adjudication"] is None
    assert len(report["grid"]) == 6
    assert report["failures"]


def test_persisted_cell_rejects_tampered_npz_summary_and_mask(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    registered, fallback = _registered()
    evidence = _registered_evidence(registered, fallback)
    cell = _write_cell(job, evidence=evidence, registered=registered)

    loaded = gate_review._audit_persisted_cell(
        job_directory=job,
        family="random",
        cell=cell,
        registered=registered,
        fit_fallback=fallback,
    )
    gate_review._assert_replay_equal(loaded, evidence)

    (job / "evidence_random.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file hash"):
        gate_review._audit_persisted_cell(
            job_directory=job,
            family="random",
            cell=cell,
            registered=registered,
            fit_fallback=fallback,
        )

    cell = _write_cell(job, evidence=evidence, registered=registered)
    cell["summary"] = {
        **cell["summary"],  # type: ignore[arg-type]
        "p_absolute_error_sum": 999.0,
    }
    with pytest.raises(ValueError, match="summary"):
        gate_review._audit_persisted_cell(
            job_directory=job,
            family="random",
            cell=cell,
            registered=registered,
            fit_fallback=fallback,
        )

    cell = _write_cell(job, evidence=evidence, registered=registered)
    cell["mask_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="mask identity"):
        gate_review._audit_persisted_cell(
            job_directory=job,
            family="random",
            cell=cell,
            registered=registered,
            fit_fallback=fallback,
        )


def test_replay_comparison_is_bitwise_for_every_case_evidence_field() -> None:
    persisted = _evidence()
    replayed_arrays = persisted.as_npz_dict()
    replayed_arrays["p_error_sum"] = np.nextafter(
        replayed_arrays["p_error_sum"], np.inf
    )
    replayed = CaseEvidence(**replayed_arrays)

    with pytest.raises(ValueError, match="p_error_sum"):
        gate_review._assert_replay_equal(persisted, replayed)


@pytest.mark.parametrize("scientific_verdict", ("kill", "proceed"))
def test_complete_grid_uses_multibundle_verdict_without_revising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_verdict: str,
) -> None:
    authority = _authority()
    monkeypatch.setattr(
        gate_review, "verify_gate_authority", lambda **kwargs: authority
    )
    specs = gate_review.build_gate_specs(tmp_path / "gate", authority)
    preflight = [
        {
            "dataset": spec.dataset,
            "seed_bundle": spec.seed_bundle,
            "job_id": spec.job_id,
            "verified": True,
            "result_file_sha256": "6" * 64,
            "manifest_file_sha256": "7" * 64,
        }
        for spec in specs
    ]
    monkeypatch.setattr(
        gate_review,
        "_preflight_grid",
        lambda actual_specs: (preflight, []),
    )
    evidence = _evidence()
    monkeypatch.setattr(
        gate_review,
        "_audit_gate_job",
        lambda *args, **kwargs: {
            family: evidence for family in _FAMILIES
        },
    )
    monkeypatch.setattr(
        gate_review,
        "_load_gate_windows",
        lambda authority, dataset: object(),
    )
    uncertainty = {"schema_version": "test"}
    adjudication = {
        "schema": "anchorcv-v1:multibundle-adjudication:v1",
        "verdict": scientific_verdict,
        "integrity": {"valid": True, "issues": []},
    }
    monkeypatch.setattr(
        gate_review,
        "analyze_extension_uncertainty",
        lambda grid: uncertainty,
    )
    monkeypatch.setattr(
        gate_review,
        "adjudicate_multibundle",
        lambda summaries, report: adjudication,
    )

    report = gate_review.review_final_gate(**_review_kwargs(tmp_path))

    assert report["status"] == "verified_complete"
    assert report["verdict"] == scientific_verdict
    assert report["uncertainty"] == uncertainty
    assert report["multibundle_adjudication"] == adjudication
    assert report["failures"] == []
    assert report["integrity"]["valid"] is True
