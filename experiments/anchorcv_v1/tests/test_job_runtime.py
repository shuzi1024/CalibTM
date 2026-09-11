from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest
import torch

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.batching import EvaluationBatch
from experiments.acil_innovation_v1.preprocessing import FitFallback
from experiments.anchorcv_v1.job_runtime import (
    evaluate_evaluation_batch,
    mask_identity_sha256,
    scientific_array_sha256,
    verify_completed_job,
)
from experiments.anchorcv_v1.job import JobSpec
from experiments.anchorcv_v1.protocol import fingerprint
from experiments.anchorcv_v1.model import PriorFreeMaskNativeExpert


def test_scientific_array_hash_is_deterministic_name_shape_dtype_and_value_bound() -> None:
    arrays = {
        "b": np.array([True, False], dtype=np.bool_),
        "a": np.array([[1.0, 2.0]], dtype="<f8"),
    }

    first = scientific_array_sha256(arrays)
    second = scientific_array_sha256(dict(reversed(tuple(arrays.items()))))
    changed = scientific_array_sha256({**arrays, "a": np.array([[1.0, 3.0]])})

    assert first == second
    assert len(first) == 64
    assert first != changed


def test_mask_identity_hash_is_grid_order_sensitive_and_json_stable() -> None:
    grid = (("a" * 64, "b" * 64), ("c" * 64, "d" * 64))

    digest = mask_identity_sha256(grid)

    assert len(digest) == 64
    assert digest == mask_identity_sha256(tuple(tuple(row) for row in grid))
    assert digest != mask_identity_sha256(tuple(reversed(grid)))
    with pytest.raises(ValueError, match="SHA-256"):
        mask_identity_sha256((("not-a-hash",),))


def _batch() -> EvaluationBatch:
    truth = torch.arange(1, 1 + 3 * 2 * 7, dtype=torch.float32).reshape(3, 2, 7)
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., (0, 3, 6)] = True
    return EvaluationBatch(
        dataset="abilene",
        cohort="tune",
        seed_bundle=1,
        family="random",
        window_indices=(0, 1, 2),
        absolute_starts=(0, 50, 100),
        mask_identities=tuple(),
        mask_sha256=(("a" * 64, "b" * 64),) * 3,
        oracle_q_sha256=(("c" * 64, "d" * 64),) * 3,
        oracle_e_sha256=(("e" * 64, "f" * 64),) * 3,
        truth=truth,
        model_input=torch.where(observed, truth, torch.full_like(truth, float("nan"))),
        observed=observed,
        target=~observed,
        controlled_gap=torch.zeros_like(observed),
        oracle_q=torch.zeros_like(observed),
        oracle_e=~observed,
        oracle_support_q=None,  # type: ignore[arg-type]
    )


def test_chunked_registered_evaluation_preserves_window_case_order() -> None:
    torch.manual_seed(9)
    prior = ACILBase(hidden=8).eval()
    neural = PriorFreeMaskNativeExpert(
        num_flows=2,
        time_steps=7,
        temporal_hidden=4,
        d_model=16,
        num_heads=4,
        num_flow_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    ).eval()

    evidence = evaluate_evaluation_batch(
        prior=prior,
        neural=neural,
        batch=_batch(),
        fit_fallback=FitFallback(mean=10.0, std=5.0),
        device=torch.device("cpu"),
        chunk_size=2,
    )

    assert evidence.case_count == 6
    assert evidence.window_index.tolist() == [0, 0, 1, 1, 2, 2]
    assert evidence.flow_index.tolist() == [0, 1, 0, 1, 0, 1]
    assert json.loads(json.dumps({"hash": scientific_array_sha256(evidence.as_npz_dict())}))["hash"]


def test_completed_job_skip_requires_exact_spec_identity_and_result_hash(tmp_path) -> None:
    spec = JobSpec(
        dataset="abilene",
        seed_bundle=1,
        stage="prototype",
        output_root=tmp_path,
        device="cuda",
        source_tree_sha256="a" * 64,
        config_sha256=fingerprint(),
    )
    spec.job_directory.mkdir(parents=True)
    result = {
        "job_id": spec.job_id,
        "status": "succeeded",
        "scientific_identity": spec.scientific_identity,
    }
    result_path = spec.job_directory / "result.json"
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    manifest = {
        **spec.scientific_identity,
        "job_id": spec.job_id,
        "status": "succeeded",
        "result_file": "result.json",
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    manifest_path = spec.job_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    assert verify_completed_job(spec) is True

    manifest["source_tree_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific identity"):
        verify_completed_job(spec)


def test_completed_job_skip_rejects_partial_manifest_only_directory(tmp_path) -> None:
    spec = JobSpec(
        dataset="abilene",
        seed_bundle=1,
        stage="prototype",
        output_root=tmp_path,
        device="cuda",
        source_tree_sha256="a" * 64,
        config_sha256=fingerprint(),
    )
    spec.job_directory.mkdir(parents=True)
    (spec.job_directory / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="partial"):
        verify_completed_job(spec)
