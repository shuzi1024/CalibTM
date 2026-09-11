from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.anchorcv_v1.job import JobSpec
from experiments.anchorcv_v1.protocol import fingerprint


def _spec(tmp_path: Path, **changes) -> JobSpec:
    values = {
        "dataset": "abilene",
        "seed_bundle": 1,
        "stage": "prototype",
        "output_root": tmp_path,
        "device": "cuda",
        "source_tree_sha256": "a" * 64,
        "config_sha256": fingerprint(),
    }
    values.update(changes)
    return JobSpec(**values)


def test_job_identity_is_content_addressed_and_operational_path_independent(tmp_path: Path) -> None:
    first = _spec(tmp_path / "a", device="cuda")
    second = _spec(tmp_path / "b", device="cuda")

    assert first.job_id == second.job_id
    assert len(first.job_id) == 64
    assert first.job_directory == (tmp_path / "a").resolve() / first.job_id
    assert json.loads(first.identity_json) == first.scientific_identity
    assert first.scientific_identity["test_access"] is False
    assert first.scientific_identity["git_available"] is False
    assert first.scientific_identity["git_commit"] is None
    assert first.scientific_identity["config_sha256"] == fingerprint()
    assert first.scientific_identity["source_tree_sha256"] == "a" * 64
    assert first.scientific_identity["compute_lane"] == "cuda_bf16_math_sdp"
    assert len(first.scientific_identity["acil_checkpoint_file_sha256"]) == 64
    assert len(first.scientific_identity["data_sha256"]) == 64


def test_job_identity_exposes_no_data_path_or_split_override(tmp_path: Path) -> None:
    identity = _spec(tmp_path).scientific_identity
    encoded = json.dumps(identity, sort_keys=True).lower()

    assert "data_path" not in encoded
    assert "test_path" not in encoded
    assert "split_override" not in encoded
    assert '"test_access": false' in encoded


@pytest.mark.parametrize(
    ("stage", "bundle"),
    [
        ("prototype", 2),
        ("prototype", 3),
        ("extension", 1),
        ("extension", 4),
    ],
)
def test_stage_bundle_mismatch_is_rejected(tmp_path: Path, stage: str, bundle: int) -> None:
    with pytest.raises(ValueError, match="bundle"):
        _spec(tmp_path, stage=stage, seed_bundle=bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset", "wsdream", "dataset"),
        ("stage", "formal", "stage"),
        ("device", "cpu", "CUDA"),
        ("device", "cuda:7", "device"),
        ("source_tree_sha256", "abc", "source_tree"),
        ("config_sha256", "b" * 64, "config"),
    ],
)
def test_invalid_job_authority_or_identity_is_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(tmp_path, **{field: value})
