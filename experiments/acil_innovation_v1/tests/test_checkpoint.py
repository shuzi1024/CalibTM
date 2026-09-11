from __future__ import annotations

import json

import pytest
import torch
from torch import nn


def test_checkpoint_roundtrip_binds_identity_and_tensor_hash(tmp_path) -> None:
    from experiments.acil_innovation_v1.checkpoint import load_checkpoint, save_checkpoint

    model = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 1))
    identity = {"protocol": "acil-innovation-v1", "job_id": "a" * 64, "epoch": 3}
    record = save_checkpoint(model, identity=identity, directory=tmp_path)
    assert record.weights_path.is_file()
    assert record.metadata_path.is_file()
    restored = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 1))
    loaded = load_checkpoint(restored, metadata_path=record.metadata_path, expected_identity=identity)
    assert loaded.tensor_sha256 == record.tensor_sha256
    for left, right in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(left, right)


def test_checkpoint_rejects_identity_drift(tmp_path) -> None:
    from experiments.acil_innovation_v1.checkpoint import load_checkpoint, save_checkpoint

    model = nn.Linear(2, 1)
    identity = {"protocol": "acil-innovation-v1", "job_id": "b" * 64, "epoch": 0}
    record = save_checkpoint(model, identity=identity, directory=tmp_path)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(
            nn.Linear(2, 1),
            metadata_path=record.metadata_path,
            expected_identity={**identity, "epoch": 1},
        )

