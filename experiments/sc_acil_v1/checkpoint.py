"""Hash-bound local checkpoints without an optional safetensors dependency."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from .protocol import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    weights_path: Path
    metadata_path: Path
    identity_sha256: str
    file_sha256: str
    tensor_sha256: str


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"sc-acil-v1:tensors:v1\x00")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        fields = (
            name.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
            tensor.view(torch.uint8).numpy().tobytes(),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def save_checkpoint(model, *, identity: Mapping[str, object], directory: Path):
    if not isinstance(model, torch.nn.Module) or not identity:
        raise TypeError("model and nonempty identity are required")
    directory.mkdir(parents=True, exist_ok=True)
    identity_sha = hashlib.sha256(
        b"sc-acil-v1:checkpoint-identity:v1\x00" + canonical_json_bytes(dict(identity))
    ).hexdigest()
    weights = directory / f"{identity_sha}.pt"
    metadata = directory / f"{identity_sha}.json"
    if weights.exists() or metadata.exists():
        raise FileExistsError("checkpoint identity already exists")
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state, weights)
    payload = {
        "schema": "sc-acil-v1:checkpoint:v1",
        "identity": dict(identity),
        "identity_sha256": identity_sha,
        "weights_file": weights.name,
        "file_sha256": _file_sha(weights),
        "tensor_sha256": _tensor_sha(state),
    }
    metadata.write_bytes(canonical_json_bytes(payload) + b"\n")
    return CheckpointRecord(
        weights, metadata, identity_sha, payload["file_sha256"], payload["tensor_sha256"]
    )


def load_checkpoint(model, *, metadata_path: Path, expected_identity: Mapping[str, object]):
    raw = metadata_path.read_bytes()
    payload = json.loads(raw.decode("ascii"))
    if raw != canonical_json_bytes(payload) + b"\n":
        raise ValueError("checkpoint metadata is not canonical")
    if payload.get("identity") != dict(expected_identity):
        raise ValueError("checkpoint identity mismatch")
    identity_sha = hashlib.sha256(
        b"sc-acil-v1:checkpoint-identity:v1\x00"
        + canonical_json_bytes(dict(expected_identity))
    ).hexdigest()
    if payload.get("identity_sha256") != identity_sha:
        raise ValueError("checkpoint identity hash mismatch")
    weights = metadata_path.with_name(str(payload["weights_file"]))
    if _file_sha(weights) != payload.get("file_sha256"):
        raise ValueError("checkpoint file hash mismatch")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or _tensor_sha(state) != payload.get("tensor_sha256"):
        raise ValueError("checkpoint tensor hash mismatch")
    model.load_state_dict(state, strict=True)
    return CheckpointRecord(
        weights,
        metadata_path,
        identity_sha,
        payload["file_sha256"],
        payload["tensor_sha256"],
    )


__all__ = ["CheckpointRecord", "load_checkpoint", "save_checkpoint"]

