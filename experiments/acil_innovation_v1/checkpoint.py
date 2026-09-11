"""Content-addressed safetensors checkpoints with immutable identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    weights_path: Path
    metadata_path: Path
    identity_sha256: str
    file_sha256: str
    tensor_sha256: str


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("ascii")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"acil-innovation-v1:tensors:v1\x00")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        for field in (
            name.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
            tensor.view(torch.uint8).numpy().tobytes(),
        ):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def save_checkpoint(
    model: torch.nn.Module, *, identity: Mapping[str, object], directory: Path
) -> CheckpointRecord:
    if not isinstance(model, torch.nn.Module) or not identity:
        raise TypeError("model and nonempty identity are required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity_bytes = _canonical(identity)
    identity_sha = hashlib.sha256(b"acil-innovation-v1:checkpoint-identity:v1\x00" + identity_bytes).hexdigest()
    weights = directory / f"{identity_sha}.safetensors"
    metadata = directory / f"{identity_sha}.json"
    if weights.exists() or metadata.exists():
        raise FileExistsError("checkpoint identity already exists")
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, str(weights))
    file_sha = _file_sha(weights)
    tensor_sha = _tensor_sha(state)
    payload = {
        "schema": "acil-innovation-v1:checkpoint:v1",
        "identity": dict(identity),
        "identity_sha256": identity_sha,
        "weights_file": weights.name,
        "file_sha256": file_sha,
        "tensor_sha256": tensor_sha,
    }
    metadata.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return CheckpointRecord(weights, metadata, identity_sha, file_sha, tensor_sha)


def load_checkpoint(
    model: torch.nn.Module,
    *,
    metadata_path: Path,
    expected_identity: Mapping[str, object],
) -> CheckpointRecord:
    metadata_path = Path(metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("identity") != dict(expected_identity):
        raise ValueError("checkpoint identity mismatch")
    identity_sha = hashlib.sha256(
        b"acil-innovation-v1:checkpoint-identity:v1\x00" + _canonical(expected_identity)
    ).hexdigest()
    if payload.get("identity_sha256") != identity_sha:
        raise ValueError("checkpoint identity hash mismatch")
    weights = metadata_path.with_name(payload["weights_file"])
    if _file_sha(weights) != payload.get("file_sha256"):
        raise ValueError("checkpoint file hash mismatch")
    state = load_file(str(weights), device="cpu")
    if _tensor_sha(state) != payload.get("tensor_sha256"):
        raise ValueError("checkpoint tensor hash mismatch")
    model.load_state_dict(state, strict=True)
    return CheckpointRecord(
        weights_path=weights,
        metadata_path=metadata_path,
        identity_sha256=identity_sha,
        file_sha256=payload["file_sha256"],
        tensor_sha256=payload["tensor_sha256"],
    )


__all__ = ["CheckpointRecord", "load_checkpoint", "save_checkpoint"]
