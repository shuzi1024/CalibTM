"""Write-once result helpers for the tiny-KAN development comparison."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .jobs import PROTOCOL_ID, canonical_json


RESULT_SCHEMA = f"{PROTOCOL_ID}:result:v1"
MANIFEST_SCHEMA = f"{PROTOCOL_ID}:result-manifest:v1"
_CHECKPOINT_IDENTITY_DOMAIN = b"sc-acil-v1:checkpoint-identity:v1\x00"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        encoded = memoryview(canonical_json(payload) + b"\n")
        while encoded:
            written = os.write(descriptor, encoded)
            if written <= 0:
                raise OSError("write made no progress")
            encoded = encoded[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def write_result(path: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    output = Path(path)
    write_exclusive(output, result)
    manifest = {
        "job": result.get("job"),
        "protocol": PROTOCOL_ID,
        "result_file": output.name,
        "result_sha256": file_sha256(output),
        "schema": MANIFEST_SCHEMA,
        "source_tree_sha256": result.get("source_tree_sha256"),
    }
    write_exclusive(output.with_name(output.name + ".manifest.json"), manifest)
    return manifest


def _hex_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_child(root: Path, relative: object) -> Path | None:
    """Resolve one checkpoint path without permitting escape or symlinks."""

    if not isinstance(relative, str) or not relative:
        return None
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None
    candidate = root / relative_path
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if resolved.parent.parent != resolved_root or resolved.parent.name != "checkpoints":
        return None
    if candidate.is_symlink() or not candidate.is_file():
        return None
    return candidate


def _verified_checkpoint(output: Path, payload: dict[str, Any]) -> bool:
    checkpoint = payload.get("checkpoint")
    training = payload.get("training")
    if not isinstance(checkpoint, dict) or not isinstance(training, dict):
        return False
    identity = checkpoint.get("identity")
    expected_identity = {
        "architecture": payload.get("architecture"),
        "best_epoch": training.get("best_epoch"),
        "job": payload.get("job"),
        "source_tree_sha256": payload.get("source_tree_sha256"),
    }
    if not isinstance(identity, dict) or identity != expected_identity:
        return False
    identity_sha256 = hashlib.sha256(
        _CHECKPOINT_IDENTITY_DOMAIN + canonical_json(identity)
    ).hexdigest()
    if checkpoint.get("identity_sha256") != identity_sha256:
        return False
    weights = _regular_child(output.parent, checkpoint.get("weights"))
    metadata_path = _regular_child(output.parent, checkpoint.get("metadata"))
    if weights is None or metadata_path is None:
        return False
    try:
        raw_metadata = metadata_path.read_bytes()
        metadata = json.loads(raw_metadata.decode("ascii"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return False
    if (
        not isinstance(metadata, dict)
        or raw_metadata != canonical_json(metadata) + b"\n"
        or metadata.get("schema") != "sc-acil-v1:checkpoint:v1"
        or metadata.get("identity") != identity
        or metadata.get("identity_sha256") != identity_sha256
        or metadata.get("weights_file") != weights.name
        or metadata.get("file_sha256") != checkpoint.get("file_sha256")
        or metadata.get("tensor_sha256") != checkpoint.get("tensor_sha256")
        or not _hex_sha256(checkpoint.get("file_sha256"))
        or not _hex_sha256(checkpoint.get("tensor_sha256"))
        or file_sha256(weights) != checkpoint.get("file_sha256")
    ):
        return False
    return True


def verified_success(path: str | Path) -> dict[str, Any] | None:
    output = Path(path)
    manifest_path = output.with_name(output.name + ".manifest.json")
    if (
        output.is_symlink()
        or manifest_path.is_symlink()
        or not output.is_file()
        or not manifest_path.is_file()
    ):
        return None
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != RESULT_SCHEMA
        or payload.get("protocol") != PROTOCOL_ID
        or payload.get("status") != "succeeded"
        or payload.get("test_access") is not False
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("protocol") != PROTOCOL_ID
        or manifest.get("result_file") != output.name
        or manifest.get("result_sha256") != file_sha256(output)
        or manifest.get("job") != payload.get("job")
        or manifest.get("source_tree_sha256")
        != payload.get("source_tree_sha256")
        or not _verified_checkpoint(output, payload)
    ):
        return None
    return payload


__all__ = [
    "MANIFEST_SCHEMA",
    "RESULT_SCHEMA",
    "file_sha256",
    "verified_success",
    "write_exclusive",
    "write_result",
]
