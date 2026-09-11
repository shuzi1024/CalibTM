"""Immutable source, protocol and permitted-data identity for this prototype."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.anchorcv_v1.data_access import canonical_data_identity

from .run_job import expected_job_identities


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_FREEZE_PATH = _PACKAGE_ROOT / "freezes/minimal_calibrator_v1.json"
_SOURCE_NAMES = (
    "__init__.py",
    "adjudicate.py",
    "freeze.py",
    "model.py",
    "research_card.json",
    "run_job.py",
)
_DEPENDENCY_PATHS = (
    "experiments/acil_mechanism_v1/model.py",
    "experiments/acil_mechanism_v1/run_job.py",
    "experiments/acil_innovation_v1/acil.py",
    "experiments/acil_innovation_v1/acil_core.py",
    "experiments/acil_innovation_v1/batching.py",
    "experiments/acil_innovation_v1/config.py",
    "experiments/acil_innovation_v1/configs/protocol_v1.json",
    "experiments/acil_innovation_v1/data.py",
    "experiments/acil_innovation_v1/masks.py",
    "experiments/acil_innovation_v1/models.py",
    "experiments/acil_innovation_v1/oracle_model.py",
    "experiments/acil_innovation_v1/preprocessing.py",
    "experiments/acil_innovation_v1/registries.py",
    "experiments/acil_innovation_v1/training.py",
    "experiments/anchorcv_v1/data_access.py",
    "experiments/sc_acil_v1/checkpoint.py",
    "experiments/sc_acil_v1/protocol.py",
)
_TREE_DOMAIN = b"minimal-calibrator-v1:source-tree:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"freeze input must be a regular file: {path}")
    before = path.stat()
    encoded = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"freeze input changed while read: {path}")
    return encoded


def _file_record(relative: str) -> dict[str, object]:
    encoded = _stable_bytes(_REPO_ROOT / relative)
    return {
        "bytes": len(encoded),
        "path": relative,
        "sha256": _sha256_bytes(encoded),
    }


def _current_record() -> dict[str, Any]:
    source_files = [
        _file_record(f"experiments/minimal_calibrator_v1/{name}")
        for name in _SOURCE_NAMES
    ]
    dependency_files = [_file_record(path) for path in _DEPENDENCY_PATHS]
    all_files = source_files + dependency_files
    return {
        "data_identities": {
            dataset: canonical_data_identity(dataset)
            for dataset in ("abilene", "geant")
        },
        "dependency_files": dependency_files,
        "expected_jobs": list(expected_job_identities()),
        "git_available": False,
        "git_commit": None,
        "hash_algorithm": "sha256",
        "hash_scope": "enumerated_runtime_source_and_dependencies",
        "protocol": "minimal-calibrator-v1",
        "research_card_sha256": next(
            item["sha256"]
            for item in source_files
            if item["path"].endswith("/research_card.json")
        ),
        "schema_version": 1,
        "source_files": source_files,
        "source_tree_sha256": _sha256_bytes(
            _TREE_DOMAIN + _canonical_json(all_files)
        ),
    }


def create_freeze_record(path: str | Path) -> dict[str, Any]:
    output = Path(path)
    first = _current_record()
    if first != _current_record():
        raise ValueError("freeze inputs drifted across deterministic repeat")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(first)
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        os.write(descriptor, encoded)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return first


def verify_freeze_record(path: str | Path) -> dict[str, Any]:
    freeze_path = Path(path)
    encoded = _stable_bytes(freeze_path)
    try:
        record = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("freeze record must be canonical JSON") from exc
    if not isinstance(record, dict) or _canonical_json(record) != encoded:
        raise ValueError("freeze record must be canonical JSON")
    if record.get("schema_version") != 1:
        raise ValueError("freeze record schema drifted")
    if record != _current_record():
        raise ValueError("freeze source/card/data identity drifted")
    return record


def freeze_file_sha256(path: str | Path) -> str:
    return _sha256_bytes(_stable_bytes(Path(path)))


def probe_identity(
    path: str | Path = DEFAULT_FREEZE_PATH,
) -> dict[str, str]:
    record = verify_freeze_record(path)
    return {
        "freeze_file_sha256": freeze_file_sha256(path),
        "research_card_sha256": record["research_card_sha256"],
        "source_tree_sha256": record["source_tree_sha256"],
    }


__all__ = [
    "DEFAULT_FREEZE_PATH",
    "create_freeze_record",
    "freeze_file_sha256",
    "probe_identity",
    "verify_freeze_record",
]
