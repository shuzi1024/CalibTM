"""Deterministic source/protocol/data manifest for a Git-less checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .data import permitted_parent_hashes
from .jobs import planned_jobs
from .protocol import canonical_json_bytes, protocol_sha256


_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
_OLD = _HERE.parent / "acil_innovation_v1"
_GENERATED = _HERE / "generated" / "freeze_v2"


def _source_files() -> tuple[Path, ...]:
    new_files = [
        path
        for path in _HERE.rglob("*")
        if path.is_file()
        and not any(part in {"generated", "runs", "__pycache__", ".pytest_cache"} for part in path.parts)
        and path.suffix in {".py", ".json", ".md", ".sh"}
    ]
    old_files = list(_OLD.glob("*.py")) + [_OLD / "configs" / "protocol_v1.json"]
    files = tuple(sorted({path.resolve() for path in (*new_files, *old_files)}))
    if not files or any(not path.is_file() for path in files):
        raise ValueError("source dependency registry is incomplete")
    return files


def _tree_hash(files: Iterable[Path]) -> str:
    digest = hashlib.sha256(b"sc-acil-v1:source-tree:v1\x00")
    for path in files:
        relative = path.relative_to(_ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def source_tree_sha256() -> str:
    return _tree_hash(_source_files())


def build_manifest() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "sc-acil-v1",
        "protocol_sha256": protocol_sha256(),
        "git_available": False,
        "git_commit": None,
        "source_tree_sha256": source_tree_sha256(),
        "permitted_parent_semantic_sha256": permitted_parent_hashes(),
        "jobs": [
            job.to_json()
            for stage in ("fit_acil", "formal_gate")
            for job in planned_jobs(stage)
        ],
    }
    digest = hashlib.sha256(
        b"sc-acil-v1:manifest:v1\x00" + canonical_json_bytes(payload)
    ).hexdigest()
    return {**payload, "manifest_sha256": digest}


def write_freeze() -> Path:
    manifest = build_manifest()
    _GENERATED.mkdir(parents=True, exist_ok=False)
    path = _GENERATED / "manifest.json"
    content = canonical_json_bytes(manifest) + b"\n"
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
    return path


def load_and_verify_freeze() -> dict[str, object]:
    path = _GENERATED / "manifest.json"
    try:
        raw = path.read_bytes()
        frozen = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SC-ACIL freeze_v1 is absent or invalid") from exc
    if raw != canonical_json_bytes(frozen) + b"\n":
        raise ValueError("SC-ACIL freeze is not canonical")
    current = build_manifest()
    if frozen != current:
        raise ValueError("SC-ACIL source/protocol/data identity drifted after freeze")
    return current


__all__ = [
    "build_manifest",
    "load_and_verify_freeze",
    "source_tree_sha256",
    "write_freeze",
]
