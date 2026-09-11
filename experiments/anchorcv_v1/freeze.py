"""Deterministic, immutable source freeze for the AnchorCV v1 protocol.

The data side of this record is deliberately semantic and path-free:
``canonical_data_identity`` is the only data-identity capability called here.
In particular, this module never enumerates canonical arrays, raw CSV files,
legacy caches, or any sealed split.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from . import data_access, protocol


_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parents[1]
_SOURCE_TREE_DOMAIN = b"anchorcv-v1:source-tree:v1\x00"
_EXCLUDED_DIRECTORY_NAMES = (
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "freezes",
    "generated",
    "handoffs",
    "results",
    "runs",
    "tests",
)
_OPERATIONAL_EXCLUDED_FILES = (
    "experiments/anchorcv_v1/occupancy.py",
)
_SOURCE_INVENTORY_POLICY = {
    "anchorcv_root": "experiments/anchorcv_v1",
    "anchorcv_source_suffixes": [".py"],
    "excluded_directory_names": list(_EXCLUDED_DIRECTORY_NAMES),
    "operational_excluded_files": list(_OPERATIONAL_EXCLUDED_FILES),
    "research_card": "experiments/anchorcv_v1/research_card.yaml",
}

# These are source/config dependencies imported by the registered data,
# batching, ACIL-prior, and neural-training paths.  Keeping the list literal
# makes the upstream boundary reviewable; no broad upstream tree is crawled.
UPSTREAM_DEPENDENCY_PATHS = (
    "experiments/acil_innovation_v1/__init__.py",
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
    "experiments/od_orbit_gpt_v1/__init__.py",
    "experiments/od_orbit_gpt_v1/config.py",
    "experiments/od_orbit_gpt_v1/configs/protocol_v1.json",
    "experiments/od_orbit_gpt_v1/data.py",
    "experiments/od_orbit_gpt_v1/protocol.py",
    "experiments/sc2_ari_v1/__init__.py",
    "experiments/sc2_ari_v1/data.py",
    "experiments/sc2_ari_v1/initialization.py",
)

_ROOT_FIELDS = {
    "config_sha256",
    "data_identities",
    "git_available",
    "git_commit",
    "hash_algorithm",
    "protocol",
    "schema_version",
    "source_files",
    "source_inventory_policy",
    "source_tree_sha256",
    "upstream_dependency_paths",
}
_FILE_FIELDS = {"bytes", "path", "role", "sha256"}
_DATA_IDENTITY_FIELDS = {
    "data_sha256",
    "dataset",
    "hash_scope",
    "parsed_array_sha256",
}
_DATASETS = ("abilene", "geant")
_DATA_HASH_SCOPE = "canonical_parsed_permitted_split_arrays_only"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("freeze record must contain canonical finite JSON") from exc


def _exact_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} fields drifted; missing={missing!r}, unknown={unknown!r}"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _plain_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is inaccessible: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} symlink is forbidden: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory: {path}")


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_file(path: Path, *, label: str) -> bytes:
    """Read a stable regular file through a no-follow descriptor."""

    try:
        path_before = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is inaccessible: {path}") from exc
    if stat.S_ISLNK(path_before.st_mode):
        raise ValueError(f"{label} symlink is forbidden: {path}")
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open stable {label}: {path}") from exc
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        if (
            descriptor_before.st_dev,
            descriptor_before.st_ino,
        ) != (path_before.st_dev, path_before.st_ino):
            raise ValueError(f"{label} changed while being opened: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
        if _stat_identity(descriptor_before) != _stat_identity(descriptor_after):
            raise ValueError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)

    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} disappeared while being read: {path}") from exc
    if _stat_identity(path_before) != _stat_identity(path_after):
        raise ValueError(f"{label} changed while being read: {path}")
    return b"".join(chunks)


def _repo_relative(path: Path) -> str:
    try:
        relative = path.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"source path escapes fixed repository root: {path}") from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"source path is not canonical: {path}")
    return relative.as_posix()


def _anchorcv_paths() -> list[tuple[Path, str]]:
    _plain_directory(_PACKAGE_ROOT, "AnchorCV source root")
    results: list[tuple[Path, str]] = []
    stack = [_PACKAGE_ROOT]
    card_path = _PACKAGE_ROOT / "research_card.yaml"
    while stack:
        directory = stack.pop()
        _plain_directory(directory, "AnchorCV source directory")
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError(
                f"cannot enumerate AnchorCV source directory: {directory}"
            ) from exc
        child_directories: list[Path] = []
        for child in children:
            if child.name in _EXCLUDED_DIRECTORY_NAMES:
                continue
            if _repo_relative(child) in _OPERATIONAL_EXCLUDED_FILES:
                continue
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise ValueError(f"cannot stat AnchorCV source entry: {child}") from exc
            is_candidate = child.suffix == ".py" or child == card_path
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"AnchorCV source symlink is forbidden: {child}")
            if stat.S_ISDIR(metadata.st_mode):
                child_directories.append(child)
            elif is_candidate:
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(
                        f"AnchorCV source must be a regular file: {child}"
                    )
                role = "research_card" if child == card_path else "anchorcv_source"
                results.append((child, role))
        # Reverse because this is a LIFO stack; final sorting remains authoritative.
        stack.extend(reversed(child_directories))
    if not any(path == card_path for path, _ in results):
        raise ValueError("AnchorCV research_card.yaml is missing")
    return sorted(results, key=lambda item: _repo_relative(item[0]))


def _upstream_paths() -> list[tuple[Path, str]]:
    if tuple(sorted(UPSTREAM_DEPENDENCY_PATHS)) != UPSTREAM_DEPENDENCY_PATHS:
        raise ValueError("upstream dependency paths must be sorted")
    if len(set(UPSTREAM_DEPENDENCY_PATHS)) != len(UPSTREAM_DEPENDENCY_PATHS):
        raise ValueError("upstream dependency paths must be unique")
    results: list[tuple[Path, str]] = []
    for raw in UPSTREAM_DEPENDENCY_PATHS:
        if not isinstance(raw, str):
            raise ValueError("upstream dependency path must be a string")
        relative = Path(raw)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("upstream dependency path must be canonical repo-relative")
        path = _REPO_ROOT / relative
        # The stable reader performs the authoritative regular/symlink check.
        _read_regular_file(path, label="upstream dependency")
        results.append((path, "upstream_dependency"))
    return results


def _source_record(path: Path, role: str) -> dict[str, object]:
    content = _read_regular_file(path, label=role.replace("_", " "))
    return {
        "bytes": len(content),
        "path": _repo_relative(path),
        "role": role,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _source_records() -> list[dict[str, object]]:
    identified = _anchorcv_paths() + _upstream_paths()
    paths = [_repo_relative(path) for path, _ in identified]
    if len(paths) != len(set(paths)):
        raise ValueError("source inventory contains duplicate paths")
    return [
        _source_record(path, role)
        for path, role in sorted(identified, key=lambda item: _repo_relative(item[0]))
    ]


def _source_tree_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256(_SOURCE_TREE_DOMAIN)
    encoded = _canonical_json(list(records))
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _validated_data_identity(
    identity: object, *, expected_dataset: str
) -> dict[str, object]:
    if not isinstance(identity, Mapping):
        raise ValueError("canonical data identity must be an object")
    _exact_fields(identity, _DATA_IDENTITY_FIELDS, "canonical data identity")
    if identity.get("dataset") != expected_dataset:
        raise ValueError("canonical data identity dataset drifted")
    if identity.get("hash_scope") != _DATA_HASH_SCOPE:
        raise ValueError("canonical data identity hash scope drifted")
    parsed = identity.get("parsed_array_sha256")
    if not isinstance(parsed, Mapping):
        raise ValueError("parsed-array identities must be an object")
    _exact_fields(parsed, {"train", "val"}, "parsed-array identity")
    normalized_parsed = {
        split: _require_sha256(parsed.get(split), f"{expected_dataset} {split} data")
        for split in ("train", "val")
    }
    return {
        "data_sha256": _require_sha256(
            identity.get("data_sha256"), f"{expected_dataset} data identity"
        ),
        "dataset": expected_dataset,
        "hash_scope": _DATA_HASH_SCOPE,
        "parsed_array_sha256": normalized_parsed,
    }


def _data_identities(datasets: Sequence[str]) -> dict[str, object]:
    if tuple(datasets) != _DATASETS:
        raise ValueError("AnchorCV protocol datasets drifted from abilene/geant")
    return {
        dataset: _validated_data_identity(
            data_access.canonical_data_identity(dataset),
            expected_dataset=dataset,
        )
        for dataset in datasets
    }


def _current_record() -> dict[str, object]:
    spec = protocol.load_protocol()
    if spec.protocol_id != "anchorcv-v1":
        raise ValueError("AnchorCV protocol identity drifted")
    config_sha256 = _require_sha256(
        protocol.fingerprint(spec), "config_sha256"
    )
    source_files = _source_records()
    return {
        "config_sha256": config_sha256,
        "data_identities": _data_identities(spec.datasets),
        "git_available": False,
        "git_commit": None,
        "hash_algorithm": "sha256",
        "protocol": "anchorcv-v1",
        "schema_version": 1,
        "source_files": source_files,
        "source_inventory_policy": dict(_SOURCE_INVENTORY_POLICY),
        "source_tree_sha256": _source_tree_sha256(source_files),
        "upstream_dependency_paths": list(UPSTREAM_DEPENDENCY_PATHS),
    }


def _write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _canonical_json(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"cannot create freeze parent directory: {path.parent}") from exc
    _plain_directory(path.parent, "freeze parent directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError:
        raise FileExistsError(
            f"freeze record already exists; overwrite is forbidden: {path}"
        ) from None
    except OSError as exc:
        raise ValueError(f"cannot exclusively create freeze record: {path}") from exc
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short freeze-record write")
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def create_freeze_record(output_path: str | Path) -> dict[str, object]:
    """Build and exclusively write one deterministic AnchorCV freeze JSON."""

    record = _current_record()
    confirmation = _current_record()
    if record != confirmation:
        raise ValueError("freeze inputs drifted across deterministic repeat")
    _write_exclusive(Path(output_path), record)
    return record


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate freeze-record key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite freeze-record constant {token!r} is forbidden")


def _load_record(path: Path) -> dict[str, Any]:
    encoded = _read_regular_file(path, label="freeze record")
    try:
        payload = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("freeze record must be canonical JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("freeze record must be a JSON object")
    if _canonical_json(payload) != encoded:
        raise ValueError("freeze record JSON is not in canonical form")
    return payload


def _validate_record_schema(record: Mapping[str, Any]) -> None:
    _exact_fields(record, _ROOT_FIELDS, "freeze record")
    if record.get("schema_version") != 1 or isinstance(
        record.get("schema_version"), bool
    ):
        raise ValueError("freeze record schema_version must be integer 1")
    if record.get("protocol") != "anchorcv-v1":
        raise ValueError("freeze record protocol drifted")
    if record.get("git_available") is not False:
        raise ValueError("freeze record git_available must be false")
    if record.get("git_commit") is not None:
        raise ValueError("freeze record git_commit must be null")
    if record.get("hash_algorithm") != "sha256":
        raise ValueError("freeze record hash algorithm drifted")
    _require_sha256(record.get("config_sha256"), "config_sha256")
    _require_sha256(record.get("source_tree_sha256"), "source_tree_sha256")
    if record.get("source_inventory_policy") != _SOURCE_INVENTORY_POLICY:
        raise ValueError("freeze record source inventory policy drifted")
    if record.get("upstream_dependency_paths") != list(
        UPSTREAM_DEPENDENCY_PATHS
    ):
        raise ValueError("freeze record upstream dependency inventory drifted")

    source_files = record.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("freeze record source files must be a nonempty list")
    previous = ""
    seen: set[str] = set()
    for item in source_files:
        if not isinstance(item, Mapping):
            raise ValueError("source-file fields are missing")
        _exact_fields(item, _FILE_FIELDS, "source-file")
        path = item.get("path")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError("source-file path must be canonical repo-relative")
        if path <= previous or path in seen:
            raise ValueError("source-file paths must be sorted and unique")
        previous = path
        seen.add(path)
        if item.get("role") not in {
            "anchorcv_source",
            "research_card",
            "upstream_dependency",
        }:
            raise ValueError("source-file role is unknown")
        byte_count = item.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("source-file byte count must be nonnegative")
        _require_sha256(item.get("sha256"), f"source-file {path}")

    identities = record.get("data_identities")
    if not isinstance(identities, Mapping):
        raise ValueError("data identities must be an object")
    _exact_fields(identities, set(_DATASETS), "data identities")
    for dataset in _DATASETS:
        _validated_data_identity(
            identities.get(dataset), expected_dataset=dataset
        )


def verify_freeze_record(path: str | Path) -> dict[str, object]:
    """Recompute protocol, data, and every source file; fail closed on drift."""

    record = _load_record(Path(path))
    _validate_record_schema(record)

    spec = protocol.load_protocol()
    current_config = _require_sha256(
        protocol.fingerprint(spec), "current config_sha256"
    )
    if record["config_sha256"] != current_config:
        raise ValueError("freeze record config_sha256 drifted")

    current_data = _data_identities(spec.datasets)
    if record["data_identities"] != current_data:
        raise ValueError("freeze record data identities drifted")

    current_sources = _source_records()
    if record["source_files"] != current_sources:
        raise ValueError("freeze record source file inventory or content drifted")
    current_tree = _source_tree_sha256(current_sources)
    if record["source_tree_sha256"] != current_tree:
        raise ValueError("freeze record source tree drifted")
    return record


__all__ = [
    "UPSTREAM_DEPENDENCY_PATHS",
    "create_freeze_record",
    "verify_freeze_record",
]
