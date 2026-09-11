"""Deterministic, fixed-root provenance for ACIL-Innovation v1.

The builder deliberately does not enumerate or open ``canonical/``, raw source
files, legacy caches, or result directories. Parsed arrays are represented by
their previously audited semantic identities; only fixed GPT assets are read.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any, Mapping

from .config import canonical_json_bytes
from .data import (
    _load_canonical_array,
    expected_parsed_array_sha256,
    parsed_array_sha256,
)


_EXPERIMENT_ROOT = Path(__file__).resolve().parent
_GPT_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
_EXPECTED_GPT_HASHES = {
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
}
_UPSTREAM_PROTOCOL_CONFIG_SHA256 = (
    "30689bc52176a095b307ab47ec38dfd502e40704c20d92bd6a326f6db2018fad"
)
_ACIL_CORE_UPSTREAM_SHA256 = (
    "5bb37678a94e9253703d2050efa679bfbf463beb9633e0cf12544f6fbf5a1450"
)
_PARSED_ARRAY_RECORDS = (
    {
        "dataset": "abilene",
        "dtype": "<f4",
        "order": "C",
        "sha256": "5ec5799c6a57c3a3971ab8b33fe95a8ea7c93a01955bec92440a75f38115d8d2",
        "shape": [33884, 144],
        "split": "train",
    },
    {
        "dataset": "abilene",
        "dtype": "<f4",
        "order": "C",
        "sha256": "561be18adfff24be226c23e5a8937b9bceed5a5804bdc8c56c530ef8afe5626e",
        "shape": [7250, 144],
        "split": "val",
    },
    {
        "dataset": "geant",
        "dtype": "<f4",
        "order": "C",
        "sha256": "233cb96106d01c6ef483ae01041d71188a0a37c65aa73fb1074b635f75e796f2",
        "shape": [7572, 462],
        "split": "train",
    },
    {
        "dataset": "geant",
        "dtype": "<f4",
        "order": "C",
        "sha256": "337d1a0b6d1583d94332b80e22d2ecaf297827e3450666e3cc4f678eb45e05dc",
        "shape": [1600, 462],
        "split": "val",
    },
)
_FILE_HASH_DOMAIN = b"acil-innovation-v1:file:v1\x00"
_PRODUCTION_HASH_DOMAIN = b"acil-innovation-v1:production:v1\x00"
_TEST_HASH_DOMAIN = b"acil-innovation-v1:tests:v1\x00"
_PARSED_HASH_DOMAIN = b"acil-innovation-v1:parsed-semantics:v1\x00"
_GPT_HASH_DOMAIN = b"acil-innovation-v1:gpt-assets:v1\x00"
_RUNTIME_HASH_DOMAIN = b"acil-innovation-v1:runtime:v1\x00"
_PROVENANCE_HASH_DOMAIN = b"acil-innovation-v1:provenance:v1\x00"
_REQUIRED_RUNTIME_MODULES = {
    "numpy": "numpy",
    "safetensors": "safetensors",
    "torch": "torch",
    "transformers": "transformers",
}
_SUPPORTED_RELEASE_RANGES = {
    "numpy": ((1, 24, 0), (2, 0, 0)),
    "safetensors": ((0, 3, 0), (1, 0, 0)),
    "torch": ((2, 1, 0), (2, 5, 0)),
    "transformers": ((4, 30, 0), (4, 31, 0)),
}


def _domain_hash(domain: bytes, value: Any) -> str:
    encoded = canonical_json_bytes(value)
    digest = hashlib.sha256(domain)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _plain_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot stat fixed {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} symlink is forbidden: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return metadata


def _hash_file(root: Path, path: Path, *, label: str) -> dict[str, Any]:
    before = _plain_regular_file(path, label=label)
    content = path.read_bytes()
    after = _plain_regular_file(path, label=label)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"{label} changed while being hashed: {path}")
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha256(_FILE_HASH_DOMAIN)
    digest.update(len(relative.encode("utf-8")).to_bytes(8, "big"))
    digest.update(relative.encode("utf-8"))
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)
    return {
        "bytes": len(content),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "path": relative,
        "sha256": digest.hexdigest(),
    }


def _direct_python_files(directory: Path, root: Path) -> tuple[Path, ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"production source directory is invalid: {directory}")
    result: list[Path] = []
    for path in directory.iterdir():
        if path.suffix != ".py":
            continue
        if path.is_symlink():
            raise ValueError(f"production source symlink is forbidden: {path}")
        _plain_regular_file(path, label="production source")
        result.append(path)
    return tuple(sorted(result, key=lambda value: value.relative_to(root).as_posix()))


def _production_paths(root: Path) -> tuple[Path, ...]:
    paths = list(_direct_python_files(root, root))
    scripts = root / "scripts"
    if scripts.exists() or scripts.is_symlink():
        paths.extend(_direct_python_files(scripts, root))
    return tuple(sorted(paths, key=lambda value: value.relative_to(root).as_posix()))


def _test_paths(root: Path) -> tuple[Path, ...]:
    tests = root / "tests"
    if tests.is_symlink() or not tests.is_dir():
        raise ValueError("tests source directory must be a regular directory")
    paths: list[Path] = []
    for path in tests.iterdir():
        if not (path.name.startswith("test_") and path.suffix == ".py"):
            continue
        if path.is_symlink():
            raise ValueError(f"test source symlink is forbidden: {path}")
        _plain_regular_file(path, label="test source")
        paths.append(path)
    if not paths:
        raise ValueError("complete test source inventory is empty")
    return tuple(sorted(paths, key=lambda value: value.relative_to(root).as_posix()))


def _top_level_local_imports(path: Path, production_modules: set[str]) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ValueError(f"cannot parse production import closure: {path}") from exc
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module:
            candidate = node.module.split(".", 1)[0]
            if candidate in production_modules:
                imports.add(candidate)
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            for alias in node.names:
                candidate = alias.name.split(".", 1)[0]
                if candidate in production_modules:
                    imports.add(candidate)
    return sorted(imports)


def _config_record(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "configs" / "protocol_v1.json"
    record = _hash_file(root, path, label="protocol config")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("protocol config is not valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("protocol config must be an object")
    protocol = config.get("protocol")
    freeze = config.get("freeze")
    if not isinstance(protocol, dict) or protocol.get("id") != "acil-innovation-v1":
        raise ValueError("protocol config identity drifted")
    if protocol.get("status") != "draft_protocol":
        raise ValueError("freeze may only be built from draft_protocol config status")
    if not isinstance(freeze, dict):
        raise ValueError("protocol config has no freeze metadata")
    if freeze.get("git_available") is not False or freeze.get("git_commit") is not None:
        raise ValueError("protocol config Git metadata drifted")
    config_sha256 = hashlib.sha256(
        b"acil-innovation-v1:protocol-config:v1\x00" + canonical_json_bytes(config)
    ).hexdigest()
    return record, {"semantic_sha256": config_sha256, "status": protocol["status"]}


def _gpt_records(
    root: Path, expected_hashes: Mapping[str, str] | None
) -> dict[str, Any]:
    asset_root = root / "assets" / "gpt2" / _GPT_REVISION
    if asset_root.is_symlink() or not asset_root.is_dir():
        raise ValueError("fixed GPT asset directory is missing or symlinked")
    expected_names = ("config.json", "model.safetensors")
    actual_names = tuple(
        sorted(
            path.name
            for path in asset_root.iterdir()
            if path.is_file() or path.is_symlink()
        )
    )
    if actual_names != expected_names:
        raise ValueError("fixed GPT asset inventory must contain exactly two files")
    files = tuple(
        _hash_file(root, asset_root / name, label="GPT asset")
        for name in expected_names
    )
    if expected_hashes is not None:
        for record in files:
            name = Path(record["path"]).name
            # The file record uses a path-bound digest and carries a content digest.
            content_sha256 = record["content_sha256"]
            if content_sha256 != expected_hashes[name]:
                raise ValueError(f"fixed GPT asset content SHA-256 drifted: {name}")
    result = {
        "files": list(files),
        "revision": _GPT_REVISION,
    }
    result["sha256"] = _domain_hash(_GPT_HASH_DOMAIN, result)
    return result


def _release_tuple(value: str, distribution: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"required runtime distribution {distribution} is missing")
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        raise ValueError(
            f"required runtime distribution {distribution} has an invalid version"
        )
    return tuple(int(part or 0) for part in match.groups())


def validate_runtime_versions(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the sole supported v1 interpreter and package envelope."""

    if not isinstance(runtime, Mapping) or set(runtime) != {
        "implementation",
        "packages",
        "python",
    }:
        raise ValueError("runtime identity has missing or extra fields")
    if runtime.get("implementation") != "CPython":
        raise ValueError("runtime implementation must be CPython")
    python = runtime.get("python")
    if not isinstance(python, str) or re.match(r"^3\.10\.\d+(?:$|[+-])", python) is None:
        raise ValueError("runtime Python must be a supported CPython 3.10 release")
    packages = runtime.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(
        _REQUIRED_RUNTIME_MODULES
    ):
        raise ValueError("runtime package registry must contain exactly four dependencies")
    canonical_packages: dict[str, str] = {}
    for distribution in _REQUIRED_RUNTIME_MODULES:
        value = packages.get(distribution)
        if value is None:
            raise ValueError(f"required runtime distribution {distribution} is missing")
        release = _release_tuple(value, distribution)
        lower, upper = _SUPPORTED_RELEASE_RANGES[distribution]
        if not lower <= release < upper:
            raise ValueError(
                f"required runtime distribution {distribution} version {value!r} "
                "is outside the supported v1 range"
            )
        canonical_packages[distribution] = str(value)
    return {
        "implementation": "CPython",
        "packages": canonical_packages,
        "python": python,
    }


def runtime_versions() -> dict[str, Any]:
    """Capture only distributions that are both installed and importable."""

    packages: dict[str, str | None] = {}
    for distribution, module_name in _REQUIRED_RUNTIME_MODULES.items():
        try:
            importlib.import_module(module_name)
            packages[distribution] = importlib.metadata.version(distribution)
        except (ImportError, ModuleNotFoundError, importlib.metadata.PackageNotFoundError):
            packages[distribution] = None
    return {
        "implementation": platform.python_implementation(),
        "packages": packages,
        "python": platform.python_version(),
    }


def _verified_parsed_array_records(canonical_root: Path) -> list[dict[str, Any]]:
    canonical_root = Path(canonical_root)
    try:
        metadata = canonical_root.lstat()
    except OSError as exc:
        raise ValueError("canonical permitted-array directory is inaccessible") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("canonical permitted-array root must be a regular directory")
    records: list[dict[str, Any]] = []
    for pinned in _PARSED_ARRAY_RECORDS:
        dataset = str(pinned["dataset"])
        split = str(pinned["split"])
        expected = expected_parsed_array_sha256(dataset, split)
        if pinned["sha256"] != expected:
            raise ValueError("parsed-array provenance registry drifted from data registry")
        path = canonical_root / f"{dataset}_{split}.npy"
        values = _load_canonical_array(path, dataset, split, expected)
        actual = parsed_array_sha256(values, dataset, split)
        if actual != expected:
            raise ValueError("canonical permitted array semantic identity drifted")
        records.append(
            {
                **dict(pinned),
                "content_verified": True,
                "file": f"canonical/{dataset}_{split}.npy",
            }
        )
    return records


def _build_provenance(
    root: Path,
    *,
    runtime_versions: Mapping[str, Any],
    expected_gpt_hashes: Mapping[str, str] | None,
    canonical_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("experiment provenance root must be a regular directory")

    runtime = validate_runtime_versions(runtime_versions)
    parsed_records = _verified_parsed_array_records(
        root / "canonical" if canonical_root is None else Path(canonical_root)
    )

    production_paths = _production_paths(root)
    production_files = [
        _hash_file(root, path, label="production source")
        for path in production_paths
    ]
    test_files = [
        _hash_file(root, path, label="test source") for path in _test_paths(root)
    ]
    config_file, config_identity = _config_record(root)
    card_file = _hash_file(
        root, root / "research_card_v1.md", label="research card"
    )
    production_modules = {path.stem for path in production_paths}
    import_edges = {
        path.relative_to(root).as_posix(): _top_level_local_imports(
            path, production_modules
        )
        for path in production_paths
    }
    production = {
        "files": production_files,
        "import_closure": [item["path"] for item in production_files],
        "import_edges": import_edges,
    }
    production["sha256"] = _domain_hash(
        _PRODUCTION_HASH_DOMAIN,
        {
            "config": config_file,
            "files": production_files,
            "research_card": card_file,
        },
    )
    tests = {"files": test_files}
    tests["sha256"] = _domain_hash(_TEST_HASH_DOMAIN, test_files)
    parsed_arrays = {
        "payload_bytes_read": True,
        "records": parsed_records,
        "upstream": {
            "config_sha256": _UPSTREAM_PROTOCOL_CONFIG_SHA256,
            "protocol": "od-orbit-gpt-v1",
            "provider": "experiments.od_orbit_gpt_v1.protocol",
        },
    }
    parsed_arrays["sha256"] = _domain_hash(_PARSED_HASH_DOMAIN, parsed_arrays)
    gpt_assets = _gpt_records(root, expected_gpt_hashes)
    runtime = json.loads(canonical_json_bytes(runtime))
    runtime_sha256 = _domain_hash(_RUNTIME_HASH_DOMAIN, runtime)
    payload = {
        "acil_core_upstream": {
            "locator": "Imputation/models/geoanchor_v2_modules.py",
            "sha256": _ACIL_CORE_UPSTREAM_SHA256,
            "upstream_bytes_read": False,
        },
        "config": {
            "file": config_file,
            "status": config_identity["status"],
        },
        "config_sha256": config_identity["semantic_sha256"],
        "git_available": False,
        "git_commit": None,
        "gpt_assets": gpt_assets,
        "parsed_arrays": parsed_arrays,
        "production": production,
        "protocol": "acil-innovation-v1",
        "research_card": card_file,
        "runtime": runtime,
        "runtime_sha256": runtime_sha256,
        "schema_version": 1,
        "tests": tests,
    }
    payload["provenance_sha256"] = _domain_hash(_PROVENANCE_HASH_DOMAIN, payload)
    return payload


def build_provenance() -> dict[str, Any]:
    """Build provenance from the sole fixed package root and asset revision."""

    return _build_provenance(
        _EXPERIMENT_ROOT,
        runtime_versions=runtime_versions(),
        expected_gpt_hashes=_EXPECTED_GPT_HASHES,
        canonical_root=None,
    )


__all__ = ["build_provenance", "runtime_versions", "validate_runtime_versions"]
