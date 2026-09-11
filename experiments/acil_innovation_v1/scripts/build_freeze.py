"""Build and exclusively write the ACIL-Innovation implementation freeze v2.

The scientific protocol remains ``acil-innovation-v1``.  ``freeze_v2`` is a
new source/runtime implementation snapshot after the abandoned ``freeze_v1``;
the old directory is neither overwritten nor consulted as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Mapping

from ..config import (
    CUBLAS_WORKSPACE_CONFIG,
    canonical_json_bytes,
    load_protocol_config,
)
from ..manifest import _build_manifest, validate_manifest
from ..provenance import build_provenance, validate_runtime_versions


_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_FIXED_GENERATED_ROOT = _EXPERIMENT_ROOT / "generated"
_FREEZE_DIRECTORY = "freeze_v2"
_FREEZE_VERSION = "v2"
_FREEZE_STATE = "protocol_v2_frozen"
_FREEZE_HASH_DOMAIN = b"acil-innovation-v1:freeze-record:v2\x00"


@dataclass(frozen=True, slots=True)
class FreezeBundle:
    record: dict[str, Any]
    freeze_sha256: str


@dataclass(frozen=True, slots=True)
class FreezeArtifact:
    root: Path
    provenance_path: Path
    manifest_path: Path
    freeze_path: Path
    anchor_path: Path


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _execution_supported_handlers() -> frozenset[tuple[str, str]]:
    """Load the source-pinned static worker coverage registry fail closed."""

    try:
        from ..execution import SUPPORTED_JOB_HANDLERS
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValueError("worker handler coverage registry is unavailable") from exc
    if not isinstance(SUPPORTED_JOB_HANDLERS, frozenset):
        raise ValueError("worker handler coverage registry must be a frozenset")
    handlers: set[tuple[str, str]] = set()
    for value in SUPPORTED_JOB_HANDLERS:
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not all(isinstance(item, str) and item for item in value)
        ):
            raise ValueError("worker handler coverage registry is malformed")
        handlers.add((value[0], value[1]))
    return frozenset(handlers)


def _freeze_preflight(
    provenance: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    supported_handlers: frozenset[tuple[str, str]],
) -> dict[str, Any]:
    """Verify runnable identities without claiming tests or smoke were executed."""

    validate_runtime_versions(provenance.get("runtime", {}))
    parsed = provenance.get("parsed_arrays")
    if not isinstance(parsed, Mapping) or parsed.get("payload_bytes_read") is not True:
        raise ValueError("parsed-array payload verification is absent")
    records = parsed.get("records")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("parsed-array verification must contain exactly four parents")
    identities = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("content_verified") is not True:
            raise ValueError("parsed-array content verification is incomplete")
        identity = (record.get("dataset"), record.get("split"))
        if not all(isinstance(item, str) for item in identity):
            raise ValueError("parsed-array verification identity is malformed")
        identities.append(identity)
    expected_parents = {
        ("abilene", "train"),
        ("abilene", "val"),
        ("geant", "train"),
        ("geant", "val"),
    }
    if len(set(identities)) != 4 or set(identities) != expected_parents:
        raise ValueError("parsed-array verification identity grid drifted")

    tests = provenance.get("tests")
    if not isinstance(tests, Mapping) or not isinstance(tests.get("files"), list):
        raise ValueError("test-source provenance is absent")
    if not tests["files"]:
        raise ValueError("test-source provenance inventory is empty")
    _require_sha256(tests.get("sha256"), "tests SHA-256")

    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("planned manifest job grid is absent")
    planned_handlers: set[tuple[str, str]] = set()
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ValueError("planned manifest job identity is malformed")
        stage, method = job.get("stage"), job.get("method")
        if not isinstance(stage, str) or not stage or not isinstance(method, str) or not method:
            raise ValueError("planned manifest job handler identity is malformed")
        planned_handlers.add((stage, method))
    if frozenset(planned_handlers) != supported_handlers:
        missing = sorted(planned_handlers - set(supported_handlers))
        extra = sorted(set(supported_handlers) - planned_handlers)
        raise ValueError(
            f"worker handler coverage differs from active manifest: "
            f"missing={missing!r}, extra={extra!r}"
        )
    return {
        "deterministic_gpu_smoke_repeat_required_before_launch": True,
        "full_test_suite_required_before_launch": True,
        "handler_coverage": "verified",
        "parsed_array_payloads": "verified",
        "runtime_dependencies": "verified",
        "test_execution_claimed": False,
        "test_sources": "content_hashed_not_execution_evidence",
    }


def _build_freeze_record(
    provenance: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    supported_handlers: frozenset[tuple[str, str]] | None = None,
) -> FreezeBundle:
    if not isinstance(provenance, Mapping) or not isinstance(manifest, Mapping):
        raise TypeError("provenance and manifest must be mappings")
    if load_protocol_config()["protocol"]["status"] != "draft_protocol":
        raise ValueError("freeze requires the immutable draft_protocol config")
    manifest_sha256 = _require_sha256(
        manifest.get("manifest_sha256"), "manifest SHA-256"
    )
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    validate_manifest(manifest_payload, expected_provenance=provenance)
    preflight = _freeze_preflight(
        provenance,
        manifest,
        supported_handlers=(
            _execution_supported_handlers()
            if supported_handlers is None
            else supported_handlers
        ),
    )
    record = {
        "config_sha256": _require_sha256(
            provenance.get("config_sha256"), "config SHA-256"
        ),
        "config_status_at_freeze": "draft_protocol",
        "cuda_determinism": {
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
            "torch_deterministic_algorithms": True,
        },
        "git_available": False,
        "git_commit": None,
        "gpt_assets_sha256": _require_sha256(
            provenance.get("gpt_assets", {}).get("sha256"),
            "GPT assets SHA-256",
        ),
        "manifest_sha256": manifest_sha256,
        "parsed_arrays_sha256": _require_sha256(
            provenance.get("parsed_arrays", {}).get("sha256"),
            "parsed arrays SHA-256",
        ),
        "production_sha256": _require_sha256(
            provenance.get("production", {}).get("sha256"),
            "production SHA-256",
        ),
        "preflight": preflight,
        "protocol": "acil-innovation-v1",
        "provenance_sha256": _require_sha256(
            provenance.get("provenance_sha256"), "provenance SHA-256"
        ),
        "runtime_sha256": _require_sha256(
            provenance.get("runtime_sha256"), "runtime SHA-256"
        ),
        "schema_version": 1,
        "state": _FREEZE_STATE,
        "tests_sha256": _require_sha256(
            provenance.get("tests", {}).get("sha256"), "tests SHA-256"
        ),
        "version": _FREEZE_VERSION,
    }
    encoded = canonical_json_bytes(record)
    digest = hashlib.sha256(_FREEZE_HASH_DOMAIN)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return FreezeBundle(record=record, freeze_sha256=digest.hexdigest())


def build_freeze_record() -> FreezeBundle:
    """Build the fixed in-memory freeze twice identically when sources are stable."""

    provenance = build_provenance()
    manifest = _build_manifest(provenance)
    return _build_freeze_record(provenance, manifest)


def _plain_directory(path: Path, *, create: bool) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if not create:
            raise
        path.mkdir(mode=0o700)
        metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"freeze directory is not a plain directory: {path}")


def _exclusive_write(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short freeze artifact write")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bundle(
    generated_root: Path,
    provenance: Mapping[str, Any],
    manifest: Mapping[str, Any],
    bundle: FreezeBundle,
) -> FreezeArtifact:
    generated_root = Path(generated_root)
    _plain_directory(generated_root, create=True)
    freeze_root = generated_root / _FREEZE_DIRECTORY
    try:
        freeze_root.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(
            f"{_FREEZE_DIRECTORY} already exists; overwrite is forbidden"
        ) from None
    created = True
    try:
        provenance_path = freeze_root / (
            f"provenance-{provenance['provenance_sha256']}.json"
        )
        manifest_path = freeze_root / f"manifest-{manifest['manifest_sha256']}.json"
        freeze_path = freeze_root / f"freeze-{bundle.freeze_sha256}.json"
        anchor_path = freeze_root / "anchor.json"
        _exclusive_write(provenance_path, provenance)
        _exclusive_write(manifest_path, manifest)
        _exclusive_write(freeze_path, bundle.record)
        anchor = {
            "freeze_file": freeze_path.name,
            "freeze_sha256": bundle.freeze_sha256,
            "manifest_file": manifest_path.name,
            "manifest_sha256": manifest["manifest_sha256"],
            "protocol": "acil-innovation-v1",
            "provenance_file": provenance_path.name,
            "provenance_sha256": provenance["provenance_sha256"],
            "schema_version": 1,
            "version": _FREEZE_VERSION,
        }
        _exclusive_write(anchor_path, anchor)
        directory_descriptor = os.open(
            freeze_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        created = False
        return FreezeArtifact(
            root=freeze_root,
            provenance_path=provenance_path,
            manifest_path=manifest_path,
            freeze_path=freeze_path,
            anchor_path=anchor_path,
        )
    finally:
        if created:
            shutil.rmtree(freeze_root, ignore_errors=True)


def _write_freeze_for_test(
    root: Path,
    provenance: Mapping[str, Any],
    manifest: Mapping[str, Any],
    bundle: FreezeBundle,
) -> FreezeArtifact:
    """Test-only injected parent; production callers cannot select it."""

    return _write_bundle(Path(root), provenance, manifest, bundle)


def write_freeze() -> FreezeArtifact:
    """Write once to ``generated/freeze_v2``; an existing v2 root is fatal."""

    provenance = build_provenance()
    manifest = _build_manifest(provenance)
    bundle = _build_freeze_record(provenance, manifest)
    return _write_bundle(_FIXED_GENERATED_ROOT, provenance, manifest, bundle)


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("build_freeze accepts no arguments")
    artifact = write_freeze()
    sys.stdout.write(
        canonical_json_bytes(
            {"freeze_root": str(artifact.root), "status": "written"}
        ).decode("ascii")
        + "\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module command
    raise SystemExit(main())
