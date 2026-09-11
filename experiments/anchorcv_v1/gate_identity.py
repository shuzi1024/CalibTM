"""Sealed content identity for the six-job AnchorCV final gate.

This module is deliberately an authority layer, not a gate-data loader.  It
accepts only a completed, independently audited full-tune extension report and
the exact verified method freeze.  No gate cohort, array, mask, or result is
opened here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any

from .freeze import verify_freeze_record


_DATASETS = ("abilene", "geant")
_BUNDLES = (1, 2, 3)
_EXPECTED_KEYS = tuple(
    (bundle, dataset)
    for bundle in _BUNDLES
    for dataset in _DATASETS
)
_FLOWS = {"abilene": 144, "geant": 462}
_MASK_FAMILIES = ("random", "internal_block", "two_burst")
_REPORT_SCHEMA = "anchorcv-v1:extension-review:v1"
_AUTHORITY_PAYLOAD_SCHEMA = "anchorcv-v1:gate-authority-payload:v1"
_AUTHORITY_RECORD_SCHEMA = "anchorcv-v1:gate-authority-record:v1"
_GATE_ID = "anchorcv-v1:final-gate:v1"
_METHOD_FREEZE_DOMAIN = b"anchorcv-v1:gate-method-freeze:v1\x00"
_GATE_JOB_DOMAIN = b"anchorcv-v1:gate-job:v1\x00"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_REPORT_FIELDS = {
    "schema",
    "status",
    "verdict",
    "confirmation_authorized",
    "job_count",
    "checkpoint_bindings",
    "grid",
    "integrity",
    "prototype",
    "uncertainty",
    "multibundle_adjudication",
    "interpretation",
}
_REPORT_INTEGRITY_FIELDS = {
    "valid",
    "source_tree_sha256",
    "config_sha256",
    "git_available",
    "git_commit",
    "test_access",
}
_BINDING_FIELDS = {
    "dataset",
    "seed_bundle",
    "source_training_stage",
    "source_training_job_id",
    "source_result_sha256",
    "source_manifest_sha256",
    "neural_checkpoint_file_sha256",
    "neural_checkpoint_tensor_sha256",
    "acil_checkpoint_file_sha256",
    "best_epoch",
    "best_source_dev_nmae",
}
_GRID_FIELDS = {"dataset", "job_id", "seed_bundle", "stage"}
_PAYLOAD_FIELDS = {
    "schema",
    "protocol",
    "gate",
    "source_tree_sha256",
    "config_sha256",
    "git_available",
    "git_commit",
    "test_access",
    "freeze_record_file_sha256",
    "extension_report_canonical_sha256",
    "extension_report_schema",
    "extension_report_verdict",
    "checkpoint_bindings",
}
_RECORD_FIELDS = {
    "schema",
    "method_freeze_sha256",
    "authority_payload",
}


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
        raise ValueError("gate authority requires canonical finite JSON") from exc


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token!r} is forbidden")


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_file(
    path: Path,
    *,
    label: str,
    require_readonly: bool = False,
) -> bytes:
    """Read one stable regular file without following its final component."""

    path = Path(path)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is inaccessible: {path}") from exc
    if stat.S_ISLNK(path_before.st_mode):
        raise ValueError(f"{label} symlink is forbidden: {path}")
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    if require_readonly and path_before.st_mode & 0o222:
        raise ValueError(f"{label} must be read-only: {path}")

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


def _decode_json(encoded: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid finite JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields drifted; "
            f"missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _expected_stage(bundle: int) -> str:
    return "prototype" if bundle == 1 else "extension"


def _verified_freeze(
    path: Path,
) -> tuple[dict[str, object], str]:
    """Bind verified current source/config state to exact freeze-file bytes."""

    path = Path(path)
    encoded_before = _read_regular_file(path, label="freeze record")
    parsed = _decode_json(encoded_before, label="freeze record")
    verified = verify_freeze_record(path)
    if not isinstance(verified, Mapping):
        raise ValueError("verified freeze record must be an object")
    encoded_after = _read_regular_file(path, label="freeze record")
    if encoded_before != encoded_after:
        raise ValueError("freeze record changed during verification")
    if parsed != verified:
        raise ValueError("freeze record bytes differ from verified current record")

    required = {
        "schema_version",
        "protocol",
        "source_tree_sha256",
        "config_sha256",
        "git_available",
        "git_commit",
    }
    if not required.issubset(verified):
        raise ValueError("verified freeze record authority fields are incomplete")
    if (
        verified.get("schema_version") != 1
        or isinstance(verified.get("schema_version"), bool)
        or verified.get("protocol") != "anchorcv-v1"
        or verified.get("git_available") is not False
        or verified.get("git_commit") is not None
    ):
        raise ValueError("verified freeze record authority identity drifted")
    _require_sha256(
        verified.get("source_tree_sha256"),
        label="freeze source_tree_sha256",
    )
    _require_sha256(
        verified.get("config_sha256"),
        label="freeze config_sha256",
    )
    return dict(verified), hashlib.sha256(encoded_before).hexdigest()


def _validated_report(
    path: Path,
    *,
    freeze: Mapping[str, object],
) -> tuple[dict[str, object], str, list[dict[str, object]]]:
    encoded = _read_regular_file(Path(path), label="extension report")
    report = _decode_json(encoded, label="extension report")
    _exact_fields(report, _REPORT_FIELDS, label="extension report")
    if report.get("schema") != _REPORT_SCHEMA:
        raise ValueError("extension report schema/version drifted")
    if report.get("status") != "verified_complete":
        raise ValueError("extension report is not verified complete")
    if report.get("verdict") != "proceed":
        raise ValueError("extension report verdict must be proceed")
    if report.get("confirmation_authorized") is not True:
        raise ValueError("extension report does not authorize confirmation")
    if report.get("job_count") != 6 or isinstance(report.get("job_count"), bool):
        raise ValueError("extension report must bind exactly six jobs")
    if not isinstance(report.get("prototype"), Mapping):
        raise ValueError("extension report prototype must be an object")
    if not isinstance(report.get("uncertainty"), Mapping):
        raise ValueError("extension report uncertainty must be an object")
    multibundle = report.get("multibundle_adjudication")
    if not isinstance(multibundle, Mapping):
        raise ValueError(
            "extension report multi-bundle adjudication must be an object"
        )
    multibundle_integrity = multibundle.get("integrity")
    if (
        multibundle.get("schema")
        != "anchorcv-v1:multibundle-adjudication:v1"
        or multibundle.get("verdict") != "proceed"
        or multibundle.get("verdict") != report.get("verdict")
        or not isinstance(multibundle_integrity, Mapping)
        or multibundle_integrity.get("valid") is not True
    ):
        raise ValueError(
            "extension report multi-bundle adjudication must be a valid "
            "proceed verdict aligned with the root verdict"
        )
    if (
        not isinstance(report.get("interpretation"), str)
        or not report["interpretation"]
    ):
        raise ValueError("extension report interpretation must be nonempty")

    integrity = report.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("extension report integrity must be an object")
    _exact_fields(
        integrity,
        _REPORT_INTEGRITY_FIELDS,
        label="extension report integrity",
    )
    expected_integrity = {
        "valid": True,
        "source_tree_sha256": freeze.get("source_tree_sha256"),
        "config_sha256": freeze.get("config_sha256"),
        "git_available": False,
        "git_commit": None,
        "test_access": False,
    }
    for field, expected in expected_integrity.items():
        if integrity.get(field) != expected:
            raise ValueError(
                f"extension report integrity {field} differs from the freeze"
            )

    raw_bindings = report.get("checkpoint_bindings")
    if not isinstance(raw_bindings, list) or len(raw_bindings) != 6:
        raise ValueError("extension report must contain six checkpoint bindings")
    bindings: list[dict[str, object]] = []
    keys: list[tuple[int, str]] = []
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, Mapping):
            raise ValueError(f"checkpoint binding {index} must be an object")
        _exact_fields(
            raw,
            _BINDING_FIELDS,
            label=f"checkpoint binding {index}",
        )
        dataset = raw.get("dataset")
        bundle = raw.get("seed_bundle")
        if dataset not in _DATASETS:
            raise ValueError(f"checkpoint binding {index} dataset is unregistered")
        if (
            isinstance(bundle, bool)
            or not isinstance(bundle, int)
            or bundle not in _BUNDLES
        ):
            raise ValueError(f"checkpoint binding {index} bundle is unregistered")
        key = (bundle, str(dataset))
        if raw.get("source_training_stage") != _expected_stage(bundle):
            raise ValueError(
                f"checkpoint binding {index} source training stage drifted"
            )
        for field in (
            "source_training_job_id",
            "source_result_sha256",
            "source_manifest_sha256",
            "neural_checkpoint_file_sha256",
            "neural_checkpoint_tensor_sha256",
            "acil_checkpoint_file_sha256",
        ):
            _require_sha256(
                raw.get(field),
                label=f"checkpoint binding {index} {field}",
            )
        best_epoch = raw.get("best_epoch")
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or not 0 <= best_epoch < 20
        ):
            raise ValueError(
                f"checkpoint binding {index} best_epoch must be in [0, 20)"
            )
        best_nmae = raw.get("best_source_dev_nmae")
        if (
            isinstance(best_nmae, bool)
            or not isinstance(best_nmae, (int, float))
            or not math.isfinite(float(best_nmae))
            or float(best_nmae) < 0.0
        ):
            raise ValueError(
                f"checkpoint binding {index} best_source_dev_nmae is invalid"
            )
        bindings.append(dict(raw))
        keys.append(key)
    if tuple(keys) != _EXPECTED_KEYS:
        raise ValueError(
            "checkpoint binding grid must be the fixed bundle-major six-job grid"
        )

    raw_grid = report.get("grid")
    if not isinstance(raw_grid, list) or len(raw_grid) != 6:
        raise ValueError("extension report grid must contain exactly six jobs")
    for index, (raw, binding) in enumerate(zip(raw_grid, bindings)):
        if not isinstance(raw, Mapping):
            raise ValueError(f"extension report grid item {index} must be an object")
        _exact_fields(raw, _GRID_FIELDS, label=f"extension report grid item {index}")
        expected = {
            "dataset": binding["dataset"],
            "job_id": binding["source_training_job_id"],
            "seed_bundle": binding["seed_bundle"],
            "stage": binding["source_training_stage"],
        }
        if dict(raw) != expected:
            raise ValueError(
                f"extension report grid item {index} differs from checkpoint binding"
            )

    canonical_sha256 = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report, canonical_sha256, bindings


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _method_freeze_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _METHOD_FREEZE_DOMAIN + _canonical_json(payload)
    ).hexdigest()


_AUTHORITY_SEAL = object()


class VerifiedGateAuthority:
    """Deeply immutable authority value constructible only by this module."""

    __slots__ = (
        "_payload",
        "_method_freeze_sha256",
        "_record_bytes",
        "_locked",
    )

    def __init__(
        self,
        payload: Mapping[str, object],
        method_freeze_sha256: str,
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _AUTHORITY_SEAL:
            raise TypeError(
                "VerifiedGateAuthority is available only from the verified factory"
            )
        normalized = json.loads(_canonical_json(payload).decode("ascii"))
        record = {
            "schema": _AUTHORITY_RECORD_SCHEMA,
            "method_freeze_sha256": method_freeze_sha256,
            "authority_payload": normalized,
        }
        object.__setattr__(self, "_payload", _deep_freeze(normalized))
        object.__setattr__(self, "_method_freeze_sha256", method_freeze_sha256)
        object.__setattr__(self, "_record_bytes", _canonical_json(record))
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("VerifiedGateAuthority is immutable")
        object.__setattr__(self, name, value)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("VerifiedGateAuthority cannot be subclassed")

    @property
    def payload(self) -> MappingProxyType:
        return self._payload  # type: ignore[return-value]

    @property
    def method_freeze_sha256(self) -> str:
        return self._method_freeze_sha256

    def checkpoint_binding(
        self,
        dataset: str,
        seed_bundle: int,
    ) -> MappingProxyType:
        if dataset not in _DATASETS:
            raise ValueError("dataset is outside the fixed final-gate grid")
        if (
            isinstance(seed_bundle, bool)
            or not isinstance(seed_bundle, int)
            or seed_bundle not in _BUNDLES
        ):
            raise ValueError("seed bundle is outside the fixed final-gate grid")
        bindings = self._payload["checkpoint_bindings"]
        for binding in bindings:  # type: ignore[union-attr]
            if (
                binding["dataset"] == dataset
                and binding["seed_bundle"] == seed_bundle
            ):
                return binding
        raise RuntimeError("verified authority lost a fixed checkpoint binding")


def create_verified_gate_authority(
    *,
    extension_report: str | Path,
    freeze_record: str | Path,
    prototype_output_root: str | Path,
    extension_output_root: str | Path,
) -> VerifiedGateAuthority:
    """Re-audit exact artifacts, verify current state, and seal authority."""

    freeze, freeze_file_sha256 = _verified_freeze(Path(freeze_record))
    report, report_sha256, bindings = _validated_report(
        Path(extension_report),
        freeze=freeze,
    )
    # Delayed by design: importing the artifact auditor must not make this
    # identity-only module load data or artifact machinery at module import.
    from .review import review_extension

    reviewed = review_extension(
        prototype_output_root=Path(prototype_output_root),
        extension_output_root=Path(extension_output_root),
        freeze_record=Path(freeze_record),
    )
    if not isinstance(reviewed, Mapping):
        raise ValueError("artifact re-review did not return an extension report")
    try:
        reviewed_canonical = _canonical_json(reviewed)
    except ValueError as exc:
        raise ValueError("artifact re-review returned invalid JSON evidence") from exc
    if reviewed_canonical != _canonical_json(report):
        raise ValueError(
            "current extension report was not exactly rederived by artifact "
            "re-review"
        )
    payload: dict[str, object] = {
        "schema": _AUTHORITY_PAYLOAD_SCHEMA,
        "protocol": "anchorcv-v1",
        "gate": _GATE_ID,
        "source_tree_sha256": freeze["source_tree_sha256"],
        "config_sha256": freeze["config_sha256"],
        "git_available": False,
        "git_commit": None,
        "test_access": False,
        "freeze_record_file_sha256": freeze_file_sha256,
        "extension_report_canonical_sha256": report_sha256,
        "extension_report_schema": _REPORT_SCHEMA,
        "extension_report_verdict": "proceed",
        "checkpoint_bindings": bindings,
    }
    _exact_fields(payload, _PAYLOAD_FIELDS, label="gate authority payload")
    method_freeze = _method_freeze_sha256(payload)
    return VerifiedGateAuthority(
        payload,
        method_freeze,
        _seal=_AUTHORITY_SEAL,
    )


def _plain_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = path.parent.lstat()
    except OSError as exc:
        raise ValueError(
            f"cannot create gate-authority parent directory: {path.parent}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("gate-authority parent symlink is forbidden")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("gate-authority parent must be a directory")


def write_gate_authority(
    authority: VerifiedGateAuthority,
    output_path: str | Path,
) -> None:
    """Exclusively persist canonical read-only authority bytes."""

    if (
        type(authority) is not VerifiedGateAuthority
        or not hasattr(authority, "_record_bytes")
    ):
        raise TypeError("authority must come from the verified factory")
    path = Path(output_path)
    _plain_parent(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError:
        raise FileExistsError(
            f"gate authority already exists; overwrite is forbidden: {path}"
        ) from None
    except OSError as exc:
        raise ValueError(f"cannot exclusively create gate authority: {path}") from exc
    try:
        remaining = memoryview(authority._record_bytes)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short gate-authority write")
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _validate_authority_record(record: Mapping[str, object]) -> None:
    _exact_fields(record, _RECORD_FIELDS, label="gate authority record")
    if record.get("schema") != _AUTHORITY_RECORD_SCHEMA:
        raise ValueError("gate authority record schema/version drifted")
    method_hash = _require_sha256(
        record.get("method_freeze_sha256"),
        label="method_freeze_sha256",
    )
    payload = record.get("authority_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("gate authority payload must be an object")
    _exact_fields(payload, _PAYLOAD_FIELDS, label="gate authority payload")
    fixed = {
        "schema": _AUTHORITY_PAYLOAD_SCHEMA,
        "protocol": "anchorcv-v1",
        "gate": _GATE_ID,
        "git_available": False,
        "git_commit": None,
        "test_access": False,
        "extension_report_schema": _REPORT_SCHEMA,
        "extension_report_verdict": "proceed",
    }
    for field, expected in fixed.items():
        if payload.get(field) != expected:
            raise ValueError(f"gate authority payload {field} drifted")
    for field in (
        "source_tree_sha256",
        "config_sha256",
        "freeze_record_file_sha256",
        "extension_report_canonical_sha256",
    ):
        _require_sha256(payload.get(field), label=f"gate authority {field}")
    bindings = payload.get("checkpoint_bindings")
    if not isinstance(bindings, list) or len(bindings) != 6:
        raise ValueError("gate authority payload must bind exactly six checkpoints")
    if _method_freeze_sha256(payload) != method_hash:
        raise ValueError("gate authority method_freeze_sha256 mismatch")


def verify_gate_authority(
    *,
    authority_record: str | Path,
    extension_report: str | Path,
    freeze_record: str | Path,
    prototype_output_root: str | Path,
    extension_output_root: str | Path,
) -> VerifiedGateAuthority:
    """Strictly rebind a stored authority to current source/config/report."""

    encoded = _read_regular_file(
        Path(authority_record),
        label="gate authority record",
        require_readonly=True,
    )
    record = _decode_json(encoded, label="gate authority record")
    if _canonical_json(record) != encoded:
        raise ValueError("gate authority record JSON is not canonical")
    _validate_authority_record(record)
    current = create_verified_gate_authority(
        extension_report=extension_report,
        freeze_record=freeze_record,
        prototype_output_root=prototype_output_root,
        extension_output_root=extension_output_root,
    )
    if current._record_bytes != encoded:
        raise ValueError(
            "gate authority differs from the current freeze or extension report"
        )
    return current


@dataclass(frozen=True, slots=True)
class GateJobSpec:
    """One of exactly six fixed checkpoint-reuse final-gate jobs."""

    dataset: str
    seed_bundle: int
    output_root: Path
    authority: VerifiedGateAuthority
    window_schedule_sha256: str

    def __post_init__(self) -> None:
        if self.dataset not in _DATASETS:
            raise ValueError("dataset is outside the fixed final-gate grid")
        if (
            isinstance(self.seed_bundle, bool)
            or not isinstance(self.seed_bundle, int)
            or self.seed_bundle not in _BUNDLES
        ):
            raise ValueError("seed bundle is outside the fixed final-gate grid")
        if (
            type(self.authority) is not VerifiedGateAuthority
            or not hasattr(self.authority, "_record_bytes")
        ):
            raise TypeError("authority must come from the verified authority factory")
        _require_sha256(
            self.window_schedule_sha256,
            label="window_schedule_sha256",
        )
        output_root = Path(self.output_root)
        if output_root == Path("/"):
            raise ValueError("output_root cannot be the filesystem root")
        resolved = output_root.resolve()
        if resolved == Path("/"):
            raise ValueError("output_root cannot resolve to the filesystem root")
        object.__setattr__(self, "output_root", resolved)

    @property
    def scientific_identity(self) -> dict[str, object]:
        payload = self.authority.payload
        binding = self.authority.checkpoint_binding(
            self.dataset,
            self.seed_bundle,
        )
        return {
            "protocol": "anchorcv-v1",
            "gate": _GATE_ID,
            "stage": "final_gate",
            "evaluation_cohort": "gate",
            "dataset": self.dataset,
            "seed_bundle": self.seed_bundle,
            "flows": _FLOWS[self.dataset],
            "mask_families": list(_MASK_FAMILIES),
            "checkpoint_reuse": "exact_verified_full_tune",
            "training": False,
            "compute_lane": "cuda_bf16_math_sdp",
            "source_training_stage": binding["source_training_stage"],
            "source_training_job_id": binding["source_training_job_id"],
            "source_result_sha256": binding["source_result_sha256"],
            "source_manifest_sha256": binding["source_manifest_sha256"],
            "source_neural_checkpoint_file_sha256": binding[
                "neural_checkpoint_file_sha256"
            ],
            "source_neural_checkpoint_tensor_sha256": binding[
                "neural_checkpoint_tensor_sha256"
            ],
            "source_acil_checkpoint_file_sha256": binding[
                "acil_checkpoint_file_sha256"
            ],
            "method_freeze_sha256": self.authority.method_freeze_sha256,
            "window_schedule_sha256": self.window_schedule_sha256,
            "source_tree_sha256": payload["source_tree_sha256"],
            "config_sha256": payload["config_sha256"],
            "git_available": False,
            "git_commit": None,
            "test_access": False,
        }

    @property
    def identity_json(self) -> str:
        return _canonical_json(self.scientific_identity).decode("ascii")

    @property
    def job_id(self) -> str:
        return hashlib.sha256(
            _GATE_JOB_DOMAIN + self.identity_json.encode("ascii")
        ).hexdigest()

    @property
    def job_directory(self) -> Path:
        return self.output_root / self.job_id


__all__ = [
    "GateJobSpec",
    "VerifiedGateAuthority",
    "create_verified_gate_authority",
    "verify_gate_authority",
    "write_gate_authority",
]
