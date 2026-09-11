"""Immutable, content-verified result artifacts for frozen planned jobs."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file

from .checkpoint import CheckpointRecord, _tensor_sha
from .checkpoint_validation import (
    embedded_acil_source_job,
    validate_embedded_acil_state,
    validate_named_artifact_registry,
    validate_registered_state_schema,
    validate_succeeded_job_inventory,
)
from .config import canonical_json_bytes, load_protocol_config, protocol_config_sha256
from .jobs import PlannedJob, planned_jobs
from .masks import build_evaluation_mask, factory_oracle_partition
from .metrics import MetricRecord, target_sha256
from .registries import dataset_spec, ordered_window_starts, stage_spec


_METRIC_KEYS = {
    "absolute_error_sum",
    "absolute_truth_sum",
    "identity",
    "squared_error_sum",
    "squared_truth_sum",
    "target_count",
    "target_set_sha256",
}
_ARTIFACT_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_RESERVED_ARTIFACT_PATHS = {"records.jsonl", "result.json", "started.json"}


@dataclass(frozen=True, slots=True)
class FileArtifact:
    path: Path
    sha256: str
    bytes: int
    count: int | None = None


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


def _plain_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} directory is inaccessible") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a regular directory")


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _read_regular_bytes(path: Path) -> tuple[bytes, str]:
    path = Path(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"result artifact is not an accessible regular file: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"result artifact must be a regular file: {path}")
    try:
        content = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ValueError(f"result artifact cannot be read: {path}") from exc
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
        raise ValueError(f"result artifact changed while being read: {path}")
    return content, hashlib.sha256(content).hexdigest()


def regular_file_sha256(path: Path, *, expected_sha256: str | None = None) -> str:
    """Hash one non-symlink regular result file and optionally verify its hash."""

    _, actual = _read_regular_bytes(Path(path))
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, "expected file hash")
        if actual != expected:
            raise ValueError("regular result file hash mismatch")
    return actual


def _exclusive_bytes(path: Path, content: bytes, *, count: int | None) -> FileArtifact:
    path = Path(path)
    _plain_directory(path.parent, label="result artifact parent")
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise
    actual_content, actual_sha256 = _read_regular_bytes(path)
    if actual_content != content:
        raise ValueError("result artifact changed immediately after exclusive write")
    return FileArtifact(
        path=path,
        sha256=actual_sha256,
        bytes=len(actual_content),
        count=count,
    )


def write_canonical_json_exclusive(path: Path, value: Any) -> FileArtifact:
    """Write one canonical JSON object using exclusive file creation."""

    return _exclusive_bytes(
        Path(path), canonical_json_bytes(value) + b"\n", count=None
    )


def write_canonical_jsonl_exclusive(
    path: Path, rows: Iterable[Mapping[str, object]]
) -> FileArtifact:
    """Write canonical JSON Lines in order using exclusive file creation."""

    values = tuple(rows)
    content = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in values)
    return _exclusive_bytes(Path(path), content, count=len(values))


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json(content: bytes, *, label: str) -> Any:
    try:
        value = json.loads(
            content.decode("ascii"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {item!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    return value


def _read_canonical_json(path: Path) -> tuple[Any, FileArtifact]:
    content, sha256 = _read_regular_bytes(path)
    if not content.endswith(b"\n"):
        raise ValueError("canonical JSON must end in exactly one record newline")
    value = _decode_json(content[:-1], label="result JSON")
    if content != canonical_json_bytes(value) + b"\n":
        raise ValueError("result JSON byte representation is not canonical")
    return value, FileArtifact(path=Path(path), sha256=sha256, bytes=len(content))


def _read_canonical_jsonl(path: Path) -> tuple[tuple[dict[str, Any], ...], FileArtifact]:
    content, sha256 = _read_regular_bytes(path)
    if not content:
        return (), FileArtifact(path=Path(path), sha256=sha256, bytes=0, count=0)
    lines = content.splitlines(keepends=True)
    if any(not line.endswith(b"\n") or line == b"\n" for line in lines):
        raise ValueError("JSONL contains an unterminated or empty record")
    rows: list[dict[str, Any]] = []
    for line in lines:
        value = _decode_json(line[:-1], label="result JSONL row")
        if not isinstance(value, dict):
            raise ValueError("each result JSONL row must be an object")
        if line != canonical_json_bytes(value) + b"\n":
            raise ValueError("result JSONL row byte representation is not canonical")
        rows.append(value)
    return tuple(rows), FileArtifact(
        path=Path(path), sha256=sha256, bytes=len(content), count=len(rows)
    )


def _metric_payload(record: MetricRecord | Mapping[str, object]) -> dict[str, object]:
    if isinstance(record, MetricRecord):
        return record.to_json()
    if not isinstance(record, Mapping):
        raise TypeError("metric record must be a MetricRecord or mapping")
    return dict(record)


def _validate_metric_payload(payload: Mapping[str, object]) -> dict[str, object]:
    if set(payload) != _METRIC_KEYS:
        raise ValueError("metric record schema has missing or extra fields")
    identity = payload["identity"]
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("metric record identity must be a nonempty object")
    for name in (
        "absolute_error_sum",
        "absolute_truth_sum",
        "squared_error_sum",
        "squared_truth_sum",
    ):
        value = payload[name]
        if isinstance(value, bool):
            raise ValueError(f"metric record {name} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"metric record {name} must be finite and nonnegative")
    count = payload["target_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("metric record target_count must be a positive integer")
    _require_sha256(payload["target_set_sha256"], "metric target-set hash")
    canonical = canonical_json_bytes(dict(payload))
    return json.loads(canonical.decode("ascii"))


def _identity_key(identity: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(dict(identity))


def _ordered_complete_metric_rows(
    records: Iterable[MetricRecord | Mapping[str, object]],
    expected_identities: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    expected_values = tuple(dict(identity) for identity in expected_identities)
    expected_keys = tuple(_identity_key(identity) for identity in expected_values)
    expected_set = set(expected_keys)
    if len(expected_keys) != len(expected_set):
        raise ValueError("expected metric identity registry contains duplicates")
    by_identity: dict[bytes, dict[str, object]] = {}
    for record in records:
        payload = _validate_metric_payload(_metric_payload(record))
        key = _identity_key(payload["identity"])
        if key in by_identity:
            raise ValueError("metric JSONL contains a duplicate identity")
        if key not in expected_set:
            raise ValueError("metric JSONL contains an unexpected identity")
        by_identity[key] = payload
    missing = [key for key in expected_keys if key not in by_identity]
    if missing:
        raise ValueError("metric JSONL has missing registered identities")
    return tuple(by_identity[key] for key in expected_keys)


def write_metric_records_exclusive(
    path: Path,
    records: Iterable[MetricRecord | Mapping[str, object]],
    *,
    expected_identities: Sequence[Mapping[str, object]],
) -> FileArtifact:
    """Validate a complete unique identity grid and write it in registry order."""

    ordered = _ordered_complete_metric_rows(records, expected_identities)
    return write_canonical_jsonl_exclusive(Path(path), ordered)


def validate_metric_records_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_identities: Sequence[Mapping[str, object]],
) -> FileArtifact:
    """Re-read and validate a canonical complete metric JSONL artifact."""

    expected_sha256 = _require_sha256(expected_sha256, "records file hash")
    rows, artifact = _read_canonical_jsonl(Path(path))
    if artifact.sha256 != expected_sha256:
        raise ValueError("metric records file hash mismatch")
    ordered = _ordered_complete_metric_rows(rows, expected_identities)
    if tuple(rows) != ordered:
        raise ValueError("metric JSONL identity order is not canonical")
    return artifact


def _resolve_job(stage: str, job_id: str) -> PlannedJob:
    if not isinstance(stage, str) or not stage:
        raise TypeError("stage must be a nonempty string")
    if stage not in tuple(load_protocol_config()["freeze"]["active_manifest_stages"]):
        raise ValueError("stage is not part of the active manifest")
    _require_sha256(job_id, "job ID")
    matches = [job for job in planned_jobs(stage) if job.job_id == job_id]
    if len(matches) != 1:
        raise ValueError("job ID is not registered for the requested stage")
    return matches[0]


def _carried_methods(job: PlannedJob) -> tuple[str, ...]:
    if not isinstance(job, PlannedJob):
        raise TypeError("job must be a PlannedJob")
    stage = stage_spec(job.stage)
    try:
        carried = stage.result_carriers[job.method]
    except KeyError:
        raise ValueError("planned job has no frozen result-carrier registration") from None
    return tuple(carried)


@lru_cache(maxsize=128)
def _expected_record_identities(job: PlannedJob) -> tuple[dict[str, object], ...]:
    stage = stage_spec(job.stage)
    methods = _carried_methods(job)
    if stage.eval_cohort is None:
        if methods:
            raise RuntimeError("training-only job unexpectedly carries result methods")
        return ()
    starts = ordered_window_starts(job.dataset, stage.eval_cohort)
    flow_count = dataset_spec(job.dataset).flows
    families = tuple(load_protocol_config()["masks"]["families"])
    oracle = job.stage == "stage_h"
    return tuple(
        {
            "method": method,
            "seed_bundle": job.seed_bundle,
            "dataset": job.dataset,
            "mask_family": family,
            "window_start": start,
            "flow": flow,
            "oracle": oracle,
        }
        for method in methods
        for family in families
        for start in starts
        for flow in range(flow_count)
    )


@lru_cache(maxsize=131072)
def _registered_target_contract(
    dataset: str,
    cohort: str,
    window_start: int,
    flow: int,
    family: str,
    seed_bundle: int,
    oracle: bool,
) -> tuple[int, str]:
    bundle = build_evaluation_mask(
        dataset=dataset,
        cohort=cohort,
        window_start=window_start,
        flow_index=flow,
        family=family,
        seed_bundle=seed_bundle,
    )
    target = (
        factory_oracle_partition(bundle).evaluation if oracle else bundle.target
    )
    return int(target.sum()), target_sha256(target)


def _validate_job_record_targets(
    records: Iterable[MetricRecord | Mapping[str, object]], *, job: PlannedJob
) -> None:
    stage = stage_spec(job.stage)
    if stage.eval_cohort is None:
        if tuple(records):
            raise ValueError("training-only job cannot contain metric records")
        return
    carried = set(_carried_methods(job))
    for record in records:
        payload = _validate_metric_payload(_metric_payload(record))
        identity = payload["identity"]
        if set(identity) != {
            "method",
            "seed_bundle",
            "dataset",
            "mask_family",
            "window_start",
            "flow",
            "oracle",
        }:
            raise ValueError("job metric identity must contain exactly seven frozen fields")
        if (
            identity["method"] not in carried
            or identity["dataset"] != job.dataset
            or identity["seed_bundle"] != job.seed_bundle
            or identity["oracle"] is not (job.stage == "stage_h")
        ):
            raise ValueError("job metric identity differs from its carrier registration")
        try:
            expected_count, expected_sha256 = _registered_target_contract(
                job.dataset,
                stage.eval_cohort,
                int(identity["window_start"]),
                int(identity["flow"]),
                str(identity["mask_family"]),
                job.seed_bundle,
                job.stage == "stage_h",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("job metric identity is absent from the deterministic registry") from exc
        if payload["target_count"] != expected_count:
            raise ValueError("job metric target count differs from deterministic registry")
        if payload["target_set_sha256"] != expected_sha256:
            raise ValueError("job metric target-set hash differs from deterministic registry")


def _validate_job_records(
    path: Path, *, job: PlannedJob, expected_sha256: str
) -> FileArtifact:
    identities = _expected_record_identities(job)
    artifact = validate_metric_records_file(
        path,
        expected_sha256=expected_sha256,
        expected_identities=identities,
    )
    rows, _ = _read_canonical_jsonl(path)
    _validate_job_record_targets(rows, job=job)
    return artifact


def _checkpoint_identity_sha256(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(
        b"acil-innovation-v1:checkpoint-identity:v1\x00"
        + canonical_json_bytes(dict(identity))
    ).hexdigest()


_CHECKPOINT_IDENTITY_KEYS = {
    "protocol",
    "config_sha256",
    "stage",
    "job_id",
    "method",
    "dataset",
    "seed_bundle",
    "epoch",
    "source_dev_nmae",
    "manifest_sha256",
    "provenance_sha256",
}
_DEPENDENCY_BINDING_KEYS = {
    "source_stage",
    "source_job_id",
    "checkpoint_identity_sha256",
    "checkpoint_file_sha256",
    "checkpoint_tensor_sha256",
}


def _source_job(job: PlannedJob, binding) -> PlannedJob:
    matches = tuple(
        candidate
        for candidate in planned_jobs(binding.source_stage)
        if candidate.method == binding.source_method
        and candidate.dataset == job.dataset
        and candidate.seed_bundle == job.seed_bundle
    )
    if len(matches) != 1:
        raise RuntimeError("checkpoint dependency source job is not uniquely registered")
    return matches[0]


def _embedded_checkpoint_binding(job: PlannedJob):
    prefix = f"{job.method}."
    matches = tuple(
        binding
        for binding in stage_spec(job.stage).checkpoint_hash_bindings
        if any(role.startswith(prefix) for role in binding.consumer_roles)
    )
    if len(matches) > 1:
        raise RuntimeError("learned job has multiple embedded checkpoint bindings")
    if matches:
        binding = matches[0]
        roles = tuple(
            role for role in binding.consumer_roles if role.startswith(prefix)
        )
        if roles != (f"{job.method}.acil_base",) or binding.source_method != "acil":
            raise RuntimeError("embedded checkpoint binding registry drifted")
        return binding
    return None


def _dependency_binding_name(binding, consumer_role: str | None) -> str:
    if binding.source_method == "acil":
        return "acil_base"
    if consumer_role is None or "." in consumer_role:
        raise RuntimeError("non-ACIL dependency has no top-level carrier identity")
    return consumer_role


def _expected_dependency_bindings(job: PlannedJob) -> dict[str, object]:
    carried = set(_carried_methods(job))
    result: dict[str, object] = {}
    prefix = f"{job.method}."
    for binding in stage_spec(job.stage).checkpoint_hash_bindings:
        consumers: list[str | None] = [
            role for role in binding.consumer_roles if role in carried
        ]
        if any(role.startswith(prefix) for role in binding.consumer_roles):
            consumers.append(None)
        for consumer in consumers:
            name = _dependency_binding_name(binding, consumer)
            previous = result.get(name)
            if previous is not None and previous != binding:
                raise RuntimeError("dependency artifact name maps to multiple bindings")
            result[name] = binding
    return result


def _validate_dependency_binding(
    value: object,
    *,
    job: PlannedJob,
    specification,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint dependency binding must be an object")
    required = set(_DEPENDENCY_BINDING_KEYS)
    if specification.source_method != "acil":
        required.add("source_method")
    if set(value) != required:
        raise ValueError("checkpoint dependency binding has missing or extra fields")
    source = _source_job(job, specification)
    if value.get("source_stage") != source.stage:
        raise ValueError("checkpoint dependency source stage drifted")
    if value.get("source_job_id") != source.job_id:
        raise ValueError("checkpoint dependency source job drifted")
    if specification.source_method != "acil" and value.get(
        "source_method"
    ) != specification.source_method:
        raise ValueError("checkpoint dependency source method drifted")
    result = dict(value)
    for field in (
        "checkpoint_identity_sha256",
        "checkpoint_file_sha256",
        "checkpoint_tensor_sha256",
    ):
        result[field] = _require_sha256(
            value.get(field), f"checkpoint dependency {field}"
        )
    return result


def _validate_checkpoint_identity(
    identity: object,
    *,
    job: PlannedJob,
    config_sha256: str,
    manifest_sha256: str,
    provenance_sha256: str,
) -> dict[str, object]:
    if not isinstance(identity, Mapping):
        raise ValueError("checkpoint identity must be an object")
    embedded = _embedded_checkpoint_binding(job)
    required = set(_CHECKPOINT_IDENTITY_KEYS)
    if embedded is not None:
        required.add("base_checkpoint")
    if set(identity) != required:
        raise ValueError("checkpoint identity has missing or extra fields")
    expected = {
        "protocol": "acil-innovation-v1",
        "config_sha256": _require_sha256(config_sha256, "expected config hash"),
        "stage": job.stage,
        "job_id": job.job_id,
        "method": job.method,
        "dataset": job.dataset,
        "seed_bundle": job.seed_bundle,
        "manifest_sha256": _require_sha256(
            manifest_sha256, "expected manifest hash"
        ),
        "provenance_sha256": _require_sha256(
            provenance_sha256, "expected provenance hash"
        ),
    }
    if any(identity.get(name) != value for name, value in expected.items()):
        raise ValueError("checkpoint identity differs from its result/freeze binding")
    epoch = identity.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < 20:
        raise ValueError("checkpoint identity must bind the selected zero-based epoch")
    source_dev = identity.get("source_dev_nmae")
    if (
        isinstance(source_dev, bool)
        or not isinstance(source_dev, (int, float))
        or not math.isfinite(float(source_dev))
        or float(source_dev) < 0.0
    ):
        raise ValueError("checkpoint identity source-dev NMAE must be finite and nonnegative")
    result = dict(identity)
    if embedded is not None:
        result["base_checkpoint"] = _validate_dependency_binding(
            identity.get("base_checkpoint"), job=job, specification=embedded
        )
    return result


def _validate_checkpoint_descriptor(
    root: Path,
    *,
    job: PlannedJob,
    descriptor: Mapping[str, object],
    config_sha256: str,
    manifest_sha256: str,
    provenance_sha256: str,
) -> dict[str, object]:
    required = {
        "metadata",
        "metadata_sha256",
        "weights",
        "identity",
        "identity_sha256",
        "file_sha256",
        "tensor_sha256",
    }
    if set(descriptor) != required:
        raise ValueError("checkpoint result descriptor has missing or extra fields")
    metadata_relative = descriptor["metadata"]
    weights_relative = descriptor["weights"]
    if not isinstance(metadata_relative, str) or not isinstance(weights_relative, str):
        raise ValueError("checkpoint result paths must be relative strings")
    identity_sha = _require_sha256(descriptor["identity_sha256"], "checkpoint identity hash")
    if metadata_relative != f"checkpoints/{identity_sha}.json":
        raise ValueError("checkpoint metadata relative path drifted")
    if weights_relative != f"checkpoints/{identity_sha}.safetensors":
        raise ValueError("checkpoint weights relative path drifted")
    root = Path(root)
    metadata_path = root / metadata_relative
    weights_path = root / weights_relative
    metadata_payload, metadata_artifact = _read_canonical_json(metadata_path)
    if metadata_artifact.sha256 != _require_sha256(
        descriptor["metadata_sha256"], "checkpoint metadata file hash"
    ):
        raise ValueError("checkpoint metadata file hash mismatch")
    if not isinstance(metadata_payload, dict) or set(metadata_payload) != {
        "schema",
        "identity",
        "identity_sha256",
        "weights_file",
        "file_sha256",
        "tensor_sha256",
    }:
        raise ValueError("checkpoint metadata schema drifted")
    identity = _validate_checkpoint_identity(
        descriptor["identity"],
        job=job,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )
    if metadata_payload["identity"] != identity:
        raise ValueError("checkpoint identity differs between result and metadata")
    computed_identity_sha = _checkpoint_identity_sha256(identity)
    if computed_identity_sha != identity_sha:
        raise ValueError("checkpoint identity hash mismatch")
    if metadata_payload["identity_sha256"] != identity_sha:
        raise ValueError("checkpoint metadata identity hash mismatch")
    if metadata_payload["weights_file"] != weights_path.name:
        raise ValueError("checkpoint metadata weights filename drifted")
    weights_sha = regular_file_sha256(weights_path)
    if weights_sha != _require_sha256(descriptor["file_sha256"], "checkpoint file hash"):
        raise ValueError("checkpoint weights file hash mismatch")
    if metadata_payload["file_sha256"] != weights_sha:
        raise ValueError("checkpoint metadata file hash mismatch")
    state = load_file(str(weights_path), device="cpu")
    tensor_sha = _tensor_sha(state)
    if tensor_sha != _require_sha256(descriptor["tensor_sha256"], "checkpoint tensor hash"):
        raise ValueError("checkpoint tensor hash mismatch")
    if metadata_payload["tensor_sha256"] != tensor_sha:
        raise ValueError("checkpoint metadata tensor hash mismatch")
    validate_registered_state_schema(job, state)
    return {
        "metadata": metadata_relative,
        "metadata_sha256": metadata_artifact.sha256,
        "weights": weights_relative,
        "identity": dict(identity),
        "identity_sha256": identity_sha,
        "file_sha256": weights_sha,
        "tensor_sha256": tensor_sha,
    }


def _validate_dependency_source_result(
    binding: Mapping[str, object],
    *,
    job: PlannedJob,
    specification,
    output_root: Path,
    manifest_sha256: str,
    provenance_sha256: str,
) -> None:
    source = _source_job(job, specification)
    try:
        payload = load_job_result(
            stage=source.stage,
            job_id=source.job_id,
            output_root=output_root,
        )
    except (OSError, ValueError) as exc:
        raise ValueError("checkpoint dependency source result is invalid") from exc
    if payload.get("status") != "succeeded":
        raise ValueError("checkpoint dependency source job did not succeed")
    if (
        payload.get("manifest_sha256") != manifest_sha256
        or payload.get("provenance_sha256") != provenance_sha256
    ):
        raise ValueError("checkpoint dependency source belongs to another freeze")
    source_checkpoint = payload.get("checkpoint")
    if not isinstance(source_checkpoint, Mapping):
        raise ValueError("checkpoint dependency source has no valid checkpoint")
    for binding_field, source_field in (
        ("checkpoint_identity_sha256", "identity_sha256"),
        ("checkpoint_file_sha256", "file_sha256"),
        ("checkpoint_tensor_sha256", "tensor_sha256"),
    ):
        if binding[binding_field] != source_checkpoint.get(source_field):
            raise ValueError("checkpoint dependency hash differs from source result")


def _validate_dependencies_contract(
    root: Path,
    *,
    job: PlannedJob,
    artifacts: Mapping[str, Mapping[str, object]],
    checkpoint: Mapping[str, object],
    output_root: Path,
    manifest_sha256: str,
    provenance_sha256: str,
) -> None:
    expected = _expected_dependency_bindings(job)
    descriptor = artifacts.get("dependencies")
    needs_artifact = any(name != "acil_base" for name in expected)
    if not expected:
        if descriptor is not None:
            raise ValueError("job has an unregistered dependencies artifact")
        return
    identity = checkpoint.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("checkpoint identity is unavailable for dependency validation")
    validated: dict[str, dict[str, object]] = {}
    if descriptor is None:
        if needs_artifact:
            raise ValueError("job is missing its registered dependencies artifact")
        base = identity.get("base_checkpoint")
        specification = expected.get("acil_base")
        if specification is None:
            raise RuntimeError("dependency registry cannot be represented by base checkpoint")
        validated["acil_base"] = _validate_dependency_binding(
            base, job=job, specification=specification
        )
    else:
        if descriptor.get("path") != "dependencies.json":
            raise ValueError("dependencies artifact path must be dependencies.json")
        payload, artifact = _read_canonical_json(Path(root) / "dependencies.json")
        if (
            artifact.sha256 != descriptor.get("sha256")
            or artifact.bytes != descriptor.get("bytes")
        ):
            raise ValueError("dependencies artifact descriptor drifted")
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema",
            "job",
            "dependencies",
        }:
            raise ValueError("dependencies artifact schema has missing or extra fields")
        if payload.get("schema") != "acil-innovation-v1:job-dependencies:v1":
            raise ValueError("dependencies artifact schema identity drifted")
        if payload.get("job") != _job_payload(job):
            raise ValueError("dependencies artifact job identity drifted")
        dependencies = payload.get("dependencies")
        if not isinstance(dependencies, Mapping) or set(dependencies) != set(expected):
            raise ValueError("dependencies artifact has missing or extra bindings")
        for name, specification in expected.items():
            validated[name] = _validate_dependency_binding(
                dependencies[name], job=job, specification=specification
            )
        if "acil_base" in validated and validated["acil_base"] != identity.get(
            "base_checkpoint"
        ):
            raise ValueError("dependencies artifact ACIL base differs from checkpoint identity")
    for name, specification in expected.items():
        _validate_dependency_source_result(
            validated[name],
            job=job,
            specification=specification,
            output_root=Path(output_root),
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
        )


def _validate_embedded_checkpoint_state(
    root: Path,
    *,
    job: PlannedJob,
    checkpoint: Mapping[str, object],
    output_root: Path,
) -> None:
    """Re-read candidate/source tensors and prove exact embedded ACIL reuse."""

    source = embedded_acil_source_job(job)
    if source is None:
        return
    identity = checkpoint.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(
        identity.get("base_checkpoint"), Mapping
    ):
        raise ValueError("embedded ACIL checkpoint has no source binding")
    binding = identity["base_checkpoint"]
    source_result = load_job_result(
        stage=source.stage,
        job_id=source.job_id,
        output_root=Path(output_root),
    )
    source_checkpoint = source_result.get("checkpoint")
    if not isinstance(source_checkpoint, Mapping):
        raise ValueError("embedded ACIL source result has no checkpoint")
    for binding_field, source_field in (
        ("checkpoint_identity_sha256", "identity_sha256"),
        ("checkpoint_file_sha256", "file_sha256"),
        ("checkpoint_tensor_sha256", "tensor_sha256"),
    ):
        if binding.get(binding_field) != source_checkpoint.get(source_field):
            raise ValueError("embedded ACIL binding differs from source checkpoint")
    candidate_weights = checkpoint.get("weights")
    source_weights = source_checkpoint.get("weights")
    if not isinstance(candidate_weights, str) or not isinstance(source_weights, str):
        raise ValueError("embedded ACIL checkpoint weights path is invalid")
    candidate_state = load_file(str(Path(root) / candidate_weights), device="cpu")
    source_state = load_file(
        str(Path(output_root) / source.stage / source.job_id / source_weights),
        device="cpu",
    )
    validate_embedded_acil_state(job, candidate_state, source_state)


def _job_payload(job: PlannedJob) -> dict[str, object]:
    return job.to_json()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _plain_directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    """Return the stable filesystem identity of one non-symlink directory."""

    _plain_directory(path, label=label)
    try:
        metadata = path.lstat()
    except OSError as exc:  # The directory changed after the first lstat.
        raise ValueError(f"{label} directory changed during publish") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a regular directory")
    return metadata.st_dev, metadata.st_ino


def _verify_directory_publish(
    source: Path,
    destination: Path,
    *,
    source_identity: tuple[int, int],
    source_parent_identity: tuple[int, int],
    destination_parent_identity: tuple[int, int],
) -> None:
    """Fail closed unless rename moved exactly the checked source directory."""

    if _lexists(source):
        raise ValueError("publish source still exists after directory rename")
    if _plain_directory_identity(
        source.parent, label="publish source parent"
    ) != source_parent_identity:
        raise ValueError("publish source parent identity changed during rename")
    if _plain_directory_identity(
        destination.parent, label="publish destination parent"
    ) != destination_parent_identity:
        raise ValueError("publish destination parent identity changed during rename")
    if _plain_directory_identity(
        destination, label="published destination"
    ) != source_identity:
        raise ValueError("published destination is not the checked source directory")


def _portable_rename_directory_noreplace(
    source: Path,
    destination: Path,
    *,
    source_identity: tuple[int, int],
    source_parent_identity: tuple[int, int],
    destination_parent_identity: tuple[int, int],
) -> None:
    """Best available fallback for a frozen single-publisher job namespace."""

    # renameat2 may have executed filesystem-specific validation before
    # reporting that RENAME_NOREPLACE is unsupported.  Recheck every bound
    # inode before entering the weaker POSIX fallback, not only after publish.
    if _plain_directory_identity(source, label="publish source") != source_identity:
        raise ValueError("publish source identity changed before portable rename")
    if _plain_directory_identity(
        source.parent, label="publish source parent"
    ) != source_parent_identity:
        raise ValueError("publish source parent identity changed before portable rename")
    if _plain_directory_identity(
        destination.parent, label="publish destination parent"
    ) != destination_parent_identity:
        raise ValueError(
            "publish destination parent identity changed before portable rename"
        )

    # The frozen queue permits one protocol-controlled publisher for a job ID.
    # This final lstat prevents replacing any destination already present here,
    # including a dangling symlink (which Path.exists() would miss).
    if _lexists(destination):
        raise FileExistsError("final planned-job directory already exists")
    try:
        os.rename(source, destination)
    except OSError as exc:
        try:
            current_source_identity = _plain_directory_identity(
                source, label="publish source"
            )
        except ValueError as identity_error:
            raise ValueError(
                "publish syscall failed after source identity changed"
            ) from identity_error
        if current_source_identity != source_identity:
            raise ValueError("publish source identity changed during rename") from exc
        if _plain_directory_identity(
            source.parent, label="publish source parent"
        ) != source_parent_identity:
            raise ValueError("publish source parent identity changed during rename") from exc
        if _plain_directory_identity(
            destination.parent, label="publish destination parent"
        ) != destination_parent_identity:
            raise ValueError(
                "publish destination parent identity changed during rename"
            ) from exc
        if _lexists(destination):
            raise FileExistsError(
                "final planned-job directory already exists"
            ) from exc
        raise


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish under the protocol-controlled single publisher contract.

    ``RENAME_NOREPLACE`` provides kernel no-replace when supported.  The GPFS
    fallback keeps final-directory visibility atomic and rejects every target
    visible to its final ``lstat``.  POSIX ``rename`` cannot, however, exclude a
    non-cooperating process that creates an empty directory after that check;
    this adversarial race is explicitly outside the frozen queue guarantee.
    """

    source = Path(source)
    destination = Path(destination)
    source_identity = _plain_directory_identity(source, label="publish source")
    source_parent_identity = _plain_directory_identity(
        source.parent, label="publish source parent"
    )
    destination_parent_identity = _plain_directory_identity(
        destination.parent, label="publish destination parent"
    )
    if _lexists(destination):
        raise FileExistsError("final planned-job directory already exists")

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    use_portable_fallback = renameat2 is None
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result != 0:
            error = ctypes.get_errno()
            if result != -1 or error == 0:
                raise RuntimeError(
                    "ambiguous renameat2 failure status; refusing portable fallback"
                )
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError("final planned-job directory already exists")
            unsupported = {
                errno.EINVAL,
                errno.ENOSYS,
                getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
                errno.ENOTSUP,
            }
            if error not in unsupported:
                raise OSError(error, os.strerror(error), str(destination))
            use_portable_fallback = True
    if use_portable_fallback:
        _portable_rename_directory_noreplace(
            source,
            destination,
            source_identity=source_identity,
            source_parent_identity=source_parent_identity,
            destination_parent_identity=destination_parent_identity,
        )
    _verify_directory_publish(
        source,
        destination,
        source_identity=source_identity,
        source_parent_identity=source_parent_identity,
        destination_parent_identity=destination_parent_identity,
    )


def _artifact_relative_path(root: Path, path: Path) -> str:
    root_absolute = Path(root).absolute()
    path_absolute = Path(path).absolute()
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError:
        raise ValueError("named artifact must be inside the fixed job staging directory") from None
    if not relative.parts:
        raise ValueError("named artifact path must name a regular file")
    current = root_absolute
    _plain_directory(current, label="job staging")
    for part in relative.parts[:-1]:
        current = current / part
        _plain_directory(current, label="named artifact parent")
    relative_string = relative.as_posix()
    if (
        relative_string in _RESERVED_ARTIFACT_PATHS
        or relative.parts[0] == "checkpoints"
    ):
        raise ValueError("named artifact path collides with a reserved job artifact")
    return relative_string


def _validate_named_artifact(
    root: Path, *, name: str, descriptor: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(name, str) or _ARTIFACT_NAME.fullmatch(name) is None:
        raise ValueError("named artifact name is not a stable lowercase identifier")
    if set(descriptor) != {"path", "sha256", "bytes"}:
        raise ValueError("named artifact descriptor has missing or extra fields")
    relative = descriptor["path"]
    byte_count = descriptor["bytes"]
    if not isinstance(relative, str) or not relative:
        raise ValueError("named artifact relative path must be a nonempty string")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError("named artifact byte count must be a nonnegative integer")
    path = Path(root) / relative
    canonical_relative = _artifact_relative_path(root, path)
    if canonical_relative != relative:
        raise ValueError("named artifact relative path is not canonical")
    content, sha256 = _read_regular_bytes(path)
    if sha256 != _require_sha256(descriptor["sha256"], "named artifact hash"):
        raise ValueError("named artifact file hash mismatch")
    if len(content) != byte_count:
        raise ValueError("named artifact byte count mismatch")
    return {"path": relative, "sha256": sha256, "bytes": len(content)}


_TRAINING_KEYS = {
    "schema",
    "protocol",
    "config_sha256",
    "job",
    "fit_fallback",
    "epochs_completed",
    "optimizer_updates",
    "best_epoch",
    "best_source_dev_nmae",
    "epochs",
}
_TRAINING_EPOCH_KEYS = {
    "epoch",
    "physical_batches",
    "optimizer_updates",
    "mean_loss",
    "source_dev_absolute_error_sum",
    "source_dev_absolute_truth_sum",
    "source_dev_nmae",
    "selected_as_best",
}


def _training_number(
    value: object, *, label: str, minimum: float | None = None, strict: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"training {label} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"training {label} must be finite")
    if minimum is not None and (
        result <= minimum if strict else result < minimum
    ):
        relation = "strictly above" if strict else "at least"
        raise ValueError(f"training {label} must be {relation} {minimum}")
    return result


def _training_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"training {label} must be an integer")
    return value


def _validate_training_payload(
    payload: object,
    *,
    job: PlannedJob,
    checkpoint_identity: Mapping[str, object],
) -> dict[str, object]:
    """Validate the complete fixed training lane and its selected checkpoint."""

    if not isinstance(payload, Mapping) or set(payload) != _TRAINING_KEYS:
        raise ValueError("training schema has missing or extra fields")
    if payload.get("schema") != "acil-innovation-v1:training-history:v1":
        raise ValueError("training schema identity drifted")
    if payload.get("protocol") != "acil-innovation-v1":
        raise ValueError("training protocol identity drifted")
    if payload.get("config_sha256") != protocol_config_sha256():
        raise ValueError("training config SHA-256 drifted")
    if payload.get("job") != _job_payload(job):
        raise ValueError("training job identity drifted")
    fallback = payload.get("fit_fallback")
    if not isinstance(fallback, Mapping) or set(fallback) != {"mean", "std"}:
        raise ValueError("training fallback schema drifted")
    _training_number(fallback["mean"], label="fallback mean", minimum=0.0)
    _training_number(
        fallback["std"], label="fallback std", minimum=1e-6
    )
    if _training_integer(
        payload.get("epochs_completed"), label="epochs completed"
    ) != 20:
        raise ValueError("training must contain exactly 20 epochs")
    if _training_integer(
        payload.get("optimizer_updates"), label="optimizer updates"
    ) != 320:
        raise ValueError("training must contain exactly 320 optimizer updates")
    best_epoch = _training_integer(payload.get("best_epoch"), label="best epoch")
    if not 0 <= best_epoch < 20:
        raise ValueError("training best epoch must be a zero-based epoch in [0,20)")
    best_nmae = _training_number(
        payload.get("best_source_dev_nmae"),
        label="best source-dev NMAE",
        minimum=0.0,
    )
    epochs = payload.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != 20:
        raise ValueError("training must contain exactly 20 epochs")

    raw_nmae: list[float] = []
    selected: list[int] = []
    for expected_epoch, row in enumerate(epochs):
        if not isinstance(row, Mapping) or set(row) != _TRAINING_EPOCH_KEYS:
            raise ValueError("training epoch schema has missing or extra fields")
        if _training_integer(row.get("epoch"), label="epoch") != expected_epoch:
            raise ValueError("training epoch registry must be exactly 0..19")
        if _training_integer(
            row.get("physical_batches"), label="physical batches"
        ) != 64:
            raise ValueError("training epoch must contain exactly 64 physical batches")
        if _training_integer(
            row.get("optimizer_updates"), label="epoch optimizer updates"
        ) != 16:
            raise ValueError("training epoch must contain exactly 16 optimizer updates")
        _training_number(row.get("mean_loss"), label="mean loss", minimum=0.0)
        error = _training_number(
            row.get("source_dev_absolute_error_sum"),
            label="source-dev absolute-error sum",
            minimum=0.0,
        )
        truth = _training_number(
            row.get("source_dev_absolute_truth_sum"),
            label="source-dev absolute-truth sum",
            minimum=0.0,
            strict=True,
        )
        recorded = _training_number(
            row.get("source_dev_nmae"),
            label="source-dev NMAE",
            minimum=0.0,
        )
        computed = error / truth
        if recorded != computed:
            raise ValueError("training raw source-dev NMAE differs from its sums")
        flag = row.get("selected_as_best")
        if not isinstance(flag, bool):
            raise ValueError("training selected_as_best must be boolean")
        if flag:
            selected.append(expected_epoch)
        raw_nmae.append(computed)

    raw_argmin = min(range(20), key=lambda epoch: (raw_nmae[epoch], epoch))
    if best_epoch != raw_argmin:
        raise ValueError("training best epoch is not the earliest raw argmin")
    if best_nmae != raw_nmae[raw_argmin]:
        raise ValueError("training best source-dev NMAE differs from raw argmin")
    if selected != [raw_argmin]:
        raise ValueError("training selected_as_best must mark exactly the raw argmin")
    if checkpoint_identity.get("epoch") != best_epoch:
        raise ValueError("checkpoint epoch differs from training best epoch")
    checkpoint_nmae = _training_number(
        checkpoint_identity.get("source_dev_nmae"),
        label="checkpoint source-dev NMAE",
        minimum=0.0,
    )
    if checkpoint_nmae != best_nmae:
        raise ValueError("checkpoint source-dev NMAE differs from training best value")
    return json.loads(canonical_json_bytes(dict(payload)).decode("ascii"))


def _validate_training_artifact(
    root: Path,
    *,
    job: PlannedJob,
    artifacts: Mapping[str, Mapping[str, object]],
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    descriptor = artifacts.get("training")
    if not isinstance(descriptor, Mapping):
        raise ValueError("successful learned job is missing its training artifact")
    if descriptor.get("path") != "training.json":
        raise ValueError("training artifact path must be training.json")
    payload, artifact = _read_canonical_json(Path(root) / "training.json")
    if (
        artifact.sha256 != descriptor.get("sha256")
        or artifact.bytes != descriptor.get("bytes")
    ):
        raise ValueError("training artifact descriptor drifted")
    identity = checkpoint.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("checkpoint identity is unavailable for training validation")
    return _validate_training_payload(
        payload, job=job, checkpoint_identity=identity
    )


class JobResultWriter:
    """One exclusive in-progress directory that finalizes exactly once."""

    def __init__(
        self,
        *,
        job: PlannedJob,
        output_root: Path,
        manifest_sha256: str,
        provenance_sha256: str,
    ) -> None:
        self._job = job
        self._manifest_sha256 = _require_sha256(manifest_sha256, "manifest hash")
        self._provenance_sha256 = _require_sha256(provenance_sha256, "provenance hash")
        output_root = Path(output_root)
        if _lexists(output_root):
            _plain_directory(output_root, label="output root")
        else:
            output_root.mkdir(parents=True, exist_ok=False)
            _plain_directory(output_root, label="output root")
        self._stage_directory = output_root / job.stage
        if _lexists(self._stage_directory):
            _plain_directory(self._stage_directory, label="stage output")
        else:
            self._stage_directory.mkdir(exist_ok=False)
        self._final_directory = self._stage_directory / job.job_id
        self._staging_directory = self._stage_directory / f".{job.job_id}.inprogress"
        if _lexists(self._final_directory) or _lexists(self._staging_directory):
            raise FileExistsError("planned job already has a final or in-progress directory")
        self._staging_directory.mkdir(mode=0o700)
        self._checkpoint_directory = self._staging_directory / "checkpoints"
        self._checkpoint_directory.mkdir()
        self._finalized = False
        self._records: FileArtifact | None = None
        self._checkpoint: dict[str, object] | None = None
        self._artifacts: dict[str, dict[str, object]] = {}
        started = {
            "schema": "acil-innovation-v1:job-started:v1",
            "status": "started",
            "job": _job_payload(job),
            "config_sha256": protocol_config_sha256(),
            "manifest_sha256": self._manifest_sha256,
            "provenance_sha256": self._provenance_sha256,
        }
        self._started = write_canonical_json_exclusive(
            self._staging_directory / "started.json", started
        )

    @property
    def staging_directory(self) -> Path:
        return self._staging_directory

    @property
    def checkpoint_directory(self) -> Path:
        return self._checkpoint_directory

    def _require_active(self) -> None:
        if self._finalized:
            raise RuntimeError("job result writer is already finalized")

    def register_records(
        self,
        records: Iterable[MetricRecord | Mapping[str, object]],
        expected_identities: Sequence[Mapping[str, object]],
    ) -> FileArtifact:
        self._require_active()
        if self._records is not None:
            raise FileExistsError("job records were already written")
        frozen_identities = _expected_record_identities(self._job)
        supplied_keys = tuple(_identity_key(value) for value in expected_identities)
        frozen_keys = tuple(_identity_key(value) for value in frozen_identities)
        if len(supplied_keys) != len(set(supplied_keys)):
            raise ValueError("supplied expected record identities contain duplicates")
        if set(supplied_keys) != set(frozen_keys):
            raise ValueError("supplied expected record identities drift from the frozen job grid")
        values = tuple(records)
        _validate_job_record_targets(values, job=self._job)
        self._records = write_metric_records_exclusive(
            self._staging_directory / "records.jsonl",
            values,
            expected_identities=frozen_identities,
        )
        return self._records

    def write_records(
        self, records: Iterable[MetricRecord | Mapping[str, object]]
    ) -> FileArtifact:
        """Compatibility spelling using the sole frozen expected identity grid."""

        return self.register_records(records, _expected_record_identities(self._job))

    def register_artifact(self, name: str, path: Path) -> Mapping[str, object]:
        self._require_active()
        if not isinstance(name, str) or _ARTIFACT_NAME.fullmatch(name) is None:
            raise ValueError("named artifact name is not a stable lowercase identifier")
        if name in self._artifacts:
            raise FileExistsError("named artifact is already registered")
        relative = _artifact_relative_path(self._staging_directory, Path(path))
        if any(value["path"] == relative for value in self._artifacts.values()):
            raise FileExistsError("named artifact file is already registered")
        content, sha256 = _read_regular_bytes(Path(path))
        descriptor = {"path": relative, "sha256": sha256, "bytes": len(content)}
        self._artifacts[name] = _validate_named_artifact(
            self._staging_directory, name=name, descriptor=descriptor
        )
        return dict(self._artifacts[name])

    def register_checkpoint(self, checkpoint: CheckpointRecord) -> Mapping[str, object]:
        self._require_active()
        if self._checkpoint is not None:
            raise FileExistsError("job checkpoint was already registered")
        if not isinstance(checkpoint, CheckpointRecord):
            raise TypeError("checkpoint must be a CheckpointRecord")
        metadata = Path(checkpoint.metadata_path)
        weights = Path(checkpoint.weights_path)
        if metadata.parent != self._checkpoint_directory or weights.parent != self._checkpoint_directory:
            raise ValueError("checkpoint files must be inside the fixed job checkpoint directory")
        descriptor = {
            "metadata": metadata.relative_to(self._staging_directory).as_posix(),
            "metadata_sha256": regular_file_sha256(metadata),
            "weights": weights.relative_to(self._staging_directory).as_posix(),
            "identity": _read_canonical_json(metadata)[0]["identity"],
            "identity_sha256": checkpoint.identity_sha256,
            "file_sha256": checkpoint.file_sha256,
            "tensor_sha256": checkpoint.tensor_sha256,
        }
        self._checkpoint = _validate_checkpoint_descriptor(
            self._staging_directory,
            job=self._job,
            descriptor=descriptor,
            config_sha256=protocol_config_sha256(),
            manifest_sha256=self._manifest_sha256,
            provenance_sha256=self._provenance_sha256,
        )
        return dict(self._checkpoint)

    def _finalize(self) -> Path:
        self._require_active()
        if _lexists(self._final_directory):
            raise FileExistsError("final planned-job directory already exists")
        _fsync_directory(self._staging_directory)
        _rename_directory_noreplace(self._staging_directory, self._final_directory)
        _fsync_directory(self._stage_directory)
        self._finalized = True
        return self._final_directory

    def succeed(self) -> Path:
        self._require_active()
        if self._checkpoint is None:
            raise ValueError("successful learned job is missing its checkpoint")
        checkpoint = _validate_checkpoint_descriptor(
            self._staging_directory,
            job=self._job,
            descriptor=self._checkpoint,
            config_sha256=protocol_config_sha256(),
            manifest_sha256=self._manifest_sha256,
            provenance_sha256=self._provenance_sha256,
        )
        artifacts = {
            name: _validate_named_artifact(
                self._staging_directory, name=name, descriptor=descriptor
            )
            for name, descriptor in sorted(self._artifacts.items())
        }
        _validate_training_artifact(
            self._staging_directory,
            job=self._job,
            artifacts=artifacts,
            checkpoint=checkpoint,
        )
        _validate_dependencies_contract(
            self._staging_directory,
            job=self._job,
            artifacts=artifacts,
            checkpoint=checkpoint,
            output_root=self._stage_directory.parent,
            manifest_sha256=self._manifest_sha256,
            provenance_sha256=self._provenance_sha256,
        )
        validate_named_artifact_registry(self._job, artifacts)
        _validate_embedded_checkpoint_state(
            self._staging_directory,
            job=self._job,
            checkpoint=checkpoint,
            output_root=self._stage_directory.parent,
        )
        expected_count = len(_expected_record_identities(self._job))
        if self._records is None:
            if expected_count != 0:
                raise ValueError("successful job is missing its complete metric records")
            self.register_records((), ())
        records = _validate_job_records(
            self._records.path,
            job=self._job,
            expected_sha256=self._records.sha256,
        )
        validate_succeeded_job_inventory(
            self._staging_directory,
            job=self._job,
            checkpoint=checkpoint,
            artifacts=artifacts,
            finalized=False,
        )
        result = {
            "schema": "acil-innovation-v1:job-succeeded:v1",
            "status": "succeeded",
            "job": _job_payload(self._job),
            "config_sha256": protocol_config_sha256(),
            "manifest_sha256": self._manifest_sha256,
            "provenance_sha256": self._provenance_sha256,
            "started_sha256": self._started.sha256,
            "checkpoint": checkpoint,
            "artifacts": artifacts,
            "records": {
                "jsonl": "records.jsonl",
                "sha256": records.sha256,
                "count": records.count,
            },
        }
        write_canonical_json_exclusive(self._staging_directory / "result.json", result)
        validate_succeeded_job_inventory(
            self._staging_directory,
            job=self._job,
            checkpoint=checkpoint,
            artifacts=artifacts,
            finalized=True,
        )
        return self._finalize()

    def fail_algorithmically(self, *, failure_type: str, message: str) -> Path:
        self._require_active()
        if not isinstance(failure_type, str) or not failure_type or len(failure_type) > 256:
            raise ValueError("algorithmic failure type must be a short nonempty string")
        if not isinstance(message, str) or not message or len(message) > 4096:
            raise ValueError("algorithmic failure message must be a nonempty bounded string")
        result = {
            "schema": "acil-innovation-v1:job-algorithmic-failure:v1",
            "status": "algorithmic_failure",
            "job": _job_payload(self._job),
            "config_sha256": protocol_config_sha256(),
            "manifest_sha256": self._manifest_sha256,
            "provenance_sha256": self._provenance_sha256,
            "started_sha256": self._started.sha256,
            "failure": {"type": failure_type, "message": message},
            "artifacts": {
                name: _validate_named_artifact(
                    self._staging_directory, name=name, descriptor=descriptor
                )
                for name, descriptor in sorted(self._artifacts.items())
            },
        }
        write_canonical_json_exclusive(self._staging_directory / "result.json", result)
        return self._finalize()


def begin_job(
    *,
    stage: str,
    job_id: str,
    output_root: Path,
    manifest_sha256: str,
    provenance_sha256: str,
) -> JobResultWriter:
    """Create the sole exclusive in-progress directory for one planned job."""

    return JobResultWriter(
        job=_resolve_job(stage, job_id),
        output_root=Path(output_root),
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )


def _validate_result_common(payload: Mapping[str, object], job: PlannedJob) -> None:
    if payload.get("job") != _job_payload(job):
        raise ValueError("job result identity differs from the planned job")
    if payload.get("config_sha256") != protocol_config_sha256():
        raise ValueError("job result config hash drifted")
    _require_sha256(payload.get("manifest_sha256"), "job result manifest hash")
    _require_sha256(payload.get("provenance_sha256"), "job result provenance hash")
    _require_sha256(payload.get("started_sha256"), "job started file hash")


def load_job_result(*, stage: str, job_id: str, output_root: Path) -> dict[str, object]:
    """Load and fully revalidate one finalized planned-job directory."""

    job = _resolve_job(stage, job_id)
    root = Path(output_root) / stage / job_id
    _plain_directory(root, label="final job")
    result, _ = _read_canonical_json(root / "result.json")
    if not isinstance(result, dict):
        raise ValueError("job result must be a JSON object")
    status = result.get("status")
    common = {
        "schema",
        "status",
        "job",
        "config_sha256",
        "manifest_sha256",
        "provenance_sha256",
        "started_sha256",
    }
    if status == "succeeded":
        if set(result) != common | {"artifacts", "checkpoint", "records"}:
            raise ValueError("succeeded job result schema has missing or extra fields")
        if result["schema"] != "acil-innovation-v1:job-succeeded:v1":
            raise ValueError("succeeded job result schema identity drifted")
    elif status == "algorithmic_failure":
        if set(result) != common | {"artifacts", "failure"}:
            raise ValueError("algorithmic-failure result schema has missing or extra fields")
        if result["schema"] != "acil-innovation-v1:job-algorithmic-failure:v1":
            raise ValueError("algorithmic-failure result schema identity drifted")
    else:
        raise ValueError("job result has an unknown status")
    _validate_result_common(result, job)
    started, started_artifact = _read_canonical_json(root / "started.json")
    expected_started = {
        "schema": "acil-innovation-v1:job-started:v1",
        "status": "started",
        "job": _job_payload(job),
        "config_sha256": result["config_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "provenance_sha256": result["provenance_sha256"],
    }
    if started != expected_started or started_artifact.sha256 != result["started_sha256"]:
        raise ValueError("job started record or hash drifted")
    artifacts = result["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError("job named artifacts registry must be an object")
    seen_artifact_paths: set[str] = set()
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError("job named artifact descriptor must be an object")
        validated = _validate_named_artifact(root, name=name, descriptor=descriptor)
        if validated["path"] in seen_artifact_paths:
            raise ValueError("job named artifacts contain a duplicate relative path")
        seen_artifact_paths.add(str(validated["path"]))
    if status == "succeeded":
        records = result["records"]
        if not isinstance(records, Mapping) or set(records) != {"jsonl", "sha256", "count"}:
            raise ValueError("job records descriptor schema drifted")
        if records["jsonl"] != "records.jsonl":
            raise ValueError("job records relative path drifted")
        checkpoint = result["checkpoint"]
        if not isinstance(checkpoint, Mapping):
            raise ValueError("job checkpoint descriptor must be an object")
        checkpoint = _validate_checkpoint_descriptor(
            root,
            job=job,
            descriptor=checkpoint,
            config_sha256=str(result["config_sha256"]),
            manifest_sha256=str(result["manifest_sha256"]),
            provenance_sha256=str(result["provenance_sha256"]),
        )
        _validate_training_artifact(
            root,
            job=job,
            artifacts=artifacts,
            checkpoint=checkpoint,
        )
        _validate_dependencies_contract(
            root,
            job=job,
            artifacts=artifacts,
            checkpoint=checkpoint,
            output_root=Path(output_root),
            manifest_sha256=str(result["manifest_sha256"]),
            provenance_sha256=str(result["provenance_sha256"]),
        )
        validate_named_artifact_registry(job, artifacts)
        _validate_embedded_checkpoint_state(
            root,
            job=job,
            checkpoint=checkpoint,
            output_root=Path(output_root),
        )
        artifact = _validate_job_records(
            root / "records.jsonl",
            job=job,
            expected_sha256=records["sha256"],
        )
        if records["count"] != artifact.count:
            raise ValueError("job records complete line count drifted")
        validate_succeeded_job_inventory(
            root,
            job=job,
            checkpoint=checkpoint,
            artifacts=artifacts,
            finalized=True,
        )
    else:
        failure = result["failure"]
        if not isinstance(failure, Mapping) or set(failure) != {"type", "message"}:
            raise ValueError("algorithmic failure schema drifted")
        if not all(isinstance(failure[name], str) and failure[name] for name in failure):
            raise ValueError("algorithmic failure fields must be nonempty strings")
    return json.loads(canonical_json_bytes(result).decode("ascii"))


__all__ = [
    "FileArtifact",
    "JobResultWriter",
    "begin_job",
    "load_job_result",
    "regular_file_sha256",
    "validate_metric_records_file",
    "write_canonical_json_exclusive",
    "write_canonical_jsonl_exclusive",
    "write_metric_records_exclusive",
]
