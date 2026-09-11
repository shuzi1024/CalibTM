"""Fail-closed architecture, embedded-state, and result-inventory checks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PurePosixPath
import stat
import threading
from typing import Mapping

import torch
from torch import Tensor, nn

from .acil import ACILBase
from .checkpoint import _tensor_sha
from .jobs import PlannedJob, planned_jobs
from .model_factory import build_registered_model, require_active_registered_job
from .registries import stage_spec


_STRICT_LOAD_LOCK = threading.RLock()


def build_registered_checkpoint_model(job: PlannedJob) -> nn.Module:
    """Expose the training-shared model factory at the validation boundary."""

    return build_registered_model(job)


def expected_named_artifact_paths(job: PlannedJob) -> dict[str, str]:
    """Return the exact successful named-artifact registry for one active job."""

    job = require_active_registered_job(job)
    expected = {"training": "training.json"}
    if job.stage == "stage_i" and job.method == "global_loo":
        expected.update(
            {
                "derangement_plans": "derangement_plans.json",
                "derangement_records": "derangement_records.jsonl",
            }
        )
    if job.stage == "full_tune":
        expected["dependencies"] = "dependencies.json"
    return dict(sorted(expected.items()))


def validate_named_artifact_registry(
    job: PlannedJob, artifacts: Mapping[str, Mapping[str, object]]
) -> None:
    """Reject a missing, extra, renamed, or rebound successful artifact."""

    expected = expected_named_artifact_paths(job)
    if not isinstance(artifacts, Mapping):
        raise ValueError("successful named artifact registry must be an object")
    if set(artifacts) != set(expected):
        raise ValueError("successful named artifact registry has missing or extra entries")
    for name, relative in expected.items():
        descriptor = artifacts[name]
        if not isinstance(descriptor, Mapping):
            raise ValueError("successful named artifact descriptor must be an object")
        if descriptor.get("path") != relative:
            raise ValueError("successful named artifact path differs from its registry")


def _plain_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is missing from the successful inventory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a plain non-symlink directory")


def _plain_regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is missing from the successful inventory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a plain non-symlink regular file")


def _checkpoint_inventory_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"successful checkpoint {label} path is invalid")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.parts[:1] != ("checkpoints",)
        or len(relative.parts) != 2
        or relative.parts[1] in {"", ".", ".."}
    ):
        raise ValueError(f"successful checkpoint {label} escapes its exact inventory")
    return relative


def validate_succeeded_job_inventory(
    root: Path,
    *,
    job: PlannedJob,
    checkpoint: Mapping[str, object],
    artifacts: Mapping[str, Mapping[str, object]],
    finalized: bool,
) -> None:
    """Require the exact files and sole checkpoint pair for a successful job."""

    require_active_registered_job(job)
    if not isinstance(finalized, bool):
        raise TypeError("finalized must be boolean")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("successful checkpoint descriptor must be an object")
    validate_named_artifact_registry(job, artifacts)
    metadata = _checkpoint_inventory_path(checkpoint.get("metadata"), label="metadata")
    weights = _checkpoint_inventory_path(checkpoint.get("weights"), label="weights")
    if metadata == weights:
        raise ValueError("successful checkpoint inventory aliases metadata and weights")

    root = Path(root)
    _plain_directory(root, label="successful job root")
    expected_top = {
        "started.json",
        "records.jsonl",
        "checkpoints",
        *(str(descriptor["path"]) for descriptor in artifacts.values()),
    }
    if finalized:
        expected_top.add("result.json")
    actual_top = {path.name for path in root.iterdir()}
    if actual_top != expected_top:
        missing = sorted(expected_top - actual_top)
        extra = sorted(actual_top - expected_top)
        raise ValueError(
            f"successful job inventory has missing={missing!r}, extra={extra!r}"
        )

    checkpoint_root = root / "checkpoints"
    _plain_directory(checkpoint_root, label="successful checkpoint directory")
    expected_checkpoint = {metadata.name, weights.name}
    actual_checkpoint = {path.name for path in checkpoint_root.iterdir()}
    if actual_checkpoint != expected_checkpoint:
        missing = sorted(expected_checkpoint - actual_checkpoint)
        extra = sorted(actual_checkpoint - expected_checkpoint)
        raise ValueError(
            f"successful checkpoint inventory has missing={missing!r}, extra={extra!r}"
        )

    for name in sorted(expected_top - {"checkpoints"}):
        _plain_regular_file(root / name, label=f"successful artifact {name}")
    for name in sorted(expected_checkpoint):
        _plain_regular_file(
            checkpoint_root / name, label=f"successful checkpoint file {name}"
        )


def _validate_state_against_model(
    model: nn.Module, state: Mapping[str, Tensor], *, label: str
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("registered architecture must be a torch module")
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} state must be a tensor mapping")
    if any(not isinstance(name, str) or not name for name in state):
        raise ValueError(f"{label} state contains an invalid tensor name")
    if any(not isinstance(value, Tensor) for value in state.values()):
        raise ValueError(f"{label} state contains a non-tensor value")

    expected = model.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(
            f"{label} state differs from registered architecture: "
            f"missing={missing!r}, extra={extra!r}"
        )
    for name, expected_tensor in expected.items():
        supplied = state[name]
        if supplied.shape != expected_tensor.shape or supplied.dtype != expected_tensor.dtype:
            raise ValueError(
                f"{label} state tensor {name!r} differs from registered architecture"
            )
    try:
        incompatible = model.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} state cannot strict-load registered architecture") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"{label} state cannot strict-load registered architecture")


def validate_registered_state_schema(
    job: PlannedJob, state: Mapping[str, Tensor]
) -> None:
    """Strict-load a state into the exact architecture registered for its job."""

    job = require_active_registered_job(job)
    with _STRICT_LOAD_LOCK:
        model = _registered_architecture_template(job.stage, job.method)
        _validate_state_against_model(
            model, state, label="checkpoint registered architecture"
        )


@lru_cache(maxsize=None)
def _registered_architecture_template(stage: str, method: str) -> nn.Module:
    """Materialize each registered architecture once per validating process."""

    matches = tuple(
        job for job in planned_jobs(stage) if job.method == method
    )
    if not matches:
        raise ValueError("model architecture is absent from the active manifest")
    return build_registered_checkpoint_model(matches[0])


def _embedded_acil_binding(job: PlannedJob):
    job = require_active_registered_job(job)
    role = f"{job.method}.acil_base"
    matches = tuple(
        binding
        for binding in stage_spec(job.stage).checkpoint_hash_bindings
        if role in binding.consumer_roles
    )
    if len(matches) > 1:
        raise RuntimeError("job has multiple embedded ACIL checkpoint bindings")
    if not matches:
        return None
    binding = matches[0]
    if binding.source_method != "acil":
        raise RuntimeError("embedded ACIL role is not bound to the ACIL method")
    return binding


def embedded_acil_source_job(job: PlannedJob) -> PlannedJob | None:
    """Resolve the sole exact ACIL source job embedded in a candidate state."""

    binding = _embedded_acil_binding(job)
    if binding is None:
        return None
    matches = tuple(
        source
        for source in planned_jobs(binding.source_stage)
        if source.method == binding.source_method
        and source.dataset == job.dataset
        and source.seed_bundle == job.seed_bundle
    )
    if len(matches) != 1:
        raise RuntimeError("embedded ACIL source job is not uniquely registered")
    return matches[0]


def validate_embedded_acil_state(
    job: PlannedJob,
    candidate_state: Mapping[str, Tensor],
    source_state: Mapping[str, Tensor],
) -> None:
    """Prove every frozen ``acil.*`` tensor equals the bound source tensor."""

    if embedded_acil_source_job(job) is None:
        raise ValueError("job has no registered embedded ACIL checkpoint")
    _validate_state_against_model(
        ACILBase(), source_state, label="source ACIL checkpoint"
    )
    if not isinstance(candidate_state, Mapping):
        raise ValueError("candidate state must be a tensor mapping")
    prefix = "acil."
    embedded = {
        name[len(prefix) :]: value
        for name, value in candidate_state.items()
        if isinstance(name, str) and name.startswith(prefix)
    }
    if set(embedded) != set(source_state):
        raise ValueError("embedded ACIL state has missing or extra tensors")
    for name, source in source_state.items():
        candidate = embedded[name]
        if (
            not isinstance(candidate, Tensor)
            or not isinstance(source, Tensor)
            or candidate.shape != source.shape
            or candidate.dtype != source.dtype
            or not torch.equal(candidate.detach().cpu(), source.detach().cpu())
        ):
            raise ValueError(f"embedded ACIL tensor {name!r} differs from source")
    if _tensor_sha(embedded) != _tensor_sha(source_state):
        raise ValueError("embedded ACIL semantic tensor hash differs from source")


__all__ = [
    "build_registered_checkpoint_model",
    "embedded_acil_source_job",
    "expected_named_artifact_paths",
    "validate_embedded_acil_state",
    "validate_named_artifact_registry",
    "validate_registered_state_schema",
    "validate_succeeded_job_inventory",
]
