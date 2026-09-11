"""Checkpoint-reuse-only execution for one sealed AnchorCV final-gate job.

The final gate is an evaluation lane.  It reconstructs and verifies the exact
training-job identity authorized by :mod:`gate_identity`, verifies every
source artifact and model tensor before opening the gate cohort, and never
calls a training routine or writes a checkpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Mapping

import numpy as np
from safetensors.torch import load_file as load_safetensors
import torch

from experiments.acil_innovation_v1.batching import build_evaluation_batch
from experiments.acil_innovation_v1.registries import dataset_spec, seed_bundle
from experiments.sc2_ari_v1.initialization import load_frozen_acil

from .data_access import permitted_fit_fallback
from .evaluation import summarize_case_evidence
from .gate_data import _load_gate_windows, gate_window_schedule_sha256
from .gate_identity import GateJobSpec
from .job import JobSpec
from .job_runtime import (
    evaluate_evaluation_batch,
    mask_identity_sha256,
    scientific_array_sha256,
    scientific_tensor_sha256,
)
from .model import PriorFreeMaskNativeExpert
from .review import safe_job_artifact_path, verify_neural_state


_MASK_FAMILIES = ("random", "internal_block", "two_burst")
_RESULT_SCHEMA = "anchorcv-v1:final-gate-result:v1"
_MANIFEST_SCHEMA = "anchorcv-v1:final-gate-manifest:v1"
_ATTEMPT_SCHEMA = "anchorcv-v1:final-gate-attempt:v1"
_RUNTIME_SCHEMA = "anchorcv-v1:final-gate-runtime:v1"
_FAILURE_SCHEMA = "anchorcv-v1:final-gate-failure:v1"

_SOURCE_RESULT_FIELDS = {
    "job_id",
    "status",
    "scientific_identity",
    "neural_parameter_count",
    "neural_checkpoint",
    "prior_checkpoint",
    "training",
    "cells",
}
_SOURCE_NEURAL_FIELDS = {
    "file",
    "file_sha256",
    "tensor_sha256",
    "best_epoch",
    "best_source_dev_nmae",
}
_SOURCE_PRIOR_FIELDS = {"file_sha256"}
_SOURCE_TRAINING_FIELDS = {
    "epochs_completed",
    "optimizer_updates",
    "best_epoch",
    "best_source_dev_nmae",
    "epoch_records",
}
_SOURCE_MANIFEST_ADDITIONAL_FIELDS = {
    "job_id",
    "status",
    "result_file",
    "result_sha256",
    "neural_checkpoint_file_sha256",
    "neural_checkpoint_tensor_sha256",
}
_RESULT_FIELDS = {
    "schema",
    "job_id",
    "status",
    "scientific_identity",
    "training_performed",
    "source_binding",
    "neural_parameter_count",
    "prior_checkpoint",
    "cells",
}
_CELL_FIELDS = {
    "summary",
    "evidence_file",
    "evidence_file_sha256",
    "evidence_content_sha256",
    "mask_sha256",
    "oracle_q_sha256",
    "oracle_e_sha256",
}
_MANIFEST_ADDITIONAL_FIELDS = {
    "schema",
    "job_id",
    "status",
    "result_file",
    "result_sha256",
    "training_performed",
    "source_binding",
}
_RUNTIME_FIELDS = {
    "schema",
    "device",
    "finished_at",
    "host",
    "python",
    "torch",
    "cuda_runtime",
    "gpu_name",
    "training_seconds",
    "evaluation_seconds",
    "total_seconds",
}
_ATTEMPT_ADDITIONAL_FIELDS = {
    "schema",
    "job_id",
    "started_at",
    "status",
    "training_performed",
    "source_binding",
    "result_sha256",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path: Path, value: object) -> None:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot decode {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} schema drifted; "
            f"missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )


def _plain_source_root(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} source root must be a regular non-symlink directory")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} source root cannot be resolved") from exc


def _build_source_spec(
    spec: GateJobSpec,
    *,
    prototype_output_root: Path,
    extension_output_root: Path,
) -> tuple[JobSpec, dict[str, object]]:
    binding = dict(
        spec.authority.checkpoint_binding(spec.dataset, spec.seed_bundle)
    )
    expected_stage = "prototype" if spec.seed_bundle == 1 else "extension"
    if binding.get("source_training_stage") != expected_stage:
        raise ValueError("authorized source training stage drifted")
    selected_root = (
        prototype_output_root
        if expected_stage == "prototype"
        else extension_output_root
    )
    source_root = _plain_source_root(
        Path(selected_root),
        label=expected_stage,
    )
    payload = spec.authority.payload
    source_spec = JobSpec(
        dataset=spec.dataset,
        seed_bundle=spec.seed_bundle,
        stage=expected_stage,
        output_root=source_root,
        device="cuda",
        source_tree_sha256=str(payload["source_tree_sha256"]),
        config_sha256=str(payload["config_sha256"]),
    )
    if source_spec.job_id != binding.get("source_training_job_id"):
        raise ValueError("authorized source training job identity drifted")
    if source_spec.stage != binding.get("source_training_stage"):
        raise ValueError("authorized source training stage differs from JobSpec")
    return source_spec, binding


def _verify_source_checkpoint(
    source_spec: JobSpec,
    binding: Mapping[str, object],
) -> tuple[Mapping[str, torch.Tensor], int]:
    directory = source_spec.job_directory
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("authorized source job directory is absent or is a symlink")
    result_path = safe_job_artifact_path(
        directory,
        "result.json",
        expected_name="result.json",
    )
    manifest_path = safe_job_artifact_path(
        directory,
        "manifest.json",
        expected_name="manifest.json",
    )
    if _file_sha256(result_path) != binding.get("source_result_sha256"):
        raise ValueError("authorized source result file hash mismatch")
    if _file_sha256(manifest_path) != binding.get("source_manifest_sha256"):
        raise ValueError("authorized source manifest file hash mismatch")

    result = _read_json_object(result_path, label="source result")
    manifest = _read_json_object(manifest_path, label="source manifest")
    _exact_fields(result, _SOURCE_RESULT_FIELDS, label="source result")
    _exact_fields(
        manifest,
        set(source_spec.scientific_identity)
        | _SOURCE_MANIFEST_ADDITIONAL_FIELDS,
        label="source manifest",
    )
    expected_identity = source_spec.scientific_identity
    if (
        result.get("job_id") != source_spec.job_id
        or manifest.get("job_id") != source_spec.job_id
        or result.get("scientific_identity") != expected_identity
        or {
            field: manifest.get(field)
            for field in expected_identity
        }
        != expected_identity
    ):
        raise ValueError("authorized source artifact scientific identity mismatch")
    if (
        result.get("status") != "succeeded"
        or manifest.get("status") != "succeeded"
        or manifest.get("result_file") != "result.json"
        or manifest.get("result_sha256") != binding.get("source_result_sha256")
    ):
        raise ValueError("authorized source artifact status/result binding drifted")

    neural = result.get("neural_checkpoint")
    prior = result.get("prior_checkpoint")
    training = result.get("training")
    if not isinstance(neural, Mapping):
        raise ValueError("source neural checkpoint record must be an object")
    if not isinstance(prior, Mapping):
        raise ValueError("source prior checkpoint record must be an object")
    if not isinstance(training, Mapping):
        raise ValueError("source training record must be an object")
    _exact_fields(neural, _SOURCE_NEURAL_FIELDS, label="source neural checkpoint")
    _exact_fields(prior, _SOURCE_PRIOR_FIELDS, label="source prior checkpoint")
    _exact_fields(training, _SOURCE_TRAINING_FIELDS, label="source training")
    if neural.get("file") != "neural_best.safetensors":
        raise ValueError(
            "source checkpoint reported filename differs from the fixed filename"
        )
    checkpoint_path = safe_job_artifact_path(
        directory,
        neural["file"],
        expected_name="neural_best.safetensors",
    )
    expected_file_hash = binding.get("neural_checkpoint_file_sha256")
    if (
        _file_sha256(checkpoint_path) != expected_file_hash
        or neural.get("file_sha256") != expected_file_hash
        or manifest.get("neural_checkpoint_file_sha256")
        != expected_file_hash
    ):
        raise ValueError("authorized source neural checkpoint file hash mismatch")
    expected_tensor_hash = binding.get("neural_checkpoint_tensor_sha256")
    if (
        neural.get("tensor_sha256") != expected_tensor_hash
        or manifest.get("neural_checkpoint_tensor_sha256")
        != expected_tensor_hash
    ):
        raise ValueError("authorized source neural checkpoint tensor binding drifted")
    if (
        neural.get("best_epoch") != binding.get("best_epoch")
        or training.get("best_epoch") != binding.get("best_epoch")
        or neural.get("best_source_dev_nmae")
        != binding.get("best_source_dev_nmae")
        or training.get("best_source_dev_nmae")
        != binding.get("best_source_dev_nmae")
    ):
        raise ValueError("authorized source checkpoint selection record drifted")
    expected_acil_hash = binding.get("acil_checkpoint_file_sha256")
    if (
        prior.get("file_sha256") != expected_acil_hash
        or expected_identity.get("acil_checkpoint_file_sha256")
        != expected_acil_hash
    ):
        raise ValueError("authorized source ACIL checkpoint binding drifted")

    try:
        state = load_safetensors(str(checkpoint_path), device="cpu")
    except Exception as exc:
        raise ValueError("cannot load authorized source neural checkpoint") from exc
    if scientific_tensor_sha256(state) != expected_tensor_hash:
        raise ValueError("authorized source neural checkpoint tensor hash mismatch")
    parameter_count = result.get("neural_parameter_count")
    verify_neural_state(
        source_spec.dataset,
        state,
        reported_parameter_count=parameter_count,
    )
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
    ):
        raise ValueError("authorized source neural parameter count is invalid")
    return state, parameter_count


def _build_frozen_neural(dataset: str) -> PriorFreeMaskNativeExpert:
    """Build the exact registered architecture without using a training API."""

    return PriorFreeMaskNativeExpert(
        num_flows=dataset_spec(dataset).flows,
        time_steps=50,
        temporal_hidden=128,
        d_model=256,
        num_heads=8,
        num_flow_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    )


def _seed_evaluation(bundle: int) -> None:
    seeds = seed_bundle(bundle)
    random.seed(seeds.model)
    np.random.seed(seeds.model % (2**32))
    torch.manual_seed(seeds.model)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds.model)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True)


def _expected_output_names() -> set[str]:
    return {
        "attempt_manifest.json",
        "result.json",
        "runtime.json",
        "manifest.json",
        *(f"evidence_{family}.npz" for family in _MASK_FAMILIES),
    }


def verify_completed_gate_job(spec: GateJobSpec) -> bool:
    """Strictly verify a complete immutable final-gate result before reuse."""

    if type(spec) is not GateJobSpec:
        raise TypeError("spec must be exactly GateJobSpec")
    directory = spec.job_directory
    if not directory.exists():
        return False
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("existing gate job path must be a regular non-symlink directory")
    actual_names = {entry.name for entry in directory.iterdir()}
    expected_names = _expected_output_names()
    if actual_names != expected_names:
        raise ValueError(
            "partial or unexpected gate job directory cannot be reused"
        )
    for name in expected_names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("gate job artifact must be a regular non-symlink file")

    result_path = safe_job_artifact_path(
        directory, "result.json", expected_name="result.json"
    )
    manifest_path = safe_job_artifact_path(
        directory, "manifest.json", expected_name="manifest.json"
    )
    result = _read_json_object(result_path, label="existing gate result")
    manifest = _read_json_object(manifest_path, label="existing gate manifest")
    _exact_fields(result, _RESULT_FIELDS, label="existing gate result")
    _exact_fields(
        manifest,
        set(spec.scientific_identity) | _MANIFEST_ADDITIONAL_FIELDS,
        label="existing gate manifest",
    )
    expected_binding = dict(
        spec.authority.checkpoint_binding(spec.dataset, spec.seed_bundle)
    )
    expected_identity = spec.scientific_identity
    if (
        result.get("scientific_identity") != expected_identity
        or {
            field: manifest.get(field)
            for field in expected_identity
        }
        != expected_identity
    ):
        raise ValueError("existing gate scientific identity mismatch")
    if (
        result.get("schema") != _RESULT_SCHEMA
        or manifest.get("schema") != _MANIFEST_SCHEMA
        or result.get("job_id") != spec.job_id
        or manifest.get("job_id") != spec.job_id
        or result.get("status") != "succeeded"
        or manifest.get("status") != "succeeded"
        or result.get("training_performed") is not False
        or manifest.get("training_performed") is not False
        or result.get("source_binding") != expected_binding
        or manifest.get("source_binding") != expected_binding
        or manifest.get("result_file") != "result.json"
    ):
        raise ValueError("existing gate identity/status/source binding mismatch")
    if manifest.get("result_sha256") != _file_sha256(result_path):
        raise ValueError("existing gate result hash mismatch")
    prior = result.get("prior_checkpoint")
    cells = result.get("cells")
    if (
        not isinstance(prior, Mapping)
        or set(prior) != {"file_sha256"}
        or prior.get("file_sha256")
        != expected_binding["acil_checkpoint_file_sha256"]
        or not isinstance(cells, Mapping)
        or set(cells) != set(_MASK_FAMILIES)
    ):
        raise ValueError("existing gate nested result schema drifted")
    for family in _MASK_FAMILIES:
        cell = cells[family]
        if not isinstance(cell, Mapping):
            raise ValueError("existing gate cell must be an object")
        _exact_fields(cell, _CELL_FIELDS, label=f"existing gate {family} cell")
        evidence_name = f"evidence_{family}.npz"
        if cell.get("evidence_file") != evidence_name:
            raise ValueError("existing gate evidence filename drifted")
        evidence_path = safe_job_artifact_path(
            directory,
            cell["evidence_file"],
            expected_name=evidence_name,
        )
        if cell.get("evidence_file_sha256") != _file_sha256(evidence_path):
            raise ValueError("existing gate evidence file hash mismatch")

    runtime = _read_json_object(
        safe_job_artifact_path(
            directory, "runtime.json", expected_name="runtime.json"
        ),
        label="existing gate runtime",
    )
    _exact_fields(runtime, _RUNTIME_FIELDS, label="existing gate runtime")
    if (
        runtime.get("schema") != _RUNTIME_SCHEMA
        or runtime.get("device") != "cuda"
        or runtime.get("training_seconds") != 0.0
    ):
        raise ValueError("existing gate runtime schema/lane drifted")
    attempt = _read_json_object(
        safe_job_artifact_path(
            directory,
            "attempt_manifest.json",
            expected_name="attempt_manifest.json",
        ),
        label="existing gate attempt manifest",
    )
    _exact_fields(
        attempt,
        set(spec.scientific_identity) | _ATTEMPT_ADDITIONAL_FIELDS,
        label="existing gate attempt manifest",
    )
    if (
        attempt.get("schema") != _ATTEMPT_SCHEMA
        or attempt.get("job_id") != spec.job_id
        or attempt.get("status") != "running"
        or attempt.get("training_performed") is not False
        or attempt.get("source_binding") != expected_binding
        or attempt.get("result_sha256") is not None
    ):
        raise ValueError("existing gate attempt manifest identity drifted")
    return True


def _load_completed_gate_job(
    spec: GateJobSpec,
) -> dict[str, object] | None:
    if not verify_completed_gate_job(spec):
        return None
    return _read_json_object(
        spec.job_directory / "result.json",
        label="existing gate result",
    )


def execute_gate_job(
    spec: GateJobSpec,
    *,
    prototype_output_root: Path,
    extension_output_root: Path,
) -> dict[str, object]:
    """Evaluate exactly one authorized checkpoint on the fixed gate cohort."""

    if type(spec) is not GateJobSpec:
        raise TypeError("spec must be exactly GateJobSpec")
    completed = _load_completed_gate_job(spec)
    if completed is not None:
        return completed
    if spec.job_directory.exists():
        raise FileExistsError(
            "partial gate job directory already exists; retain it and use "
            "an audited retry root"
        )
    spec.job_directory.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    started = time.monotonic()
    binding = dict(
        spec.authority.checkpoint_binding(spec.dataset, spec.seed_bundle)
    )
    attempt = {
        **spec.scientific_identity,
        "schema": _ATTEMPT_SCHEMA,
        "job_id": spec.job_id,
        "started_at": started_at,
        "status": "running",
        "training_performed": False,
        "source_binding": binding,
        "result_sha256": None,
    }
    _exclusive_json(spec.job_directory / "attempt_manifest.json", attempt)

    try:
        expected_schedule = gate_window_schedule_sha256(spec.dataset)
        if spec.window_schedule_sha256 != expected_schedule:
            raise ValueError("gate window schedule identity drifted")
        source_spec, reconstructed_binding = _build_source_spec(
            spec,
            prototype_output_root=Path(prototype_output_root),
            extension_output_root=Path(extension_output_root),
        )
        if reconstructed_binding != binding:
            raise ValueError("gate source binding changed during execution")
        state, parameter_count = _verify_source_checkpoint(source_spec, binding)

        if not torch.cuda.is_available():
            raise RuntimeError("final gate requires the registered CUDA lane")
        device = torch.device("cuda")
        _seed_evaluation(spec.seed_bundle)
        neural = _build_frozen_neural(spec.dataset)
        try:
            neural.load_state_dict(dict(state), strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "authorized source neural checkpoint cannot load strictly"
            ) from exc
        neural.requires_grad_(False)
        neural.eval().to(device)

        prior, prior_record = load_frozen_acil(
            spec.dataset,
            spec.seed_bundle,
            device=device,
        )
        if prior_record.file_sha256 != binding[
            "acil_checkpoint_file_sha256"
        ]:
            raise ValueError("loaded ACIL checkpoint differs from gate authority")
        prior.eval()

        # This is deliberately the first and only gate-cohort access.  All
        # source identities, files, tensors, and models are verified above.
        gate_windows = _load_gate_windows(spec.authority, spec.dataset)
        fallback = permitted_fit_fallback(spec.dataset)
        cell_results: dict[str, object] = {}
        for family in _MASK_FAMILIES:
            registered = build_evaluation_batch(
                gate_windows,
                seed_bundle=spec.seed_bundle,
                family=family,
            )
            evidence = evaluate_evaluation_batch(
                prior=prior,
                neural=neural,
                batch=registered,
                fit_fallback=fallback,
                device=device,
                chunk_size=8,
            )
            evidence_arrays = evidence.as_npz_dict()
            evidence_path = (
                spec.job_directory / f"evidence_{family}.npz"
            )
            with evidence_path.open("xb") as handle:
                np.savez_compressed(handle, **evidence_arrays)
            cell_results[family] = {
                "summary": summarize_case_evidence(evidence),
                "evidence_file": evidence_path.name,
                "evidence_file_sha256": _file_sha256(evidence_path),
                "evidence_content_sha256": scientific_array_sha256(
                    evidence_arrays
                ),
                "mask_sha256": mask_identity_sha256(
                    registered.mask_sha256
                ),
                "oracle_q_sha256": mask_identity_sha256(
                    registered.oracle_q_sha256
                ),
                "oracle_e_sha256": mask_identity_sha256(
                    registered.oracle_e_sha256
                ),
            }
        torch.cuda.synchronize(device)
        finished = time.monotonic()
        evaluation_seconds = finished - started
        if (
            not math.isfinite(evaluation_seconds)
            or evaluation_seconds < 0.0
        ):
            raise RuntimeError("gate runtime duration accounting became invalid")
        result: dict[str, object] = {
            "schema": _RESULT_SCHEMA,
            "job_id": spec.job_id,
            "status": "succeeded",
            "scientific_identity": spec.scientific_identity,
            "training_performed": False,
            "source_binding": binding,
            "neural_parameter_count": parameter_count,
            "prior_checkpoint": {
                "file_sha256": prior_record.file_sha256,
            },
            "cells": cell_results,
        }
        result_path = spec.job_directory / "result.json"
        _exclusive_json(result_path, result)
        result_hash = _file_sha256(result_path)
        runtime = {
            "schema": _RUNTIME_SCHEMA,
            "device": "cuda",
            "finished_at": _utc_now(),
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "training_seconds": 0.0,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": evaluation_seconds,
        }
        _exclusive_json(spec.job_directory / "runtime.json", runtime)
        manifest = {
            **spec.scientific_identity,
            "schema": _MANIFEST_SCHEMA,
            "job_id": spec.job_id,
            "status": "succeeded",
            "result_file": "result.json",
            "result_sha256": result_hash,
            "training_performed": False,
            "source_binding": binding,
        }
        _exclusive_json(spec.job_directory / "manifest.json", manifest)
        return result
    except Exception as exc:
        failure_path = spec.job_directory / "failure.json"
        if not failure_path.exists():
            _exclusive_json(
                failure_path,
                {
                    "schema": _FAILURE_SCHEMA,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "job_id": spec.job_id,
                    "status": "failed",
                    "time": _utc_now(),
                    "training_performed": False,
                    "source_binding": binding,
                },
            )
        raise


__all__ = [
    "execute_gate_job",
    "verify_completed_gate_job",
]
