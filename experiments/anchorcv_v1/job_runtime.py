"""Execution and immutable artifacts for one frozen AnchorCV job."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Mapping

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors

from experiments.acil_innovation_v1.batching import (
    EvaluationBatch,
    build_evaluation_batch,
)
from experiments.acil_innovation_v1.registries import seed_bundle
from experiments.sc2_ari_v1.initialization import load_frozen_acil

from .data_access import load_permitted_windows, permitted_fit_fallback
from .evaluation import (
    CaseEvidence,
    concatenate_case_evidence,
    evaluate_tensor_batch,
    summarize_case_evidence,
)
from .job import JobSpec
from .model import PriorFreeMaskNativeExpert
from .protocol import load_protocol
from .training_runtime import (
    build_neural_expert,
    model_parameter_count,
    seed_everything,
    train_registered_expert,
)


_ARRAY_DOMAIN = b"anchorcv-v1:scientific-arrays:v1\x00"
_MASK_DOMAIN = b"anchorcv-v1:mask-grid:v1\x00"
_TENSOR_DOMAIN = b"anchorcv-v1:model-tensors:v1\x00"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_array_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("one or more named arrays are required")
    digest = hashlib.sha256(_ARRAY_DOMAIN)
    for name in sorted(arrays):
        if not isinstance(name, str) or not name:
            raise ValueError("array names must be nonempty strings")
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise ValueError("object arrays are forbidden")
        canonical = np.ascontiguousarray(value)
        parts = (
            name.encode("utf-8"),
            canonical.dtype.str.encode("ascii"),
            _canonical_json_bytes(list(canonical.shape)),
            canonical.view(np.uint8).tobytes(order="C"),
        )
        for part in parts:
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def scientific_tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("one or more model tensors are required")
    digest = hashlib.sha256(_TENSOR_DOMAIN)
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        payloads = (
            name.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            _canonical_json_bytes(list(tensor.shape)),
            tensor.view(torch.uint8).numpy().tobytes(order="C"),
        )
        for payload in payloads:
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def mask_identity_sha256(grid: tuple[tuple[str, ...], ...]) -> str:
    if not isinstance(grid, tuple) or not grid:
        raise ValueError("mask identity grid must be a nonempty tuple")
    for row in grid:
        if not isinstance(row, tuple) or not row:
            raise ValueError("mask identity grid rows must be nonempty tuples")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in row
        ):
            raise ValueError("mask identity values must be lowercase SHA-256 strings")
    return hashlib.sha256(
        _MASK_DOMAIN + _canonical_json_bytes(grid)
    ).hexdigest()


def evaluate_evaluation_batch(
    *,
    prior,
    neural: PriorFreeMaskNativeExpert,
    batch: EvaluationBatch,
    fit_fallback,
    device: torch.device,
    chunk_size: int,
) -> CaseEvidence:
    if not isinstance(batch, EvaluationBatch):
        raise TypeError("batch must be EvaluationBatch")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    parts = []
    for start in range(0, int(batch.truth.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(batch.truth.shape[0]))
        parts.append(
            evaluate_tensor_batch(
                prior=prior,
                neural=neural,
                model_input=batch.model_input[start:stop].to(
                    device, non_blocking=False
                ),
                truth=batch.truth[start:stop].to(device, non_blocking=False),
                observed=batch.observed[start:stop].to(
                    device, non_blocking=False
                ),
                fit_fallback=fit_fallback,
                window_offset=start,
            )
        )
    return concatenate_case_evidence(*parts)


def _exclusive_json(path: Path, payload: object) -> None:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot decode {label}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def verify_completed_job(spec: JobSpec) -> bool:
    """Verify the immutable identity and result hash before reusing a job."""

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    directory = spec.job_directory
    if not directory.exists():
        return False
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("existing job path must be a regular directory")
    manifest_path = spec.job_directory / "manifest.json"
    result_path = spec.job_directory / "result.json"
    if not manifest_path.exists() or not result_path.exists():
        raise ValueError("partial job directory cannot be treated as completed")
    manifest = _read_json_object(manifest_path, label="existing manifest")
    result = _read_json_object(result_path, label="existing result")
    expected_identity = spec.scientific_identity
    manifest_identity = {
        name: manifest.get(name) for name in expected_identity
    }
    if (
        manifest_identity != expected_identity
        or result.get("scientific_identity") != expected_identity
    ):
        raise ValueError("existing job scientific identity mismatch")
    if (
        manifest.get("job_id") != spec.job_id
        or result.get("job_id") != spec.job_id
    ):
        raise ValueError("existing job content-addressed identity mismatch")
    if (
        manifest.get("status") != "succeeded"
        or result.get("status") != "succeeded"
        or manifest.get("result_file") != "result.json"
    ):
        raise ValueError("existing result identity/status mismatch")
    if manifest.get("result_sha256") != _file_sha256(result_path):
        raise ValueError("existing result hash mismatch")
    return True


def _load_completed(spec: JobSpec) -> dict[str, object] | None:
    if not verify_completed_job(spec):
        return None
    result_path = spec.job_directory / "result.json"
    result = _read_json_object(result_path, label="existing result")
    return result


def execute_job(spec: JobSpec) -> dict[str, object]:
    """Train N once, then evaluate P/N/oracle/AnchorCV on all tune masks."""

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    completed = _load_completed(spec)
    if completed is not None:
        return completed
    if spec.job_directory.exists():
        raise FileExistsError(
            "partial job directory already exists; retain it and use an audited retry root"
        )
    spec.job_directory.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    attempt_manifest = {
        **spec.scientific_identity,
        "job_id": spec.job_id,
        "started_at": started_at,
        "status": "running",
        "result_sha256": None,
    }
    _exclusive_json(spec.job_directory / "attempt_manifest.json", attempt_manifest)

    try:
        if spec.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA job requested but CUDA is unavailable")
        device = torch.device(spec.device)
        seeds = seed_bundle(spec.seed_bundle)
        seed_everything(seeds.model, deterministic=True)
        model = build_neural_expert(spec.dataset).to(device)
        training = train_registered_expert(
            model,
            dataset=spec.dataset,
            seed_bundle=spec.seed_bundle,
            device=device,
        )
        model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        trained_at = time.monotonic()

        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        weights_path = spec.job_directory / "neural_best.safetensors"
        save_safetensors(state, str(weights_path))
        checkpoint = {
            "file": weights_path.name,
            "file_sha256": _file_sha256(weights_path),
            "tensor_sha256": scientific_tensor_sha256(state),
            "best_epoch": training.best_epoch,
            "best_source_dev_nmae": training.best_source_dev_nmae,
        }

        prior, prior_record = load_frozen_acil(
            spec.dataset, spec.seed_bundle, device=device
        )
        prior.eval()
        tune_windows = load_permitted_windows(spec.dataset, "tune")
        fallback = permitted_fit_fallback(spec.dataset)
        protocol = load_protocol()
        cell_results: dict[str, object] = {}
        for family in protocol.mask_families:
            evaluation_batch = build_evaluation_batch(
                tune_windows,
                seed_bundle=spec.seed_bundle,
                family=family,
            )
            evidence = evaluate_evaluation_batch(
                prior=prior,
                neural=model,
                batch=evaluation_batch,
                fit_fallback=fallback,
                device=device,
                chunk_size=8,
            )
            evidence_path = spec.job_directory / f"evidence_{family}.npz"
            np.savez_compressed(evidence_path, **evidence.as_npz_dict())
            evidence_arrays = evidence.as_npz_dict()
            cell_results[family] = {
                "summary": summarize_case_evidence(evidence),
                "evidence_file": evidence_path.name,
                "evidence_file_sha256": _file_sha256(evidence_path),
                "evidence_content_sha256": scientific_array_sha256(
                    evidence_arrays
                ),
                "mask_sha256": mask_identity_sha256(
                    evaluation_batch.mask_sha256
                ),
                "oracle_q_sha256": mask_identity_sha256(
                    evaluation_batch.oracle_q_sha256
                ),
                "oracle_e_sha256": mask_identity_sha256(
                    evaluation_batch.oracle_e_sha256
                ),
            }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluated_at = time.monotonic()
        epoch_records = [asdict(record) for record in training.epochs]
        if (
            training.epochs_completed != 20
            or training.optimizer_updates != 320
            or len(epoch_records) != 20
        ):
            raise RuntimeError("training result differs from the frozen budget")
        result: dict[str, object] = {
            "job_id": spec.job_id,
            "status": "succeeded",
            "scientific_identity": spec.scientific_identity,
            "neural_parameter_count": model_parameter_count(model),
            "neural_checkpoint": checkpoint,
            "prior_checkpoint": {
                "file_sha256": prior_record.file_sha256,
            },
            "training": {
                "epochs_completed": training.epochs_completed,
                "optimizer_updates": training.optimizer_updates,
                "best_epoch": training.best_epoch,
                "best_source_dev_nmae": training.best_source_dev_nmae,
                "epoch_records": epoch_records,
            },
            "cells": cell_results,
        }
        result_path = spec.job_directory / "result.json"
        _exclusive_json(result_path, result)
        result_sha256 = _file_sha256(result_path)
        runtime = {
            "device": spec.device,
            "finished_at": _utc_now(),
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "training_seconds": trained_at - started_monotonic,
            "evaluation_seconds": evaluated_at - trained_at,
            "total_seconds": evaluated_at - started_monotonic,
        }
        if not all(
            math.isfinite(float(runtime[name])) and float(runtime[name]) >= 0.0
            for name in ("training_seconds", "evaluation_seconds", "total_seconds")
        ):
            raise RuntimeError("runtime duration accounting became invalid")
        _exclusive_json(spec.job_directory / "runtime.json", runtime)
        manifest = {
            **spec.scientific_identity,
            "job_id": spec.job_id,
            "status": "succeeded",
            "result_file": result_path.name,
            "result_sha256": result_sha256,
            "neural_checkpoint_file_sha256": checkpoint["file_sha256"],
            "neural_checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
        }
        required = set(load_protocol().required_hash_fields)
        if not required.issubset(manifest):
            raise RuntimeError("final manifest lacks required integrity hashes")
        _exclusive_json(spec.job_directory / "manifest.json", manifest)
        return result
    except Exception as exc:
        failure_path = spec.job_directory / "failure.json"
        if not failure_path.exists():
            _exclusive_json(
                failure_path,
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "job_id": spec.job_id,
                    "status": "failed",
                    "time": _utc_now(),
                },
            )
        raise


__all__ = [
    "evaluate_evaluation_batch",
    "execute_job",
    "mask_identity_sha256",
    "scientific_array_sha256",
    "scientific_tensor_sha256",
    "verify_completed_job",
]
