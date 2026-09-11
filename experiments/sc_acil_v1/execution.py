"""Frozen training/evaluation worker for the SC-ACIL Stage-A grid."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.batching import (
    EvaluationBatch,
    TrainingBatch,
    build_evaluation_batch,
    build_training_batch,
)
from experiments.acil_innovation_v1.config import CUBLAS_WORKSPACE_CONFIG
from experiments.acil_innovation_v1.data import fit_fallback
from experiments.acil_innovation_v1.model_factory import seed_runtime
from experiments.acil_innovation_v1.preprocessing import (
    FitFallback,
    observation_only_linear_fill,
    observation_statistics,
)
from experiments.acil_innovation_v1.registries import seed_bundle as registered_seed_bundle
from experiments.acil_innovation_v1.training import (
    ProtocolTrainingResult,
    SourceDevResult,
    acil_paired_objective_separated,
    normalized_target_mae,
    run_protocol_training,
)

from .data import load_windows
from .checkpoint import load_checkpoint, save_checkpoint
from .evidence import mask_registry_sha256, window_metric_rows, write_rows
from .freeze import load_and_verify_freeze
from .jobs import PlannedJob, planned_jobs, resolve_job
from .model import SCACILModel, new_sc_acil_model
from .protocol import canonical_json_bytes, protocol_sha256


_CARRIERS = {
    ("fit_acil", "acil_only"): (),
    ("formal_gate", "sc_acil_u0"): (
        "linear_interpolation",
        "acil_only",
        "sc_acil_u0",
    ),
    ("formal_gate", "sc_acil"): ("sc_acil",),
}


def _acil_training_loss(
    model: ACILBase,
    *,
    model_input: Tensor,
    truth: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> Tensor:
    loss, _ = acil_paired_objective_separated(
        model,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=fit_fallback,
    )
    return loss


def _residual_training_loss(
    model: SCACILModel,
    *,
    model_input: Tensor,
    truth: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> Tensor:
    prediction = model(model_input, observed, fit_fallback)
    scale = observation_statistics(model_input, observed, fit_fallback).std
    return normalized_target_mae(prediction, truth, ~observed, scale)


def _predict_evaluation(
    *,
    method: str,
    model: nn.Module | None,
    batch: EvaluationBatch,
    fallback: FitFallback,
    device: torch.device,
) -> Tensor:
    if model is not None:
        model.eval()
    predictions = []
    for start in range(0, batch.truth.shape[0], 8):
        stop = min(start + 8, batch.truth.shape[0])
        model_input = batch.model_input[start:stop].to(device)
        observed = batch.observed[start:stop].to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if method == "linear_fill":
                prediction = observation_only_linear_fill(model_input, observed, fallback)
            elif method == "acil":
                if not isinstance(model, ACILBase):
                    raise TypeError("ACIL evaluation requires ACILBase")
                prediction = model(model_input, observed, fallback).prediction
            elif method == "local_loo":
                if not isinstance(model, SCACILModel):
                    raise TypeError("SC-ACIL evaluation requires SCACILModel")
                prediction = model(model_input, observed, fallback)
            else:
                raise ValueError("unsupported SC-ACIL evaluation method")
        canonical = prediction.detach().float().cpu().contiguous()
        observed_cpu = batch.observed[start:stop]
        if not torch.equal(
            canonical.masked_select(observed_cpu),
            batch.truth[start:stop].masked_select(observed_cpu),
        ):
            raise RuntimeError("observed hard projection drifted")
        if not torch.isfinite(canonical).all().item() or (canonical < 0).any().item():
            raise FloatingPointError("evaluation prediction is nonfinite or negative")
        predictions.append(canonical)
    return torch.cat(predictions, dim=0)


def _source_dev_result(
    *,
    model: nn.Module,
    method: str,
    batches: Mapping[str, EvaluationBatch],
    fallback: FitFallback,
    device: torch.device,
    epoch: int,
) -> SourceDevResult:
    rows = []
    for family in ("random", "internal_block", "two_burst"):
        batch = batches[family]
        prediction = _predict_evaluation(
            method=method,
            model=model,
            batch=batch,
            fallback=fallback,
            device=device,
        )
        difference = (prediction.double() - batch.truth.double()).abs()
        truth = batch.truth.double().abs()
        rows.append(
            {
                "epoch": epoch,
                "mask_family": family,
                "ae": float(difference.masked_select(batch.target).sum(dtype=torch.float64).item()),
                "truth": float(truth.masked_select(batch.target).sum(dtype=torch.float64).item()),
            }
        )
    return SourceDevResult(rows=tuple(rows))


def carrier_labels(stage: str, method: str) -> tuple[str, ...]:
    try:
        return _CARRIERS[(stage, method)]
    except KeyError:
        raise ValueError("job has no frozen execution handler") from None


def _read_canonical_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read canonical result {path}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"result {path} is not canonical")
    return value


def _write_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    content = canonical_json_bytes(dict(value)) + b"\n"
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _result_path(output_root: Path, job: PlannedJob) -> Path:
    return Path(output_root) / job.stage / job.job_id / "result.json"


def require_all_acil_dependencies(
    output_root: Path, *, manifest_sha256: str
) -> None:
    succeeded = 0
    for job in planned_jobs("fit_acil"):
        path = _result_path(Path(output_root), job)
        try:
            payload = _read_canonical_json(path)
        except ValueError:
            continue
        if (
            payload.get("status") == "succeeded"
            and payload.get("manifest_sha256") == manifest_sha256
            and isinstance(payload.get("checkpoint"), dict)
        ):
            succeeded += 1
    if succeeded != 6:
        raise RuntimeError(
            "formal_gate requires all six succeeded exact-freeze ACIL dependencies"
        )


def _epoch_batches(windows, job: PlannedJob, epoch: int):
    for batch_index in range(64):
        yield build_training_batch(
            windows,
            seed_bundle=job.seed_bundle,
            epoch=epoch,
            batch_index=batch_index,
        )


def _to_device(batch: TrainingBatch, device: torch.device):
    return (
        batch.model_input.to(device),
        batch.truth.to(device),
        batch.observed.to(device),
    )


def _source_batches(job: PlannedJob) -> dict[str, EvaluationBatch]:
    windows = load_windows(job.dataset, "source_dev")
    return {
        family: build_evaluation_batch(
            windows, seed_bundle=job.seed_bundle, family=family
        )
        for family in ("random", "internal_block", "two_burst")
    }


def _train_acil(job: PlannedJob, device: torch.device):
    seeds = registered_seed_bundle(job.seed_bundle)
    seed_runtime(seeds.model)
    model = ACILBase().to(device)
    fit = load_windows(job.dataset, "fit")
    fallback = fit_fallback(job.dataset)
    source = _source_batches(job)

    def loss(active: nn.Module, value: object) -> Tensor:
        if not isinstance(active, ACILBase) or not isinstance(value, TrainingBatch):
            raise TypeError("ACIL worker received an invalid model or batch")
        model_input, truth, observed = _to_device(value, device)
        return _acil_training_loss(
            active,
            model_input=model_input,
            truth=truth,
            observed=observed,
            fit_fallback=fallback,
        )

    result = run_protocol_training(
        model,
        pretrained_parameters=(),
        epoch_batches=lambda epoch: _epoch_batches(fit, job, epoch),
        compute_loss=loss,
        evaluate_source_dev=lambda active, epoch: _source_dev_result(
            model=active,
            method="acil",
            batches=source,
            fallback=fallback,
            device=device,
            epoch=epoch,
        ),
        device_type=device.type,
    )
    return model, result, fallback


def _train_residual(
    job: PlannedJob, *, acil: ACILBase, device: torch.device
):
    seeds = registered_seed_bundle(job.seed_bundle)
    model = new_sc_acil_model(
        job.method, acil=acil, model_seed=seeds.model
    ).to(device)
    fit = load_windows(job.dataset, "fit")
    fallback = fit_fallback(job.dataset)
    source = _source_batches(job)

    def loss(active: nn.Module, value: object) -> Tensor:
        if not isinstance(active, SCACILModel) or not isinstance(value, TrainingBatch):
            raise TypeError("SC-ACIL worker received an invalid model or batch")
        model_input, truth, observed = _to_device(value, device)
        return _residual_training_loss(
            active,
            model_input=model_input,
            truth=truth,
            observed=observed,
            fit_fallback=fallback,
        )

    result = run_protocol_training(
        model,
        pretrained_parameters=(),
        epoch_batches=lambda epoch: _epoch_batches(fit, job, epoch),
        compute_loss=loss,
        evaluate_source_dev=lambda active, epoch: _source_dev_result(
            model=active,
            method="local_loo",
            batches=source,
            fallback=fallback,
            device=device,
            epoch=epoch,
        ),
        device_type=device.type,
    )
    return model, result, fallback


def _training_payload(
    result: ProtocolTrainingResult, job: PlannedJob, fallback: FitFallback
) -> dict[str, object]:
    return {
        "schema": "sc-acil-v1:training:v1",
        "protocol": "sc-acil-v1",
        "protocol_sha256": protocol_sha256(),
        "job": job.to_json(),
        "fit_fallback": {"mean": float(fallback.mean), "std": float(fallback.std)},
        "epochs_completed": result.epochs_completed,
        "optimizer_updates": result.optimizer_updates,
        "best_epoch": result.best_epoch,
        "best_source_dev_nmae": result.best_source_dev_nmae,
        "epochs": [asdict(row) for row in result.epochs],
    }


def _checkpoint_descriptor(
    model: nn.Module,
    *,
    result: ProtocolTrainingResult,
    job: PlannedJob,
    staging: Path,
    manifest_sha256: str,
    base_checkpoint: Mapping[str, object] | None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "protocol": "sc-acil-v1",
        "protocol_sha256": protocol_sha256(),
        "manifest_sha256": manifest_sha256,
        "job": job.to_json(),
        "epoch": result.best_epoch,
        "source_dev_nmae": result.best_source_dev_nmae,
    }
    if base_checkpoint is not None:
        identity["base_checkpoint"] = dict(base_checkpoint)
    record = save_checkpoint(
        model, identity=identity, directory=staging / "checkpoints"
    )
    return {
        "metadata": f"checkpoints/{record.metadata_path.name}",
        "weights": f"checkpoints/{record.weights_path.name}",
        "identity": identity,
        "identity_sha256": record.identity_sha256,
        "file_sha256": record.file_sha256,
        "tensor_sha256": record.tensor_sha256,
    }


def _find_acil_job(dataset: str, seed_bundle: int) -> PlannedJob:
    matches = tuple(
        job
        for job in planned_jobs("fit_acil")
        if job.dataset == dataset and job.seed_bundle == seed_bundle
    )
    if len(matches) != 1:
        raise RuntimeError("ACIL dependency registry is not unique")
    return matches[0]


def _load_acil_dependency(
    output_root: Path,
    *,
    dataset: str,
    seed_bundle: int,
    manifest_sha256: str,
    device: torch.device,
):
    job = _find_acil_job(dataset, seed_bundle)
    result_path = _result_path(output_root, job)
    payload = _read_canonical_json(result_path)
    if payload.get("status") != "succeeded" or payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("ACIL dependency belongs to another status or freeze")
    descriptor = payload.get("checkpoint")
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("identity"), dict):
        raise ValueError("ACIL checkpoint descriptor is invalid")
    model = ACILBase()
    record = load_checkpoint(
        model,
        metadata_path=result_path.parent / str(descriptor["metadata"]),
        expected_identity=descriptor["identity"],
    )
    for actual, field in (
        (record.identity_sha256, "identity_sha256"),
        (record.file_sha256, "file_sha256"),
        (record.tensor_sha256, "tensor_sha256"),
    ):
        if actual != descriptor.get(field):
            raise ValueError("ACIL checkpoint hash binding drifted")
    model.to(device).eval().requires_grad_(False)
    binding = {
        "source_stage": "fit_acil",
        "source_job_id": job.job_id,
        "checkpoint_identity_sha256": record.identity_sha256,
        "checkpoint_file_sha256": record.file_sha256,
        "checkpoint_tensor_sha256": record.tensor_sha256,
    }
    return model, binding


def _gate_rows(
    job: PlannedJob,
    *,
    acil: ACILBase,
    model: SCACILModel,
    fallback: FitFallback,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    windows = load_windows(job.dataset, "gate")
    if job.method == "sc_acil_u0":
        methods = (
            ("linear_interpolation", "linear_fill", None),
            ("acil_only", "acil", acil),
            ("sc_acil_u0", "local_loo", model),
        )
    else:
        methods = (("sc_acil", "local_loo", model),)
    rows = []
    for family in ("random", "internal_block", "two_burst"):
        batch = build_evaluation_batch(
            windows, seed_bundle=job.seed_bundle, family=family
        )
        registry_sha = mask_registry_sha256(batch.mask_sha256)
        for label, evaluation_method, active in methods:
            prediction = _predict_evaluation(
                method=evaluation_method,
                model=active,
                batch=batch,
                fallback=fallback,
                device=device,
            )
            rows.extend(
                window_metric_rows(
                    method=label,
                    seed_bundle=job.seed_bundle,
                    dataset=job.dataset,
                    mask_family=family,
                    absolute_starts=batch.absolute_starts,
                    truth=batch.truth,
                    prediction=prediction,
                    observed=batch.observed,
                    mask_registry_sha256=registry_sha,
                )
            )
    expected = len(windows.absolute_starts) * 3 * len(methods)
    if len(rows) != expected:
        raise RuntimeError("formal evidence row count drifted")
    return tuple(rows)


def _begin(output_root: Path, job: PlannedJob) -> tuple[Path, Path]:
    stage = output_root / job.stage
    stage.mkdir(parents=True, exist_ok=True)
    final = stage / job.job_id
    if final.exists():
        raise FileExistsError("planned job result already exists")
    staging = stage / f".{job.job_id}.partial-{os.getpid()}"
    staging.mkdir(mode=0o755, exist_ok=False)
    return staging, final


def _publish(staging: Path, final: Path, payload: Mapping[str, object]) -> Path:
    _write_canonical_json(staging / "result.json", payload)
    os.replace(staging, final)
    return final


def execute_job(stage: str, job_id: str, output_root: Path):
    job = resolve_job(stage, job_id)
    carrier_labels(job.stage, job.method)
    freeze = load_and_verify_freeze()
    manifest_sha = str(freeze["manifest_sha256"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from the deterministic contract")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each formal worker requires exactly one visible CUDA device")
    output_root = Path(output_root)
    if stage == "formal_gate":
        require_all_acil_dependencies(output_root, manifest_sha256=manifest_sha)
    staging, final = _begin(output_root, job)
    device = torch.device("cuda:0")
    try:
        if stage == "fit_acil":
            model, training, fallback = _train_acil(job, device)
            base_binding = None
            evidence = None
        else:
            acil, base_binding = _load_acil_dependency(
                output_root,
                dataset=job.dataset,
                seed_bundle=job.seed_bundle,
                manifest_sha256=manifest_sha,
                device=device,
            )
            model, training, fallback = _train_residual(job, acil=acil, device=device)
            rows = _gate_rows(
                job,
                acil=acil,
                model=model,
                fallback=fallback,
                device=device,
            )
            evidence = write_rows(staging / "window_metrics.jsonl", rows)
        _write_canonical_json(
            staging / "training.json", _training_payload(training, job, fallback)
        )
        checkpoint = _checkpoint_descriptor(
            model,
            result=training,
            job=job,
            staging=staging,
            manifest_sha256=manifest_sha,
            base_checkpoint=base_binding,
        )
        payload: dict[str, object] = {
            "schema": "sc-acil-v1:job-result:v1",
            "status": "succeeded",
            "protocol": "sc-acil-v1",
            "protocol_sha256": protocol_sha256(),
            "manifest_sha256": manifest_sha,
            "job": job.to_json(),
            "carried_methods": list(carrier_labels(stage, job.method)),
            "checkpoint": checkpoint,
            "training": "training.json",
            "evidence": evidence,
        }
        return _publish(staging, final, payload)
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
        payload = {
            "schema": "sc-acil-v1:job-result:v1",
            "status": "algorithmic_failure",
            "protocol": "sc-acil-v1",
            "protocol_sha256": protocol_sha256(),
            "manifest_sha256": manifest_sha,
            "job": job.to_json(),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc) or type(exc).__name__,
        }
        return _publish(staging, final, payload)
    except BaseException as exc:
        try:
            _write_canonical_json(
                staging / "infrastructure_failure.json",
                {
                    "job": job.to_json(),
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "manifest_sha256": manifest_sha,
                },
            )
        finally:
            raise


__all__ = [
    "carrier_labels",
    "execute_job",
    "require_all_acil_dependencies",
]
