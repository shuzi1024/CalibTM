"""Manifest-resolved training/evaluation worker for frozen protocol jobs."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .acil import ACILBase
from .batching import EvaluationBatch, TrainingBatch, build_evaluation_batch, build_training_batch
from .checkpoint import load_checkpoint, save_checkpoint
from .config import CUBLAS_WORKSPACE_CONFIG, protocol_config_sha256
from .data import fit_fallback, load_registered_windows
from .derangement import build_derangement
from .evaluation import records_for_window
from .jobs import PlannedJob, planned_jobs
from .manifest import active_manifest_stages
from .metrics import MetricRecord, metric_record
from .model_factory import (
    new_residual_model as _new_residual_model,
    seed_runtime as _seed_runtime,
)
from .models import QueryResidualModel
from .oracle_model import OracleSupportQ, TruthQDeepSets
from .preprocessing import (
    FitFallback,
    observation_only_linear_fill,
    observation_statistics,
)
from .registries import seed_bundle as registered_seed_bundle, stage_spec
from .result_io import (
    begin_job,
    load_job_result,
    write_canonical_json_exclusive,
    write_metric_records_exclusive,
)
from .training import (
    ProtocolTrainingResult,
    SourceDevResult,
    acil_paired_objective_separated,
    normalized_target_mae,
    run_protocol_training,
)


SUPPORTED_JOB_HANDLERS = frozenset(
    {
        ("stage0_acil_tune", "acil"),
        ("stage_h", "truth_q_deepsets"),
        ("stage_i", "local_loo"),
        ("stage_i", "global_loo"),
        ("full_tune", "full_u0"),
        ("full_tune", "full_scratch"),
        ("full_tune", "full_gpt2"),
    }
)
_DIRECT_UPSTREAM = {
    "stage_h": "stage0_acil_tune",
    "stage_i": "stage_h",
    "full_tune": "stage_i",
}


def resolve_planned_job(stage: str, job_id: str) -> PlannedJob:
    """Resolve exactly one immutable job; no method/data override is accepted."""

    if not isinstance(stage, str) or not stage:
        raise TypeError("stage must be a nonempty string")
    if not isinstance(job_id, str) or not job_id:
        raise TypeError("job_id must be a nonempty string")
    if stage not in active_manifest_stages():
        raise ValueError(f"stage {stage!r} is outside the active manifest")
    try:
        jobs = planned_jobs(stage)
    except ValueError as exc:
        raise ValueError(f"unknown frozen stage {stage!r}") from exc
    matches = tuple(job for job in jobs if job.job_id == job_id)
    if len(matches) != 1:
        raise ValueError("job_id is not a planned job in the requested stage")
    return matches[0]


def acil_training_loss(
    model: ACILBase,
    *,
    model_input: Tensor,
    truth: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> Tensor:
    """Frozen K=3/K=2 ACIL objective with a NaN-sealed model operand."""

    loss, _ = acil_paired_objective_separated(
        model,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=fit_fallback,
    )
    return loss


def oracle_training_loss(
    model: TruthQDeepSets,
    *,
    model_input: Tensor,
    truth: Tensor,
    observed: Tensor,
    support_q: OracleSupportQ,
    evaluation_target: Tensor,
    fit_fallback: FitFallback,
) -> Tensor:
    """Score the truth-Q diagnostic on immutable E only."""

    if not isinstance(model, TruthQDeepSets):
        raise TypeError("oracle model must be TruthQDeepSets")
    if (
        model_input.shape != truth.shape
        or truth.shape != observed.shape
        or truth.shape != evaluation_target.shape
    ):
        raise ValueError("oracle model, truth, observed, and E shapes must match")
    if observed.dtype is not torch.bool or evaluation_target.dtype is not torch.bool:
        raise TypeError("oracle observed and E tensors must be boolean")
    if not torch.isfinite(truth).all().item() or (truth < 0).any().item():
        raise ValueError("oracle loss truth must be finite and nonnegative")
    q_mask = support_q.support_mask(observed)
    if torch.any(q_mask & evaluation_target).item():
        raise ValueError("oracle Q and E must be disjoint")
    if not torch.equal(q_mask | evaluation_target, ~observed):
        raise ValueError("oracle Q union E must equal the unobserved complement")
    prediction = model(model_input, observed, support_q, fit_fallback)
    scale = observation_statistics(model_input, observed, fit_fallback).std
    return normalized_target_mae(prediction, truth, evaluation_target, scale)


def residual_training_loss(
    model: QueryResidualModel,
    *,
    model_input: Tensor,
    truth: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> Tensor:
    """Score a deployable residual model on all 47 missing timestamps."""

    if not isinstance(model, QueryResidualModel):
        raise TypeError("deployable residual model must be QueryResidualModel")
    if model_input.shape != truth.shape or truth.shape != observed.shape:
        raise ValueError("residual model input, truth, and observed shapes must match")
    if observed.dtype is not torch.bool:
        raise TypeError("residual observed tensor must be boolean")
    if not torch.isfinite(truth).all().item() or (truth < 0).any().item():
        raise ValueError("residual loss truth must be finite and nonnegative")
    prediction = model(model_input, observed, fit_fallback)
    scale = observation_statistics(model_input, observed, fit_fallback).std
    return normalized_target_mae(prediction, truth, ~observed, scale)


def _to_device(batch: TrainingBatch, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        batch.model_input.to(device=device, non_blocking=False),
        batch.truth.to(device=device, non_blocking=False),
        batch.observed.to(device=device, non_blocking=False),
    )


def _oracle_support_slice(
    batch: EvaluationBatch, start: int, stop: int, device: torch.device
) -> OracleSupportQ:
    return OracleSupportQ(
        indices=batch.oracle_support_q.indices[start:stop].to(device),
        values=batch.oracle_support_q.values[start:stop].to(device),
    )


def _predict_evaluation(
    *,
    method: str,
    model: nn.Module | None,
    batch: EvaluationBatch,
    fallback: FitFallback,
    device: torch.device,
    physical_batch_size: int = 8,
) -> Tensor:
    """Return CPU float32 predictions in the registered window order."""

    predictions: list[Tensor] = []
    if model is not None:
        model.eval()
    for start in range(0, batch.truth.shape[0], physical_batch_size):
        stop = min(start + physical_batch_size, batch.truth.shape[0])
        model_input = batch.model_input[start:stop].to(device)
        observed = batch.observed[start:stop].to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if method == "linear_fill":
                prediction = observation_only_linear_fill(
                    model_input, observed, fallback
                )
            elif method == "acil":
                if not isinstance(model, ACILBase):
                    raise TypeError("ACIL evaluation requires ACILBase")
                prediction = model(model_input, observed, fallback).prediction
            elif method == "truth_q_deepsets":
                if not isinstance(model, TruthQDeepSets):
                    raise TypeError("oracle evaluation requires TruthQDeepSets")
                support = _oracle_support_slice(batch, start, stop, device)
                prediction = model(model_input, observed, support, fallback)
            elif method in {
                "local_loo",
                "global_loo",
                "loo_deepsets",
                "full_u0",
                "full_scratch",
                "full_gpt2",
            }:
                if not isinstance(model, QueryResidualModel):
                    raise TypeError("residual evaluation requires QueryResidualModel")
                prediction = model(model_input, observed, fallback)
            else:
                raise ValueError(f"unsupported evaluation method {method!r}")
        canonical = prediction.detach().float().cpu().contiguous()
        truth = batch.truth[start:stop]
        observed_cpu = batch.observed[start:stop]
        if not torch.equal(
            canonical.masked_select(observed_cpu),
            truth.masked_select(observed_cpu),
        ):
            raise RuntimeError("evaluation observed hard projection drifted")
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
    oracle: bool,
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
        target = batch.oracle_e if oracle else batch.target
        difference = (prediction.double() - batch.truth.double()).abs()
        truth = batch.truth.double().abs()
        rows.append(
            {
                "epoch": epoch,
                "mask_family": family,
                "ae": float(difference.masked_select(target).sum(dtype=torch.float64).item()),
                "truth": float(truth.masked_select(target).sum(dtype=torch.float64).item()),
            }
        )
    return SourceDevResult(rows=tuple(rows))


def _metric_records(
    *,
    stage: str,
    method: str,
    job: PlannedJob,
    batch: EvaluationBatch,
    prediction: Tensor,
    oracle: bool,
) -> tuple[MetricRecord, ...]:
    if prediction.shape != batch.truth.shape:
        raise ValueError("prediction shape differs from evaluation registry")
    target = batch.oracle_e if oracle else batch.target
    rows: list[MetricRecord] = []
    for window, absolute_start in enumerate(batch.absolute_starts):
        for flow in range(batch.truth.shape[1]):
            identity: dict[str, object] = {
                "method": method,
                "seed_bundle": job.seed_bundle,
                "dataset": job.dataset,
                "mask_family": batch.family,
                "window_start": absolute_start,
                "flow": flow,
                "oracle": oracle,
            }
            rows.append(
                metric_record(
                    identity=identity,
                    truth=batch.truth[window, flow].numpy(),
                    prediction=prediction[window, flow].numpy(),
                    observed=batch.observed[window, flow].numpy(),
                    target=target[window, flow].numpy(),
                )
            )
    return tuple(rows)


def _evaluation_records_for_methods(
    *,
    stage: str,
    job: PlannedJob,
    methods: Mapping[str, nn.Module | None],
    fallback: FitFallback,
    device: torch.device,
    oracle: bool,
) -> tuple[MetricRecord, ...]:
    cohort = stage_spec(stage).eval_cohort
    if cohort is None:
        raise ValueError("training-only stage has no evaluation record cohort")
    windows = load_registered_windows(stage, job.dataset, cohort)
    rows: list[MetricRecord] = []
    for family in ("random", "internal_block", "two_burst"):
        batch = build_evaluation_batch(
            windows, seed_bundle=job.seed_bundle, family=family
        )
        for method, model in methods.items():
            prediction = _predict_evaluation(
                method=method,
                model=model,
                batch=batch,
                fallback=fallback,
                device=device,
            )
            rows.extend(
                _metric_records(
                    stage=stage,
                    method=method,
                    job=job,
                    batch=batch,
                    prediction=prediction,
                    oracle=oracle,
                )
            )
    return tuple(rows)


def _deranged_records(
    *,
    job: PlannedJob,
    model: QueryResidualModel,
    fallback: FitFallback,
    device: torch.device,
) -> tuple[tuple[MetricRecord, ...], tuple[dict[str, object], ...]]:
    if job.stage != "stage_i" or job.method != "global_loo":
        raise ValueError("derangement is registered only for Stage-I global_loo")
    windows = load_registered_windows(job.stage, job.dataset, "tune")
    all_rows: list[MetricRecord] = []
    plans: list[dict[str, object]] = []
    for family in ("random", "internal_block", "two_burst"):
        batch = build_evaluation_batch(
            windows, seed_bundle=job.seed_bundle, family=family
        )
        innovations = []
        for start in range(0, batch.truth.shape[0], 8):
            stop = min(start + 8, batch.truth.shape[0])
            model_input = batch.model_input[start:stop].to(device)
            observed = batch.observed[start:stop].to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = model.prepare_features(model_input, observed, fallback)
            innovations.append(
                features.loo.innovation_normalized.detach().float().cpu().contiguous()
            )
        unshuffled = torch.cat(innovations, dim=0)
        plan = build_derangement(job.dataset, job.seed_bundle, family)
        shuffled = plan.apply(unshuffled)
        predictions = []
        for start in range(0, batch.truth.shape[0], 8):
            stop = min(start + 8, batch.truth.shape[0])
            model_input = batch.model_input[start:stop].to(device)
            observed = batch.observed[start:stop].to(device)
            override = shuffled[start:stop].to(device=device, dtype=model_input.dtype)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model.forward_with_innovation(
                    model_input,
                    observed,
                    fallback,
                    innovation_normalized=override,
                )
            predictions.append(prediction.detach().float().cpu().contiguous())
        canonical = torch.cat(predictions, dim=0)
        all_rows.extend(
            _metric_records(
                stage=job.stage,
                method="global_loo_deranged",
                job=job,
                batch=batch,
                prediction=canonical,
                oracle=False,
            )
        )
        plans.append(
            {
                "identity": asdict(plan.identity),
                "permutation_sha256": plan.permutation_sha256,
            }
        )
    return tuple(all_rows), tuple(plans)


def _training_epoch_batches(
    *,
    windows,
    job: PlannedJob,
    epoch: int,
):
    for batch_index in range(64):
        yield build_training_batch(
            windows,
            seed_bundle=job.seed_bundle,
            epoch=epoch,
            batch_index=batch_index,
        )


def _train_acil(job: PlannedJob, device: torch.device):
    seeds = registered_seed_bundle(job.seed_bundle)
    _seed_runtime(seeds.model)
    model = ACILBase().to(device)
    fit_windows = load_registered_windows(job.stage, job.dataset, "fit")
    source_windows = load_registered_windows(job.stage, job.dataset, "source_dev")
    fallback = fit_fallback(job.dataset)
    source_batches = {
        family: build_evaluation_batch(
            source_windows, seed_bundle=job.seed_bundle, family=family
        )
        for family in ("random", "internal_block", "two_burst")
    }

    def loss(active: nn.Module, value: object) -> Tensor:
        if not isinstance(active, ACILBase) or not isinstance(value, TrainingBatch):
            raise TypeError("ACIL worker received an invalid model or batch")
        model_input, truth, observed = _to_device(value, device)
        return acil_training_loss(
            active,
            model_input=model_input,
            truth=truth,
            observed=observed,
            fit_fallback=fallback,
        )

    result = run_protocol_training(
        model,
        pretrained_parameters=(),
        epoch_batches=lambda epoch: _training_epoch_batches(
            windows=fit_windows, job=job, epoch=epoch
        ),
        compute_loss=loss,
        evaluate_source_dev=lambda active, epoch: _source_dev_result(
            model=active,
            method="acil",
            batches=source_batches,
            fallback=fallback,
            device=device,
            epoch=epoch,
            oracle=False,
        ),
        device_type=device.type,
    )
    return model, result, fallback


def _train_residual(
    job: PlannedJob,
    *,
    acil: ACILBase,
    device: torch.device,
):
    seeds = registered_seed_bundle(job.seed_bundle)
    model, pretrained = _new_residual_model(
        job.method, acil=acil, model_seed=seeds.model
    )
    model = model.to(device)
    fit_windows = load_registered_windows(job.stage, job.dataset, "fit")
    source_windows = load_registered_windows(job.stage, job.dataset, "source_dev")
    fallback = fit_fallback(job.dataset)
    source_batches = {
        family: build_evaluation_batch(
            source_windows, seed_bundle=job.seed_bundle, family=family
        )
        for family in ("random", "internal_block", "two_burst")
    }

    def loss(active: nn.Module, value: object) -> Tensor:
        if not isinstance(active, QueryResidualModel) or not isinstance(
            value, TrainingBatch
        ):
            raise TypeError("residual worker received an invalid model or batch")
        model_input, truth, observed = _to_device(value, device)
        return residual_training_loss(
            active,
            model_input=model_input,
            truth=truth,
            observed=observed,
            fit_fallback=fallback,
        )

    result = run_protocol_training(
        model,
        pretrained_parameters=pretrained,
        epoch_batches=lambda epoch: _training_epoch_batches(
            windows=fit_windows, job=job, epoch=epoch
        ),
        compute_loss=loss,
        evaluate_source_dev=lambda active, epoch: _source_dev_result(
            model=active,
            method=job.method,
            batches=source_batches,
            fallback=fallback,
            device=device,
            epoch=epoch,
            oracle=False,
        ),
        device_type=device.type,
    )
    return model, result, fallback


def _train_oracle(
    job: PlannedJob,
    *,
    acil: ACILBase,
    device: torch.device,
):
    seeds = registered_seed_bundle(job.seed_bundle)
    _seed_runtime(seeds.model)
    model = TruthQDeepSets(acil).to(device)
    fit_windows = load_registered_windows(job.stage, job.dataset, "fit")
    source_windows = load_registered_windows(job.stage, job.dataset, "source_dev")
    fallback = fit_fallback(job.dataset)
    source_batches = {
        family: build_evaluation_batch(
            source_windows, seed_bundle=job.seed_bundle, family=family
        )
        for family in ("random", "internal_block", "two_burst")
    }

    def loss(active: nn.Module, value: object) -> Tensor:
        if not isinstance(active, TruthQDeepSets) or not isinstance(
            value, TrainingBatch
        ):
            raise TypeError("oracle worker received an invalid model or batch")
        model_input, truth, observed = _to_device(value, device)
        support = OracleSupportQ(
            indices=value.oracle_support_q.indices.to(device),
            values=value.oracle_support_q.values.to(device),
        )
        return oracle_training_loss(
            active,
            model_input=model_input,
            truth=truth,
            observed=observed,
            support_q=support,
            evaluation_target=value.oracle_e.to(device),
            fit_fallback=fallback,
        )

    result = run_protocol_training(
        model,
        pretrained_parameters=(),
        epoch_batches=lambda epoch: _training_epoch_batches(
            windows=fit_windows, job=job, epoch=epoch
        ),
        compute_loss=loss,
        evaluate_source_dev=lambda active, epoch: _source_dev_result(
            model=active,
            method="truth_q_deepsets",
            batches=source_batches,
            fallback=fallback,
            device=device,
            epoch=epoch,
            oracle=True,
        ),
        device_type=device.type,
    )
    return model, result, fallback


def _training_payload(
    result: ProtocolTrainingResult,
    *,
    job: PlannedJob,
    fallback: FitFallback,
) -> dict[str, object]:
    return {
        "schema": "acil-innovation-v1:training-history:v1",
        "protocol": "acil-innovation-v1",
        "config_sha256": protocol_config_sha256(),
        "job": job.to_json(),
        "fit_fallback": {"mean": float(fallback.mean), "std": float(fallback.std)},
        "epochs_completed": result.epochs_completed,
        "optimizer_updates": result.optimizer_updates,
        "best_epoch": result.best_epoch,
        "best_source_dev_nmae": result.best_source_dev_nmae,
        "epochs": [asdict(row) for row in result.epochs],
    }


def _checkpoint_identity(
    *,
    job: PlannedJob,
    training: ProtocolTrainingResult,
    manifest_sha256: str,
    provenance_sha256: str,
    base_checkpoint: Mapping[str, object] | None = None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "protocol": "acil-innovation-v1",
        "config_sha256": protocol_config_sha256(),
        "stage": job.stage,
        "job_id": job.job_id,
        "method": job.method,
        "dataset": job.dataset,
        "seed_bundle": job.seed_bundle,
        "epoch": training.best_epoch,
        "source_dev_nmae": training.best_source_dev_nmae,
        "manifest_sha256": manifest_sha256,
        "provenance_sha256": provenance_sha256,
    }
    if base_checkpoint is not None:
        identity["base_checkpoint"] = dict(base_checkpoint)
    return identity


def _read_json_exact(path: Path) -> object:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot decode fixed freeze artifact {path.name}") from exc
    from .config import canonical_json_bytes

    if raw != canonical_json_bytes(value):
        raise ValueError(f"freeze artifact {path.name} is not canonical")
    return value


def _active_freeze() -> tuple[str, str]:
    """Rebuild scientific identities and bind them only to fixed freeze_v2."""

    from .manifest import _build_manifest
    from .provenance import build_provenance
    from .scripts.build_freeze import (
        _FREEZE_DIRECTORY,
        _FREEZE_VERSION,
        _build_freeze_record,
    )

    root = Path(__file__).resolve().parent / "generated" / _FREEZE_DIRECTORY
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            f"fixed generated/{_FREEZE_DIRECTORY} is absent or invalid"
        )
    provenance = build_provenance()
    manifest = _build_manifest(provenance)
    bundle = _build_freeze_record(provenance, manifest)
    anchor_path = root / "anchor.json"
    anchor = _read_json_exact(anchor_path)
    expected_anchor = {
        "freeze_file": f"freeze-{bundle.freeze_sha256}.json",
        "freeze_sha256": bundle.freeze_sha256,
        "manifest_file": f"manifest-{manifest['manifest_sha256']}.json",
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol": "acil-innovation-v1",
        "provenance_file": f"provenance-{provenance['provenance_sha256']}.json",
        "provenance_sha256": provenance["provenance_sha256"],
        "schema_version": 1,
        "version": _FREEZE_VERSION,
    }
    if anchor != expected_anchor:
        raise ValueError(
            f"active source/runtime identities differ from {_FREEZE_DIRECTORY}"
        )
    if _read_json_exact(root / expected_anchor["provenance_file"]) != provenance:
        raise ValueError("frozen provenance payload drifted")
    if _read_json_exact(root / expected_anchor["manifest_file"]) != manifest:
        raise ValueError("frozen manifest payload drifted")
    if _read_json_exact(root / expected_anchor["freeze_file"]) != bundle.record:
        raise ValueError("frozen record payload drifted")
    return str(manifest["manifest_sha256"]), str(provenance["provenance_sha256"])


def _find_job(stage: str, *, method: str, dataset: str, seed_bundle: int) -> PlannedJob:
    matches = tuple(
        job
        for job in planned_jobs(stage)
        if job.method == method
        and job.dataset == dataset
        and job.seed_bundle == seed_bundle
    )
    if len(matches) != 1:
        raise RuntimeError("frozen dependency job registry is not unique")
    return matches[0]


def _load_acil_dependency(
    *,
    source_stage: str,
    output_root: Path,
    dataset: str,
    seed_bundle: int,
    manifest_sha256: str,
    provenance_sha256: str,
    device: torch.device,
) -> tuple[ACILBase, dict[str, object]]:
    dependency = _find_job(
        source_stage, method="acil", dataset=dataset, seed_bundle=seed_bundle
    )
    payload = load_job_result(
        stage=source_stage, job_id=dependency.job_id, output_root=output_root
    )
    if payload.get("status") != "succeeded":
        raise ValueError("required ACIL dependency did not succeed")
    if (
        payload.get("manifest_sha256") != manifest_sha256
        or payload.get("provenance_sha256") != provenance_sha256
    ):
        raise ValueError("required ACIL dependency belongs to another freeze")
    descriptor = payload["checkpoint"]
    if not isinstance(descriptor, Mapping):
        raise ValueError("required ACIL checkpoint descriptor is invalid")
    identity = descriptor["identity"]
    if not isinstance(identity, Mapping):
        raise ValueError("required ACIL checkpoint identity is invalid")
    model = ACILBase()
    record = load_checkpoint(
        model,
        metadata_path=output_root
        / source_stage
        / dependency.job_id
        / str(descriptor["metadata"]),
        expected_identity=identity,
    )
    for actual, field in (
        (record.identity_sha256, "identity_sha256"),
        (record.file_sha256, "file_sha256"),
        (record.tensor_sha256, "tensor_sha256"),
    ):
        if actual != descriptor[field]:
            raise ValueError("required ACIL checkpoint hash binding drifted")
    model.to(device).eval().requires_grad_(False)
    binding = {
        "source_stage": source_stage,
        "source_job_id": dependency.job_id,
        "checkpoint_identity_sha256": record.identity_sha256,
        "checkpoint_file_sha256": record.file_sha256,
        "checkpoint_tensor_sha256": record.tensor_sha256,
    }
    return model, binding


def _load_residual_dependency(
    *,
    source_stage: str,
    source_method: str,
    output_root: Path,
    dataset: str,
    seed_bundle: int,
    manifest_sha256: str,
    provenance_sha256: str,
    device: torch.device,
) -> tuple[QueryResidualModel, dict[str, object]]:
    dependency = _find_job(
        source_stage,
        method=source_method,
        dataset=dataset,
        seed_bundle=seed_bundle,
    )
    payload = load_job_result(
        stage=source_stage, job_id=dependency.job_id, output_root=output_root
    )
    if payload.get("status") != "succeeded":
        raise ValueError("required residual dependency did not succeed")
    if (
        payload.get("manifest_sha256") != manifest_sha256
        or payload.get("provenance_sha256") != provenance_sha256
    ):
        raise ValueError("required residual dependency belongs to another freeze")
    descriptor = payload["checkpoint"]
    if not isinstance(descriptor, Mapping) or not isinstance(
        descriptor.get("identity"), Mapping
    ):
        raise ValueError("required residual checkpoint descriptor is invalid")
    seeds = registered_seed_bundle(seed_bundle)
    model, _ = _new_residual_model(
        source_method,
        acil=ACILBase(),
        model_seed=seeds.model,
    )
    record = load_checkpoint(
        model,
        metadata_path=output_root
        / source_stage
        / dependency.job_id
        / str(descriptor["metadata"]),
        expected_identity=descriptor["identity"],
    )
    for actual, field in (
        (record.identity_sha256, "identity_sha256"),
        (record.file_sha256, "file_sha256"),
        (record.tensor_sha256, "tensor_sha256"),
    ):
        if actual != descriptor[field]:
            raise ValueError("required residual checkpoint hash binding drifted")
    model.to(device).eval().requires_grad_(False)
    return model, {
        "source_stage": source_stage,
        "source_job_id": dependency.job_id,
        "source_method": source_method,
        "checkpoint_identity_sha256": record.identity_sha256,
        "checkpoint_file_sha256": record.file_sha256,
        "checkpoint_tensor_sha256": record.tensor_sha256,
    }


def _write_success(
    *,
    writer,
    job: PlannedJob,
    model: nn.Module,
    training: ProtocolTrainingResult,
    fallback: FitFallback,
    records: tuple[MetricRecord, ...],
    manifest_sha256: str,
    provenance_sha256: str,
    base_checkpoint: Mapping[str, object] | None = None,
) -> Path:
    identity = _checkpoint_identity(
        job=job,
        training=training,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        base_checkpoint=base_checkpoint,
    )
    checkpoint = save_checkpoint(
        model, identity=identity, directory=writer.checkpoint_directory
    )
    writer.register_checkpoint(checkpoint)
    writer.write_records(records)
    history = write_canonical_json_exclusive(
        writer.staging_directory / "training.json",
        _training_payload(training, job=job, fallback=fallback),
    )
    writer.register_artifact("training", history.path)
    return writer.succeed()


def _register_derangement_artifacts(
    *,
    writer,
    job: PlannedJob,
    records: tuple[MetricRecord, ...],
    plans: tuple[dict[str, object], ...],
) -> None:
    starts = load_registered_windows("stage_i", job.dataset, "tune").absolute_starts
    flows = len(records) // (3 * len(starts))
    expected = tuple(
        {
            "method": "global_loo_deranged",
            "seed_bundle": job.seed_bundle,
            "dataset": job.dataset,
            "mask_family": family,
            "window_start": start,
            "flow": flow,
            "oracle": False,
        }
        for family in ("random", "internal_block", "two_burst")
        for start in starts
        for flow in range(flows)
    )
    artifact = write_metric_records_exclusive(
        writer.staging_directory / "derangement_records.jsonl",
        records,
        expected_identities=expected,
    )
    writer.register_artifact("derangement_records", artifact.path)
    plan_artifact = write_canonical_json_exclusive(
        writer.staging_directory / "derangement_plans.json",
        {
            "schema": "acil-innovation-v1:stage-i-derangement-plans:v1",
            "job": job.to_json(),
            "plans": list(plans),
        },
    )
    writer.register_artifact("derangement_plans", plan_artifact.path)


def _register_dependencies(
    *, writer, job: PlannedJob, dependencies: Mapping[str, Mapping[str, object]]
) -> None:
    artifact = write_canonical_json_exclusive(
        writer.staging_directory / "dependencies.json",
        {
            "schema": "acil-innovation-v1:job-dependencies:v1",
            "job": job.to_json(),
            "dependencies": {
                name: dict(value) for name, value in sorted(dependencies.items())
            },
        },
    )
    writer.register_artifact("dependencies", artifact.path)


def _preflight_job(
    *,
    job: PlannedJob,
    output_root: Path,
    manifest_sha256: str,
    provenance_sha256: str,
    device: torch.device,
) -> dict[str, object]:
    """Validate dispatch, upstream authorization, and dependencies before staging."""

    if (job.stage, job.method) not in SUPPORTED_JOB_HANDLERS:
        raise RuntimeError(
            f"worker integration for {job.stage}/{job.method} is not frozen"
        )
    upstream = _DIRECT_UPSTREAM.get(job.stage)
    if upstream is not None:
        from .adjudication import load_stage_adjudication

        decision = load_stage_adjudication(upstream, output_root)
        if decision.get("stage") != upstream or decision.get("verdict") != "proceed":
            raise RuntimeError(
                f"stage {job.stage!r} requires an exact upstream proceed adjudication"
            )

    dependencies: dict[str, object] = {}
    if job.stage in {"stage_h", "stage_i", "full_tune"}:
        acil, binding = _load_acil_dependency(
            source_stage="stage0_acil_tune",
            output_root=output_root,
            dataset=job.dataset,
            seed_bundle=job.seed_bundle,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            device=device,
        )
        dependencies["acil"] = acil
        dependencies["acil_binding"] = binding
    if job.stage == "full_tune" and job.method == "full_u0":
        deepsets, binding = _load_residual_dependency(
            source_stage="stage_i",
            source_method="global_loo",
            output_root=output_root,
            dataset=job.dataset,
            seed_bundle=job.seed_bundle,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            device=device,
        )
        dependencies["loo_deepsets"] = deepsets
        dependencies["loo_deepsets_binding"] = binding
    return dependencies


def _preflight_dependency(
    dependencies: Mapping[str, object], name: str, expected_type: type
):
    try:
        value = dependencies[name]
    except KeyError:
        raise RuntimeError(f"preflight dependency {name!r} is absent") from None
    if not isinstance(value, expected_type):
        raise TypeError(f"preflight dependency {name!r} has an invalid type")
    return value


def execute_job(stage: str, job_id: str, output_root: Path):
    """Execute one frozen GPU job and atomically retain its evidence."""

    job = resolve_planned_job(stage, job_id)
    output_root = Path(output_root)
    manifest_sha256, provenance_sha256 = _active_freeze()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must equal the frozen CUDA determinism contract"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("formal planned jobs require one visible CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each planned worker must see exactly one CUDA device")
    device = torch.device("cuda:0")
    preflight = _preflight_job(
        job=job,
        output_root=output_root,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        device=device,
    )
    writer = begin_job(
        stage=stage,
        job_id=job_id,
        output_root=output_root,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )
    try:
        if stage == "stage0_acil_tune" and job.method == "acil":
            model, training, fallback = _train_acil(job, device)
            records = _evaluation_records_for_methods(
                stage=stage,
                job=job,
                methods={"linear_fill": None, "acil": model},
                fallback=fallback,
                device=device,
                oracle=False,
            )
            return _write_success(
                writer=writer,
                job=job,
                model=model,
                training=training,
                fallback=fallback,
                records=records,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
            )
        if stage == "stage_h" and job.method == "truth_q_deepsets":
            acil = _preflight_dependency(preflight, "acil", ACILBase)
            binding = _preflight_dependency(
                preflight, "acil_binding", dict
            )
            model, training, fallback = _train_oracle(
                job, acil=acil, device=device
            )
            records = _evaluation_records_for_methods(
                stage=stage,
                job=job,
                methods={"acil": acil, "truth_q_deepsets": model},
                fallback=fallback,
                device=device,
                oracle=True,
            )
            return _write_success(
                writer=writer,
                job=job,
                model=model,
                training=training,
                fallback=fallback,
                records=records,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
                base_checkpoint=binding,
            )
        if stage == "stage_i" and job.method in {"local_loo", "global_loo"}:
            acil = _preflight_dependency(preflight, "acil", ACILBase)
            binding = _preflight_dependency(
                preflight, "acil_binding", dict
            )
            model, training, fallback = _train_residual(
                job, acil=acil, device=device
            )
            methods = (
                {"acil": acil, "local_loo": model}
                if job.method == "local_loo"
                else {"global_loo": model}
            )
            records = _evaluation_records_for_methods(
                stage=stage,
                job=job,
                methods=methods,
                fallback=fallback,
                device=device,
                oracle=False,
            )
            if job.method == "global_loo":
                derived, plans = _deranged_records(
                    job=job, model=model, fallback=fallback, device=device
                )
                _register_derangement_artifacts(
                    writer=writer, job=job, records=derived, plans=plans
                )
            return _write_success(
                writer=writer,
                job=job,
                model=model,
                training=training,
                fallback=fallback,
                records=records,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
                base_checkpoint=binding,
            )
        if stage == "full_tune" and job.method in {
            "full_u0",
            "full_scratch",
            "full_gpt2",
        }:
            acil = _preflight_dependency(preflight, "acil", ACILBase)
            acil_binding = _preflight_dependency(
                preflight, "acil_binding", dict
            )
            model, training, fallback = _train_residual(
                job, acil=acil, device=device
            )
            dependencies: dict[str, Mapping[str, object]] = {
                "acil_base": acil_binding
            }
            if job.method == "full_u0":
                deepsets = _preflight_dependency(
                    preflight, "loo_deepsets", QueryResidualModel
                )
                deepsets_binding = _preflight_dependency(
                    preflight, "loo_deepsets_binding", dict
                )
                dependencies["loo_deepsets"] = deepsets_binding
                methods = {
                    "acil": acil,
                    "loo_deepsets": deepsets,
                    "full_u0": model,
                }
            else:
                methods = {job.method: model}
            records = _evaluation_records_for_methods(
                stage=stage,
                job=job,
                methods=methods,
                fallback=fallback,
                device=device,
                oracle=False,
            )
            _register_dependencies(
                writer=writer, job=job, dependencies=dependencies
            )
            return _write_success(
                writer=writer,
                job=job,
                model=model,
                training=training,
                fallback=fallback,
                records=records,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
                base_checkpoint=acil_binding,
            )
        if stage == "formal_acil" and job.method == "acil":
            model, training, fallback = _train_acil(job, device)
            return _write_success(
                writer=writer,
                job=job,
                model=model,
                training=training,
                fallback=fallback,
                records=(),
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
            )
        if stage == "formal_gate" and job.method in {
            "loo_deepsets",
            "full_u0",
            "full_scratch",
            "full_gpt2",
        }:
            acil, acil_binding = _load_acil_dependency(
                source_stage="formal_acil",
                output_root=output_root,
                dataset=job.dataset,
                seed_bundle=job.seed_bundle,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
                device=device,
            )
            model, training, fallback = _train_residual(
                job, acil=acil, device=device
            )
            methods = (
                {"linear_fill": None, "acil": acil, "full_u0": model}
                if job.method == "full_u0"
                else {job.method: model}
            )
            records = _evaluation_records_for_methods(
                stage=stage,
                job=job,
                methods=methods,
                fallback=fallback,
                device=device,
                oracle=False,
            )
            _register_dependencies(
                writer=writer,
                job=job,
                dependencies={"acil_base": acil_binding},
            )
            return _write_success(
                writer=writer,
                job=job,
                model=model,
                training=training,
                fallback=fallback,
                records=records,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
                base_checkpoint=acil_binding,
            )
        raise RuntimeError(
            f"worker integration for {job.stage}/{job.method} is not frozen yet"
        )
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
        return writer.fail_algorithmically(
            failure_type=type(exc).__name__, message=str(exc) or type(exc).__name__
        )


__all__ = [
    "SUPPORTED_JOB_HANDLERS",
    "acil_training_loss",
    "execute_job",
    "oracle_training_loss",
    "residual_training_loss",
    "resolve_planned_job",
]
