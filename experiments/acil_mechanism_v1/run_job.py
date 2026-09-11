"""Run one fixed, branch-sealed ACIL feature-ablation job."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
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
from experiments.acil_innovation_v1.preprocessing import (
    FitFallback,
    observation_only_linear_fill,
)
from experiments.acil_innovation_v1.registries import seed_bundle
from experiments.acil_innovation_v1.training import (
    SourceDevResult,
    acil_paired_objective_separated,
    run_protocol_training,
)
from experiments.anchorcv_v1.data_access import (
    canonical_data_identity,
    load_permitted_windows,
    permitted_fit_fallback,
)
from experiments.sc_acil_v1.checkpoint import save_checkpoint

from .model import new_acil


_DATASETS = ("abilene", "geant")
_SEEDS = (1, 2, 3)
_METHODS = ("full", "value_only", "no_anchor")
_FAMILIES = ("random", "internal_block", "two_burst")
_JOB_DOMAIN = b"acil-mechanism-v1:job:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def registered_job(
    dataset: str, seed_bundle_id: int, method: str
) -> dict[str, object]:
    if dataset not in _DATASETS:
        raise ValueError("dataset is outside the frozen grid")
    if (
        isinstance(seed_bundle_id, bool)
        or not isinstance(seed_bundle_id, int)
        or seed_bundle_id not in _SEEDS
    ):
        raise ValueError("seed bundle is outside the frozen grid")
    if method not in _METHODS:
        raise ValueError("method is outside the frozen grid")
    stage = "prototype" if seed_bundle_id == 1 else "extension"
    identity: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "protocol": "acil-mechanism-v1",
        "seed_bundle": seed_bundle_id,
        "stage": stage,
    }
    identity["job_id"] = hashlib.sha256(
        _JOB_DOMAIN + _canonical_json(identity)
    ).hexdigest()
    return identity


def expected_job_identities(stage: str) -> tuple[dict[str, object], ...]:
    if stage == "prototype":
        seeds = (1,)
    elif stage == "extension":
        seeds = _SEEDS
    else:
        raise ValueError("stage must be prototype or extension")
    return tuple(
        registered_job(dataset, bundle, method)
        for dataset in _DATASETS
        for bundle in seeds
        for method in _METHODS
    )


def write_result(path: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    output = Path(path)
    _write_json_exclusive(output, result)
    manifest = {
        "job": result.get("job"),
        "probe_freeze": result.get("probe_freeze"),
        "protocol": "acil-mechanism-v1",
        "result_file": output.name,
        "result_sha256": _file_sha256(output),
        "schema": "acil-mechanism-v1:result-manifest:v1",
    }
    _write_json_exclusive(
        output.with_name(output.name + ".manifest.json"),
        manifest,
    )
    return manifest


def _window_error_sums(
    prediction: object,
    truth: object,
    target: object,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    selected = np.asarray(target)
    if (
        candidate.ndim != 3
        or actual.shape != candidate.shape
        or selected.shape != candidate.shape
        or selected.dtype != np.dtype(np.bool_)
    ):
        raise ValueError("prediction, truth and target must align [W,F,T]")
    if (
        not np.isfinite(candidate[selected]).all()
        or not np.isfinite(actual[selected]).all()
    ):
        raise ValueError("selected prediction and truth must be finite")
    difference = np.zeros_like(candidate)
    denominator_values = np.zeros_like(actual)
    np.subtract(candidate, actual, out=difference, where=selected)
    np.copyto(denominator_values, actual, where=selected)
    errors = np.abs(difference).sum(axis=(1, 2), dtype=np.float64)
    denominators = np.abs(denominator_values).sum(
        axis=(1, 2), dtype=np.float64
    )
    if np.any(errors < 0.0) or np.any(denominators <= 0.0):
        raise ValueError("window sums must have nonnegative error and positive truth")
    return errors, denominators


def _mask_grid_sha256(batch: EvaluationBatch) -> str:
    payload = {
        "absolute_starts": list(batch.absolute_starts),
        "cohort": batch.cohort,
        "dataset": batch.dataset,
        "family": batch.family,
        "mask_sha256": [list(row) for row in batch.mask_sha256],
        "seed_bundle": batch.seed_bundle,
    }
    return hashlib.sha256(
        b"acil-mechanism-v1:mask-grid:v1\x00"
        + _canonical_json(payload)
    ).hexdigest()


def _load_windows(dataset: str, cohort: str):
    """Expose exactly fit/source-dev/tune; no caller-controlled path."""

    return load_permitted_windows(dataset, cohort)


def _epoch_batches(dataset: str, bundle: int, epoch: int):
    fit = _load_windows(dataset, "fit")
    for batch_index in range(64):
        yield build_training_batch(
            fit,
            seed_bundle=bundle,
            epoch=epoch,
            batch_index=batch_index,
        )


def _to_device(
    batch: TrainingBatch, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        batch.model_input.to(device),
        batch.truth.to(device),
        batch.observed.to(device),
    )


def _training_loss(
    model: ACILBase,
    batch: TrainingBatch,
    fallback: FitFallback,
    device: torch.device,
) -> Tensor:
    model_input, truth, observed = _to_device(batch, device)
    loss, _ = acil_paired_objective_separated(
        model,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=fallback,
    )
    return loss


def _predict(
    *,
    model: ACILBase | None,
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
            if model is None:
                prediction = observation_only_linear_fill(
                    model_input, observed, fallback
                )
            else:
                prediction = model(
                    model_input, observed, fallback
                ).prediction
        canonical = prediction.detach().float().cpu().contiguous()
        observed_cpu = batch.observed[start:stop]
        if not torch.equal(
            canonical.masked_select(observed_cpu),
            batch.truth[start:stop].masked_select(observed_cpu),
        ):
            raise RuntimeError("observed hard projection drifted")
        if (
            not torch.isfinite(canonical).all().item()
            or (canonical < 0).any().item()
        ):
            raise FloatingPointError("evaluation prediction is invalid")
        predictions.append(canonical)
    return torch.cat(predictions, dim=0)


def _evaluation_batches(
    dataset: str, bundle: int, cohort: str
) -> dict[str, EvaluationBatch]:
    windows = _load_windows(dataset, cohort)
    return {
        family: build_evaluation_batch(
            windows,
            seed_bundle=bundle,
            family=family,
        )
        for family in _FAMILIES
    }


def _source_dev_result(
    *,
    model: ACILBase,
    batches: dict[str, EvaluationBatch],
    fallback: FitFallback,
    device: torch.device,
    epoch: int,
) -> SourceDevResult:
    rows = []
    for family in _FAMILIES:
        batch = batches[family]
        prediction = _predict(
            model=model,
            batch=batch,
            fallback=fallback,
            device=device,
        )
        difference = (prediction.double() - batch.truth.double()).abs()
        truth = batch.truth.double().abs()
        rows.append(
            {
                "ae": float(
                    difference.masked_select(batch.target).sum(
                        dtype=torch.float64
                    ).item()
                ),
                "epoch": epoch,
                "mask_family": family,
                "truth": float(
                    truth.masked_select(batch.target).sum(
                        dtype=torch.float64
                    ).item()
                ),
            }
        )
    return SourceDevResult(rows=tuple(rows))


def _tune_evidence(
    *,
    dataset: str,
    bundle: int,
    model: ACILBase,
    fallback: FitFallback,
    device: torch.device,
) -> dict[str, object]:
    batches = _evaluation_batches(dataset, bundle, "tune")
    cells: dict[str, object] = {}
    for family in _FAMILIES:
        batch = batches[family]
        candidate = _predict(
            model=model,
            batch=batch,
            fallback=fallback,
            device=device,
        )
        linear = _predict(
            model=None,
            batch=batch,
            fallback=fallback,
            device=device,
        )
        truth = batch.truth.detach().float().cpu().numpy()
        target = batch.target.detach().cpu().numpy()
        model_error, truth_sum = _window_error_sums(
            candidate.numpy(), truth, target
        )
        linear_error, linear_truth = _window_error_sums(
            linear.numpy(), truth, target
        )
        if not np.array_equal(truth_sum, linear_truth):
            raise RuntimeError("paired methods changed the truth denominator")
        cells[family] = {
            "linear_error_sum_per_window": [
                float(value) for value in linear_error
            ],
            "mask_grid_sha256": _mask_grid_sha256(batch),
            "model_error_sum_per_window": [
                float(value) for value in model_error
            ],
            "target_count_per_window": [
                int(value)
                for value in target.sum(axis=(1, 2), dtype=np.int64)
            ],
            "truth_sum_per_window": [
                float(value) for value in truth_sum
            ],
            "window_starts": list(batch.absolute_starts),
        }
    return cells


def run_job(
    *,
    dataset: str,
    seed_bundle_id: int,
    method: str,
    output: str | Path,
) -> dict[str, object]:
    job = registered_job(dataset, seed_bundle_id, method)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG differs from the deterministic contract"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each formal worker requires exactly one visible CUDA device"
        )

    from .freeze import probe_identity

    freeze = probe_identity()
    started = time.monotonic()
    device = torch.device("cuda:0")
    seeds = seed_bundle(seed_bundle_id)
    _seed_runtime(seeds.model)
    model = new_acil(method, model_seed=seeds.model).to(device)
    fallback = permitted_fit_fallback(dataset)
    source_batches = _evaluation_batches(
        dataset, seed_bundle_id, "source_dev"
    )

    def loss(active: nn.Module, value: object) -> Tensor:
        if not isinstance(active, ACILBase):
            raise TypeError("training received the wrong model")
        if not isinstance(value, TrainingBatch):
            raise TypeError("training received the wrong batch")
        return _training_loss(active, value, fallback, device)

    training = run_protocol_training(
        model,
        pretrained_parameters=(),
        epoch_batches=lambda epoch: _epoch_batches(
            dataset, seed_bundle_id, epoch
        ),
        compute_loss=loss,
        evaluate_source_dev=lambda active, epoch: _source_dev_result(
            model=active,
            batches=source_batches,
            fallback=fallback,
            device=device,
            epoch=epoch,
        ),
        device_type=device.type,
    )
    cells = _tune_evidence(
        dataset=dataset,
        bundle=seed_bundle_id,
        model=model,
        fallback=fallback,
        device=device,
    )

    checkpoint_identity = {
        "best_epoch": training.best_epoch,
        "feature_set": model.extractor.feature_set,
        "job": job,
        "probe_freeze": freeze,
        "source_dev_nmae": training.best_source_dev_nmae,
    }
    output_path = Path(output)
    checkpoint = save_checkpoint(
        model,
        identity=checkpoint_identity,
        directory=output_path.parent / "checkpoints",
    )
    result: dict[str, object] = {
        "cells": cells,
        "checkpoint": {
            "file_sha256": checkpoint.file_sha256,
            "identity": checkpoint_identity,
            "identity_sha256": checkpoint.identity_sha256,
            "metadata": checkpoint.metadata_path.relative_to(
                output_path.parent
            ).as_posix(),
            "tensor_sha256": checkpoint.tensor_sha256,
            "weights": checkpoint.weights_path.relative_to(
                output_path.parent
            ).as_posix(),
        },
        "data_identity": canonical_data_identity(dataset),
        "evaluation_cohort": "tune",
        "feature_set": model.extractor.feature_set,
        "fit_access": True,
        "job": job,
        "operational": {
            "cuda_device_name": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.monotonic() - started,
            "torch_version": torch.__version__,
        },
        "probe_freeze": freeze,
        "protocol": "acil-mechanism-v1",
        "schema": "acil-mechanism-v1:result:v1",
        "selection_cohort": "source_dev",
        "status": "succeeded",
        "test_access": False,
        "training": {
            "best_epoch": training.best_epoch,
            "best_source_dev_nmae": training.best_source_dev_nmae,
            "epochs": [asdict(row) for row in training.epochs],
            "epochs_completed": training.epochs_completed,
            "optimizer_updates": training.optimizer_updates,
            "parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
    }
    write_result(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one frozen ACIL matched-feature job."
    )
    parser.add_argument("--dataset", choices=_DATASETS, required=True)
    parser.add_argument(
        "--seed-bundle",
        choices=_SEEDS,
        required=True,
        type=int,
    )
    parser.add_argument("--method", choices=_METHODS, required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_job(
        dataset=args.dataset,
        seed_bundle_id=args.seed_bundle,
        method=args.method,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "job_id": result["job"]["job_id"],
                "output": str(args.output),
                "status": result["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_load_windows",
    "_window_error_sums",
    "build_parser",
    "expected_job_identities",
    "registered_job",
    "run_job",
    "write_result",
]
