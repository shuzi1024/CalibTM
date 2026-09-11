"""Run one fixed branch-sealed minimal-calibrator prototype job."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor, nn

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.config import CUBLAS_WORKSPACE_CONFIG
from experiments.acil_innovation_v1.registries import seed_bundle
from experiments.acil_innovation_v1.training import run_protocol_training
from experiments.acil_mechanism_v1.run_job import (
    _epoch_batches,
    _evaluation_batches,
    _seed_runtime,
    _source_dev_result,
    _training_loss,
    _tune_evidence,
)
from experiments.anchorcv_v1.data_access import (
    canonical_data_identity,
    load_permitted_windows,
    permitted_fit_fallback,
)
from experiments.sc_acil_v1.checkpoint import save_checkpoint

from .model import new_calibrator


_DATASETS = ("abilene", "geant")
_METHODS = ("static_zero", "blinear_only", "value_only")
_JOB_DOMAIN = b"minimal-calibrator-v1:job:v1\x00"


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


def registered_job(dataset: str, method: str) -> dict[str, object]:
    if dataset not in _DATASETS:
        raise ValueError("dataset is outside the frozen prototype grid")
    if method not in _METHODS:
        raise ValueError("method is outside the frozen prototype grid")
    identity: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "protocol": "minimal-calibrator-v1",
        "seed_bundle": 1,
        "stage": "prototype",
    }
    identity["job_id"] = hashlib.sha256(
        _JOB_DOMAIN + _canonical_json(identity)
    ).hexdigest()
    return identity


def expected_job_identities() -> tuple[dict[str, object], ...]:
    return tuple(
        registered_job(dataset, method)
        for dataset in _DATASETS
        for method in _METHODS
    )


def write_result(path: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    output = Path(path)
    _write_json_exclusive(output, result)
    manifest = {
        "job": result.get("job"),
        "probe_freeze": result.get("probe_freeze"),
        "protocol": "minimal-calibrator-v1",
        "result_file": output.name,
        "result_sha256": _file_sha256(output),
        "schema": "minimal-calibrator-v1:result-manifest:v1",
    }
    _write_json_exclusive(
        output.with_name(output.name + ".manifest.json"),
        manifest,
    )
    return manifest


def _load_windows(dataset: str, cohort: str):
    """Expose only fit/source-dev/tune from the sealed discovery loader."""

    return load_permitted_windows(dataset, cohort)


def run_job(
    *,
    dataset: str,
    method: str,
    output: str | Path,
) -> dict[str, object]:
    job = registered_job(dataset, method)
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
    seeds = seed_bundle(1)
    _seed_runtime(seeds.model)
    model = new_calibrator(method, model_seed=seeds.model).to(device)
    fallback = permitted_fit_fallback(dataset)
    source_batches = _evaluation_batches(dataset, 1, "source_dev")

    def loss(active: nn.Module, value: object) -> Tensor:
        if not isinstance(active, ACILBase):
            raise TypeError("training received the wrong model")
        return _training_loss(active, value, fallback, device)

    training = run_protocol_training(
        model,
        pretrained_parameters=(),
        epoch_batches=lambda epoch: _epoch_batches(dataset, 1, epoch),
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
        bundle=1,
        model=model,
        fallback=fallback,
        device=device,
    )
    if probe_identity() != freeze:
        raise RuntimeError("freeze identity drifted during training")

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
        "protocol": "minimal-calibrator-v1",
        "schema": "minimal-calibrator-v1:result:v1",
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
        description="Run one frozen minimal-calibrator prototype job."
    )
    parser.add_argument("--dataset", choices=_DATASETS, required=True)
    parser.add_argument("--method", choices=_METHODS, required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_job(
        dataset=args.dataset,
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
    "build_parser",
    "expected_job_identities",
    "registered_job",
    "run_job",
    "write_result",
]
