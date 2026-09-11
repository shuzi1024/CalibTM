"""Run one fixed tiny-KAN development-collision job."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    permitted_fit_fallback,
)
from experiments.sc_acil_v1.checkpoint import save_checkpoint

from .jobs import DATASETS, METHODS, PROTOCOL_ID, SEED_BUNDLES, registered_job
from .model import new_calibrator
from .result_io import RESULT_SCHEMA, write_result
from .source_identity import source_record, source_tree_sha256


def _architecture_record(model: ACILBase) -> dict[str, object]:
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    custom = getattr(model, "architecture_record", None)
    details = custom() if callable(custom) else {}
    if not isinstance(details, dict):
        raise TypeError("architecture_record must return a dictionary")
    return {
        "class": type(model).__name__,
        "details": details,
        "parameter_count": parameter_count,
    }


def run_registered_job(
    *,
    dataset: str,
    seed_bundle_id: int,
    method: str,
    output: str | Path,
) -> dict[str, object]:
    job = registered_job(dataset, seed_bundle_id, method)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from the fixed contract")
    if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") != "python":
        raise RuntimeError("the remote protobuf compatibility mode must be python")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each worker requires exactly one visible CUDA device")

    source_before = source_record()
    started = time.monotonic()
    device = torch.device("cuda:0")
    seeds = seed_bundle(seed_bundle_id)
    _seed_runtime(seeds.model)
    model = new_calibrator(method, model_seed=seeds.model).to(device)
    architecture = _architecture_record(model)
    if method == "mlp_value" and architecture["parameter_count"] != 5475:
        raise RuntimeError("the exact MLP control no longer has 5,475 parameters")
    if method.startswith("kan_") and not (
        0.95 * 5475 <= architecture["parameter_count"] <= 1.05 * 5475
    ):
        raise RuntimeError("KAN parameter count is outside the frozen +/-5% band")

    fallback = permitted_fit_fallback(dataset)
    source_batches = _evaluation_batches(dataset, seed_bundle_id, "source_dev")

    def loss(active: nn.Module, value: object) -> Tensor:
        if not isinstance(active, ACILBase):
            raise TypeError("training received the wrong model")
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
    if source_record() != source_before:
        raise RuntimeError("runtime source drifted during the job")

    output_path = Path(output)
    checkpoint_identity = {
        "architecture": architecture,
        "best_epoch": training.best_epoch,
        "job": job.to_json(),
        "source_tree_sha256": source_before["aggregate_sha256"],
    }
    checkpoint = save_checkpoint(
        model,
        identity=checkpoint_identity,
        directory=output_path.parent / "checkpoints",
    )
    result: dict[str, object] = {
        "architecture": architecture,
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
        "evidence_boundary": {
            "candidate_confirmation": False,
            "cohort": "tune",
            "project_wide_pristine": False,
            "role": "post-project one-shot development collision",
        },
        "fit_access": True,
        "job": job.to_json(),
        "operational": {
            "cuda_device_name": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.monotonic() - started,
            "torch_version": torch.__version__,
        },
        "protocol": PROTOCOL_ID,
        "schema": RESULT_SCHEMA,
        "selection_cohort": "source_dev",
        "source_tree_sha256": source_tree_sha256(),
        "status": "succeeded",
        "test_access": False,
        "training": {
            "best_epoch": training.best_epoch,
            "best_source_dev_nmae": training.best_source_dev_nmae,
            "epochs": [asdict(row) for row in training.epochs],
            "epochs_completed": training.epochs_completed,
            "optimizer_updates": training.optimizer_updates,
            "parameter_count": architecture["parameter_count"],
        },
    }
    write_result(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed-bundle", choices=SEED_BUNDLES, type=int, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_registered_job(
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


__all__ = ["build_parser", "run_registered_job"]
