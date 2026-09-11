"""Deterministic fit-only forward/backward smoke; it never loads gate windows."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.batching import build_training_batch
from experiments.acil_innovation_v1.config import CUBLAS_WORKSPACE_CONFIG
from experiments.acil_innovation_v1.data import fit_fallback
from experiments.acil_innovation_v1.model_factory import seed_runtime
from experiments.acil_innovation_v1.registries import seed_bundle as registered_seed_bundle

from .data import load_windows
from .execution import _residual_training_loss
from .freeze import load_and_verify_freeze
from .model import new_sc_acil_model
from .protocol import canonical_json_bytes


def tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"sc-acil-v1:smoke-tensors:v1\x00")
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        for field in (
            name.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
            tensor.view(torch.uint8).numpy().tobytes(),
        ):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def _repeat(method: str, device: torch.device) -> dict[str, object]:
    seeds = registered_seed_bundle(4)
    # ACILBase owns learned parameters, so the temporary smoke base must be
    # constructed inside the same deterministic seed boundary as the residual.
    seed_runtime(seeds.model)
    model = new_sc_acil_model(
        method, acil=ACILBase(), model_seed=seeds.model
    ).to(device)
    batch = build_training_batch(
        load_windows("geant", "fit"), seed_bundle=4, epoch=0, batch_index=0
    )
    model_input = batch.model_input.to(device)
    truth = batch.truth.to(device)
    observed = batch.observed.to(device)
    fallback = fit_fallback("geant")
    model.train()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        prediction = model(model_input, observed, fallback)
        loss = _residual_training_loss(
            model,
            model_input=model_input,
            truth=truth,
            observed=observed,
            fit_fallback=fallback,
        )
    loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    if not gradients:
        raise RuntimeError("smoke produced no gradients")
    return {
        "loss": float(loss.detach().float().cpu().item()),
        "prediction_sha256": tensor_digest({"prediction": prediction.detach()}),
        "gradient_sha256": tensor_digest(gradients),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def run_smoke(output_root: Path, *, device_name: str) -> Path:
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("smoke device must be cpu or cuda")
    freeze = load_and_verify_freeze()
    if device_name == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
            raise RuntimeError("CUDA smoke requires the deterministic CUBLAS contract")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA smoke requires exactly one visible GPU")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    methods = {}
    for method in ("sc_acil_u0", "sc_acil"):
        first = _repeat(method, device)
        second = _repeat(method, device)
        if first != second:
            raise RuntimeError(f"{method} deterministic smoke repeat drifted")
        methods[method] = first
    if methods["sc_acil"]["trainable_parameter_count"] != methods["sc_acil_u0"]["trainable_parameter_count"]:
        raise RuntimeError("matched-capacity smoke contract drifted")
    payload = {
        "schema": "sc-acil-v1:deterministic-smoke:v1",
        "manifest_sha256": freeze["manifest_sha256"],
        "device": device_name,
        "dataset": "geant",
        "cohort": "fit",
        "seed_bundle": 4,
        "epoch": 0,
        "batch_index": 0,
        "methods": methods,
        "gate_access": False,
        "sealed_test_access": False,
    }
    directory = Path(output_root) / "_smoke"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{device_name}.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
    return path


__all__ = ["run_smoke", "tensor_digest"]
