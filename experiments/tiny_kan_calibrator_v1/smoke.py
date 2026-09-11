"""Data-free CUDA arithmetic and memory smoke for the frozen tiny-KAN arms."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from experiments.acil_innovation_v1.config import CUBLAS_WORKSPACE_CONFIG
from experiments.acil_innovation_v1.preprocessing import FitFallback

from .model import METHODS, count_parameters, new_calibrator
from .result_io import write_exclusive
from .source_identity import source_tree_sha256


SMOKE_SCHEMA = "tiny-kan-calibrator-v1:cuda-smoke:v1"


def run_smoke(output: str | Path) -> dict[str, object]:
    """Run one synthetic optimizer update per arm without opening any dataset."""

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from the fixed contract")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the smoke requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(2026081107)
    batch, flows, steps = 8, 462, 50
    truth = torch.rand((batch, flows, steps), generator=generator) * 4.0 + 0.25
    observed = torch.zeros((batch, flows, steps), dtype=torch.bool)
    observed[..., (0, 24, 49)] = True
    payload = torch.where(observed, truth, torch.full_like(truth, float("nan")))
    payload = payload.to(device)
    observed = observed.to(device)
    truth = truth.to(device)
    fallback = FitFallback(mean=2.0, std=1.5)

    rows: dict[str, object] = {}
    for index, method in enumerate(METHODS):
        torch.manual_seed(41001 + index)
        model = new_calibrator(method, model_seed=41001).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(payload, observed, fallback)
            missing_prediction = result.prediction.masked_select(~observed)
            missing_truth = truth.masked_select(~observed)
            loss = torch.mean(torch.abs(missing_prediction - missing_truth))
        if not torch.isfinite(loss):
            raise RuntimeError(f"{method} produced a non-finite loss")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not gradients or any(
            gradient is None or not torch.isfinite(gradient).all()
            for gradient in gradients
        ):
            raise RuntimeError(f"{method} produced an invalid gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        observed_error = (
            result.prediction.masked_select(observed)
            - payload.masked_select(observed)
        ).abs().max()
        negative_count = int(
            (result.prediction.masked_select(~observed) < 0).sum().item()
        )
        if float(observed_error.item()) != 0.0 or negative_count != 0:
            raise RuntimeError(f"{method} violated output feasibility")
        peak = int(torch.cuda.max_memory_reserved(device))
        if peak <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"{method} produced invalid operational telemetry")
        rows[method] = {
            "elapsed_seconds": elapsed,
            "gradient_tensors": len(gradients),
            "loss_finite": True,
            "negative_output_count": negative_count,
            "observed_hard_copy_max_error": float(observed_error.item()),
            "parameter_count": count_parameters(model),
            "peak_memory_reserved_bytes": peak,
        }
        del optimizer, model, result, loss, gradients
        torch.cuda.empty_cache()

    report = {
        "cuda_device_name": torch.cuda.get_device_name(0),
        "data_access": False,
        "methods": rows,
        "schema": SMOKE_SCHEMA,
        "shape": [batch, flows, steps],
        "source_tree_sha256": source_tree_sha256(),
        "test_access": False,
    }
    write_exclusive(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_smoke(args.output)
    print(
        json.dumps(
            {
                "data_access": report["data_access"],
                "output": str(args.output),
                "status": "succeeded",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SMOKE_SCHEMA", "build_parser", "run_smoke"]
