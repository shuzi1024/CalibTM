"""Deterministic training lane for the prior-free AnchorCV neural expert."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor

from experiments.acil_innovation_v1.batching import (
    EvaluationBatch,
    TrainingBatch,
    build_evaluation_batch,
    build_training_batch,
)
from experiments.acil_innovation_v1.preprocessing import FitFallback
from experiments.acil_innovation_v1.registries import dataset_spec
from experiments.acil_innovation_v1.training import (
    ProtocolTrainingResult,
    SourceDevResult,
    run_protocol_training,
)

from .data_access import (
    load_permitted_windows,
    permitted_fit_fallback,
)
from .model import PriorFreeMaskNativeExpert
from .objective import (
    PairedObjectiveDetails,
    paired_mask_native_objective,
    predict_raw,
)
from .protocol import load_protocol


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be boolean")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # PyTorch's fused/flash SDPA kernels are numerically stable but did not
    # reproduce bit-for-bit across the four H800 devices in the smoke audit.
    # The math backend is slower but gives this small prototype a stricter,
    # reviewable deterministic lane with ample memory headroom.
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(not deterministic)
    torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)
    torch.use_deterministic_algorithms(deterministic)


def build_neural_expert(
    dataset: str,
    *,
    num_flows_override: int | None = None,
    time_steps_override: int | None = None,
    tiny: bool = False,
) -> PriorFreeMaskNativeExpert:
    """Build the frozen formal architecture (or a test-only tiny twin)."""

    protocol = load_protocol()
    if dataset not in protocol.datasets:
        raise ValueError("dataset must be exactly abilene or geant")
    if not isinstance(tiny, bool):
        raise TypeError("tiny must be boolean")
    flows = dataset_spec(dataset).flows
    times = 50
    if num_flows_override is not None:
        if not tiny:
            raise ValueError("num_flows_override is test-only")
        flows = int(num_flows_override)
    if time_steps_override is not None:
        if not tiny:
            raise ValueError("time_steps_override is test-only")
        times = int(time_steps_override)
    if tiny:
        return PriorFreeMaskNativeExpert(
            num_flows=flows,
            time_steps=times,
            temporal_hidden=8,
            d_model=32,
            num_heads=4,
            num_flow_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
    return PriorFreeMaskNativeExpert(
        num_flows=flows,
        time_steps=times,
        temporal_hidden=128,
        d_model=256,
        num_heads=8,
        num_flow_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    )


def model_parameter_count(model: PriorFreeMaskNativeExpert) -> int:
    if not isinstance(model, PriorFreeMaskNativeExpert):
        raise TypeError("model must be PriorFreeMaskNativeExpert")
    return sum(int(parameter.numel()) for parameter in model.parameters())


def compute_batch_loss(
    model: PriorFreeMaskNativeExpert,
    batch: TrainingBatch,
    *,
    fit_fallback: FitFallback,
    device: torch.device,
) -> tuple[Tensor, PairedObjectiveDetails]:
    if not isinstance(batch, TrainingBatch):
        raise TypeError("batch must be TrainingBatch")
    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device")
    model_input = batch.model_input.to(device, non_blocking=False)
    truth = batch.truth.to(device, non_blocking=False)
    observed = batch.observed.to(device, non_blocking=False)
    return paired_mask_native_objective(
        model,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=fit_fallback,
        drop_rank=1,
    )


def evaluate_neural_error_sums(
    model: PriorFreeMaskNativeExpert,
    batch: EvaluationBatch,
    *,
    fit_fallback: FitFallback,
    device: torch.device,
    chunk_size: int,
) -> dict[str, float | int]:
    """Evaluate the fixed K=3 missing complement in bounded-memory chunks."""

    if not isinstance(batch, EvaluationBatch):
        raise TypeError("batch must be EvaluationBatch")
    if model.training:
        raise ValueError("model must be in evaluation mode")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    error_parts: list[float] = []
    truth_parts: list[float] = []
    target_count = 0
    with torch.no_grad():
        for start in range(0, int(batch.truth.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(batch.truth.shape[0]))
            truth = batch.truth[start:stop].to(device, non_blocking=False)
            observed = batch.observed[start:stop].to(device, non_blocking=False)
            model_input = batch.model_input[start:stop].to(
                device, non_blocking=False
            )
            target = ~observed
            predicted = predict_raw(
                model, model_input, observed, fit_fallback
            ).prediction
            selected_error = (predicted - truth).abs().masked_select(target)
            selected_truth = truth.abs().masked_select(target)
            if (
                not torch.isfinite(selected_error).all().item()
                or not torch.isfinite(selected_truth).all().item()
            ):
                raise FloatingPointError("evaluation operands became nonfinite")
            error_parts.append(float(selected_error.double().sum().item()))
            truth_parts.append(float(selected_truth.double().sum().item()))
            target_count += int(target.sum().item())
    absolute_error = math.fsum(error_parts)
    absolute_truth = math.fsum(truth_parts)
    if absolute_truth <= 0.0 or target_count <= 0:
        raise ValueError("evaluation target denominator must be positive")
    return {
        "absolute_error_sum": absolute_error,
        "absolute_truth_sum": absolute_truth,
        "target_count": target_count,
        "nmae": absolute_error / absolute_truth,
    }


def train_registered_expert(
    model: PriorFreeMaskNativeExpert,
    *,
    dataset: str,
    seed_bundle: int,
    device: torch.device,
) -> ProtocolTrainingResult:
    """Run the exact inherited 20-epoch/320-update registered lane."""

    protocol = load_protocol()
    if dataset not in protocol.datasets:
        raise ValueError("dataset must be exactly abilene or geant")
    if seed_bundle not in protocol.prototype_bundles + protocol.extension_bundles:
        raise ValueError("seed bundle is outside the frozen AnchorCV grid")
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be a CPU or CUDA torch.device")
    if model.num_flows != dataset_spec(dataset).flows or model.time_steps != 50:
        raise ValueError("formal model shape differs from the dataset registry")
    fit_windows = load_permitted_windows(dataset, "fit")
    source_windows = load_permitted_windows(dataset, "source_dev")
    fallback = permitted_fit_fallback(dataset)

    def epoch_batches(epoch: int):
        for batch_index in range(64):
            yield build_training_batch(
                fit_windows,
                seed_bundle=seed_bundle,
                epoch=epoch,
                batch_index=batch_index,
            )

    def compute_loss(
        current_model: torch.nn.Module, batch: TrainingBatch
    ) -> Tensor:
        if current_model is not model:
            raise RuntimeError("training callback model identity drifted")
        loss, _ = compute_batch_loss(
            model,
            batch,
            fit_fallback=fallback,
            device=device,
        )
        return loss

    def evaluate_source_dev(
        current_model: torch.nn.Module, epoch: int
    ) -> SourceDevResult:
        if current_model is not model or not 0 <= int(epoch) < 20:
            raise RuntimeError("source-dev callback identity drifted")
        rows: list[dict[str, Any]] = []
        for family in protocol.mask_families:
            evaluation_batch = build_evaluation_batch(
                source_windows,
                seed_bundle=seed_bundle,
                family=family,
            )
            sums = evaluate_neural_error_sums(
                model,
                evaluation_batch,
                fit_fallback=fallback,
                device=device,
                chunk_size=10,
            )
            rows.append(
                {
                    "epoch": epoch,
                    "mask_family": family,
                    "ae": sums["absolute_error_sum"],
                    "truth": sums["absolute_truth_sum"],
                }
            )
        return SourceDevResult(rows=tuple(rows))

    return run_protocol_training(
        model,
        pretrained_parameters=(),
        epoch_batches=epoch_batches,
        compute_loss=compute_loss,
        evaluate_source_dev=evaluate_source_dev,
        device_type=device.type,
    )


__all__ = [
    "build_neural_expert",
    "compute_batch_loss",
    "evaluate_neural_error_sums",
    "model_parameter_count",
    "seed_everything",
    "train_registered_expert",
]
