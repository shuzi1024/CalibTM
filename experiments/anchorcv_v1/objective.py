"""Observation-only preprocessing and the frozen AnchorCV neural objective.

This module deliberately contains no interpolation call.  Missing payloads are
masked to zero before the prior-free expert, while hidden truth is kept on a
separate loss-only path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from experiments.acil_innovation_v1.preprocessing import (
    FitFallback,
    ObservationStatistics,
    observation_statistics,
)

from .model import PriorFreeMaskNativeExpert, hard_project_observations


@dataclass(frozen=True, slots=True)
class PreparedMaskNativeInput:
    visible_normalized: Tensor
    observed: Tensor
    normalized_time: Tensor
    mean: Tensor
    std: Tensor
    statistics: ObservationStatistics


@dataclass(frozen=True, slots=True)
class DroppedAnchor:
    submask: Tensor
    anchor_index: Tensor
    anchor_mask: Tensor
    rank: int


@dataclass(frozen=True, slots=True)
class RawPrediction:
    prediction: Tensor
    normalized_prediction: Tensor
    decoder_output: Tensor
    prepared: PreparedMaskNativeInput


@dataclass(frozen=True, slots=True)
class PairedObjectiveDetails:
    k3_missing_loss: Tensor
    k2_all_missing_loss: Tensor
    k3_target_count: int
    k2_target_count: int
    drop_rank: int
    submask: Tensor
    anchor_index: Tensor


def _validate_values_and_mask(values: Tensor, observed: Tensor) -> tuple[int, int, int]:
    if not isinstance(values, Tensor) or not isinstance(observed, Tensor):
        raise TypeError("values and observed must be tensors")
    if values.ndim != 3 or observed.ndim != 3 or values.shape != observed.shape:
        raise ValueError("values and observed must share shape [B,F,T]")
    if not values.is_floating_point():
        raise TypeError("values must use a floating dtype")
    if observed.dtype is not torch.bool:
        raise TypeError("observed must be boolean")
    if values.device != observed.device:
        raise ValueError("values and observed must share a device")
    if any(int(size) < 1 for size in values.shape):
        raise ValueError("values and observed must be nonempty")
    visible = values.masked_select(observed)
    if not torch.isfinite(visible).all().item():
        raise ValueError("observed values must be finite")
    if (visible < 0.0).any().item():
        raise ValueError("observed traffic values must be nonnegative")
    return tuple(int(size) for size in values.shape)  # type: ignore[return-value]


def prepare_mask_native_input(
    model_input: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> PreparedMaskNativeInput:
    """Normalize only currently observed values; every missing input is zero."""

    _, _, times = _validate_values_and_mask(model_input, observed)
    statistics = observation_statistics(model_input, observed, fit_fallback)
    visible_normalized = torch.where(
        observed,
        (model_input - statistics.mean) / statistics.std,
        torch.zeros((), dtype=model_input.dtype, device=model_input.device),
    )
    if not torch.isfinite(visible_normalized).all().item():
        raise FloatingPointError("observation-only normalization became nonfinite")
    normalized_time = torch.linspace(
        -1.0,
        1.0,
        times,
        dtype=model_input.dtype,
        device=model_input.device,
    )
    return PreparedMaskNativeInput(
        visible_normalized=visible_normalized,
        observed=observed.detach().clone(),
        normalized_time=normalized_time,
        mean=statistics.mean,
        std=statistics.std,
        statistics=statistics,
    )


def drop_observed_anchor(observed: Tensor, *, rank: int) -> DroppedAnchor:
    """Hide one of the three sorted observed anchors in every flow."""

    if not isinstance(observed, Tensor) or observed.ndim != 3:
        raise ValueError("observed must have shape [B,F,T]")
    if observed.dtype is not torch.bool:
        raise TypeError("observed must be boolean")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in {0, 1, 2}:
        raise ValueError("rank must be exactly 0, 1, or 2")
    counts = observed.sum(dim=-1)
    if not torch.equal(counts, torch.full_like(counts, 3)):
        raise ValueError("anchor dropout requires exactly three observations per flow")
    times = int(observed.shape[-1])
    positions = torch.arange(times, device=observed.device, dtype=torch.long)
    positions = positions.reshape(1, 1, times).expand_as(observed)
    sorted_positions = torch.where(
        observed, positions, torch.full_like(positions, times)
    ).sort(dim=-1).values[..., :3]
    anchor_index = sorted_positions[..., rank]
    anchor_mask = torch.zeros_like(observed)
    anchor_mask.scatter_(-1, anchor_index.unsqueeze(-1), True)
    submask = observed & ~anchor_mask
    return DroppedAnchor(
        submask=submask,
        anchor_index=anchor_index,
        anchor_mask=anchor_mask,
        rank=rank,
    )


def predict_raw(
    model: PriorFreeMaskNativeExpert,
    model_input: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> RawPrediction:
    """Run the neural expert and return a nonnegative, hard-projected result."""

    if not isinstance(model, PriorFreeMaskNativeExpert):
        raise TypeError("model must be PriorFreeMaskNativeExpert")
    prepared = prepare_mask_native_input(model_input, observed, fit_fallback)
    decoder_output = model(
        prepared.visible_normalized,
        observed,
        prepared.normalized_time,
    )
    raw_unprojected = decoder_output.float() * prepared.std + prepared.mean
    prediction = hard_project_observations(
        raw_unprojected.clamp_min(0.0),
        model_input,
        observed,
    )
    if not torch.isfinite(prediction).all().item():
        raise FloatingPointError("mask-native expert produced nonfinite predictions")
    normalized_prediction = (prediction - prepared.mean) / prepared.std
    return RawPrediction(
        prediction=prediction,
        normalized_prediction=normalized_prediction,
        decoder_output=decoder_output,
        prepared=prepared,
    )


def _normalized_mae(
    prediction: Tensor,
    truth: Tensor,
    target: Tensor,
    scale: Tensor,
) -> Tensor:
    if prediction.shape != truth.shape or target.shape != truth.shape:
        raise ValueError("prediction, truth, and target must have aligned shapes")
    if target.dtype is not torch.bool:
        raise TypeError("target must be boolean")
    if scale.shape != truth.shape[:-1] + (1,):
        raise ValueError("scale must have shape [B,F,1]")
    if not target.any().item() or (scale <= 0.0).any().item():
        raise ValueError("target must be nonempty and scale must be positive")
    selected = torch.abs((prediction - truth) / scale).masked_select(target)
    if not torch.isfinite(selected).all().item():
        raise FloatingPointError("normalized loss became nonfinite")
    return selected.mean()


def paired_mask_native_objective(
    model: PriorFreeMaskNativeExpert,
    *,
    model_input: Tensor,
    truth: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
    drop_rank: int,
) -> tuple[Tensor, PairedObjectiveDetails]:
    """Train K=3 deployment and fair middle-anchor K=2 behavior together.

    The frozen 0.50/0.50 objective exactly matches the ACIL prior's registered
    K=3 plus sorted-middle K=2 training contract.  The held middle anchor is
    already part of the complete K=2 missing complement and is not counted a
    second time.
    """

    _validate_values_and_mask(model_input, observed)
    if truth.shape != model_input.shape or not truth.is_floating_point():
        raise ValueError("truth must be a floating tensor aligned with model input")
    if truth.device != model_input.device:
        raise ValueError("truth and model input must share a device")
    if not torch.isfinite(truth).all().item() or (truth < 0.0).any().item():
        raise ValueError("loss truth must be finite and nonnegative")
    if drop_rank != 1:
        raise ValueError("formal paired objective requires sorted middle rank 1")
    dropped = drop_observed_anchor(observed, rank=drop_rank)
    submask_input = torch.where(
        dropped.submask,
        model_input,
        torch.full((), float("nan"), dtype=model_input.dtype, device=model_input.device),
    )
    k3 = predict_raw(model, model_input, observed, fit_fallback)
    k2 = predict_raw(model, submask_input, dropped.submask, fit_fallback)
    k3_target = ~observed
    k2_target = ~dropped.submask
    k3_loss = _normalized_mae(
        k3.prediction, truth, k3_target, k3.prepared.std
    )
    k2_loss = _normalized_mae(
        k2.prediction, truth, k2_target, k2.prepared.std
    )
    details = PairedObjectiveDetails(
        k3_missing_loss=k3_loss,
        k2_all_missing_loss=k2_loss,
        k3_target_count=int(k3_target.sum().item()),
        k2_target_count=int(k2_target.sum().item()),
        drop_rank=drop_rank,
        submask=dropped.submask,
        anchor_index=dropped.anchor_index,
    )
    total = 0.50 * k3_loss + 0.50 * k2_loss
    return total, details


__all__ = [
    "DroppedAnchor",
    "PairedObjectiveDetails",
    "PreparedMaskNativeInput",
    "RawPrediction",
    "drop_observed_anchor",
    "paired_mask_native_objective",
    "predict_raw",
    "prepare_mask_native_input",
]
