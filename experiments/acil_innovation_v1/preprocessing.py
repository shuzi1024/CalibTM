"""Observation-only interpolation and normalization in ``[B,F,T]`` layout."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


_SCALE_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class FitFallback:
    """Dataset scalars computed from the permitted fit cohort only."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        for name, value in (("mean", self.mean), ("std", self.std)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"fit fallback {name} must be a finite scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"fit fallback {name} must be a finite scalar")
        if float(self.mean) < 0.0:
            raise ValueError("fit fallback mean must be nonnegative")
        if float(self.std) < _SCALE_FLOOR:
            raise ValueError("fit fallback std must be at least 1e-6")


@dataclass(frozen=True, slots=True)
class ObservationStatistics:
    mean: Tensor
    std: Tensor
    all_missing: Tensor


@dataclass(frozen=True, slots=True)
class PreparedObservations:
    linear_fill: Tensor
    normalized_fill: Tensor
    observed: Tensor
    statistics: ObservationStatistics


def _validate_inputs(values: Tensor, observed: Tensor) -> tuple[int, int, int]:
    if not isinstance(values, Tensor) or not isinstance(observed, Tensor):
        raise TypeError("values and observed must be torch tensors")
    if values.ndim != 3 or observed.ndim != 3:
        raise ValueError("values and observed must have rank 3 [B,F,T]")
    if values.shape != observed.shape:
        raise ValueError("values and observed must have the same shape")
    if any(size < 1 for size in values.shape):
        raise ValueError("values and observed must be nonempty [B,F,T] tensors")
    if not values.is_floating_point():
        raise TypeError("values must use a floating dtype")
    if observed.dtype is not torch.bool:
        raise TypeError("observed mask must have dtype bool")
    if values.device != observed.device:
        raise ValueError("values and observed must share one device")
    observed_values = values.masked_select(observed)
    if not torch.isfinite(observed_values).all().item():
        raise ValueError("observed values must be finite")
    if (observed_values < 0.0).any().item():
        raise ValueError("observed traffic values must be nonnegative")
    return tuple(int(size) for size in values.shape)  # type: ignore[return-value]


def _require_fallback(value: FitFallback) -> FitFallback:
    if not isinstance(value, FitFallback):
        raise TypeError("fit_fallback must be a FitFallback")
    return value


def observation_only_linear_fill(
    values: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> Tensor:
    """Linear interpolation with nearest-edge extension and fit-only fallback.

    Values at unobserved positions are never read. They may contain NaNs or any
    finite payload without affecting the returned tensor.
    """

    _, _, times = _validate_inputs(values, observed)
    fallback = _require_fallback(fit_fallback)
    safe_values = torch.where(observed, values, torch.zeros_like(values))
    positions = torch.arange(times, device=values.device, dtype=torch.long)
    positions = positions.reshape(1, 1, times).expand_as(observed)

    previous_seed = torch.where(observed, positions, torch.full_like(positions, -1))
    previous = torch.cummax(previous_seed, dim=-1).values
    following_seed = torch.where(
        observed, positions, torch.full_like(positions, times)
    )
    following = torch.cummin(following_seed.flip(-1), dim=-1).values.flip(-1)

    previous_safe = previous.clamp(min=0)
    following_safe = following.clamp(max=times - 1)
    previous_values = safe_values.gather(-1, previous_safe)
    following_values = safe_values.gather(-1, following_safe)
    has_previous = previous >= 0
    has_following = following < times
    denominator = (following - previous).clamp(min=1).to(values.dtype)
    fraction = (positions - previous).to(values.dtype) / denominator
    interpolated = previous_values + (following_values - previous_values) * fraction
    filled = torch.where(
        has_previous & has_following,
        interpolated,
        torch.where(
            has_previous,
            previous_values,
            torch.where(has_following, following_values, torch.zeros_like(values)),
        ),
    )
    all_missing = ~observed.any(dim=-1, keepdim=True)
    fallback_mean = torch.as_tensor(
        float(fallback.mean), device=values.device, dtype=values.dtype
    )
    filled = torch.where(all_missing, fallback_mean, filled)
    # Final projection is deliberately last so observed raw bits are retained.
    return torch.where(observed, values, filled)


def observation_statistics(
    values: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> ObservationStatistics:
    """Observation-only per-flow location and stable population scale."""

    _validate_inputs(values, observed)
    fallback = _require_fallback(fit_fallback)
    counts = observed.sum(dim=-1, keepdim=True)
    all_missing = counts == 0
    safe_counts = counts.clamp(min=1).to(values.dtype)
    safe_values = torch.where(observed, values, torch.zeros_like(values))
    observed_mean = safe_values.sum(dim=-1, keepdim=True) / safe_counts
    centered = torch.where(
        observed, safe_values - observed_mean, torch.zeros_like(values)
    )
    variance = centered.square().sum(dim=-1, keepdim=True) / safe_counts
    observed_std = variance.clamp(min=0.0).sqrt()

    fallback_mean = torch.as_tensor(
        float(fallback.mean), device=values.device, dtype=values.dtype
    )
    fallback_std = torch.as_tensor(
        float(fallback.std), device=values.device, dtype=values.dtype
    )
    mean = torch.where(all_missing, fallback_mean, observed_mean)
    std = torch.where(
        all_missing | (observed_std < _SCALE_FLOOR),
        fallback_std,
        observed_std.clamp_min(_SCALE_FLOOR),
    )
    return ObservationStatistics(mean=mean, std=std, all_missing=all_missing)


def prepare_observations(
    values: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> PreparedObservations:
    """Build the complete deployable preprocessing record."""

    statistics = observation_statistics(values, observed, fit_fallback)
    linear_fill = observation_only_linear_fill(values, observed, fit_fallback)
    normalized_fill = (linear_fill - statistics.mean) / statistics.std
    if not torch.isfinite(normalized_fill).all().item():
        raise FloatingPointError("observation normalization produced nonfinite values")
    return PreparedObservations(
        linear_fill=linear_fill,
        normalized_fill=normalized_fill,
        observed=observed.detach().clone(),
        statistics=statistics,
    )


__all__ = [
    "FitFallback",
    "ObservationStatistics",
    "PreparedObservations",
    "observation_only_linear_fill",
    "observation_statistics",
    "prepare_observations",
]
