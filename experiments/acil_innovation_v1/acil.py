"""Normalized-coordinate ACIL base and legal middle-anchor LOO innovation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .acil_core import (
    AnchorConditionedInterpolationLayer,
    ObservationGeometryExtractorV2,
)
from .preprocessing import (
    FitFallback,
    ObservationStatistics,
    prepare_observations,
)


@dataclass(frozen=True, slots=True)
class ACILResult:
    prediction: Tensor
    normalized_prediction: Tensor
    linear_fill: Tensor
    features: Tensor
    geometry: Mapping[str, Tensor]
    statistics: ObservationStatistics


@dataclass(frozen=True, slots=True)
class LOOInnovation:
    full_result: ACILResult
    submask_result: ACILResult
    submask: Tensor
    middle_index: Tensor
    innovation_raw: Tensor
    innovation_normalized: Tensor


def _bft_geometry(
    geometry: Mapping[str, Tensor], batch: int, times: int, flows: int
) -> dict[str, Tensor]:
    converted: dict[str, Tensor] = {}
    for name, value in geometry.items():
        if value.ndim >= 3 and tuple(value.shape[:3]) == (batch, times, flows):
            if value.ndim == 3:
                converted[name] = value.permute(0, 2, 1).contiguous()
            elif value.ndim == 4:
                converted[name] = value.permute(0, 2, 1, 3).contiguous()
            else:  # no current ACIL geometry has rank > 4
                raise RuntimeError(f"unsupported ACIL geometry rank for {name!r}")
        else:
            converted[name] = value
    return converted


class ACILBase(nn.Module):
    """ACIL prior operating on normalized observation-only ``[B,F,T]`` inputs."""

    def __init__(
        self,
        *,
        feature_set: str = "full",
        hidden: int = 64,
        beta_r: float = 0.25,
        beta_o: float = 0.10,
        beta_e: float = 0.10,
        use_edge_extrapolation: bool = True,
    ) -> None:
        super().__init__()
        self.extractor = ObservationGeometryExtractorV2(feature_set=feature_set)
        self.interpolator = AnchorConditionedInterpolationLayer(
            feature_dim=self.extractor.feature_dim,
            hidden=hidden,
            beta_r=beta_r,
            beta_o=beta_o,
            beta_e=beta_e,
            use_edge_extrapolation=use_edge_extrapolation,
        )

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        fit_fallback: FitFallback,
    ) -> ACILResult:
        prepared = prepare_observations(values, observed, fit_fallback)
        statistics = prepared.statistics
        normalized_observed = torch.where(
            observed,
            (values - statistics.mean) / statistics.std,
            torch.zeros_like(values),
        )
        batch, flows, times = values.shape
        b_linear = prepared.normalized_fill.permute(0, 2, 1).contiguous()
        x_obs = normalized_observed.permute(0, 2, 1).contiguous()
        mask_btf = observed.permute(0, 2, 1).contiguous()
        geometry_btf = self.extractor(x_obs, mask_btf, b_linear=b_linear)
        normalized_btf, _ = self.interpolator(
            b_linear, x_obs, mask_btf, geometry_btf
        )
        normalized = normalized_btf.permute(0, 2, 1).contiguous()
        raw = normalized * statistics.std + statistics.mean
        missing = raw.clamp_min(0.0)
        # Deliberately final: raw observed traffic retains the exact source bits.
        prediction = torch.where(observed, values, missing)
        if not torch.isfinite(prediction).all().item():
            raise FloatingPointError("ACIL produced nonfinite raw predictions")
        normalized_prediction = (prediction - statistics.mean) / statistics.std
        geometry = _bft_geometry(geometry_btf, batch, times, flows)
        return ACILResult(
            prediction=prediction,
            normalized_prediction=normalized_prediction,
            linear_fill=prepared.linear_fill,
            features=geometry["features"],
            geometry=geometry,
            statistics=statistics,
        )


def middle_anchor_leave_one_out(
    acil: ACILBase,
    values: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> LOOInnovation:
    """Remove only the sorted middle of exactly three observed anchors per flow."""

    if not isinstance(acil, ACILBase):
        raise TypeError("acil must be an ACILBase")
    # ACIL validates the remaining tensor contract; this check fixes LOO semantics.
    if not isinstance(observed, Tensor) or observed.ndim != 3:
        raise ValueError("observed must be a boolean [B,F,T] tensor")
    if observed.dtype is not torch.bool:
        raise TypeError("observed must be boolean")
    counts = observed.sum(dim=-1)
    if not torch.equal(counts, torch.full_like(counts, 3)):
        raise ValueError("middle-anchor LOO requires exactly three observations per flow")

    full_result = acil(values, observed, fit_fallback)
    times = observed.shape[-1]
    positions = torch.arange(times, device=observed.device, dtype=torch.long)
    positions = positions.reshape(1, 1, times).expand_as(observed)
    observed_positions = torch.where(
        observed, positions, torch.full_like(positions, times)
    )
    sorted_positions = observed_positions.sort(dim=-1).values[..., :3]
    middle_index = sorted_positions[..., 1]
    submask = observed.detach().clone()
    submask.scatter_(-1, middle_index.unsqueeze(-1), False)
    submask_result = acil(values, submask, fit_fallback)
    middle_truth = values.gather(-1, middle_index.unsqueeze(-1))
    middle_prediction = submask_result.prediction.gather(
        -1, middle_index.unsqueeze(-1)
    )
    innovation_raw = middle_truth - middle_prediction
    innovation_normalized = innovation_raw / full_result.statistics.std
    if not torch.isfinite(innovation_normalized).all().item():
        raise FloatingPointError("LOO innovation is nonfinite")
    return LOOInnovation(
        full_result=full_result,
        submask_result=submask_result,
        submask=submask,
        middle_index=middle_index,
        innovation_raw=innovation_raw,
        innovation_normalized=innovation_normalized,
    )


__all__ = [
    "ACILBase",
    "ACILResult",
    "LOOInnovation",
    "middle_anchor_leave_one_out",
]
