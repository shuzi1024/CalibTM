"""Vendored ACIL geometry and interpolation core.

The two modules below are an isolated, layout-preserving subset of
``Imputation/models/geoanchor_v2_modules.py`` at upstream SHA-256
``5bb37678a94e9253703d2050efa679bfbf463beb9633e0cf12544f6fbf5a1450``.
The experiment-facing wrapper in :mod:`acil` supplies normalized coordinates
and converts between ``[B,F,T]`` and the upstream ``[B,T,F]`` layout.
"""

from __future__ import annotations

import torch
from torch import nn


class ObservationGeometryExtractorV2(nn.Module):
    """Extract observation-only anchor/gap features in ``[B,T,F]`` layout."""

    FEATURE_NAMES = (
        "B_linear",
        "x_obs",
        "m",
        "d_left",
        "d_right",
        "gap_length",
        "relative_position",
        "gap_type",
        "is_edge_gap",
        "x_left_anchor",
        "x_right_anchor",
        "anchor_delta",
        "anchor_slope",
        "local_volatility",
        "interpolation_uncertainty",
        "distance_to_nearest_anchor",
    )

    def __init__(self, feature_set: str = "full", eps: float = 1e-6) -> None:
        super().__init__()
        if feature_set not in {"full", "value_only", "no_anchor_values"}:
            raise ValueError(f"unsupported ACIL feature_set {feature_set!r}")
        self.feature_set = feature_set
        self.eps = float(eps)
        self._feature_index = {
            name: index for index, name in enumerate(self.FEATURE_NAMES)
        }

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def _apply_feature_set(self, features: torch.Tensor) -> torch.Tensor:
        if self.feature_set == "full":
            return features
        out = features.clone()
        if self.feature_set == "value_only":
            keep_names = {"B_linear", "x_obs", "m", "local_volatility"}
        else:
            keep_names = set(self.FEATURE_NAMES) - {
                "x_left_anchor",
                "x_right_anchor",
                "anchor_delta",
                "anchor_slope",
            }
        keep = {self._feature_index[name] for name in keep_names}
        for index in range(out.shape[-1]):
            if index not in keep:
                out[..., index] = 0.0
        return out

    def forward(
        self,
        x_obs: torch.Tensor,
        observed_mask: torch.Tensor,
        b_linear: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(x_obs, torch.Tensor) or x_obs.ndim != 3:
            raise ValueError("ACIL x_obs must be [B,T,F]")
        if observed_mask.shape != x_obs.shape:
            raise ValueError("ACIL observed_mask must match x_obs")
        if b_linear is None:
            b_linear = x_obs
        if b_linear.shape != x_obs.shape:
            raise ValueError("ACIL b_linear must match x_obs")
        if not x_obs.is_floating_point() or not b_linear.is_floating_point():
            raise TypeError("ACIL values must be floating point")

        mask = observed_mask.to(device=x_obs.device, dtype=x_obs.dtype).clamp(0, 1)
        mask_bool = mask > 0.5
        batch, steps, flows = x_obs.shape
        dtype = x_obs.dtype
        device = x_obs.device
        source = torch.where(mask_bool, x_obs, torch.zeros_like(x_obs))
        missing = ~mask_bool
        positions = torch.arange(steps, device=device).view(1, steps, 1)
        positions = positions.expand(batch, steps, flows)

        left_seed = torch.where(mask_bool, positions, torch.full_like(positions, -1))
        left_index = torch.cummax(left_seed, dim=1).values
        left_exists = left_index >= 0
        right_seed = torch.where(
            mask_bool, positions, torch.full_like(positions, steps)
        )
        right_index = torch.flip(
            torch.cummin(torch.flip(right_seed, dims=(1,)), dim=1).values,
            dims=(1,),
        )
        right_exists = right_index < steps

        left_anchor = torch.gather(source, 1, left_index.clamp(0, steps - 1))
        right_anchor = torch.gather(source, 1, right_index.clamp(0, steps - 1))
        left_anchor = torch.where(left_exists, left_anchor, torch.zeros_like(left_anchor))
        right_anchor = torch.where(
            right_exists, right_anchor, torch.zeros_like(right_anchor)
        )

        d_left_raw = (positions - left_index).to(dtype)
        d_right_raw = (right_index - positions).to(dtype)
        d_left_raw = torch.where(
            left_exists, d_left_raw, torch.full_like(d_left_raw, float(steps))
        )
        d_right_raw = torch.where(
            right_exists, d_right_raw, torch.full_like(d_right_raw, float(steps))
        )

        internal_gap = missing & left_exists & right_exists
        left_edge_gap = missing & ~left_exists & right_exists
        right_edge_gap = missing & left_exists & ~right_exists
        all_missing_gap = missing & ~left_exists & ~right_exists
        edge_gap = left_edge_gap | right_edge_gap | all_missing_gap

        internal_length = d_left_raw + d_right_raw
        left_edge_length = (right_index + 1).to(dtype)
        right_edge_length = (steps - left_index).to(dtype)
        gap_length_raw = torch.zeros_like(x_obs)
        gap_length_raw = torch.where(internal_gap, internal_length, gap_length_raw)
        gap_length_raw = torch.where(
            left_edge_gap, left_edge_length, gap_length_raw
        )
        gap_length_raw = torch.where(
            right_edge_gap, right_edge_length, gap_length_raw
        )
        gap_length_raw = torch.where(
            all_missing_gap,
            torch.full_like(gap_length_raw, float(steps)),
            gap_length_raw,
        )

        relative_position = torch.zeros_like(x_obs)
        relative_position = torch.where(
            internal_gap,
            d_left_raw / (internal_length + self.eps),
            relative_position,
        )
        relative_position = torch.where(
            right_edge_gap, torch.ones_like(relative_position), relative_position
        )

        anchor_delta = torch.abs(right_anchor - left_anchor)
        anchor_delta = torch.where(
            left_exists & right_exists, anchor_delta, torch.zeros_like(anchor_delta)
        )
        anchor_slope = (right_anchor - left_anchor) / (gap_length_raw + self.eps)
        anchor_slope = torch.where(
            left_exists & right_exists & (gap_length_raw > 0),
            anchor_slope,
            torch.zeros_like(anchor_slope),
        )
        gap_type = torch.zeros_like(x_obs)
        gap_type = torch.where(internal_gap, torch.ones_like(gap_type), gap_type)
        gap_type = torch.where(
            left_edge_gap, torch.full_like(gap_type, 2.0), gap_type
        )
        gap_type = torch.where(
            right_edge_gap, torch.full_like(gap_type, 3.0), gap_type
        )
        gap_type = torch.where(
            all_missing_gap, torch.full_like(gap_type, 4.0), gap_type
        )

        observed_count = mask.sum(dim=1, keepdim=True)
        observed_sum = (source * mask).sum(dim=1, keepdim=True)
        observed_mean = observed_sum / observed_count.clamp_min(1.0)
        observed_var = (
            ((source - observed_mean) * mask).square().sum(dim=1, keepdim=True)
            / observed_count.clamp_min(1.0)
        )
        observed_std = torch.sqrt(observed_var + self.eps)
        full_std = b_linear.std(dim=1, unbiased=False, keepdim=True)
        local_volatility = torch.where(
            observed_count > 1, observed_std, full_std
        ).expand_as(x_obs)

        d_left = torch.where(
            left_exists,
            d_left_raw / float(max(steps - 1, 1)),
            torch.ones_like(d_left_raw),
        )
        d_right = torch.where(
            right_exists,
            d_right_raw / float(max(steps - 1, 1)),
            torch.ones_like(d_right_raw),
        )
        gap_length = gap_length_raw / float(max(steps, 1))
        distance_raw = torch.minimum(d_left_raw, d_right_raw)
        distance_raw = torch.where(
            left_exists | right_exists,
            distance_raw,
            torch.full_like(distance_raw, float(steps)),
        )
        distance = distance_raw / float(max(steps, 1))

        anchor_scale = anchor_delta.amax(dim=1, keepdim=True).clamp_min(self.eps)
        anchor_delta_norm = anchor_delta / anchor_scale
        # This is intentionally the upstream window-level scalar normalization.
        # It is flow-permutation invariant but not strictly flow-independent.
        volatility_scale = local_volatility.amax(
            dim=(1, 2), keepdim=True
        ).clamp_min(self.eps)
        volatility_norm = local_volatility / volatility_scale
        gap_component = torch.log1p(gap_length_raw) / torch.log1p(
            torch.tensor(float(max(steps, 1)), device=device, dtype=dtype)
        )
        uncertainty = (
            gap_component
            + anchor_delta_norm
            + edge_gap.to(dtype)
            + volatility_norm
            + distance
        ) / 5.0
        uncertainty = torch.where(
            mask_bool, torch.zeros_like(uncertainty), uncertainty.clamp(0, 1)
        )

        features = torch.stack(
            (
                b_linear,
                source,
                mask,
                d_left.clamp(0, 1),
                d_right.clamp(0, 1),
                gap_length.clamp(0, 1),
                relative_position.clamp(0, 1),
                (gap_type / 4.0).clamp(0, 1),
                edge_gap.to(dtype),
                torch.tanh(left_anchor),
                torch.tanh(right_anchor),
                anchor_delta_norm.clamp(0, 1),
                torch.tanh(anchor_slope),
                volatility_norm.clamp(0, 1),
                uncertainty,
                distance.clamp(0, 1),
            ),
            dim=-1,
        )
        features = self._apply_feature_set(features)
        return {
            "features": torch.nan_to_num(features),
            "x_left_anchor": left_anchor,
            "x_right_anchor": right_anchor,
            "d_left_raw": d_left_raw,
            "d_right_raw": d_right_raw,
            "gap_length_raw": gap_length_raw,
            "relative_position": relative_position,
            "gap_type": gap_type,
            "is_edge_gap": edge_gap.to(dtype),
            "left_exists": left_exists,
            "right_exists": right_exists,
            "internal_gap": internal_gap,
            "left_edge_gap": left_edge_gap,
            "right_edge_gap": right_edge_gap,
            "all_missing_gap": all_missing_gap,
            "missing": missing,
            "observed": mask_bool,
            "anchor_delta": anchor_delta,
            "anchor_slope": anchor_slope,
            "local_volatility": local_volatility,
            "uncertainty_q": uncertainty,
            "distance_to_nearest_anchor": distance_raw,
            "distance_to_nearest_anchor_norm": distance,
        }


class AnchorConditionedInterpolationLayer(nn.Module):
    """Bounded learned interpolation adjustment in normalized coordinates."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 64,
        beta_r: float = 0.25,
        beta_o: float = 0.10,
        beta_e: float = 0.10,
        use_edge_extrapolation: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden = max(8, int(hidden))
        self.eps = float(eps)
        self.beta_r = float(beta_r)
        self.beta_o = float(beta_o)
        self.beta_e = float(beta_e)
        self.use_edge_extrapolation = bool(use_edge_extrapolation)
        self.trunk = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.delta_r_head = nn.Linear(hidden, 1)
        self.offset_head = nn.Linear(hidden, 1)
        self.edge_offset_head = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for head in (
            self.delta_r_head,
            self.offset_head,
            self.edge_offset_head,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        b_linear: torch.Tensor,
        x_obs: torch.Tensor,
        observed_mask: torch.Tensor,
        geometry: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = geometry["features"].to(b_linear.dtype)
        hidden = self.trunk(features)
        delta_r = self.beta_r * torch.tanh(
            self.delta_r_head(hidden).squeeze(-1)
        )
        scale = (
            geometry["anchor_delta"].to(b_linear.dtype)
            + geometry["local_volatility"].to(b_linear.dtype)
            + self.eps
        )
        offset = self.beta_o * torch.tanh(
            self.offset_head(hidden).squeeze(-1)
        ) * scale
        edge_offset = self.beta_e * torch.tanh(
            self.edge_offset_head(hidden).squeeze(-1)
        ) * geometry["local_volatility"].to(b_linear.dtype)

        d_left = geometry["d_left_raw"].to(b_linear.dtype)
        d_right = geometry["d_right_raw"].to(b_linear.dtype)
        r_linear = d_left / (d_left + d_right + self.eps)
        r_hat = (r_linear + delta_r).clamp(0, 1)
        left = geometry["x_left_anchor"].to(b_linear.dtype)
        right = geometry["x_right_anchor"].to(b_linear.dtype)
        internal = (1 - r_hat) * left + r_hat * right + offset
        edge_anchor = torch.where(geometry["left_exists"], left, right)
        edge = (
            edge_anchor + edge_offset
            if self.use_edge_extrapolation
            else edge_anchor
        )
        out = torch.where(geometry["internal_gap"], internal, b_linear)
        out = torch.where(
            geometry["left_edge_gap"] | geometry["right_edge_gap"], edge, out
        )
        out = torch.where(geometry["all_missing_gap"], torch.zeros_like(out), out)
        out = torch.where(observed_mask > 0.5, x_obs, out)
        out = torch.nan_to_num(out)
        return out, {
            "delta_r": delta_r,
            "r_hat": r_hat,
            "offset": offset,
            "edge_offset": edge_offset,
        }


__all__ = [
    "AnchorConditionedInterpolationLayer",
    "ObservationGeometryExtractorV2",
]
