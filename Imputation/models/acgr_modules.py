"""Anchor-conditioned gap refinement with boundary-preserving basis functions.

The layer in this module intentionally follows the call contract of
``AnchorConditionedInterpolationLayer`` so it can be evaluated with the same
geometry dictionary produced by ``ObservationGeometryExtractorV2``.

ACGR predicts one coefficient vector per *gap*, rather than one correction per
token.  A boundary-zero Legendre basis turns that vector into a smooth residual
inside the gap.  Consequently, internal-gap anchors remain fixed by
construction, while edge and all-missing gaps fall back to the supplied linear
baseline.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


def boundary_zero_legendre_basis(relative_position: torch.Tensor, K: int = 4) -> torch.Tensor:
    """Return the first ``K`` boundary-zero Legendre basis functions.

    ``relative_position`` is interpreted on ``[0, 1]``.  Legendre polynomials
    are evaluated at ``z = 2r - 1`` and multiplied by ``4r(1-r)``.  Every basis
    function is therefore exactly zero at both anchors (``r=0`` and ``r=1``).

    Args:
        relative_position: Tensor of arbitrary shape containing gap-relative
            positions.
        K: Number of basis functions.  The default is four.

    Returns:
        A tensor with shape ``relative_position.shape + (K,)``.
    """

    K = int(K)
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")

    r = relative_position.clamp(0.0, 1.0)
    z = 2.0 * r - 1.0
    envelope = 4.0 * r * (1.0 - r)

    polynomials = [torch.ones_like(z)]
    if K > 1:
        polynomials.append(z)
    for degree in range(2, K):
        # Bonnet's recursion for Legendre polynomials.
        current = (
            (2 * degree - 1) * z * polynomials[-1]
            - (degree - 1) * polynomials[-2]
        ) / float(degree)
        polynomials.append(current)

    return torch.stack([envelope * value for value in polynomials[:K]], dim=-1)


class AnchorConditionedGapRefinement(nn.Module):
    """Refine internal interpolation gaps with shared smooth coefficients.

    The geometry MLP consumes only features that are constant within a gap.
    Tokens belonging to the same ``(left_anchor, right_anchor)`` pair therefore
    receive exactly the same coefficient vector without scatter operations or
    Python-side grouping.  The vector is combined with a boundary-zero
    Legendre basis.

    Edge gaps and all-missing flows are deliberately left at ``b_linear``.
    Observed values are hard-preserved from ``x_obs`` after all refinement.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        hidden: int = 67,
        K: int = 4,
        beta: float = 0.25,
        eps: float = 1e-6,
        num_basis: int | None = None,
        basis_dim: int | None = None,
    ) -> None:
        super().__init__()
        if basis_dim is not None:
            K = int(basis_dim)
        if num_basis is not None:
            K = int(num_basis)
        if int(feature_dim) <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if int(K) <= 0:
            raise ValueError(f"K must be positive, got {K}")

        hidden = max(8, int(hidden))
        self.feature_dim = int(feature_dim)
        self.K = int(K)
        self.num_basis = self.K
        self.basis_dim = self.K
        self.beta = float(beta)
        self.eps = float(eps)

        # ObservationGeometryExtractorV2 feature indices that are invariant
        # inside one anchor-delimited gap: gap length/type/edge flag, left and
        # right anchor values, anchor delta/slope, and local volatility.
        self.gap_feature_indices = (5, 7, 8, 9, 10, 11, 12, 13)
        if self.feature_dim <= max(self.gap_feature_indices):
            raise ValueError(
                f"feature_dim={self.feature_dim} cannot provide ACGR gap features "
                f"through index {max(self.gap_feature_indices)}"
            )
        self.register_buffer(
            "_gap_feature_index",
            torch.tensor(self.gap_feature_indices, dtype=torch.long),
            persistent=False,
        )

        self.trunk = nn.Sequential(
            nn.LayerNorm(len(self.gap_feature_indices)),
            nn.Linear(len(self.gap_feature_indices), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.coefficient_head = nn.Linear(hidden, self.K)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Identity startup: the initial ACGR output equals the supplied prior.
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)

    def _validate_inputs(
        self,
        b_linear: torch.Tensor,
        x_obs: torch.Tensor,
        observed_mask: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
    ) -> None:
        if b_linear.dim() != 3:
            raise ValueError(f"b_linear must be [B,T,N], got {tuple(b_linear.shape)}")
        if x_obs.shape != b_linear.shape:
            raise ValueError(
                f"x_obs shape {tuple(x_obs.shape)} does not match b_linear {tuple(b_linear.shape)}"
            )
        if observed_mask.shape != b_linear.shape:
            raise ValueError(
                "observed_mask shape "
                f"{tuple(observed_mask.shape)} does not match b_linear {tuple(b_linear.shape)}"
            )
        required = {
            "features",
            "d_left_raw",
            "d_right_raw",
            "relative_position",
            "internal_gap",
        }
        missing = sorted(required.difference(geometry))
        if missing:
            raise KeyError(f"ACGR geometry is missing required fields: {missing}")
        features = geometry["features"]
        expected = (*b_linear.shape, self.feature_dim)
        if tuple(features.shape) != expected:
            raise ValueError(f"geometry features must be {expected}, got {tuple(features.shape)}")

    def _gap_ids(self, geometry: Dict[str, torch.Tensor], shape: Tuple[int, int, int]) -> torch.Tensor:
        """Build a stable integer id for each internal gap in a window-flow."""

        batch, steps, flows = shape
        device = geometry["d_left_raw"].device
        positions = torch.arange(steps, device=device).view(1, steps, 1).expand(batch, steps, flows)
        d_left = geometry["d_left_raw"].round().to(torch.long)
        d_right = geometry["d_right_raw"].round().to(torch.long)
        left_index = (positions - d_left).clamp(0, steps - 1)
        right_index = (positions + d_right).clamp(0, steps - 1)
        gap_ids = left_index * (steps + 1) + right_index
        internal = geometry["internal_gap"].to(device=device, dtype=torch.bool)
        return torch.where(internal, gap_ids, torch.full_like(gap_ids, -1))

    @staticmethod
    def _internal_gap_starts(internal_gap: torch.Tensor) -> torch.Tensor:
        """Return a vectorized mask marking the first token of each gap."""

        previous = torch.zeros_like(internal_gap)
        previous[:, 1:, :] = internal_gap[:, :-1, :]
        return internal_gap & (~previous)

    @staticmethod
    def _compatibility_diagnostics(
        b_linear: torch.Tensor,
        out: torch.Tensor,
        observed_mask: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        residual: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Expose scalar/tensor keys consumed by existing ACIL cache code."""

        dtype = b_linear.dtype
        device = b_linear.device
        missing = observed_mask.to(device=device) <= 0.5
        internal = geometry["internal_gap"].to(device=device, dtype=torch.bool) & missing
        edge = missing & (~internal)
        zeros = torch.zeros_like(b_linear)
        safe_missing = missing.to(dtype).sum().clamp_min(1.0)
        safe_internal = internal.to(dtype).sum().clamp_min(1.0)
        safe_edge = edge.to(dtype).sum().clamp_min(1.0)
        return {
            "delta_r": zeros,
            "r_hat": geometry["relative_position"].to(device=device, dtype=dtype),
            "offset": residual,
            "edge_offset": zeros,
            "mean_abs_delta_r": b_linear.new_tensor(0.0),
            "mean_signed_delta_r": b_linear.new_tensor(0.0),
            "mean_abs_offset": (residual.abs() * internal.to(dtype)).sum() / safe_internal,
            "mean_abs_edge_offset": (zeros * edge.to(dtype)).sum() / safe_edge,
            "mean_abs_B_acil_minus_B_linear": (
                (out - b_linear).abs() * missing.to(dtype)
            ).sum()
            / safe_missing,
        }

    def _identity_result(
        self,
        b_linear: torch.Tensor,
        x_obs: torch.Tensor,
        observed_mask: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask = observed_mask.to(device=b_linear.device) > 0.5
        out = torch.where(mask, x_obs.to(device=b_linear.device, dtype=b_linear.dtype), b_linear)
        basis = boundary_zero_legendre_basis(
            geometry["relative_position"].to(device=b_linear.device, dtype=b_linear.dtype),
            self.K,
        )
        zeros = b_linear.new_zeros((*b_linear.shape, self.K))
        residual = torch.zeros_like(b_linear)
        internal = geometry["internal_gap"].to(device=b_linear.device, dtype=torch.bool)
        edge = (~mask) & (~internal)
        diagnostics = {
            "basis": basis,
            "coefficients": zeros,
            "gap_coefficients": zeros,
            "gap_ids": self._gap_ids(geometry, tuple(b_linear.shape)).to(b_linear.device),
            "residual": residual,
            "internal_gap": internal,
            "edge_fallback": edge,
            "num_internal_gaps": self._internal_gap_starts(internal & (~mask)).to(b_linear.dtype).sum(),
            "mean_abs_coefficient": b_linear.new_tensor(0.0),
            "mean_abs_residual": b_linear.new_tensor(0.0),
        }
        diagnostics.update(
            self._compatibility_diagnostics(
                b_linear,
                out,
                observed_mask,
                geometry,
                residual,
            )
        )
        return out, diagnostics

    def forward(
        self,
        b_linear: torch.Tensor,
        x_obs: torch.Tensor,
        observed_mask: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        freeze_to_linear: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply ACGR using the same inputs as the existing ACIL layer."""

        self._validate_inputs(b_linear, x_obs, observed_mask, geometry)
        if freeze_to_linear:
            return self._identity_result(b_linear, x_obs, observed_mask, geometry)

        device = b_linear.device
        dtype = b_linear.dtype
        mask = observed_mask.to(device=device) > 0.5
        internal = geometry["internal_gap"].to(device=device, dtype=torch.bool) & (~mask)
        features = geometry["features"].to(device=device, dtype=dtype)
        gap_features = torch.index_select(
            features,
            dim=-1,
            index=self._gap_feature_index.to(device),
        )
        hidden = self.trunk(gap_features)

        gap_ids = self._gap_ids(geometry, tuple(b_linear.shape)).to(device)
        coefficients = self.beta * torch.tanh(self.coefficient_head(hidden))
        coefficients = torch.where(
            internal.unsqueeze(-1),
            coefficients,
            torch.zeros_like(coefficients),
        )
        basis = boundary_zero_legendre_basis(
            geometry["relative_position"].to(device=device, dtype=dtype),
            self.K,
        )

        anchor_delta = geometry.get("anchor_delta")
        local_volatility = geometry.get("local_volatility")
        if anchor_delta is None:
            anchor_delta = torch.zeros_like(b_linear)
        else:
            anchor_delta = anchor_delta.to(device=device, dtype=dtype)
        if local_volatility is None:
            local_volatility = torch.ones_like(b_linear)
        else:
            local_volatility = local_volatility.to(device=device, dtype=dtype)
        scale = anchor_delta + local_volatility + self.eps

        residual = (coefficients * basis).sum(dim=-1) * scale / math.sqrt(float(self.K))
        residual = torch.where(internal, residual, torch.zeros_like(residual))
        missing_output = torch.nan_to_num(
            b_linear + residual,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        # Hard preservation is deliberately the final operation.  Values at
        # missing positions in x_obs are never consumed by the layer.
        out = torch.where(mask, x_obs.to(device=device, dtype=dtype), missing_output)

        edge_fallback = (~mask) & (~internal)
        internal_count = internal.to(dtype).sum().clamp_min(1.0)
        coefficient_count = internal_count * float(self.K)
        diagnostics = {
            "basis": basis,
            "gap_features": gap_features,
            "coefficients": coefficients,
            "gap_coefficients": coefficients,
            "gap_ids": gap_ids,
            "residual": residual,
            "internal_gap": internal,
            "edge_fallback": edge_fallback,
            "num_internal_gaps": self._internal_gap_starts(internal).to(dtype).sum(),
            "mean_abs_coefficient": (
                coefficients.abs() * internal.unsqueeze(-1).to(dtype)
            ).sum()
            / coefficient_count,
            "mean_abs_residual": (residual.abs() * internal.to(dtype)).sum() / internal_count,
        }
        diagnostics.update(
            self._compatibility_diagnostics(
                b_linear,
                out,
                observed_mask,
                geometry,
                residual,
            )
        )
        return out, diagnostics


# Short aliases make the class convenient for experiment manifests while the
# descriptive name remains available in papers and diagnostics.
ACGRLayer = AnchorConditionedGapRefinement
ACGRPrior = AnchorConditionedGapRefinement
AnchorConditionedGapResidualLayer = AnchorConditionedGapRefinement
AnchorConstrainedGapBasisLayer = AnchorConditionedGapRefinement


__all__ = [
    "ACGRLayer",
    "ACGRPrior",
    "AnchorConditionedGapRefinement",
    "AnchorConditionedGapResidualLayer",
    "AnchorConstrainedGapBasisLayer",
    "boundary_zero_legendre_basis",
]
