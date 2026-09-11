"""Isolated one-Q-per-flow hidden-truth DeepSets diagnostic boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .acil import ACILBase, ACILResult
from .models import DeepSetsContext, TemporalFlowEncoder, anchor_envelope
from .preprocessing import FitFallback


_FEATURE_DIM = 16
_TOKEN_INPUT_DIM = 18
_QUERY_INPUT_DIM = 18
_HIDDEN_SIZE = 768


@dataclass(frozen=True, slots=True)
class OracleSupportQ:
    """Compact hidden support: one timestamp and one value per batch/flow."""

    indices: Tensor
    values: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.indices, Tensor) or not isinstance(self.values, Tensor):
            raise TypeError("OracleSupportQ indices and values must be tensors")
        if self.indices.ndim != 2 or self.indices.dtype is not torch.int64:
            raise ValueError("OracleSupportQ indices must have shape [B,F] and dtype int64")
        if self.values.shape != (*self.indices.shape, 1):
            raise ValueError("OracleSupportQ values must have shape [B,F,1]")
        if not self.values.is_floating_point():
            raise TypeError("OracleSupportQ values must be floating point")
        if self.indices.device != self.values.device:
            raise ValueError("OracleSupportQ tensors must share one device")
        if not torch.isfinite(self.values).all().item() or (self.values < 0).any().item():
            raise ValueError("OracleSupportQ values must be finite and nonnegative")

    def support_mask(self, observed: Tensor) -> Tensor:
        """Materialize exactly one Q per flow and prove Q is currently missing."""

        if not isinstance(observed, Tensor) or observed.ndim != 3:
            raise ValueError("observed must have shape [B,F,T]")
        if observed.dtype is not torch.bool:
            raise TypeError("observed must be boolean")
        if tuple(observed.shape[:2]) != tuple(self.indices.shape):
            raise ValueError("OracleSupportQ batch/flow shape conflicts with observed")
        if observed.device != self.indices.device:
            raise ValueError("OracleSupportQ and observed must share one device")
        times = observed.shape[-1]
        if ((self.indices < 0) | (self.indices >= times)).any().item():
            raise ValueError("OracleSupportQ index is outside the time window")
        if observed.gather(-1, self.indices.unsqueeze(-1)).any().item():
            raise ValueError("OracleSupportQ must select one missing entry per flow")
        mask = torch.zeros_like(observed)
        mask.scatter_(-1, self.indices.unsqueeze(-1), True)
        if not torch.equal(
            mask.sum(dim=-1), torch.ones_like(self.indices, dtype=torch.long)
        ):
            raise RuntimeError("OracleSupportQ failed to materialize exactly one support")
        return mask


@dataclass(frozen=True, slots=True)
class TruthQFeatures:
    full_result: ACILResult
    support_mask: Tensor
    innovation_raw: Tensor
    innovation_normalized: Tensor
    envelope: Tensor


class TruthQDeepSets(nn.Module):
    """Diagnostic that exposes hidden truth only through compact Q support."""

    method = "truth_q_deepsets"

    def __init__(self, acil: ACILBase) -> None:
        super().__init__()
        if not isinstance(acil, ACILBase):
            raise TypeError("acil must be an ACILBase")
        self.acil = acil
        self.acil.requires_grad_(False)
        self.flow_encoder = TemporalFlowEncoder()
        self.deepsets = DeepSetsContext()
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(_QUERY_INPUT_DIM),
            nn.Linear(_QUERY_INPUT_DIM, 128),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(_HIDDEN_SIZE + 128),
            nn.Linear(_HIDDEN_SIZE + 128, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def gpt_backbone_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def prepare_features(
        self,
        values: Tensor,
        observed: Tensor,
        support_q: OracleSupportQ,
        fit_fallback: FitFallback,
    ) -> TruthQFeatures:
        if not isinstance(support_q, OracleSupportQ):
            raise TypeError("support_q must be an OracleSupportQ")
        if support_q.values.dtype != values.dtype or support_q.values.device != values.device:
            raise ValueError("support_q values must share values dtype and device")
        with torch.no_grad():
            # The ACIL path receives only values masked by `observed`; Q is never
            # inserted into its input mask or value tensor.
            full = self.acil(values, observed, fit_fallback)
            q_mask = support_q.support_mask(observed)
            q_base = full.prediction.gather(-1, support_q.indices.unsqueeze(-1))
            innovation_raw = support_q.values - q_base
            innovation_normalized = innovation_raw / full.statistics.std
            envelope = anchor_envelope(full.geometry, observed)
        if not torch.isfinite(innovation_normalized).all().item():
            raise FloatingPointError("oracle Q innovation is nonfinite")
        return TruthQFeatures(
            full_result=full,
            support_mask=q_mask,
            innovation_raw=innovation_raw,
            innovation_normalized=innovation_normalized,
            envelope=envelope,
        )

    @staticmethod
    def _token_inputs(features: TruthQFeatures) -> Tensor:
        full = features.full_result
        if full.features.shape[-1] != _FEATURE_DIM:
            raise ValueError("ACIL feature dimension drifted")
        batch, flows, times, _ = full.features.shape
        innovation = features.innovation_normalized.unsqueeze(2).expand(
            batch, flows, times, 1
        )
        q_indicator = features.support_mask.to(full.features.dtype).unsqueeze(-1)
        tokens = torch.cat((full.features, innovation, q_indicator), dim=-1)
        if tokens.shape[-1] != _TOKEN_INPUT_DIM:
            raise RuntimeError("oracle token input dimension drifted")
        return tokens

    @staticmethod
    def _query_inputs(features: TruthQFeatures) -> Tensor:
        full = features.full_result
        queries = torch.cat(
            (
                full.features,
                full.normalized_prediction.unsqueeze(-1),
                features.envelope.unsqueeze(-1),
            ),
            dim=-1,
        )
        if queries.shape[-1] != _QUERY_INPUT_DIM:
            raise RuntimeError("oracle query input dimension drifted")
        return queries

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        support_q: OracleSupportQ,
        fit_fallback: FitFallback,
    ) -> Tensor:
        features = self.prepare_features(values, observed, support_q, fit_fallback)
        tokens = self.flow_encoder(self._token_inputs(features))
        context = self.deepsets(tokens)
        query = self.query_encoder(self._query_inputs(features))
        context = context.unsqueeze(2).expand(-1, -1, query.shape[2], -1)
        normalized_delta = torch.tanh(
            self.decoder(torch.cat((context, query), dim=-1)).squeeze(-1)
        )
        full = features.full_result
        correction = full.statistics.std * features.envelope * normalized_delta
        missing = (full.prediction + correction).clamp_min(0.0)
        prediction = torch.where(observed, values, missing)
        if not torch.isfinite(prediction).all().item():
            raise FloatingPointError("TruthQDeepSets produced nonfinite predictions")
        return prediction


__all__ = ["OracleSupportQ", "TruthQDeepSets", "TruthQFeatures"]
