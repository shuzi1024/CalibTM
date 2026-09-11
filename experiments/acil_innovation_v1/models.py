"""ACIL-innovation local, DeepSets, and noncausal GPT-set residual models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .acil import ACILBase, LOOInnovation, middle_anchor_leave_one_out
from .preprocessing import FitFallback


MODEL_METHODS = (
    "local",
    "deepsets",
    "full_u0",
    "gpt2_set",
    "gpt2_scratch",
)
_FEATURE_DIM = 16
_TOKEN_INPUT_DIM = _FEATURE_DIM + 2
_QUERY_INPUT_DIM = _FEATURE_DIM + 2
_HIDDEN_SIZE = 768
_ATTENTION_HEADS = 12
_GPT_DEPTH = 6


@dataclass(frozen=True, slots=True)
class ModelFeatures:
    loo: LOOInnovation
    envelope: Tensor


def anchor_envelope(
    geometry: Mapping[str, Tensor], observed: Tensor
) -> Tensor:
    """Return a fixed [0,1] correction envelope that vanishes at anchors."""

    if not isinstance(observed, Tensor) or observed.ndim != 3:
        raise ValueError("observed must be a boolean [B,F,T] tensor")
    if observed.dtype is not torch.bool:
        raise TypeError("observed must be boolean")
    required = (
        "relative_position",
        "internal_gap",
        "left_edge_gap",
        "right_edge_gap",
        "d_left_raw",
        "d_right_raw",
    )
    if any(name not in geometry for name in required):
        raise ValueError("ACIL geometry is missing envelope fields")
    for name in required:
        if geometry[name].shape != observed.shape:
            raise ValueError(f"ACIL geometry field {name!r} has the wrong shape")

    dtype = geometry["relative_position"].dtype
    internal = geometry["internal_gap"].to(torch.bool)
    left_edge = geometry["left_edge_gap"].to(torch.bool)
    right_edge = geometry["right_edge_gap"].to(torch.bool)
    relative = geometry["relative_position"].to(dtype)
    internal_value = 4.0 * relative * (1.0 - relative)

    d_right = geometry["d_right_raw"].to(dtype)
    d_left = geometry["d_left_raw"].to(dtype)
    left_span = torch.where(left_edge, d_right, torch.zeros_like(d_right)).amax(
        dim=-1, keepdim=True
    )
    right_span = torch.where(
        right_edge, d_left, torch.zeros_like(d_left)
    ).amax(dim=-1, keepdim=True)
    left_value = d_right / left_span.clamp_min(1.0)
    right_value = d_left / right_span.clamp_min(1.0)

    envelope = torch.zeros_like(relative)
    envelope = torch.where(internal, internal_value, envelope)
    envelope = torch.where(left_edge, left_value, envelope)
    envelope = torch.where(right_edge, right_value, envelope)
    envelope = torch.where(observed, torch.zeros_like(envelope), envelope)
    return envelope.clamp(0.0, 1.0)


class TemporalFlowEncoder(nn.Module):
    """Shared per-flow encoder; no flow identity or cross-flow operation."""

    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(_TOKEN_INPUT_DIM),
            nn.Linear(_TOKEN_INPUT_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, _HIDDEN_SIZE),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if (
            not isinstance(inputs, Tensor)
            or inputs.ndim != 4
            or inputs.shape[-1] != _TOKEN_INPUT_DIM
        ):
            raise ValueError(
                f"flow encoder inputs must have shape [B,F,T,{_TOKEN_INPUT_DIM}]"
            )
        if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
            raise ValueError("flow encoder inputs must be finite floating point")
        hidden = self.point_encoder(inputs)
        pooled = torch.cat((hidden.mean(dim=2), hidden.amax(dim=2)), dim=-1)
        return self.projection(pooled)


class DeepSetsContext(nn.Module):
    """Permutation-equivariant all-flow mean aggregation."""

    def __init__(self) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.LayerNorm(_HIDDEN_SIZE),
            nn.Linear(_HIDDEN_SIZE, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(_HIDDEN_SIZE + 256),
            nn.Linear(_HIDDEN_SIZE + 256, _HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != _HIDDEN_SIZE:
            raise ValueError("DeepSets tokens must have shape [B,F,768]")
        global_summary = self.phi(tokens).mean(dim=1, keepdim=True)
        global_summary = global_summary.expand(-1, tokens.shape[1], -1)
        return self.rho(torch.cat((tokens, global_summary), dim=-1))


def _retained_submodule(source: nn.Module, name: str) -> nn.Module:
    if hasattr(source, name):
        value = getattr(source, name)
    elif hasattr(source, "attn") and hasattr(source.attn, name):
        value = getattr(source.attn, name)
    else:
        raise ValueError(f"retained GPT-2 block is missing {name!r}")
    if not isinstance(value, nn.Module):
        raise TypeError(f"retained GPT-2 component {name!r} is not a module")
    return copy.deepcopy(value)


class NonCausalGPT2SetBlock(nn.Module):
    """Retained GPT-2 attention/MLP with full noncausal set attention."""

    def __init__(self, retained_block: nn.Module) -> None:
        super().__init__()
        if not isinstance(retained_block, nn.Module):
            raise TypeError("retained_block must be a torch module")
        self.ln_1 = _retained_submodule(retained_block, "ln_1")
        self.c_attn = _retained_submodule(retained_block, "c_attn")
        self.c_proj = _retained_submodule(retained_block, "c_proj")
        self.attn_dropout = _retained_submodule(retained_block, "attn_dropout")
        self.resid_dropout = _retained_submodule(retained_block, "resid_dropout")
        self.ln_2 = _retained_submodule(retained_block, "ln_2")
        self.mlp = _retained_submodule(retained_block, "mlp")
        self._head_size = _HIDDEN_SIZE // _ATTENTION_HEADS

    def forward(self, x: Tensor, valid_flows: Tensor) -> Tensor:
        if (
            not isinstance(x, Tensor)
            or x.ndim != 3
            or x.shape[-1] != _HIDDEN_SIZE
            or not x.is_floating_point()
            or not torch.isfinite(x).all()
        ):
            raise ValueError("set block x must be finite floating point [B,F,768]")
        batch, flows, _ = x.shape
        if (
            not isinstance(valid_flows, Tensor)
            or valid_flows.shape != (batch, flows)
            or valid_flows.dtype is not torch.bool
            or valid_flows.device != x.device
        ):
            raise ValueError("valid_flows must be boolean [B,F] on x device")
        if not valid_flows.any(dim=-1).all():
            raise ValueError("every batch row must contain a valid flow")

        normalized = self.ln_1(x)
        qkv = self.c_attn(normalized)
        if qkv.shape != (batch, flows, 3 * _HIDDEN_SIZE):
            raise RuntimeError("retained c_attn produced noncanonical QKV")
        query, key, value = qkv.split(_HIDDEN_SIZE, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch, flows, _ATTENTION_HEADS, self._head_size
            ).permute(0, 2, 1, 3)

        query, key, value = heads(query), heads(key), heads(value)
        logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
        logits = logits / math.sqrt(float(self._head_size))
        logits = logits.masked_fill(
            ~valid_flows[:, None, None, :], float("-inf")
        )
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        weights = self.attn_dropout(weights)
        context = torch.matmul(weights, value.float()).to(value.dtype)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch, flows, _HIDDEN_SIZE)
        hidden = x + self.resid_dropout(self.c_proj(context))
        hidden = hidden + self.mlp(self.ln_2(hidden))
        return hidden.masked_fill(~valid_flows.unsqueeze(-1), 0.0)


def _require_blocks(
    retained_blocks: Sequence[nn.Module] | Iterable[nn.Module] | None,
) -> tuple[nn.Module, ...]:
    if retained_blocks is None:
        raise ValueError("GPT-set methods require six retained GPT-2 blocks")
    try:
        blocks = tuple(retained_blocks)
    except TypeError as exc:
        raise TypeError("retained_blocks must be iterable") from exc
    if len(blocks) != _GPT_DEPTH:
        raise ValueError("GPT-set methods require exactly six retained GPT-2 blocks")
    if any(not isinstance(block, nn.Module) for block in blocks):
        raise TypeError("every retained GPT-2 block must be a module")
    if len({id(block) for block in blocks}) != len(blocks):
        raise ValueError("retained GPT-2 blocks must be distinct objects")
    return blocks


class QueryResidualModel(nn.Module):
    """Shared ACIL-coordinate residual model for all registered mechanisms."""

    def __init__(
        self,
        method: str,
        acil: ACILBase,
        *,
        retained_blocks: Sequence[nn.Module] | Iterable[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if method not in MODEL_METHODS:
            raise ValueError(f"unknown residual method {method!r}")
        if not isinstance(acil, ACILBase):
            raise TypeError("acil must be an ACILBase")
        self.method = method
        self.acil = acil
        self.acil.requires_grad_(False)
        self.flow_encoder = TemporalFlowEncoder()
        self.deepsets = DeepSetsContext() if method == "deepsets" else None
        if method in {"full_u0", "gpt2_set", "gpt2_scratch"}:
            sources = _require_blocks(retained_blocks)
            self.set_blocks = nn.ModuleList(
                NonCausalGPT2SetBlock(block) for block in sources
            )
        else:
            if retained_blocks is not None and tuple(retained_blocks):
                raise ValueError("local and deepsets methods do not accept GPT blocks")
            self.set_blocks = nn.ModuleList()
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
        return tuple(
            parameter
            for parameter in self.set_blocks.parameters()
            if parameter.requires_grad
        )

    def set_gpt_backbone_trainable(self, trainable: bool) -> "QueryResidualModel":
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be boolean")
        self.set_blocks.requires_grad_(trainable)
        return self

    def prepare_features(
        self,
        values: Tensor,
        observed: Tensor,
        fit_fallback: FitFallback,
    ) -> ModelFeatures:
        with torch.no_grad():
            loo = middle_anchor_leave_one_out(
                self.acil, values, observed, fit_fallback
            )
            envelope = anchor_envelope(loo.full_result.geometry, observed)
        return ModelFeatures(loo=loo, envelope=envelope)

    def token_inputs(self, features: ModelFeatures) -> Tensor:
        if not isinstance(features, ModelFeatures):
            raise TypeError("features must be ModelFeatures")
        loo = features.loo
        batch, flows, times, channels = loo.submask_result.features.shape
        if channels != _FEATURE_DIM:
            raise ValueError("submask ACIL feature dimension drifted")
        innovation = loo.innovation_normalized
        if self.method == "full_u0":
            innovation = torch.zeros_like(innovation)
        innovation = innovation.unsqueeze(2).expand(batch, flows, times, 1)
        middle = torch.zeros(
            batch,
            flows,
            times,
            dtype=loo.submask_result.features.dtype,
            device=loo.submask_result.features.device,
        )
        middle.scatter_(-1, loo.middle_index.unsqueeze(-1), 1.0)
        return torch.cat(
            (loo.submask_result.features, innovation, middle.unsqueeze(-1)),
            dim=-1,
        )

    def _context(self, token_inputs: Tensor) -> Tensor:
        tokens = self.flow_encoder(token_inputs)
        if self.deepsets is not None:
            return self.deepsets(tokens)
        if self.set_blocks:
            valid = torch.ones(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            )
            for block in self.set_blocks:
                tokens = block(tokens, valid)
        return tokens

    def _query_inputs(self, features: ModelFeatures) -> Tensor:
        full = features.loo.full_result
        if full.features.shape[-1] != _FEATURE_DIM:
            raise ValueError("full ACIL feature dimension drifted")
        return torch.cat(
            (
                full.features,
                full.normalized_prediction.unsqueeze(-1),
                features.envelope.unsqueeze(-1),
            ),
            dim=-1,
        )

    def _decode(
        self,
        *,
        features: ModelFeatures,
        token_inputs: Tensor,
        values: Tensor,
        observed: Tensor,
    ) -> Tensor:
        context = self._context(token_inputs)
        query = self.query_encoder(self._query_inputs(features))
        expanded_context = context.unsqueeze(2).expand(
            -1, -1, query.shape[2], -1
        )
        normalized_delta = torch.tanh(
            self.decoder(torch.cat((expanded_context, query), dim=-1)).squeeze(-1)
        )
        full = features.loo.full_result
        correction = (
            full.statistics.std * features.envelope * normalized_delta
        )
        raw = full.prediction + correction
        missing_prediction = raw.clamp_min(0.0)
        prediction = torch.where(observed, values, missing_prediction)
        if not torch.isfinite(prediction).all().item():
            raise FloatingPointError("residual model produced nonfinite predictions")
        return prediction

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        fit_fallback: FitFallback,
    ) -> Tensor:
        features = self.prepare_features(values, observed, fit_fallback)
        return self._decode(
            features=features,
            token_inputs=self.token_inputs(features),
            values=values,
            observed=observed,
        )

    def forward_with_innovation(
        self,
        values: Tensor,
        observed: Tensor,
        fit_fallback: FitFallback,
        *,
        innovation_normalized: Tensor,
    ) -> Tensor:
        """Run the fixed Stage-I diagnostic with an explicit scalar innovation."""

        if self.method != "deepsets":
            raise ValueError("explicit innovation is registered only for DeepSets")
        features = self.prepare_features(values, observed, fit_fallback)
        expected = features.loo.innovation_normalized.shape
        if (
            not isinstance(innovation_normalized, Tensor)
            or innovation_normalized.shape != expected
            or innovation_normalized.device != values.device
            or innovation_normalized.dtype != values.dtype
            or not torch.isfinite(innovation_normalized).all().item()
        ):
            raise ValueError(
                "innovation_normalized must be finite [B,F,1] on the input dtype/device"
            )
        tokens = self.token_inputs(features).clone()
        tokens[..., -2] = innovation_normalized.expand(-1, -1, tokens.shape[2])
        return self._decode(
            features=features,
            token_inputs=tokens,
            values=values,
            observed=observed,
        )


__all__ = [
    "MODEL_METHODS",
    "ModelFeatures",
    "NonCausalGPT2SetBlock",
    "QueryResidualModel",
    "anchor_envelope",
]
