"""Separately trained path controls, with the existing no-memory model preserved.

All constructors consume the original model's complete initialization sequence
before removing disabled modules. Shared parameter names, shapes and initial
values therefore agree exactly. These are information-path controls, not
parameter-count-matched models or inference-time interventions.
"""
from __future__ import annotations

import torch
from torch import Tensor

from experiments.spin_sync_unified_v1.models import (
    MODEL_CONFIG as ORIGINAL_MODEL_CONFIG,
    SpinSyncDirectModel,
)
from experiments.sync_delta_v1.models import WIDTH

VARIANTS = ("spin_direct", "context_only", "direct_only")
MODEL_CONFIG = {
    **ORIGINAL_MODEL_CONFIG,
    "memory_enabled": False,
    "variants": {
        "spin_direct": "original spin_direct_no_memory; decoder([Df, Db, Q(H)])",
        "context_only": "SPIN context retained; decoder([0, 0, Q(H)])",
        "direct_only": "no observation encoder or SPIN; decoder([Df, Db, Q(P)])",
    },
    "query_normalization": "original L2 normalization per 32-dimensional head",
    "shared_initialization": "construct original no-memory model, then delete inactive modules",
    "capacity_matching": False,
    "task": "offline T50 imputation; bidirectional own-flow reads; no Sync memory",
}


class SpinPathControlModel(SpinSyncDirectModel):
    """Use exactly the original prediction code with a selected input path removed."""

    def __init__(self, num_flows: int, variant: str = "spin_direct"):
        if variant not in VARIANTS:
            raise ValueError(f"Expected one of {VARIANTS}, received {variant!r}")
        # Keep creation order and all RNG draws identical to historical no-memory
        # checkpoints, including the memory modules removed by the base class.
        super().__init__(num_flows, variant="spin_direct_no_memory")
        self.variant = variant
        self.context_enabled = variant != "direct_only"
        self.direct_enabled = variant != "context_only"
        if not self.context_enabled:
            del self.h_enc, self.h_norm, self.x_skip, self.context_encoder
        if not self.direct_enabled:
            del self.direct_query, self.direct_key, self.direct_fusion

    def set_execution(self, *, node_chunk=32, gradient_checkpointing=True):
        # The base constructor also calls this before inactive modules are
        # deleted. No new state, RNG draws or parameters are introduced here.
        if type(node_chunk) is not int or node_chunk < 1:
            raise ValueError("node_chunk must be a positive integer")
        if hasattr(self, "context_encoder"):
            super().set_execution(node_chunk=node_chunk,
                                  gradient_checkpointing=gradient_checkpointing)

    def _context(self, observed: Tensor, mask: Tensor, neighbors: Tensor):
        if self.context_enabled:
            return super()._context(observed, mask, neighbors)
        if not bool(self.fit_configured):
            raise RuntimeError("Configure fit-only statistics before prediction")
        if observed.ndim != 3 or observed.shape != mask.shape:
            raise ValueError("Values and mask must have shape [B,50,F]")
        batch, length, flows = observed.shape
        if length != 50 or flows != self.num_flows:
            raise ValueError("This candidate uses a full flow graph and a 50-slot window")
        if neighbors.ndim != 2 or neighbors.shape[0] != flows or not 1 <= neighbors.shape[1] <= 9:
            raise ValueError("Expected original self-plus-neighbor table [F,J], 1<=J<=9")
        clean = torch.where(mask, observed.float(),
                            torch.zeros_like(observed, dtype=torch.float32))
        z = torch.where(mask, (clean - self.fit_mean) / self.fit_scale,
                        torch.zeros_like(clean))
        u = torch.arange(length, device=observed.device,
                         dtype=torch.float32)[None, :, None] / 49.
        position = self.u_enc(u.expand(batch, -1, -1))
        # The inherited predictor will call query(position), never query(H).
        # Neither other-flow values nor graph edges enter the target prediction.
        return position, position, z

    def _direct_reads(self, z: Tensor, mask: Tensor, position: Tensor,
                      ids: Tensor, *, trace=False):
        if self.direct_enabled:
            return super()._direct_reads(z, mask, position, ids, trace=trace)
        batch, length = z.shape[:2]
        zero = z.new_zeros(batch, length, ids.numel(), WIDTH)
        aux = {}
        if trace:
            for label in ("forward", "backward"):
                aux[f"direct_read_{label}"] = zero
        return zero, zero, aux

    def predict_block(self, *args, **kwargs):
        result = super().predict_block(*args, **kwargs)
        if kwargs.get("return_aux", False) or kwargs.get("trace", False):
            raw, aux = result
            aux["query_source"] = "spin_context" if self.context_enabled else "position"
            if not self.context_enabled:
                # Avoid mislabelling a positional token as SPIN context in traces.
                aux["position_query_input"] = aux.pop("context")
            return raw, aux
        return result


def build_model(num_flows, init_seed=41001, variant="spin_direct"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        return SpinPathControlModel(num_flows, variant=variant).float()
