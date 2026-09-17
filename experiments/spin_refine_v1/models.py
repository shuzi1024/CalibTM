"""Matched one/two-pass SPIN+Direct with an explicit observation-source flag.

The second pass is a fixed, shared-weight refinement of the first prediction.
Only its context path receives estimates. Direct always reads immutable true
observations. This is same-resolution refinement inspired by ARI, not ARI's
multi-resolution autoregressive recipe or an LLM implementation.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.spin_sync_unified_v1.models import SpinSyncDirectModel
from experiments.sync_delta_v1.models import HEADS, HEAD_DIM

VARIANTS = ("one_step", "two_step")
MODEL_CONFIG = {
    "backbone": "one SPIN context block plus bidirectional real-observation Direct",
    "variants": {"one_step": 1, "two_step": 2},
    "shared_parameters": "all parameters shared between passes",
    "context_inputs": ["fit-standardized available value", "original observation flag"],
    "pass1_context_sources": "original observed positions only",
    "pass2_context_sources": "all positions: original observations or pass1 estimates",
    "between_passes": "clamp estimates at zero, restore exact original observations; no detach",
    "direct_sources": "original observed positions and original values in both passes",
    "loss": "final-pass missing-entry raw loss only; supplied by runner",
    "new_parameters": "32 zero-initialized observation-flag input weights",
    "task": "offline full-window imputation; no cross-window state",
}


class SpinRefineModel(SpinSyncDirectModel):
    def __init__(self, num_flows: int, variant: str = "two_step"):
        if variant not in VARIANTS:
            raise ValueError("Expected one_step or two_step")
        super().__init__(num_flows, variant="spin_direct_no_memory")
        self.variant = variant
        self.refinement_steps = 1 if variant == "one_step" else 2
        # Build the original skeleton first: every old parameter remains exactly
        # seeded as before. Extend just one input projection by a provenance flag.
        original = self.h_enc.mlp[0].layer[0]
        extended = nn.Linear(2, original.out_features, bias=original.bias is not None)
        with torch.no_grad():
            extended.weight[:, :1].copy_(original.weight)
            extended.weight[:, 1:].zero_()
            if original.bias is not None:
                extended.bias.copy_(original.bias)
        self.h_enc.mlp[0].layer[0] = extended

    def _validate_inputs(self, observed: Tensor, mask: Tensor, neighbors: Tensor):
        if not bool(self.fit_configured):
            raise RuntimeError("Configure fit-only statistics before prediction")
        if observed.ndim != 3 or observed.shape != mask.shape:
            raise ValueError("Values and mask must have shape [B,50,F]")
        if observed.shape[1:] != (50, self.num_flows):
            raise ValueError("This model requires a full flow graph and a 50-slot window")
        if neighbors.ndim != 2 or neighbors.shape[0] != self.num_flows or not 1 <= neighbors.shape[1] <= 9:
            raise ValueError("Expected self-plus-neighbor table [F,J], 1<=J<=9")
        if torch.any(neighbors < -1) or torch.any(neighbors >= self.num_flows):
            raise ValueError("Neighbor indices must be -1 or valid flow indices")

    def _normalize_available(self, values: Tensor, available: Tensor):
        # Clear hidden payloads before arithmetic, including NaN/Inf hidden truth.
        clean = torch.where(available, values.float(), torch.zeros_like(values, dtype=torch.float32))
        return torch.where(available, (clean - self.fit_mean) / self.fit_scale,
                           torch.zeros_like(clean))

    def _context_from_values(self, values: Tensor, true_mask: Tensor,
                             available: Tensor, neighbors: Tensor, position: Tensor):
        z = self._normalize_available(values, available)
        inputs = torch.stack((z, true_mask.to(z.dtype)), dim=-1)
        h = self.h_enc(inputs) + position
        h = self.h_norm(torch.where(available.unsqueeze(-1), h, position))
        h = h + self.x_skip(z.unsqueeze(-1)) * available.unsqueeze(-1)
        ids = torch.arange(self.num_flows, device=values.device)
        incoming = neighbors.masked_fill(neighbors == ids[:, None], -1)
        return self.context_encoder(h, incoming, mask=available.unsqueeze(-1))

    def _decode(self, context: Tensor, forward_read: Tensor, backward_read: Tensor,
                ids: Tensor):
        batch, length = context.shape[:2]
        query = F.normalize(self.query(context.index_select(2, ids)).reshape(
            batch, length, ids.numel(), HEADS, HEAD_DIM), dim=-1, eps=1e-6)
        standardized = self.decoder(torch.cat((forward_read, backward_read,
                                                query.flatten(-2)), dim=-1)).squeeze(-1)
        return self.fit_mean + self.fit_scale * standardized, query, standardized

    @staticmethod
    def _project_context(first_raw: Tensor, observed: Tensor, true_mask: Tensor):
        """Projection preserves real observations and carries gradients through estimates."""
        return torch.where(true_mask, observed.float(), first_raw.clamp_min(0))

    def predict_block(self, x_observed, mask, stats, neighbors, target_ids,
                      serial_order_context=None, *, return_aux=False, trace=False,
                      intervention=None):
        if intervention is not None or serial_order_context is not None:
            raise ValueError("Use independently trained variants; inference interventions are unsupported")
        device = x_observed.device
        true_mask = mask.to(device=device, dtype=torch.bool)
        neighbors = torch.as_tensor(neighbors, device=device, dtype=torch.long)
        ids = torch.as_tensor(target_ids, device=device, dtype=torch.long)
        if ids.ndim != 1 or not ids.numel() or torch.any(ids < 0) or torch.any(ids >= self.num_flows):
            raise ValueError("target_ids must be a nonempty vector of valid flow indices")
        self._validate_inputs(x_observed, true_mask, neighbors)
        collect = return_aux or trace
        with torch.autocast(device_type=device.type, enabled=False):
            batch, length = x_observed.shape[:2]
            all_ids = torch.arange(self.num_flows, device=device)
            u = torch.arange(length, device=device, dtype=torch.float32)[None, :, None] / 49.
            position = self.u_enc(u.expand(batch, -1, -1))
            z_observed = self._normalize_available(x_observed, true_mask)
            # Cache the immutable Direct path once. Estimates cannot enter it.
            df, db, direct_aux = self._direct_reads(z_observed, true_mask, position,
                                                    all_ids, trace=collect)
            h1 = self._context_from_values(x_observed, true_mask, true_mask,
                                            neighbors, position)
            # Pass1 must predict every flow, even when only a target block is
            # requested: pass2 graph context needs all neighboring estimates.
            first_raw, first_q, first_standardized = self._decode(h1, df, db, all_ids)
            refinement_values = None
            if self.refinement_steps == 2:
                refinement_values = self._project_context(first_raw, x_observed, true_mask)
                final_sources = torch.ones_like(true_mask)
                h = self._context_from_values(refinement_values, true_mask, final_sources,
                                               neighbors, position)
                raw, query, standardized = self._decode(
                    h, df.index_select(2, ids), db.index_select(2, ids), ids)
            else:
                final_sources = true_mask
                h = h1
                raw = first_raw.index_select(2, ids)
                query = first_q.index_select(2, ids)
                standardized = first_standardized.index_select(2, ids)
            if not collect:
                return raw
            # Direct weights use B,F,T,T; feature/read tensors use B,T,F,C.
            aux = {name: value.index_select(1 if "weights" in name else 2, ids)
                   for name, value in direct_aux.items()}
            forward_read, backward_read = df.index_select(2, ids), db.index_select(2, ids)
            aux.update(
                context=h, position=position, q=query.permute(0, 2, 1, 3, 4),
                hf=forward_read, hb=backward_read,
                memory_hf=torch.zeros_like(forward_read), memory_hb=torch.zeros_like(backward_read),
                standardized_final=standardized, stage1_raw=first_raw,
                stage1_context=h1, refinement_input=refinement_values,
                true_observation_mask=true_mask, stage1_source_mask=true_mask,
                final_source_mask=final_sources, direct_source_mask=true_mask,
                refinement_steps=self.refinement_steps,
                query_source="spin_context", variant=self.variant,
            )
            return raw, aux


def build_model(num_flows, init_seed=41001, variant="two_step"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        return SpinRefineModel(num_flows, variant=variant).float()
