"""A jointly trained SPIN encoder feeding the original Direct+Sync reader.

This is a new model, not a checkpoint-compatible SPIN replica or output adapter.
One real masked SPIN spatiotemporal block creates context for memory K/V/Q.
Only observed positions write memory; Direct reads original observed values.
The context sees the whole observed window: scan direction is NOT causality.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.spin_comparison_v1.attention import TemporalGraphAdditiveAttention
from experiments.spin_comparison_v1.models import SPINAdapter
from experiments.spin_comparison_v1.positional import PositionalEncoder
from experiments.spin_comparison_v1.primitives import MLP
from experiments.sync_delta_v1.models import (
    HEADS, HEAD_DIM, WIDTH, SyncDeltaModel, safe_masked_softmax,
)

VARIANTS = ("spin_sync_direct", "spin_direct_no_memory")
MODEL_CONFIG = {
    "context_layers": 1, "context_hidden": 32,
    "context_temporal_attention": True, "context_spatial_masked": True,
    "memory_heads": HEADS, "memory_head_dim": HEAD_DIM,
    "direct_dim": 16, "direct_nearest_per_direction": 2,
    "normalization": "global_fit_population_mean_std",
    "initialization": "from scratch; zero final decoder; shared weights identical across variants",
    "task": "offline whole-window imputation; directional scans of bidirectional context",
}


class SpinSyncDirectModel(nn.Module):
    def __init__(self, num_flows: int, variant: str = "spin_sync_direct"):
        super().__init__()
        if num_flows < 1 or variant not in VARIANTS:
            raise ValueError("Positive flow count and a supported variant are required")
        self.num_flows, self.variant = int(num_flows), variant
        self.memory_enabled = variant == "spin_sync_direct"
        self.u_enc = PositionalEncoder(1, 32, n_layers=2, n_nodes=num_flows)
        self.h_enc = MLP(1, 32, n_layers=2)
        self.h_norm = nn.LayerNorm(32)
        self.x_skip = nn.Linear(1, 32)
        self.context_encoder = TemporalGraphAdditiveAttention(
            input_size=32, output_size=32, msg_size=32, msg_layers=1,
            root_weight=True, reweight="softmax", temporal_self_attention=True,
            mask_temporal=True, mask_spatial=True, norm=True, dropout=0.,
        )
        self.key = nn.Linear(32, WIDTH)
        self.value = nn.Sequential(nn.Linear(32, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))
        self.query = nn.Linear(32, WIDTH)
        self.direct_query = nn.Linear(32, 16)
        self.direct_key = nn.Linear(32, 16)
        self.direct_fusion = nn.Linear(3, WIDTH, bias=False)
        self.decoder = nn.Sequential(nn.Linear(3 * WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.register_buffer("fit_mean", torch.zeros(()))
        self.register_buffer("fit_scale", torch.ones(()))
        self.register_buffer("fit_configured", torch.tensor(False))
        # Instantiate before removing these modules so shared layers consume the
        # same RNG stream in both controls. They are absent from no-memory params.
        if not self.memory_enabled:
            del self.key, self.value
        self.float()
        self.set_execution()

    def configure_fit_statistics(self, statistics: Mapping):
        return SPINAdapter.configure_fit_statistics(self, statistics)

    def configure_from_bundle(self, bundle):
        return self.configure_fit_statistics(dict(
            mu=bundle.mu, C=bundle.C, global_scale=bundle.metadata["global_scale"]))

    def set_execution(self, *, node_chunk=32, gradient_checkpointing=True):
        if type(node_chunk) is not int or node_chunk < 1:
            raise ValueError("node_chunk must be a positive integer")
        self.context_encoder.node_chunk = node_chunk
        self.context_encoder.gradient_checkpointing = bool(gradient_checkpointing)

    def _context(self, observed: Tensor, mask: Tensor, neighbors: Tensor):
        if not bool(self.fit_configured):
            raise RuntimeError("Configure fit-only statistics before prediction")
        if observed.ndim != 3 or observed.shape != mask.shape:
            raise ValueError("Values and mask must have shape [B,50,F]")
        batch, length, flows = observed.shape
        if length != 50 or flows != self.num_flows:
            raise ValueError("This candidate uses a full flow graph and a 50-slot window")
        if neighbors.ndim != 2 or neighbors.shape[0] != flows or not 1 <= neighbors.shape[1] <= 9:
            raise ValueError("Expected original self-plus-neighbor table [F,J], 1<=J<=9")
        clean = torch.where(mask, observed.float(), torch.zeros_like(observed, dtype=torch.float32))
        z = torch.where(mask, (clean - self.fit_mean) / self.fit_scale,
                        torch.zeros_like(clean))
        u = torch.arange(length, device=observed.device, dtype=torch.float32)[None, :, None] / 49.
        position = self.u_enc(u.expand(batch, -1, -1))
        h = self.h_enc(z.unsqueeze(-1)) + position
        h = self.h_norm(torch.where(mask.unsqueeze(-1), h, position))
        h = h + self.x_skip(z.unsqueeze(-1)) * mask.unsqueeze(-1)
        # SPIN spatial attention excludes self, its temporal attention covers self.
        ids = torch.arange(flows, device=observed.device)
        incoming = neighbors.masked_fill(neighbors == ids[:, None], -1)
        h = self.context_encoder(h, incoming, mask=mask.unsqueeze(-1))
        return h, position, z

    def _memory_tokens(self, h: Tensor, mask: Tensor, neighbors: Tensor, ids: Tensor):
        batch, length = h.shape[:2]
        selected = neighbors.index_select(0, ids)
        valid = selected >= 0
        safe = selected.clamp_min(0)
        event_mask = (mask[:, :, safe] & valid[None, None]).permute(0, 2, 1, 3)
        event_h = h[:, :, safe].permute(0, 2, 1, 3, 4)
        shape = (batch, ids.numel(), length, selected.shape[1], HEADS, HEAD_DIM)
        k = F.normalize(self.key(event_h).reshape(shape), dim=-1, eps=1e-6)
        v = self.value(event_h).reshape(shape)
        k = torch.where(event_mask[..., None, None], k, torch.zeros_like(k)).transpose(3, 4)
        v = torch.where(event_mask[..., None, None], v, torch.zeros_like(v)).transpose(3, 4)
        return k, v, event_mask

    # Preserve the original simultaneous update and after-bucket read semantics.
    _memory_scan = SyncDeltaModel._memory_scan

    def _direct_reads(self, z: Tensor, mask: Tensor, position: Tensor, ids: Tensor, *, trace=False):
        batch, length = z.shape[:2]
        visible = mask.index_select(2, ids).permute(0, 2, 1)
        values = z.index_select(2, ids).permute(0, 2, 1)
        p = position.index_select(2, ids).permute(0, 2, 1, 3)
        query, key = self.direct_query(p), self.direct_key(p)
        slots = torch.arange(length, device=z.device)
        delta = slots[:, None] - slots[None, :]
        distance = delta.abs().float()
        logits = query @ key.transpose(-1, -2) / math.sqrt(16)
        logits = logits - torch.log1p(distance)[None, None]
        reads, aux = [], {}
        for reverse, label in ((False, "forward"), (True, "backward")):
            direction = delta <= 0 if reverse else delta >= 0
            allowed = visible[:, :, None, :] & direction
            ranked = distance[None, None].expand(batch, ids.numel(), -1, -1)
            cutoff = ranked.masked_fill(~allowed, float(length + 1)).topk(
                2, dim=-1, largest=False).values[..., -1:]
            selected = allowed & (distance[None, None] <= cutoff)
            weights = safe_masked_softmax(logits, selected)
            value_read = (weights @ values[..., None]).squeeze(-1)
            distance_read = (weights * (distance / 49.)[None, None]).sum(-1)
            available = selected.any(-1).to(dtype=z.dtype)
            features = torch.stack((value_read, distance_read, available), -1)
            read = self.direct_fusion(features).permute(0, 2, 1, 3)
            reads.append(read)
            if trace:
                aux[f"direct_weights_{label}"] = weights
                aux[f"direct_features_{label}"] = features.permute(0, 2, 1, 3)
                aux[f"direct_read_{label}"] = read
        return reads[0], reads[1], aux

    def predict_block(self, x_observed, mask, stats, neighbors, target_ids,
                      serial_order_context=None, *, return_aux=False, trace=False,
                      intervention=None):
        if intervention is not None or serial_order_context is not None:
            raise ValueError("Use separately trained variants; no inference intervention is defined")
        device = x_observed.device
        mask = mask.to(device=device, dtype=torch.bool)
        neighbors = torch.as_tensor(neighbors, device=device, dtype=torch.long)
        ids = torch.as_tensor(target_ids, device=device, dtype=torch.long)
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("target_ids must be a nonempty vector")
        collect = return_aux or trace
        with torch.autocast(device_type=device.type, enabled=False):
            # Always encode the whole graph, even when reading only target blocks.
            h, position, z = self._context(x_observed, mask, neighbors)
            batch, length = h.shape[:2]
            query = F.normalize(self.query(h.index_select(2, ids)).reshape(
                batch, length, ids.numel(), HEADS, HEAD_DIM), dim=-1, eps=1e-6)
            q = query.permute(0, 2, 1, 3, 4)
            query_flat = query.flatten(-2)
            extra = {}
            if self.memory_enabled:
                k, v, event_mask = self._memory_tokens(h, mask, neighbors, ids)
                memory_reads = []
                for reverse, label in ((False, "forward"), (True, "backward")):
                    read, aux = self._memory_scan(k, v, q, reverse=reverse, order=None,
                                                   return_aux=collect, trace=trace)
                    memory_reads.append(read.reshape(batch, ids.numel(), length, WIDTH).permute(0, 2, 1, 3))
                    if collect:
                        extra.update({f"{name}_{label}": value for name, value in aux.items()})
                mf, mb = memory_reads
                if collect:
                    extra.update(k=k, v=v, event_mask=event_mask)
            else:
                mf, mb = torch.zeros_like(query_flat), torch.zeros_like(query_flat)
            df, db, aux = self._direct_reads(z, mask, position, ids, trace=collect)
            hf, hb = mf + df, mb + db
            standardized = self.decoder(torch.cat((hf, hb, query_flat), -1)).squeeze(-1)
            raw = self.fit_mean + self.fit_scale * standardized
            if collect:
                extra.update(aux)
                extra.update(context=h, position=position, q=q, memory_hf=mf, memory_hb=mb,
                             hf=hf, hb=hb, standardized_final=standardized)
                return raw, extra
            return raw

    def forward(self, x_observed, mask, stats, neighbors):
        ids = torch.arange(self.num_flows, device=x_observed.device)
        raw = self.predict_block(x_observed, mask, stats, neighbors, ids)
        return torch.where(mask.bool(), x_observed, raw.clamp_min(0))


def build_model(num_flows, init_seed=41001, variant="spin_sync_direct"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        return SpinSyncDirectModel(num_flows, variant=variant).float()
