"""One bounded Based-inspired direct-read candidate for Sync-Delta.

Each direction reads at most two nearest observed values of the target flow.
One 16-dimensional attention head uses learned coordinate Q/K scores plus the
fixed native-slot distance bias -log(1 + distance). The weighted standardized
value, weighted distance / 49, and availability are projected into the original
128-dimensional memory readout. All frozen Sync modules retain names/shapes.

This is a direct bypass around the existing self-plus-neighbor memory, not a
replication of the Based LM or of the full-neighborhood Attention baseline.
It is not fixed Linear interpolation: weights and the final decoder are learned.
No additional gate, auxiliary loss, attention grid, or dependency is introduced.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from experiments.sync_delta_v1.models import (
    WIDTH,
    SyncDeltaModel,
    safe_masked_softmax,
)


DIRECT_DIM = 16
DIRECT_NEAREST_PER_DIRECTION = 2


class DirectSyncModel(SyncDeltaModel):
    """Sync memory plus a small, observation-only direct temporal readout."""

    def __init__(self, num_flows: int) -> None:
        super().__init__(num_flows, variant="sync_delta")
        self.direct_query = nn.Linear(24, DIRECT_DIM)
        self.direct_key = nn.Linear(24, DIRECT_DIM)
        # Default nonzero initialization avoids another zero-initialized layer
        # before the frozen zero decoder. The base decoder still guarantees the
        # common fit-mean prediction at initialization.
        self.direct_fusion = nn.Linear(3, WIDTH, bias=False)
        self.float()

    def _direct_reads(
        self,
        x_observed: Tensor,
        mask: Tensor,
        mu: Tensor,
        scale: Tensor,
        target_ids: Tensor,
        *,
        intervention: str | None,
        trace: bool,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Return [B,T,G,128] reads using only each target's own observations.

        Nearest refers to native time-slot distance, not the number of observed
        events and not continuous wall-clock time. Scores are computed over a
        small dense T-by-T matrix; support is at most two events per direction.
        """
        batch, length, _ = x_observed.shape
        groups = target_ids.numel()
        visible = mask.bool().index_select(2, target_ids).permute(0, 2, 1)
        source = x_observed.float().index_select(2, target_ids).permute(0, 2, 1)
        clean = torch.where(visible, source, torch.zeros_like(source))
        z = torch.where(
            visible,
            (clean - mu[target_ids][None, :, None]) / scale[target_ids][None, :, None],
            torch.zeros_like(clean),
        )
        if intervention == "zero_value":
            z = torch.zeros_like(z)
        coordinates = torch.cat(
            (self._time_features[:length][None].expand(groups, -1, -1),
             self.flow_embedding(target_ids)[:, None].expand(-1, length, -1)),
            dim=-1,
        )
        query = self.direct_query(coordinates)
        key = self.direct_key(coordinates)
        times = torch.arange(length, device=x_observed.device)
        delta = times[:, None] - times[None, :]
        distance = delta.abs().float()
        logits = (query @ key.transpose(-1, -2)) / math.sqrt(DIRECT_DIM)
        logits = logits - torch.log1p(distance)[None]
        reads = []
        extra: dict[str, Tensor] = {}
        for reverse, label in ((False, "forward"), (True, "backward")):
            direction = delta <= 0 if reverse else delta >= 0
            allowed = visible[:, :, None, :] & direction
            # Only distance selects the local support. Invalid sentinel ties
            # cannot make hidden events visible because allowed is reapplied.
            ranked_distance = distance[None, None].expand(batch, groups, -1, -1)
            ranked_distance = ranked_distance.masked_fill(~allowed, float(length + 1))
            cutoff = ranked_distance.topk(
                min(DIRECT_NEAREST_PER_DIRECTION, length), dim=-1, largest=False
            ).values[..., -1:]
            selected = allowed & (distance[None, None] <= cutoff)
            weights = safe_masked_softmax(logits[None], selected)
            value_read = (weights @ z[..., None]).squeeze(-1)
            distance_read = (weights * (distance / 49.0)[None, None]).sum(-1)
            available = selected.any(-1).to(dtype=z.dtype)
            features = torch.stack((value_read, distance_read, available), dim=-1)
            projected = self.direct_fusion(features).permute(0, 2, 1, 3)
            reads.append(projected)
            if trace:
                extra[f"direct_weights_{label}"] = weights
                extra[f"direct_features_{label}"] = features.permute(0, 2, 1, 3)
                extra[f"direct_read_{label}"] = projected
        return reads[0], reads[1], extra

    def predict_block(
        self,
        x_observed: Tensor,
        mask: Tensor,
        stats: Mapping[str, Tensor],
        neighbors: Tensor,
        target_ids: Tensor,
        serial_order_context: Mapping[str, Tensor] | Tensor | None = None,
        *,
        return_aux: bool = False,
        trace: bool = False,
        intervention: str | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x_observed.ndim != 3:
            raise ValueError("x_observed must have shape [B,T,F]")
        if intervention not in (None, "zero_readout", "zero_value"):
            raise ValueError(f"unknown intervention {intervention!r}")
        collect = return_aux or trace
        with torch.autocast(device_type=x_observed.device.type, enabled=False):
            keys, values, queries, event_mask, events, ids, mu, scale = self._encode(
                x_observed, mask, stats, neighbors, target_ids, intervention
            )
            batch, groups, length = keys.shape[:3]
            extra: dict[str, Tensor] = {}
            memory_reads = []
            for reverse, label in ((False, "forward"), (True, "backward")):
                read, aux = self._memory_scan(
                    keys, values, queries, reverse=reverse, order=None,
                    return_aux=collect, trace=trace,
                )
                memory_reads.append(
                    read.reshape(batch, groups, length, WIDTH).permute(0, 2, 1, 3)
                )
                extra.update({f"{name}_{label}": value for name, value in aux.items()})
            direct_forward, direct_backward, direct_aux = self._direct_reads(
                x_observed, mask, mu, scale, ids,
                intervention=intervention, trace=collect,
            )
            extra.update(direct_aux)
            hf = memory_reads[0] + direct_forward
            hb = memory_reads[1] + direct_backward
            if intervention == "zero_readout":
                hf, hb = torch.zeros_like(hf), torch.zeros_like(hb)
            query = queries.reshape(batch, groups, length, WIDTH).permute(0, 2, 1, 3)
            decoded = self.decoder(torch.cat((hf, hb, query), dim=-1)).squeeze(-1)
            raw = mu[ids][None, None] + scale[ids][None, None] * decoded
            if collect:
                extra.update(
                    hf=hf, hb=hb, q=query, h_forward=hf, h_backward=hb,
                    memory_hf=memory_reads[0], memory_hb=memory_reads[1],
                    k=keys, v=values, events=events, event_mask=event_mask,
                )
                return raw, extra
            return raw


__all__ = ["DirectSyncModel", "DIRECT_DIM", "DIRECT_NEAREST_PER_DIRECTION"]
