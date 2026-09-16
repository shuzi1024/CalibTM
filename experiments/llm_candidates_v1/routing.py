"""Observation-conditioned full-pool routing for the frozen Sync-Delta reader.

This is a task-specific adaptation of content routing, not an implementation of
Routing Transformer's online k-means. All learned parameters and the complete
bucket-wise Delta update are inherited unchanged from SyncDeltaModel.

For each scan direction, every flow has a two-component descriptor: the mean of
its visible standardized observations and its most recent visible observation.
With n observations, reliability is n/(n+4). Pair reliability is the geometric
mean, and the score is rho*exp(-mean_squared_distance/2) plus a (1-rho)*0.25
membership prior for the original fit-only neighbors. All constants below are
fixed for this one candidate. No hidden values, pairwise complete trajectories,
or learnable retrieval embeddings enter routing.

Self is always slot zero; up to eight other flows are selected from the complete
currently observed pool. Stable flow-ID tie breaking makes source slot order
irrelevant. At a time with zero target observations, the exact fit neighbor set
is used instead, with padding where needed. Every selected source only writes if
observed in the current bucket. This increases actual event access relative to
fixed self+8 followed by masking; it is not an equal-context comparison. Aux
records include valid writes and non-static writes per bucket and direction.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.sync_delta_v1.models import HEADS, HEAD_DIM, TIME_STEPS, WIDTH, SyncDeltaModel


ROUTING_PSEUDOCOUNT = 4.0
ROUTING_PRIOR_STRENGTH = 0.25
ROUTING_OTHER_NEIGHBORS = 8


class RoutingSyncModel(SyncDeltaModel):
    """Full-pool, directionally visible self+8 routing with unchanged Delta math."""

    def __init__(self, num_flows: int) -> None:
        super().__init__(num_flows, variant="sync_delta")

    def _validate_and_clean(self, x_observed, mask, stats, neighbors, target_ids):
        if x_observed.ndim != 3 or mask.shape != x_observed.shape:
            raise ValueError("x_observed/mask must share [B,T,num_flows]")
        batch, length, flows = x_observed.shape
        if flows != self.num_flows or not 1 <= length <= TIME_STEPS:
            raise ValueError("expected 1..50 times and the configured number of flows")
        device = x_observed.device
        mask = mask.to(device=device, dtype=torch.bool)
        ids = torch.as_tensor(target_ids, device=device, dtype=torch.long)
        prior = torch.as_tensor(neighbors, device=device, dtype=torch.long)
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("target_ids must be a nonempty vector")
        if prior.ndim != 2 or prior.shape[0] != flows or not 1 <= prior.shape[1] <= 9:
            raise ValueError("fit neighbors must have shape [F,J], 1<=J<=9")
        mu = torch.as_tensor(stats["mu"], device=device, dtype=torch.float32)
        scale = torch.as_tensor(stats["scale"] if "scale" in stats else stats["s"],
                                device=device, dtype=torch.float32)
        if mu.shape != (flows,) or scale.shape != (flows,):
            raise ValueError("fit mu/scale must be vectors of num_flows entries")
        # The frozen data pipeline supplies finite, strictly positive fit scales.
        # Sanitize hidden NaN/Inf before arithmetic, including routing arithmetic.
        clean = torch.where(mask, x_observed.float(), torch.zeros((), device=device))
        z = torch.where(mask, (clean - mu) / scale, torch.zeros((), device=device))
        return z, mask, mu, scale, prior, ids

    @staticmethod
    @torch.no_grad()
    def _directional_descriptors(z: Tensor, mask: Tensor, *, reverse: bool):
        """[B,T,F] descriptors, inclusive of the complete current bucket.

        Fixed additions deliberately avoid CUDA cumsum, unsupported by the
        pinned Torch runtime under deterministic algorithms. No state survives
        a call/window. Directional descriptors are detached routing decisions.
        """
        batch, length, flows = z.shape
        count = z.new_zeros(batch, flows)
        total = torch.zeros_like(count)
        latest = torch.zeros_like(count)
        means, lasts, reliabilities = [None] * length, [None] * length, [None] * length
        times = range(length - 1, -1, -1) if reverse else range(length)
        for t in times:
            count = count + mask[:, t].to(dtype=z.dtype)
            total = total + z[:, t]
            latest = torch.where(mask[:, t], z[:, t], latest)
            means[t] = total / count.clamp_min(1.0)
            lasts[t] = latest
            reliabilities[t] = count / (count + ROUTING_PSEUDOCOUNT)
        return tuple(torch.stack(items, dim=1) for items in (means, lasts, reliabilities))

    @staticmethod
    @torch.no_grad()
    def _route(descriptors, mask: Tensor, neighbors: Tensor, target_ids: Tensor) -> Tensor:
        """Choose [B,T,G,J] flow IDs; only O(B*T*G*F) scalar score storage."""
        mean, latest, reliability = descriptors
        batch, length, flows = mean.shape
        groups = target_ids.numel()
        mean_distance = mean.index_select(2, target_ids)[..., None] - mean[:, :, None, :]
        latest_distance = latest.index_select(2, target_ids)[..., None] - latest[:, :, None, :]
        similarity = torch.exp(-0.25 * (mean_distance.square() + latest_distance.square()))
        pair_reliability = (
            reliability.index_select(2, target_ids)[..., None] * reliability[:, :, None, :]
        ).sqrt()
        prior_ids = neighbors.index_select(0, target_ids)
        # Equality/reduction is invariant to prior neighbor slot permutation and
        # harmless duplicates/padding; no scatter with duplicate-index semantics.
        membership = (prior_ids[:, :, None] == torch.arange(flows, device=mean.device)[None, None]).any(dim=1)
        score = pair_reliability * similarity + (
            (1.0 - pair_reliability) * ROUTING_PRIOR_STRENGTH * membership[None, None]
        )
        own = torch.arange(flows, device=mean.device)[None, :] == target_ids[:, None]
        has_target_history = reliability.index_select(2, target_ids)[..., None] > 0
        # Dynamic routing spends the eight slots on sources actually writable
        # now. With no target evidence, strictly retain the static source set.
        allowed = torch.where(has_target_history, mask[:, :, None, :], membership[None, None])
        allowed = allowed & ~own[None, None]
        score = score.masked_fill(~allowed, -torch.inf)
        other_count = min(ROUTING_OTHER_NEIGHBORS, flows - 1)
        # Stable sorting uses the original ascending flow-ID order for exact ties.
        others = torch.argsort(score, dim=-1, descending=True, stable=True)[..., :other_count]
        others = torch.where(allowed.gather(-1, others), others, torch.full_like(others, -1))
        self_ids = target_ids[None, None, :, None].expand(batch, length, groups, 1)
        return torch.cat((self_ids, others), dim=-1)

    def _encode_pool(self, z: Tensor, mask: Tensor, target_ids: Tensor,
                     intervention: str | None):
        """Encode each source once per block, shared by all targets/directions."""
        batch, length, flows = z.shape
        time = self._time_features[:length]
        coordinates = torch.cat((
            time[:, None].expand(-1, flows, -1),
            self.flow_embedding.weight[None].expand(length, -1, -1),
        ), dim=-1)
        pool_keys = F.normalize(self.key(coordinates).reshape(length, flows, HEADS, HEAD_DIM),
                                p=2, dim=-1, eps=1e-6)
        value_z = torch.zeros_like(z) if intervention == "zero_value" else z
        value_features = torch.cat((coordinates[None].expand(batch, -1, -1, -1),
                                    value_z[..., None]), dim=-1)
        pool_values = self.value(value_features).reshape(batch, length, flows, HEADS, HEAD_DIM)
        query_coordinates = coordinates.index_select(1, target_ids).permute(1, 0, 2)
        queries = F.normalize(self.query(query_coordinates).reshape(-1, length, HEADS, HEAD_DIM),
                              p=2, dim=-1, eps=1e-6)
        queries = queries[None].expand(batch, -1, -1, -1, -1)
        return pool_keys, pool_values, queries, coordinates

    @staticmethod
    def _gather_events(pool_keys, pool_values, coordinates, z, mask, routes):
        """Routes [B,T,G,J] -> reader inputs [B,G,T,H,J,D]."""
        batch, length, groups, slots = routes.shape
        batch_ids = torch.arange(batch, device=z.device)[:, None, None, None]
        time_ids = torch.arange(length, device=z.device)[None, :, None, None]
        source_ids = routes.clamp_min(0)
        visible = mask[batch_ids, time_ids, source_ids] & (routes >= 0)
        k = pool_keys[time_ids, source_ids]
        v = pool_values[batch_ids, time_ids, source_ids]
        valid = visible[..., None, None]
        k = torch.where(valid, k, torch.zeros_like(k)).permute(0, 2, 1, 4, 3, 5)
        v = torch.where(valid, v, torch.zeros_like(v)).permute(0, 2, 1, 4, 3, 5)
        return k, v, visible.permute(0, 2, 1, 3)

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
    ):
        if intervention not in (None, "zero_readout", "zero_value"):
            raise ValueError(f"unknown intervention {intervention!r}")
        collect = return_aux or trace
        with torch.autocast(device_type=x_observed.device.type, enabled=False):
            z, mask, mu, scale, prior, ids = self._validate_and_clean(
                x_observed, mask, stats, neighbors, target_ids)
            batch, length, _ = z.shape
            keys, values, queries, coordinates = self._encode_pool(z, mask, ids, intervention)
            reads, extras = [], {}
            for reverse, label in ((False, "forward"), (True, "backward")):
                descriptors = self._directional_descriptors(z, mask, reverse=reverse)
                routes = self._route(descriptors, mask, prior, ids)
                k, v, event_mask = self._gather_events(keys, values, coordinates, z, mask, routes)
                read, aux = self._memory_scan(k, v, queries, reverse=reverse,
                    order=None, return_aux=collect, trace=trace)
                reads.append(read)
                if collect:
                    extras.update({f"{key}_{label}": value for key, value in aux.items()})
                    extras.update({f"routes_{label}": routes, f"k_{label}": k,
                                   f"v_{label}": v, f"event_mask_{label}": event_mask})
                    visible = event_mask.permute(0, 2, 1, 3)
                    static = (routes[..., None] == prior.index_select(0, ids)[None, None, :, None, :]).any(-1)
                    valid_count = visible.sum(-1)
                    nonstatic_count = (visible & ~static).sum(-1)
                    extras.update({
                        f"selected_valid_count_{label}": valid_count,
                        f"nonstatic_valid_count_{label}": nonstatic_count,
                        f"nonstatic_valid_fraction_{label}": nonstatic_count.float() / valid_count.clamp_min(1),
                    })
            hf, hb = reads
            if intervention == "zero_readout":
                hf, hb = torch.zeros_like(hf), torch.zeros_like(hb)
            groups = ids.numel()
            hf = hf.reshape(batch, groups, length, WIDTH).permute(0, 2, 1, 3)
            hb = hb.reshape(batch, groups, length, WIDTH).permute(0, 2, 1, 3)
            q = queries.reshape(batch, groups, length, WIDTH).permute(0, 2, 1, 3)
            normalized = self.decoder(torch.cat((hf, hb, q), dim=-1)).squeeze(-1)
            raw = mu[ids][None, None] + scale[ids][None, None] * normalized
            if collect:
                extras.update(hf=hf, hb=hb, q=q, h_forward=hf, h_backward=hb,
                              k=extras["k_forward"], v=extras["v_forward"],
                              event_mask=extras["event_mask_forward"])
                return raw, extras
            return raw


__all__ = ["RoutingSyncModel"]
