"""Five observation-only readers for the frozen Sync-Delta experiment.

``predict_block`` returns *unclipped raw-unit* predictions for training.  ``forward``
adds the evaluation clamp and observed-value copy.  Neither method owns optimizer,
mask-generation, fit-statistic, or random serial-order state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VARIANTS = ("sync_delta", "attention", "serial_delta", "batch_ridge", "additive")
HEADS = 4
HEAD_DIM = 32
WIDTH = HEADS * HEAD_DIM
TIME_STEPS = 50


def time_encoding(length: int = TIME_STEPS, *, device: Any = None) -> Tensor:
    """Original native-grid coordinates, including during a backward scan."""
    if not 1 <= length <= TIME_STEPS:
        raise ValueError(f"length must be in [1, {TIME_STEPS}], got {length}")
    u = torch.arange(length, dtype=torch.float32, device=device) / 49.0
    return torch.stack(
        (u, u.square(), (math.pi * u).sin(), (math.pi * u).cos(),
         (2 * math.pi * u).sin(), (2 * math.pi * u).cos(),
         (4 * math.pi * u).sin(), (4 * math.pi * u).cos()), dim=-1,
    )


def gram_denominator(keys: Tensor) -> Tensor:
    """c=max(1, max absolute Gram row sum); keys have shape [..., n, d].

    Invalid slots must already be zero.  ``amax`` distributes gradients over
    tied maxima, unlike the index-selecting reduction ``max(dim=...)``.
    """
    if keys.shape[-2] == 0:
        return keys.new_ones(keys.shape[:-2])
    gram = keys @ keys.transpose(-1, -2)
    return gram.abs().sum(dim=-1).amax(dim=-1).clamp_min(1.0)


stable_gram_denominator = gram_denominator


def sync_delta_update(
    state: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    additive: bool = False,
    return_aux: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """One simultaneous update; S[...,dv,dk], K[...,n,dk], V[...,n,dv]."""
    denominator = gram_denominator(keys)
    value_columns = values.transpose(-1, -2)
    residual = value_columns if additive else value_columns - state @ keys.transpose(-1, -2)
    update = (residual @ keys) / denominator[..., None, None]
    result = state + update
    return (result, denominator, update) if return_aux else result


sync_update = sync_delta_update


def safe_masked_softmax(logits: Tensor, allowed: Tensor, dim: int = -1) -> Tensor:
    """Exactly zero weights and finite backward for an entirely empty row."""
    allowed = allowed.to(dtype=torch.bool)
    any_allowed = allowed.any(dim=dim, keepdim=True)
    masked = logits.masked_fill(~allowed, -torch.inf)
    safe = torch.where(any_allowed, masked, torch.zeros_like(masked))
    weights = torch.softmax(safe, dim=dim)
    return torch.where(allowed, weights, torch.zeros_like(weights))


class SyncDeltaModel(nn.Module):
    """Shared encoders/decoder with one of the five frozen reader variants."""

    def __init__(self, num_flows: int, variant: str = "sync_delta") -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        if num_flows < 1:
            raise ValueError("num_flows must be positive")
        self.num_flows = int(num_flows)
        self.variant = variant
        self.flow_embedding = nn.Embedding(num_flows, 16)
        self.key = nn.Linear(25 if variant == "attention" else 24, WIDTH)
        self.query = nn.Linear(24, WIDTH)
        self.value = nn.Sequential(nn.Linear(25, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))
        self.decoder = nn.Sequential(nn.Linear(3 * WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.register_buffer("_time_features", time_encoding(), persistent=False)
        self.float()

    @torch.no_grad()
    def copy_shared_initialization_(self, reference: "SyncDeltaModel") -> "SyncDeltaModel":
        """Copy all shared parameters, preserving Attention's extra value column.

        The coordinate columns and key bias are shared too.  The extra column is
        initialized independently by ``build_model`` using a local CPU RNG.
        """
        if self.num_flows != reference.num_flows:
            raise ValueError("initialization reference must use the same flow count")
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in self.named_parameters():
            source = reference_parameters[name]
            if name == "key.weight" and parameter.shape != source.shape:
                parameter[:, :24].copy_(source[:, :24])
            else:
                if parameter.shape != source.shape:
                    raise ValueError(f"unexpected initialization mismatch for {name}")
                parameter.copy_(source)
        return self

    def _encode(
        self,
        x_observed: Tensor,
        mask: Tensor,
        stats: Mapping[str, Tensor],
        neighbors: Tensor,
        target_ids: Tensor,
        intervention: str | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, length, flows = x_observed.shape
        device = x_observed.device
        if flows != self.num_flows or mask.shape != x_observed.shape:
            raise ValueError("x_observed/mask must have shape [B,T,num_flows]")
        if not 1 <= length <= TIME_STEPS:
            raise ValueError("this specification accepts 1..50 native time steps")
        target_ids = torch.as_tensor(target_ids, dtype=torch.long, device=device)
        neighbors = torch.as_tensor(neighbors, dtype=torch.long, device=device)
        if target_ids.ndim != 1 or target_ids.numel() == 0:
            raise ValueError("target_ids must be a nonempty vector")
        if neighbors.ndim != 2 or neighbors.shape[0] != flows or not 1 <= neighbors.shape[1] <= 9:
            raise ValueError("neighbors must be [num_flows,J], 1<=J<=9, with -1 padding")
        mask = mask.to(dtype=torch.bool)
        mu = torch.as_tensor(stats["mu"], dtype=torch.float32, device=device)
        scale_input = stats["scale"] if "scale" in stats else stats["s"]
        scale = torch.as_tensor(scale_input, dtype=torch.float32, device=device)
        if mu.shape != (flows,) or scale.shape != (flows,):
            raise ValueError("fit mu and scale must both be vectors of num_flows entries")

        # Sanitize before normalization/encoding, including hidden NaNs/Infs.
        clean = torch.where(mask, x_observed.float(), torch.zeros((), device=device))
        z = torch.where(mask, (clean - mu) / scale, torch.zeros((), device=device))
        selected = neighbors.index_select(0, target_ids)
        valid_neighbor = selected >= 0
        source_ids = selected.clamp_min(0)
        group_count, group_size = source_ids.shape
        # [B,T,G,J] -> [B,G,T,J]. Missing and padding keys are excluded below.
        event_mask = (mask[:, :, source_ids] & valid_neighbor[None, None]).permute(0, 2, 1, 3)
        event_z = z[:, :, source_ids].permute(0, 2, 1, 3)
        event_z = torch.where(event_mask, event_z, torch.zeros_like(event_z))
        time = self._time_features[:length]
        source_embedding = self.flow_embedding(source_ids)
        coordinates = torch.cat(
            (time[None, :, None, :].expand(group_count, -1, group_size, -1),
             source_embedding[:, None].expand(-1, length, -1, -1)), dim=-1,
        )
        coordinates = coordinates[None].expand(batch, -1, -1, -1, -1)
        observed_features = torch.cat((coordinates, event_z[..., None]), dim=-1)
        value_z = torch.zeros_like(event_z) if intervention == "zero_value" else event_z
        value_features = torch.cat((coordinates, value_z[..., None]), dim=-1)
        key_features = observed_features if self.variant == "attention" else coordinates
        keys = F.normalize(self.key(key_features).reshape(batch, group_count, length, group_size, HEADS, HEAD_DIM),
                           p=2, dim=-1, eps=1e-6)
        values = self.value(value_features).reshape(batch, group_count, length, group_size, HEADS, HEAD_DIM)
        valid = event_mask[..., None, None]
        keys = torch.where(valid, keys, torch.zeros_like(keys)).transpose(3, 4)
        values = torch.where(valid, values, torch.zeros_like(values)).transpose(3, 4)
        query_coordinates = torch.cat(
            (time[None].expand(group_count, -1, -1),
             self.flow_embedding(target_ids)[:, None].expand(-1, length, -1)), dim=-1,
        )
        queries = F.normalize(self.query(query_coordinates).reshape(group_count, length, HEADS, HEAD_DIM),
                              p=2, dim=-1, eps=1e-6)
        queries = queries[None].expand(batch, -1, -1, -1, -1)
        events = torch.where(event_mask[..., None], observed_features, torch.zeros_like(observed_features))
        return keys, values, queries, event_mask, events, target_ids, mu, scale

    def _serial_order(
        self, context: Mapping[str, Tensor] | Tensor | None, shape: tuple[int, int, int, int],
        device: torch.device, target_ids: Tensor,
    ) -> Tensor:
        if context is None:
            raise ValueError("serial_delta requires externally generated deterministic slot order")
        order = context["order"] if isinstance(context, Mapping) else context
        order = torch.as_tensor(order, dtype=torch.long, device=device)
        batch, groups, length, group_size = shape
        # Public order layout is [B,T,G,J], or [B,T,F,J] before selecting targets.
        if order.ndim == 4 and order.shape[2] == self.num_flows and groups != self.num_flows:
            order = order.index_select(2, target_ids)
        if order.shape != (batch, length, groups, group_size):
            raise ValueError(f"serial order must be {(batch, length, groups, group_size)}, got {tuple(order.shape)}")
        return order.permute(0, 2, 1, 3)

    def _memory_scan(
        self, keys: Tensor, values: Tensor, queries: Tensor, *, reverse: bool,
        order: Tensor | None, return_aux: bool, trace: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, groups, length, heads, group_size, dimension = keys.shape
        state = keys.new_zeros(batch, groups, heads, dimension, dimension)
        readouts: list[Tensor | None] = [None] * length
        states: list[Tensor | None] = [None] * length
        state_norms: list[Tensor | None] = [None] * length
        update_norms: list[Tensor | None] = [None] * length
        denominators = gram_denominator(keys) if self.variant != "serial_delta" else None
        times = range(length - 1, -1, -1) if reverse else range(length)
        for t in times:
            k, v = keys[:, :, t], values[:, :, t]
            previous = state
            if self.variant == "serial_delta":
                assert order is not None
                slot_order = order[:, :, t].flip(-1) if reverse else order[:, :, t]
                gather_index = slot_order[:, :, None, :, None].expand(-1, -1, heads, -1, dimension)
                ordered_keys = k.gather(-2, gather_index)
                ordered_values = v.gather(-2, gather_index)
                for slot in range(group_size):
                    key = ordered_keys[..., slot, :]
                    value = ordered_values[..., slot, :]
                    residual = value - (state @ key[..., None]).squeeze(-1)
                    state = state + residual[..., :, None] * key[..., None, :]
            else:
                value_columns = v.transpose(-1, -2)
                residual = value_columns if self.variant == "additive" else value_columns - state @ k.transpose(-1, -2)
                assert denominators is not None
                state = state + (residual @ k) / denominators[:, :, t, :, None, None]
            # Current-t query always reads after the entire current-t bucket.
            readouts[t] = (state @ queries[:, :, t, :, :, None]).squeeze(-1)
            if return_aux:
                state_norms[t] = state.detach().norm(dim=(-2, -1))
                update_norms[t] = (state - previous).detach().norm(dim=(-2, -1))
            if trace:
                states[t] = state
        result = torch.stack(readouts, dim=2)
        aux: dict[str, Tensor] = {}
        if return_aux:
            aux["state_norm"] = torch.stack(state_norms, dim=2)
            aux["update_norm"] = torch.stack(update_norms, dim=2)
            if denominators is not None:
                aux["c"] = denominators
        if trace:
            aux["states"] = torch.stack(states, dim=2)
        return result, aux

    def _attention(
        self, keys: Tensor, values: Tensor, queries: Tensor, event_mask: Tensor, *, trace: bool,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        batch, groups, length, heads, group_size, dimension = keys.shape
        k = keys.permute(0, 1, 3, 2, 4, 5).reshape(batch, groups, heads, length * group_size, dimension)
        v = values.permute(0, 1, 3, 2, 4, 5).reshape(batch, groups, heads, length * group_size, dimension)
        q = queries.permute(0, 1, 3, 2, 4)
        logits = (q @ k.transpose(-1, -2)) * math.sqrt(HEAD_DIM)
        event_times = torch.arange(length, device=keys.device).repeat_interleave(group_size)
        query_times = torch.arange(length, device=keys.device)
        visible = event_mask.reshape(batch, groups, 1, 1, length * group_size)
        past = event_times[None, :] <= query_times[:, None]
        future = event_times[None, :] >= query_times[:, None]
        forward_weights = safe_masked_softmax(logits, visible & past)
        backward_weights = safe_masked_softmax(logits, visible & future)
        forward = (forward_weights @ v).permute(0, 1, 3, 2, 4)
        backward = (backward_weights @ v).permute(0, 1, 3, 2, 4)
        aux = {"weights_forward": forward_weights, "weights_backward": backward_weights} if trace else {}
        return forward, backward, aux

    def _ridge_scan(
        self, keys: Tensor, values: Tensor, queries: Tensor, *, reverse: bool,
        trace: bool, return_aux: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # SUM statistics, never mean: lambda=1 is exactly the identity below.
        bucket_gram = keys.transpose(-1, -2) @ keys
        bucket_cross = values.transpose(-1, -2) @ keys
        # CUDA cumsum has no deterministic implementation in the pinned Torch
        # 2.2.2 runtime. Fixed chronological additions preserve SUM/lambda=1 and
        # the complete autograd graph, without relaxing global determinism.
        length = keys.shape[2]
        gram_steps: list[Tensor | None] = [None] * length
        cross_steps: list[Tensor | None] = [None] * length
        running_gram = torch.zeros_like(bucket_gram[:, :, 0])
        running_cross = torch.zeros_like(bucket_cross[:, :, 0])
        times = range(length - 1, -1, -1) if reverse else range(length)
        for t in times:
            running_gram = running_gram + bucket_gram[:, :, t]
            running_cross = running_cross + bucket_cross[:, :, t]
            gram_steps[t] = running_gram
            cross_steps[t] = running_cross
        gram = torch.stack(gram_steps, dim=2)
        cross = torch.stack(cross_steps, dim=2)
        identity = torch.eye(HEAD_DIM, dtype=torch.float32, device=keys.device)
        regularized_gram = gram + identity
        factor, info = torch.linalg.cholesky_ex(regularized_gram, check_errors=False)
        if bool((info != 0).any().item()):
            failures = (info != 0).nonzero().detach().cpu().tolist()
            raise RuntimeError(f"Batch-Ridge Cholesky failed at {failures[:8]}; no jitter fallback is permitted")
        solved_query = torch.cholesky_solve(queries[..., None], factor)
        readout = (cross @ solved_query).squeeze(-1)
        aux: dict[str, Tensor] = {}
        if trace:
            aux.update(gram=regularized_gram, cross=cross)
        if return_aux:
            aux["cross_norm"] = cross.detach().norm(dim=(-2, -1))
        return readout, aux

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
        """Predict unclipped raw values [B,T,G] for independent target groups.

        ``neighbors`` is [F,J], padded with -1. ``stats`` contains fit-only ``mu``
        and ``scale`` [F]. Serial orders contain permutations of *neighbor slot*
        indices, [B,T,G,J], generated externally from stable data identities.

        Auxiliary ``hf/hb/q`` use [B,T,G,128]; ``k/v`` use [B,G,T,4,J,32].
        ``trace=True`` additionally retains states/statistics/attention weights.
        It is intended for small correctness probes, not routine training logs.
        """
        if x_observed.ndim != 3:
            raise ValueError("x_observed must have shape [B,T,F]")
        if intervention not in (None, "zero_readout", "zero_value"):
            raise ValueError(f"unknown intervention {intervention!r}")
        collect = return_aux or trace
        # The frozen implementation keeps all projected features and state algebra FP32.
        with torch.autocast(device_type=x_observed.device.type, enabled=False):
            keys, values, queries, event_mask, events, target_ids, mu, scale = self._encode(
                x_observed, mask, stats, neighbors, target_ids, intervention,
            )
            batch, groups, length, _, group_size, _ = keys.shape
            extra: dict[str, Tensor] = {}
            if self.variant == "attention":
                hf, hb, extra = self._attention(keys, values, queries, event_mask, trace=trace)
            else:
                order = self._serial_order(serial_order_context, (batch, groups, length, group_size),
                                           keys.device, target_ids) if self.variant == "serial_delta" else None
                direction_aux: dict[str, dict[str, Tensor]] = {}
                reads: list[Tensor] = []
                for reverse, label in ((False, "forward"), (True, "backward")):
                    if self.variant == "batch_ridge":
                        read, aux = self._ridge_scan(keys, values, queries, reverse=reverse,
                                                    trace=trace, return_aux=collect)
                    else:
                        read, aux = self._memory_scan(keys, values, queries, reverse=reverse,
                                                     order=order, return_aux=collect, trace=trace)
                    reads.append(read)
                    direction_aux[label] = aux
                hf, hb = reads
                for label, aux in direction_aux.items():
                    extra.update({f"{key}_{label}": value for key, value in aux.items()})
            if intervention == "zero_readout":
                hf, hb = torch.zeros_like(hf), torch.zeros_like(hb)
            hf = hf.reshape(batch, groups, length, WIDTH).permute(0, 2, 1, 3)
            hb = hb.reshape(batch, groups, length, WIDTH).permute(0, 2, 1, 3)
            q = queries.reshape(batch, groups, length, WIDTH).permute(0, 2, 1, 3)
            normalized_prediction = self.decoder(torch.cat((hf, hb, q), dim=-1)).squeeze(-1)
            raw = mu[target_ids][None, None] + scale[target_ids][None, None] * normalized_prediction
            if collect:
                extra.update(hf=hf, hb=hb, q=q, h_forward=hf, h_backward=hb,
                             k=keys, v=values, events=events, event_mask=event_mask)
                return raw, extra
            return raw

    def forward(
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
        """Evaluation output: nonnegative missing predictions and observed copy."""
        result = self.predict_block(x_observed, mask, stats, neighbors, target_ids,
                                    serial_order_context, return_aux=return_aux,
                                    trace=trace, intervention=intervention)
        raw, aux = result if isinstance(result, tuple) else (result, None)
        ids = torch.as_tensor(target_ids, dtype=torch.long, device=x_observed.device)
        prediction = torch.where(mask[:, :, ids].bool(), x_observed[:, :, ids], raw.clamp_min(0))
        if aux is not None:
            aux["raw"] = raw
            return prediction, aux
        return prediction


def build_model(num_flows: int, variant: str = "sync_delta", init_seed: int = 41001) -> SyncDeltaModel:
    """Reproduce common initialization without consuming the caller's RNG state.

    Every variant copies the same sync template. Attention's additional value
    column retains an independent, explicitly seeded default Linear initializer.
    The seed here never controls masks, data order, or serial event permutations.
    """
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
        torch.random.set_rng_state(generator.get_state())
        reference = SyncDeltaModel(num_flows, "sync_delta")
        if variant == "sync_delta":
            return reference
        extra_generator = torch.Generator(device="cpu").manual_seed(int(init_seed) + 42017)
        torch.random.set_rng_state(extra_generator.get_state())
        model = SyncDeltaModel(num_flows, variant)
        return model.copy_shared_initialization_(reference)


__all__ = ["SyncDeltaModel", "build_model", "VARIANTS", "time_encoding", "gram_denominator",
           "stable_gram_denominator", "sync_delta_update", "sync_update", "safe_masked_softmax"]
