"""Direct + Sync with the complete Sync key/value/state path removed.

The direct attention computation is reused verbatim from the frozen candidate.
The target embedding, normalized query and decoder retain their names/shapes.
``build_model`` copies their exact tensors from the matching full-model seed.
No memory parameters or persistent memory state are registered by this model.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.llm_candidates_v1.direct import DIRECT_DIM, DirectSyncModel
from experiments.llm_candidates_v1.models import build_model as build_full_model
from experiments.sync_delta_v1.models import (
    HEAD_DIM,
    HEADS,
    TIME_STEPS,
    WIDTH,
    SyncDeltaModel,
    time_encoding,
)


class DirectOnlyModel(nn.Module):
    """Own-flow observations plus the unchanged target query and decoder.

    The ``neighbors`` and serial-order arguments remain in the public protocol
    for the common training/evaluation engine; neither supplies model features.
    In particular, changing another flow's values cannot change a target output.
    Fit-only per-flow normalization and the learned flow embedding are retained.
    """

    # Reuse the frozen operations, avoiding an independently changed attention
    # path or a new evaluation rule in this single component-removal ablation.
    _direct_reads = DirectSyncModel._direct_reads
    forward = SyncDeltaModel.forward

    def __init__(self, num_flows: int) -> None:
        super().__init__()
        if num_flows < 1:
            raise ValueError("num_flows must be positive")
        self.num_flows = int(num_flows)
        self.variant = "direct_only"
        self.flow_embedding = nn.Embedding(num_flows, 16)
        self.query = nn.Linear(24, WIDTH)
        self.decoder = nn.Sequential(
            nn.Linear(3 * WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1)
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.direct_query = nn.Linear(24, DIRECT_DIM)
        self.direct_key = nn.Linear(24, DIRECT_DIM)
        self.direct_fusion = nn.Linear(3, WIDTH, bias=False)
        self.register_buffer("_time_features", time_encoding(), persistent=False)
        self.float()

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
        """Return unclipped raw-unit [B,T,G] predictions for the requested flows.

        ``zero_readout`` retains its previous diagnostic meaning: both direct
        readouts become zero. It is never used to implement Direct-only itself.
        """
        if x_observed.ndim != 3:
            raise ValueError("x_observed must have shape [B,T,F]")
        batch, length, flows = x_observed.shape
        if flows != self.num_flows or mask.shape != x_observed.shape:
            raise ValueError("x_observed/mask must have shape [B,T,num_flows]")
        if not 1 <= length <= TIME_STEPS:
            raise ValueError("this specification accepts 1..50 native time steps")
        if intervention not in (None, "zero_readout", "zero_value"):
            raise ValueError(f"unknown intervention {intervention!r}")
        device = x_observed.device
        ids = torch.as_tensor(target_ids, dtype=torch.long, device=device)
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("target_ids must be a nonempty vector")
        # The common protocol still requires a correctly shaped neighbor table.
        # Its values are not selected, encoded, or inspected by this ablation.
        if neighbors.ndim != 2 or neighbors.shape[0] != flows or not 1 <= neighbors.shape[1] <= 9:
            raise ValueError("neighbors must be [num_flows,J], 1<=J<=9, with -1 padding")
        mu = torch.as_tensor(stats["mu"], dtype=torch.float32, device=device)
        scale_input = stats["scale"] if "scale" in stats else stats["s"]
        scale = torch.as_tensor(scale_input, dtype=torch.float32, device=device)
        if mu.shape != (flows,) or scale.shape != (flows,):
            raise ValueError("fit mu and scale must both be vectors of num_flows entries")
        collect = return_aux or trace
        with torch.autocast(device_type=device.type, enabled=False):
            groups = ids.numel()
            coordinates = torch.cat(
                (self._time_features[:length][None].expand(groups, -1, -1),
                 self.flow_embedding(ids)[:, None].expand(-1, length, -1)),
                dim=-1,
            )
            # Same operations and headwise normalization as full Sync._encode.
            queries = F.normalize(
                self.query(coordinates).reshape(groups, length, HEADS, HEAD_DIM),
                p=2, dim=-1, eps=1e-6,
            )
            queries = queries[None].expand(batch, -1, -1, -1, -1)
            query = queries.reshape(batch, groups, length, WIDTH).permute(0, 2, 1, 3)
            hf, hb, extra = self._direct_reads(
                x_observed, mask, mu, scale, ids,
                intervention=intervention, trace=collect,
            )
            if intervention == "zero_readout":
                hf, hb = torch.zeros_like(hf), torch.zeros_like(hb)
            decoded = self.decoder(torch.cat((hf, hb, query), dim=-1)).squeeze(-1)
            raw = mu[ids][None, None] + scale[ids][None, None] * decoded
            if collect:
                extra.update(
                    hf=hf, hb=hb, q=query, h_forward=hf, h_backward=hb,
                    # Only target-flow observation visibility is present.
                    event_mask=mask.bool().index_select(2, ids).permute(0, 2, 1)[..., None],
                )
                return raw, extra
            return raw


def build_model(num_flows: int, init_seed: int = 41001) -> DirectOnlyModel:
    """Copy every retained tensor from the same seed's frozen Direct + Sync.

    Creating the reference is an initialization-only operation on CPU. The
    returned module never owns the reference's key/value modules or memory scan.
    The caller's CPU RNG state is unchanged by the complete factory operation.
    """
    with torch.random.fork_rng(devices=[]):
        reference = build_full_model(num_flows, "direct_sync", init_seed=int(init_seed))
        model = DirectOnlyModel(num_flows)
        retained = {name: reference.state_dict()[name] for name in model.state_dict()}
        model.load_state_dict(retained, strict=True)
        for name, tensor in model.state_dict().items():
            if not torch.equal(tensor, reference.state_dict()[name]):
                raise RuntimeError(f"Shared initialization differs: {name}")
    return model


__all__ = ["DirectOnlyModel", "build_model"]
