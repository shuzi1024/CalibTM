"""Hedgehog-inspired feature addresses for the frozen Sync-Delta reader.

This is a bounded feature-map adaptation, not a reproduction of Hedgehog or its
attention-distillation procedure.  Each head shares a learned 32-to-32 affine
map between its normalized keys and queries.  Feature-axis softmax at the fixed
sqrt(32) scale followed by L2 normalization changes their address geometry
without enlarging the memory.  Identity/zero initialization and this fixed scale
avoid flattening normalized inputs through a small random affine map.
Values, synchronous writes, bidirectional visibility and the decoder are the
original Sync-Delta implementation.  The external factory owns seeded shared
initialization; this class never changes the caller's random seed.  The added
feature map always starts at the identity with zero bias, without a grid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.sync_delta_v1.models import HEAD_DIM, HEADS, SyncDeltaModel


class FeatureSyncModel(SyncDeltaModel):
    """Sync-Delta with 4,224 additional shared Q/K feature-map parameters."""

    def __init__(self, num_flows: int) -> None:
        super().__init__(num_flows, variant="sync_delta")
        # Layout matches nn.Linear: [head, output coordinate, input coordinate].
        self.feature_weight = nn.Parameter(torch.eye(HEAD_DIM).repeat(HEADS, 1, 1))
        self.feature_bias = nn.Parameter(torch.zeros(HEADS, HEAD_DIM))

    def _map_address(self, address: Tensor) -> Tensor:
        """Map [...,head,32]; exactly the same function serves keys and queries."""
        logits = torch.einsum("...hd,hed->...he", address, self.feature_weight)
        positive = torch.softmax(math.sqrt(HEAD_DIM) * (logits + self.feature_bias), dim=-1)
        return F.normalize(positive, p=2, dim=-1, eps=1e-6)

    def _encode(
        self,
        x_observed: Tensor,
        mask: Tensor,
        stats: Mapping[str, Tensor],
        neighbors: Tensor,
        target_ids: Tensor,
        intervention: str | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        encoded = super()._encode(x_observed, mask, stats, neighbors, target_ids, intervention)
        keys, values, queries, event_mask, events, target_ids, mu, scale = encoded
        # Put heads next to the coordinate axis for the shared mapping, then
        # restore the reader layout [B,G,T,H,J,D].  Softmax maps a zero input to
        # nonzero features, so reapplying the observation mask is mandatory.
        mapped_keys = self._map_address(keys.transpose(-3, -2))
        mapped_keys = torch.where(event_mask[..., None, None], mapped_keys, torch.zeros_like(mapped_keys))
        keys = mapped_keys.transpose(-3, -2)
        queries = self._map_address(queries)
        return keys, values, queries, event_mask, events, target_ids, mu, scale


__all__ = ["FeatureSyncModel"]
