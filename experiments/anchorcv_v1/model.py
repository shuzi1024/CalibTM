"""Prior-free, mask-native neural expert for the AnchorCV prototype.

The expert receives only measurements that are currently observed, their mask,
time coordinates, and internally registered flow identities.  In particular,
there is no interpolation/prior input channel.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def hard_project_observations(
    prediction: Tensor,
    observed_values: Tensor,
    observed_mask: Tensor,
) -> Tensor:
    """Return a reconstruction that exactly preserves every observed value."""

    if prediction.ndim != 3:
        raise ValueError("prediction must have shape [B,F,T]")
    if observed_values.shape != prediction.shape:
        raise ValueError("observed_values must match prediction shape")
    if observed_mask.shape != prediction.shape:
        raise ValueError("observed_mask must match prediction shape")
    if observed_mask.dtype is not torch.bool:
        raise TypeError("observed_mask must be boolean")
    if (
        prediction.device != observed_values.device
        or prediction.device != observed_mask.device
    ):
        raise ValueError("prediction, observed_values, and observed_mask must share a device")
    return torch.where(
        observed_mask,
        observed_values.to(dtype=prediction.dtype),
        prediction,
    )


class PriorFreeMaskNativeExpert(nn.Module):
    """Directly reconstruct a traffic-matrix window from sparse observations.

    A bidirectional GRU summarizes each flow over time.  A noncausal
    Transformer encoder then exchanges information among all flows in the same
    window before a direct decoder predicts the complete normalized trajectory.
    """

    def __init__(
        self,
        *,
        num_flows: int,
        time_steps: int = 50,
        temporal_hidden: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        num_flow_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_flows", num_flows),
            ("time_steps", time_steps),
            ("temporal_hidden", temporal_hidden),
            ("d_model", d_model),
            ("num_heads", num_heads),
            ("num_flow_layers", num_flow_layers),
            ("dim_feedforward", dim_feedforward),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")

        self.num_flows = int(num_flows)
        self.time_steps = int(time_steps)
        self.d_model = int(d_model)

        # Channels: visible normalized value, visibility bit, normalized time.
        self.temporal_encoder = nn.GRU(
            input_size=3,
            hidden_size=int(temporal_hidden),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_projection = nn.Linear(2 * int(temporal_hidden), d_model)
        self.flow_embeddings = nn.Embedding(num_flows, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        flow_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.flow_encoder = nn.TransformerEncoder(
            flow_layer,
            num_layers=num_flow_layers,
            norm=nn.LayerNorm(d_model),
        )

        decoder_hidden = max(8, d_model // 2)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, decoder_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(decoder_hidden, time_steps),
        )
        nn.init.normal_(self.flow_embeddings.weight, mean=0.0, std=0.02)

    def _expand_time(
        self,
        normalized_time: Tensor,
        *,
        batch: int,
        flows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        if normalized_time.ndim == 1:
            if normalized_time.shape != (self.time_steps,):
                raise ValueError(
                    f"normalized_time must have length {self.time_steps}"
                )
            expanded = normalized_time.reshape(1, 1, self.time_steps).expand(
                batch, flows, -1
            )
        elif normalized_time.ndim == 2:
            if normalized_time.shape != (batch, self.time_steps):
                raise ValueError(
                    "normalized_time with rank 2 must have shape [B,T]"
                )
            expanded = normalized_time.reshape(batch, 1, self.time_steps).expand(
                -1, flows, -1
            )
        elif normalized_time.ndim == 3:
            if normalized_time.shape != (batch, flows, self.time_steps):
                raise ValueError(
                    "normalized_time with rank 3 must have shape [B,F,T]"
                )
            expanded = normalized_time
        else:
            raise ValueError("normalized_time must have shape [T], [B,T], or [B,F,T]")
        expanded = expanded.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(expanded).all()):
            raise ValueError("normalized_time must contain only finite values")
        return expanded

    def _validate_inputs(
        self,
        observed_values: Tensor,
        observed_mask: Tensor,
    ) -> tuple[int, int]:
        if observed_values.ndim != 3:
            raise ValueError("observed_values must have shape [B,F,T]")
        batch, flows, times = observed_values.shape
        if flows != self.num_flows:
            raise ValueError(
                f"observed_values has {flows} flows; num_flows is {self.num_flows}"
            )
        if times != self.time_steps:
            raise ValueError(
                f"observed_values has {times} steps; time_steps is {self.time_steps}"
            )
        if observed_mask.shape != observed_values.shape:
            raise ValueError("observed_mask must match observed_values shape")
        if observed_mask.dtype is not torch.bool:
            raise TypeError("observed_mask must be boolean")
        if observed_mask.device != observed_values.device:
            raise ValueError("observed_mask and observed_values must share a device")
        if not torch.is_floating_point(observed_values):
            raise TypeError("observed_values must be floating point")
        visible_values = observed_values[observed_mask]
        if visible_values.numel() and not bool(torch.isfinite(visible_values).all()):
            raise ValueError("observed entries must contain only finite values")
        return batch, flows

    def forward(
        self,
        observed_values: Tensor,
        observed_mask: Tensor,
        normalized_time: Tensor,
    ) -> Tensor:
        batch, flows = self._validate_inputs(observed_values, observed_mask)

        # Mask again inside the model so unobserved payload bytes have no
        # computational path to the output, even if a caller forgets to zero them.
        visible = torch.where(
            observed_mask,
            observed_values,
            torch.zeros((), dtype=observed_values.dtype, device=observed_values.device),
        )
        times = self._expand_time(
            normalized_time,
            batch=batch,
            flows=flows,
            dtype=observed_values.dtype,
            device=observed_values.device,
        )
        temporal_features = torch.stack(
            (visible, observed_mask.to(dtype=observed_values.dtype), times),
            dim=-1,
        )

        # Keep the recurrent front end in fp32 under mixed-precision callers;
        # the cross-flow encoder remains eligible for autocast.
        with torch.autocast(device_type=observed_values.device.type, enabled=False):
            recurrent_input = temporal_features.float().reshape(
                batch * flows, self.time_steps, 3
            )
            _, hidden = self.temporal_encoder(recurrent_input)
            bidirectional_summary = torch.cat((hidden[-2], hidden[-1]), dim=-1)
            flow_tokens = self.temporal_projection(bidirectional_summary)
        flow_tokens = flow_tokens.reshape(batch, flows, self.d_model)

        flow_ids = torch.arange(flows, device=observed_values.device)
        flow_tokens = flow_tokens + self.flow_embeddings(flow_ids).unsqueeze(0)
        encoded = self.flow_encoder(self.input_norm(flow_tokens))
        return self.decoder(encoded)


__all__ = ["PriorFreeMaskNativeExpert", "hard_project_observations"]
