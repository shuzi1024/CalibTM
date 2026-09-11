import torch
import torch.nn as nn
import torch.nn.functional as F


GEOANCHOR_VARIANTS = {
    "value_extra_control",
    "mask_only_control",
    "agt_full",
    "agt_shuffled_geometry",
    "agt_no_anchor_values",
}


def is_geoanchor_variant(variant):
    return str(variant) in GEOANCHOR_VARIANTS


class ObservationGeometryExtractor(nn.Module):
    """Build Anchor-Gap Tokenization features from the observed mask and fill B.

    Inputs and outputs use the repo's imputation layout [B, T, N]. Ground truth is
    never used here; diagnostics that need targets are computed offline.
    """

    FEATURE_NAMES = (
        "B",
        "x_obs",
        "m",
        "x_left_anchor",
        "x_right_anchor",
        "d_left",
        "d_right",
        "gap_length",
        "relative_position",
        "anchor_delta",
        "anchor_slope",
        "gap_type",
        "is_edge_gap",
        "local_volatility",
        "interpolation_uncertainty",
    )

    def __init__(self, feature_set="full", uncertainty="equal_weight", eps=1e-6):
        super().__init__()
        self.feature_set = str(feature_set)
        self.uncertainty = str(uncertainty)
        self.eps = float(eps)

    @property
    def feature_dim(self):
        return len(self.FEATURE_NAMES)

    def forward(self, baseline, observed_mask):
        if baseline.dim() != 3:
            raise ValueError(f"GeoAnchor baseline must be [B,T,N], got {tuple(baseline.shape)}")
        if observed_mask is None:
            observed_mask = torch.ones_like(baseline)
        if observed_mask.shape != baseline.shape:
            raise ValueError(
                f"GeoAnchor observed_mask shape {tuple(observed_mask.shape)} "
                f"does not match baseline {tuple(baseline.shape)}"
            )

        x = baseline
        mask = observed_mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
        mask_bool = mask > 0.5
        bsz, steps, flows = x.shape
        dtype = x.dtype
        device = x.device
        eps = self.eps

        pos = torch.arange(steps, device=device).view(1, steps, 1).expand(bsz, steps, flows)
        pos_float = pos.to(dtype=dtype)

        missing = ~mask_bool
        left_seed = torch.where(mask_bool, pos, torch.full_like(pos, -1))
        left_idx = torch.cummax(left_seed, dim=1).values
        left_exists = left_idx >= 0

        right_seed = torch.where(mask_bool, pos, torch.full_like(pos, steps))
        right_idx = torch.flip(torch.cummin(torch.flip(right_seed, dims=(1,)), dim=1).values, dims=(1,))
        right_exists = right_idx < steps

        left_gather = left_idx.clamp(0, steps - 1)
        right_gather = right_idx.clamp(0, steps - 1)
        left_anchor = torch.gather(x, dim=1, index=left_gather)
        right_anchor = torch.gather(x, dim=1, index=right_gather)
        left_anchor = torch.where(left_exists, left_anchor, torch.zeros_like(left_anchor))
        right_anchor = torch.where(right_exists, right_anchor, torch.zeros_like(right_anchor))

        d_left_raw = (pos - left_idx).to(dtype=dtype)
        d_right_raw = (right_idx - pos).to(dtype=dtype)
        d_left_raw = torch.where(left_exists, d_left_raw, torch.full_like(d_left_raw, float(steps)))
        d_right_raw = torch.where(right_exists, d_right_raw, torch.full_like(d_right_raw, float(steps)))

        internal_gap = missing & left_exists & right_exists
        left_edge_gap = missing & (~left_exists) & right_exists
        right_edge_gap = missing & left_exists & (~right_exists)
        all_missing_gap = missing & (~left_exists) & (~right_exists)
        observed = mask_bool

        internal_length = d_left_raw + d_right_raw
        left_edge_length = (right_idx + 1).to(dtype=dtype)
        right_edge_length = (steps - left_idx).to(dtype=dtype)
        gap_length_raw = torch.zeros_like(x)
        gap_length_raw = torch.where(internal_gap, internal_length, gap_length_raw)
        gap_length_raw = torch.where(left_edge_gap, left_edge_length, gap_length_raw)
        gap_length_raw = torch.where(right_edge_gap, right_edge_length, gap_length_raw)
        gap_length_raw = torch.where(all_missing_gap, torch.full_like(gap_length_raw, float(steps)), gap_length_raw)

        relative_position = torch.zeros_like(x)
        relative_position = torch.where(
            internal_gap,
            d_left_raw / (internal_length + eps),
            relative_position,
        )
        relative_position = torch.where(right_edge_gap, torch.ones_like(relative_position), relative_position)

        anchor_delta = torch.abs(right_anchor - left_anchor)
        anchor_delta = torch.where(left_exists & right_exists, anchor_delta, torch.zeros_like(anchor_delta))
        anchor_slope = (right_anchor - left_anchor) / (gap_length_raw + eps)
        anchor_slope = torch.where(left_exists & right_exists & (gap_length_raw > 0), anchor_slope, torch.zeros_like(anchor_slope))

        gap_type = torch.zeros_like(x)
        gap_type = torch.where(internal_gap, torch.ones_like(gap_type), gap_type)
        gap_type = torch.where(left_edge_gap, torch.full_like(gap_type, 2.0), gap_type)
        gap_type = torch.where(right_edge_gap, torch.full_like(gap_type, 3.0), gap_type)
        gap_type = torch.where(all_missing_gap, torch.full_like(gap_type, 4.0), gap_type)
        is_edge_gap = (left_edge_gap | right_edge_gap).to(dtype=dtype)

        observed_count = mask.sum(dim=1, keepdim=True)
        observed_sum = (x * mask).sum(dim=1, keepdim=True)
        observed_mean = observed_sum / observed_count.clamp_min(1.0)
        observed_var = (((x - observed_mean) * mask).square().sum(dim=1, keepdim=True) / observed_count.clamp_min(1.0))
        observed_std = torch.sqrt(observed_var + eps)
        full_std = x.std(dim=1, unbiased=False, keepdim=True)
        local_volatility = torch.where(observed_count > 1.0, observed_std, full_std).expand_as(x)

        denom_t = max(steps - 1, 1)
        d_left = torch.where(left_exists, d_left_raw / float(denom_t), torch.ones_like(d_left_raw))
        d_right = torch.where(right_exists, d_right_raw / float(denom_t), torch.ones_like(d_right_raw))
        gap_length = gap_length_raw / float(max(steps, 1))
        distance_nearest = torch.minimum(d_left_raw, d_right_raw) / float(max(steps, 1))

        anchor_scale = anchor_delta.amax(dim=1, keepdim=True).clamp_min(eps)
        anchor_delta_norm = anchor_delta / anchor_scale
        vol_scale = local_volatility.amax(dim=(1, 2), keepdim=True).clamp_min(eps)
        local_volatility_norm = local_volatility / vol_scale

        gap_component = torch.log1p(gap_length_raw) / torch.log1p(
            torch.tensor(float(max(steps, 1)), device=device, dtype=dtype)
        )
        q = (
            gap_component
            + anchor_delta_norm
            + is_edge_gap
            + local_volatility_norm
            + distance_nearest
        ) / 5.0
        q = torch.where(observed, torch.zeros_like(q), q.clamp(0.0, 1.0))

        features = torch.stack(
            [
                x,
                x * mask,
                mask,
                left_anchor,
                right_anchor,
                d_left.clamp(0.0, 1.0),
                d_right.clamp(0.0, 1.0),
                gap_length.clamp(0.0, 1.0),
                relative_position.clamp(0.0, 1.0),
                anchor_delta_norm.clamp(0.0, 1.0),
                torch.tanh(anchor_slope),
                gap_type / 4.0,
                is_edge_gap,
                local_volatility_norm.clamp(0.0, 1.0),
                q,
            ],
            dim=-1,
        )

        if self.feature_set == "value_extra":
            value_abs = x.abs()
            prev = F.pad(x[:, 1:, :] - x[:, :-1, :], (0, 0, 1, 0))
            nxt = F.pad(x[:, 1:, :] - x[:, :-1, :], (0, 0, 0, 1))
            value_std = x.std(dim=1, unbiased=False, keepdim=True).expand_as(x)
            value_mean = x.mean(dim=1, keepdim=True).expand_as(x)
            zeros = torch.zeros_like(x)
            features = torch.stack(
                [
                    x,
                    value_abs,
                    x.square().clamp(max=10.0) / 10.0,
                    prev,
                    nxt,
                    value_mean,
                    value_std,
                    torch.tanh(x),
                    torch.tanh(prev),
                    torch.tanh(nxt),
                    value_abs / value_abs.amax(dim=1, keepdim=True).clamp_min(eps),
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                ],
                dim=-1,
            )
        elif self.feature_set == "mask_only":
            zeros = torch.zeros_like(x)
            features = torch.stack(
                [x, mask] + [zeros for _ in range(self.feature_dim - 2)],
                dim=-1,
            )
        elif self.feature_set == "no_anchor_values":
            features = features.clone()
            for idx in (3, 4):
                features[..., idx] = 0.0

        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class GeoFlow2Vec(nn.Module):
    """Geometry-aware flow token encoder used only by GeoAnchor variants.

    The channel projector maps AGT features at each time step back to a scalar
    traffic-like series, then an LSTM with the same input/output scale as the
    default Flow2Vec turns flows into LLM tokens. This keeps memory close to the
    value-only baseline while still making the scalar series geometry-aware.
    """

    def __init__(self, feature_dim, d_model, t_steps, dropout=0.1, geo_hidden=64):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.t_steps = int(t_steps)
        self.channel_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 1),
        )
        self.flow_encoder = nn.LSTM(
            input_size=self.t_steps,
            hidden_size=int(d_model),
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        with torch.no_grad():
            linear = self.channel_proj[1]
            linear.weight.zero_()
            linear.bias.zero_()
            linear.weight[:, 0].fill_(1.0)
        for name, param in self.flow_encoder.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, features):
        if features.dim() != 4:
            raise ValueError(f"GeoFlow2Vec features must be [B,T,N,C], got {tuple(features.shape)}")
        bsz, steps, flows, channels = features.shape
        if channels != self.feature_dim:
            raise ValueError(f"GeoFlow2Vec expected C={self.feature_dim}, got {channels}")
        if steps != self.t_steps:
            raise ValueError(f"GeoFlow2Vec expected T={self.t_steps}, got {steps}")
        scalar = self.channel_proj(features).squeeze(-1)
        flow_series = scalar.permute(0, 2, 1).contiguous()
        tokens, _ = self.flow_encoder(flow_series)
        return self.dropout(tokens)
