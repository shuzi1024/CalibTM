import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


ACIL_VARIANTS = {
    "acil_frozen_linear",
    "acil_prior_only",
    "linearinterp_prior_diagnostic",
    "acil_prior_only_full",
    "acil_prior_only_value_sameparam",
    "acil_prior_only_shuffled_gap_geometry",
    "acil_prior_only_no_anchor_values",
    "acil_value_only_sameparam",
    "acil_shuffled_gap_geometry",
    "acil_shuffled_anchor_values",
    "acil_no_anchor_values",
    "acil_full",
    "acil_full_scratch",
    "acil_full_prior_init_freeze",
    "acil_full_prior_init_finetune",
    "acil_prior_freeze_baseline",
}


def is_acil_variant(variant):
    return str(variant) in ACIL_VARIANTS


class ObservationGeometryExtractorV2(nn.Module):
    """Extract per-time/per-flow anchor-gap geometry in the repo layout [B,T,N].

    The extractor only uses observed values, the observed mask, and the supplied
    interpolation baseline. It never consumes ground truth at missing positions.
    """

    FEATURE_NAMES = (
        "B_linear",
        "x_obs",
        "m",
        "d_left",
        "d_right",
        "gap_length",
        "relative_position",
        "gap_type",
        "is_edge_gap",
        "x_left_anchor",
        "x_right_anchor",
        "anchor_delta",
        "anchor_slope",
        "local_volatility",
        "interpolation_uncertainty",
        "distance_to_nearest_anchor",
    )

    GAP_GEOMETRY_FEATURES = {
        "d_left",
        "d_right",
        "gap_length",
        "relative_position",
        "gap_type",
        "is_edge_gap",
        "interpolation_uncertainty",
        "distance_to_nearest_anchor",
    }

    ANCHOR_VALUE_FEATURES = {
        "x_left_anchor",
        "x_right_anchor",
        "anchor_delta",
        "anchor_slope",
    }

    VALUE_ONLY_FEATURES = {
        "B_linear",
        "x_obs",
        "m",
        "local_volatility",
    }

    def __init__(self, feature_set="full", eps=1e-6):
        super().__init__()
        self.feature_set = str(feature_set)
        self.eps = float(eps)
        self._feature_index = {name: idx for idx, name in enumerate(self.FEATURE_NAMES)}

    @property
    def feature_dim(self):
        return len(self.FEATURE_NAMES)

    def _shuffle_token_geometry(self, features, names):
        if features.shape[0] * features.shape[2] <= 1:
            return features
        out = features.clone()
        indices = [self._feature_index[name] for name in names if name in self._feature_index]
        if not indices:
            return out
        bsz, steps, flows, channels = out.shape
        selected = out[..., indices].permute(0, 2, 1, 3).contiguous().view(bsz * flows, steps, len(indices))
        perm = torch.randperm(selected.shape[0], device=features.device)
        selected = selected[perm].view(bsz, flows, steps, len(indices)).permute(0, 2, 1, 3).contiguous()
        out[..., indices] = selected
        return out

    def _randomize_features(self, features, names):
        out = features.clone()
        indices = [self._feature_index[name] for name in names if name in self._feature_index]
        if indices:
            out[..., indices] = torch.rand_like(out[..., indices])
        return out

    def _scale_features(self, features, names, scale):
        out = features.clone()
        indices = [self._feature_index[name] for name in names if name in self._feature_index]
        if indices:
            out[..., indices] = out[..., indices] * float(scale)
        return out

    def _apply_feature_set(self, features):
        if self.feature_set == "full":
            return features
        out = features.clone()
        if self.feature_set == "value_only":
            keep = {self._feature_index[name] for name in self.VALUE_ONLY_FEATURES}
            for idx in range(out.shape[-1]):
                if idx not in keep:
                    out[..., idx] = 0.0
            return out
        if self.feature_set == "no_anchor_values":
            for name in self.ANCHOR_VALUE_FEATURES:
                out[..., self._feature_index[name]] = 0.0
            return out
        raise ValueError(f"Unsupported ACIL feature_set: {self.feature_set}")

    def forward(
        self,
        x_obs,
        observed_mask,
        b_linear=None,
        shuffle_gap_geometry=False,
        shuffle_anchor_values=False,
        corruption="none",
    ):
        if x_obs.dim() != 3:
            raise ValueError(f"ACIL x_obs must be [B,T,N], got {tuple(x_obs.shape)}")
        if observed_mask is None:
            observed_mask = torch.ones_like(x_obs)
        if observed_mask.shape != x_obs.shape:
            raise ValueError(
                f"ACIL observed_mask shape {tuple(observed_mask.shape)} does not match x_obs {tuple(x_obs.shape)}"
            )
        if b_linear is None:
            b_linear = x_obs
        if b_linear.shape != x_obs.shape:
            raise ValueError(f"ACIL b_linear shape {tuple(b_linear.shape)} does not match x_obs {tuple(x_obs.shape)}")

        mask = observed_mask.to(device=x_obs.device, dtype=x_obs.dtype).clamp(0.0, 1.0)
        mask_bool = mask > 0.5
        bsz, steps, flows = x_obs.shape
        device = x_obs.device
        dtype = x_obs.dtype
        eps = self.eps

        pos = torch.arange(steps, device=device).view(1, steps, 1).expand(bsz, steps, flows)
        source = torch.where(mask_bool, x_obs, torch.zeros_like(x_obs))
        missing = ~mask_bool

        left_seed = torch.where(mask_bool, pos, torch.full_like(pos, -1))
        left_idx = torch.cummax(left_seed, dim=1).values
        left_exists = left_idx >= 0

        right_seed = torch.where(mask_bool, pos, torch.full_like(pos, steps))
        right_idx = torch.flip(torch.cummin(torch.flip(right_seed, dims=(1,)), dim=1).values, dims=(1,))
        right_exists = right_idx < steps

        left_gather = left_idx.clamp(0, steps - 1)
        right_gather = right_idx.clamp(0, steps - 1)
        left_anchor = torch.gather(source, dim=1, index=left_gather)
        right_anchor = torch.gather(source, dim=1, index=right_gather)
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
        edge_gap = left_edge_gap | right_edge_gap | all_missing_gap

        internal_length = d_left_raw + d_right_raw
        left_edge_length = (right_idx + 1).to(dtype=dtype)
        right_edge_length = (steps - left_idx).to(dtype=dtype)
        gap_length_raw = torch.zeros_like(x_obs)
        gap_length_raw = torch.where(internal_gap, internal_length, gap_length_raw)
        gap_length_raw = torch.where(left_edge_gap, left_edge_length, gap_length_raw)
        gap_length_raw = torch.where(right_edge_gap, right_edge_length, gap_length_raw)
        gap_length_raw = torch.where(all_missing_gap, torch.full_like(gap_length_raw, float(steps)), gap_length_raw)

        relative_position = torch.zeros_like(x_obs)
        relative_position = torch.where(internal_gap, d_left_raw / (internal_length + eps), relative_position)
        relative_position = torch.where(right_edge_gap, torch.ones_like(relative_position), relative_position)

        anchor_delta = torch.abs(right_anchor - left_anchor)
        anchor_delta = torch.where(left_exists & right_exists, anchor_delta, torch.zeros_like(anchor_delta))
        anchor_slope = (right_anchor - left_anchor) / (gap_length_raw + eps)
        anchor_slope = torch.where(
            left_exists & right_exists & (gap_length_raw > 0.0),
            anchor_slope,
            torch.zeros_like(anchor_slope),
        )

        gap_type = torch.zeros_like(x_obs)
        gap_type = torch.where(internal_gap, torch.ones_like(gap_type), gap_type)
        gap_type = torch.where(left_edge_gap, torch.full_like(gap_type, 2.0), gap_type)
        gap_type = torch.where(right_edge_gap, torch.full_like(gap_type, 3.0), gap_type)
        gap_type = torch.where(all_missing_gap, torch.full_like(gap_type, 4.0), gap_type)

        observed_count = mask.sum(dim=1, keepdim=True)
        observed_sum = (source * mask).sum(dim=1, keepdim=True)
        observed_mean = observed_sum / observed_count.clamp_min(1.0)
        observed_var = ((source - observed_mean) * mask).square().sum(dim=1, keepdim=True) / observed_count.clamp_min(1.0)
        observed_std = torch.sqrt(observed_var + eps)
        full_std = b_linear.std(dim=1, unbiased=False, keepdim=True)
        local_volatility = torch.where(observed_count > 1.0, observed_std, full_std).expand_as(x_obs)

        denom_t = float(max(steps - 1, 1))
        d_left = torch.where(left_exists, d_left_raw / denom_t, torch.ones_like(d_left_raw))
        d_right = torch.where(right_exists, d_right_raw / denom_t, torch.ones_like(d_right_raw))
        gap_length = gap_length_raw / float(max(steps, 1))
        distance_nearest_raw = torch.minimum(d_left_raw, d_right_raw)
        distance_nearest_raw = torch.where(left_exists | right_exists, distance_nearest_raw, torch.full_like(d_left_raw, float(steps)))
        distance_nearest = distance_nearest_raw / float(max(steps, 1))

        anchor_scale = anchor_delta.amax(dim=1, keepdim=True).clamp_min(eps)
        anchor_delta_norm = anchor_delta / anchor_scale
        vol_scale = local_volatility.amax(dim=(1, 2), keepdim=True).clamp_min(eps)
        local_volatility_norm = local_volatility / vol_scale
        gap_component = torch.log1p(gap_length_raw) / torch.log1p(
            torch.tensor(float(max(steps, 1)), device=device, dtype=dtype)
        )
        uncertainty_q = (
            gap_component
            + anchor_delta_norm
            + edge_gap.to(dtype=dtype)
            + local_volatility_norm
            + distance_nearest
        ) / 5.0
        uncertainty_q = torch.where(mask_bool, torch.zeros_like(uncertainty_q), uncertainty_q.clamp(0.0, 1.0))

        x_observed_only = source
        features = torch.stack(
            [
                b_linear,
                x_observed_only,
                mask,
                d_left.clamp(0.0, 1.0),
                d_right.clamp(0.0, 1.0),
                gap_length.clamp(0.0, 1.0),
                relative_position.clamp(0.0, 1.0),
                (gap_type / 4.0).clamp(0.0, 1.0),
                edge_gap.to(dtype=dtype),
                torch.tanh(left_anchor),
                torch.tanh(right_anchor),
                anchor_delta_norm.clamp(0.0, 1.0),
                torch.tanh(anchor_slope),
                local_volatility_norm.clamp(0.0, 1.0),
                uncertainty_q,
                distance_nearest.clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        features = self._apply_feature_set(features)
        if shuffle_gap_geometry:
            features = self._shuffle_token_geometry(features, self.GAP_GEOMETRY_FEATURES)
        if shuffle_anchor_values:
            features = self._shuffle_token_geometry(features, self.ANCHOR_VALUE_FEATURES)

        corruption = str(corruption or "none")
        if corruption == "zero_geometry":
            zero_names = set(self.FEATURE_NAMES) - self.VALUE_ONLY_FEATURES
            for name in zero_names:
                features[..., self._feature_index[name]] = 0.0
        elif corruption == "random_geometry":
            features = self._randomize_features(features, set(self.FEATURE_NAMES) - self.VALUE_ONLY_FEATURES)
        elif corruption == "scaled_geometry_x10":
            features = self._scale_features(features, set(self.FEATURE_NAMES) - self.VALUE_ONLY_FEATURES, 10.0)
        elif corruption not in {"none", ""}:
            raise ValueError(f"Unsupported ACIL geometry corruption: {corruption}")

        return {
            "features": torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
            "x_left_anchor": left_anchor,
            "x_right_anchor": right_anchor,
            "d_left_raw": d_left_raw,
            "d_right_raw": d_right_raw,
            "gap_length_raw": gap_length_raw,
            "relative_position": relative_position,
            "gap_type": gap_type,
            "is_edge_gap": edge_gap.to(dtype=dtype),
            "left_exists": left_exists,
            "right_exists": right_exists,
            "internal_gap": internal_gap,
            "left_edge_gap": left_edge_gap,
            "right_edge_gap": right_edge_gap,
            "all_missing_gap": all_missing_gap,
            "missing": missing,
            "observed": mask_bool,
            "anchor_delta": anchor_delta,
            "anchor_slope": anchor_slope,
            "local_volatility": local_volatility,
            "uncertainty_q": uncertainty_q,
            "distance_to_nearest_anchor": distance_nearest_raw,
            "distance_to_nearest_anchor_norm": distance_nearest,
            "observed_density": mask.mean(),
            "anchor_coverage_ratio": (internal_gap.to(dtype=dtype).sum() / missing.to(dtype=dtype).sum().clamp_min(1.0)),
            "all_missing_ratio": (all_missing_gap.to(dtype=dtype).sum() / missing.to(dtype=dtype).sum().clamp_min(1.0)),
        }


class AnchorConditionedInterpolationLayer(nn.Module):
    """Learn a geometry-conditioned interpolation prior before Flow2Vec/ARI."""

    def __init__(
        self,
        feature_dim,
        hidden=64,
        beta_r=0.25,
        beta_o=0.10,
        beta_e=0.10,
        use_edge_extrapolation=True,
        eps=1e-6,
    ):
        super().__init__()
        hidden = max(8, int(hidden))
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.beta_r = float(beta_r)
        self.beta_o = float(beta_o)
        self.beta_e = float(beta_e)
        self.use_edge_extrapolation = bool(use_edge_extrapolation)
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.delta_r_head = nn.Linear(hidden, 1)
        self.offset_head = nn.Linear(hidden, 1)
        self.edge_offset_head = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for head in (self.delta_r_head, self.offset_head, self.edge_offset_head):
            nn.init.normal_(head.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(head.bias)

    def forward(self, b_linear, x_obs, observed_mask, geometry, freeze_to_linear=False):
        if freeze_to_linear:
            zeros = torch.zeros_like(b_linear)
            return b_linear, {
                "delta_r": zeros,
                "r_hat": geometry["relative_position"].to(dtype=b_linear.dtype),
                "offset": zeros,
                "edge_offset": zeros,
                "mean_abs_delta_r": b_linear.new_tensor(0.0),
                "mean_signed_delta_r": b_linear.new_tensor(0.0),
                "mean_abs_offset": b_linear.new_tensor(0.0),
                "mean_abs_edge_offset": b_linear.new_tensor(0.0),
                "mean_abs_B_acil_minus_B_linear": b_linear.new_tensor(0.0),
            }

        features = geometry["features"].to(device=b_linear.device, dtype=b_linear.dtype)
        hidden = self.trunk(features)
        delta_r = self.beta_r * torch.tanh(self.delta_r_head(hidden).squeeze(-1))
        scale_t = geometry["anchor_delta"].to(b_linear.dtype) + geometry["local_volatility"].to(b_linear.dtype) + self.eps
        offset = self.beta_o * torch.tanh(self.offset_head(hidden).squeeze(-1)) * scale_t
        edge_offset = self.beta_e * torch.tanh(self.edge_offset_head(hidden).squeeze(-1)) * geometry["local_volatility"].to(b_linear.dtype)

        d_left = geometry["d_left_raw"].to(b_linear.dtype)
        d_right = geometry["d_right_raw"].to(b_linear.dtype)
        r_linear = d_left / (d_left + d_right + self.eps)
        r_hat = (r_linear + delta_r).clamp(0.0, 1.0)

        left = geometry["x_left_anchor"].to(b_linear.dtype)
        right = geometry["x_right_anchor"].to(b_linear.dtype)
        internal_value = (1.0 - r_hat) * left + r_hat * right + offset
        edge_anchor = torch.where(geometry["left_exists"], left, right)
        edge_value = edge_anchor + edge_offset if self.use_edge_extrapolation else edge_anchor

        out = b_linear.clone()
        internal_gap = geometry["internal_gap"]
        edge_gap = geometry["left_edge_gap"] | geometry["right_edge_gap"]
        all_missing_gap = geometry["all_missing_gap"]
        out = torch.where(internal_gap, internal_value, out)
        out = torch.where(edge_gap, edge_value, out)
        out = torch.where(all_missing_gap, torch.zeros_like(out), out)
        out = torch.where(observed_mask > 0.5, x_obs, out)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        missing = geometry["missing"]
        edge_mask = geometry["left_edge_gap"] | geometry["right_edge_gap"] | geometry["all_missing_gap"]
        safe_missing = missing.to(b_linear.dtype).sum().clamp_min(1.0)
        safe_edge = edge_mask.to(b_linear.dtype).sum().clamp_min(1.0)
        diagnostics = {
            "delta_r": delta_r,
            "r_hat": r_hat,
            "offset": offset,
            "edge_offset": edge_offset,
            "mean_abs_delta_r": (delta_r.abs() * missing.to(b_linear.dtype)).sum() / safe_missing,
            "mean_signed_delta_r": (delta_r * missing.to(b_linear.dtype)).sum() / safe_missing,
            "mean_abs_offset": (offset.abs() * internal_gap.to(b_linear.dtype)).sum()
            / internal_gap.to(b_linear.dtype).sum().clamp_min(1.0),
            "mean_abs_edge_offset": (edge_offset.abs() * edge_mask.to(b_linear.dtype)).sum() / safe_edge,
            "mean_abs_B_acil_minus_B_linear": ((out - b_linear).abs() * missing.to(b_linear.dtype)).sum() / safe_missing,
        }
        return out, diagnostics


def tensor_mean_dict(diagnostics):
    out = {}
    for key, value in diagnostics.items():
        if torch.is_tensor(value) and value.dim() == 0:
            out[key] = float(value.detach().cpu().item())
    return out


def parameter_delta_from_state(module, initial_state):
    if not initial_state:
        return 0.0
    total = 0.0
    with torch.no_grad():
        for name, param in module.state_dict().items():
            if name not in initial_state:
                continue
            ref = initial_state[name].to(device=param.device, dtype=param.dtype)
            total += float((param.detach() - ref).square().sum().cpu().item())
    return math.sqrt(total)


@contextmanager
def preserve_torch_rng():
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
