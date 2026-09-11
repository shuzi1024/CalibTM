import math

import torch
import torch.nn as nn
import torch.nn.functional as F


GEOATTN_VARIANTS = {
    "acil_geoattn_true",
    "acil_geoattn_shuffled_bias",
    "acil_geoattn_random_bias",
    "acil_geoattn_zero_bias",
    "acil_geoattn_same_param_control",
}


def is_geoattn_variant(variant):
    return str(variant) in GEOATTN_VARIANTS


class ObservationGeometryDescriptor(nn.Module):
    """Build per-flow descriptors from observation-side geometry only."""

    FEATURE_NAMES = (
        "observed_density",
        "anchor_coverage_ratio",
        "mean_gap_length",
        "max_gap_length",
        "long_gap_ratio",
        "burst_gap_ratio",
        "internal_gap_ratio",
        "edge_gap_ratio",
        "gap_hist_1_2",
        "gap_hist_3_4",
        "gap_hist_5_8",
        "gap_hist_gt_8",
        "uncertainty_mean",
        "uncertainty_std",
        "uncertainty_top10_mean",
        "uncertainty_top20_mean",
        "relative_pos_0_25",
        "relative_pos_25_50",
        "relative_pos_50_75",
        "relative_pos_75_100",
    )

    def __init__(self, descriptor_set="full", normalize="batch", eps=1e-6):
        super().__init__()
        self.descriptor_set = str(descriptor_set or "full")
        self.normalize = str(normalize or "batch")
        self.eps = float(eps)
        if self.descriptor_set not in {"basic", "full"}:
            raise ValueError(f"Unsupported geoattn descriptor_set: {self.descriptor_set}")
        if self.normalize not in {"none", "batch"}:
            raise ValueError(f"Unsupported geoattn descriptor normalize mode: {self.normalize}")

    @property
    def feature_dim(self):
        return 12 if self.descriptor_set == "basic" else len(self.FEATURE_NAMES)

    def _top_fraction_mean(self, values, valid, fraction):
        bsz, flows, steps = values.shape
        k = max(1, int(math.ceil(float(steps) * float(fraction))))
        masked = values.masked_fill(~valid, -1.0)
        top = torch.topk(masked, k=min(k, steps), dim=-1).values
        keep = top >= 0.0
        return (top.clamp_min(0.0) * keep.to(values.dtype)).sum(dim=-1) / keep.to(values.dtype).sum(dim=-1).clamp_min(1.0)

    def forward(self, observed_mask):
        if observed_mask is None:
            raise ValueError("GeoAttn requires observed_mask to build observation geometry descriptors")
        if observed_mask.dim() != 3:
            raise ValueError(f"GeoAttn observed_mask must be [B,T,N], got {tuple(observed_mask.shape)}")

        mask = observed_mask.float().clamp(0.0, 1.0)
        bsz, steps, flows = mask.shape
        device = mask.device
        dtype = mask.dtype
        eps = self.eps

        mask_bool = mask > 0.5
        missing = ~mask_bool
        pos = torch.arange(steps, device=device).view(1, steps, 1).expand(bsz, steps, flows)

        left_seed = torch.where(mask_bool, pos, torch.full_like(pos, -1))
        left_idx = torch.cummax(left_seed, dim=1).values
        left_exists = left_idx >= 0

        right_seed = torch.where(mask_bool, pos, torch.full_like(pos, steps))
        right_idx = torch.flip(torch.cummin(torch.flip(right_seed, dims=(1,)), dim=1).values, dims=(1,))
        right_exists = right_idx < steps

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
        gap_length_raw = torch.zeros_like(mask)
        gap_length_raw = torch.where(internal_gap, internal_length, gap_length_raw)
        gap_length_raw = torch.where(left_edge_gap, left_edge_length, gap_length_raw)
        gap_length_raw = torch.where(right_edge_gap, right_edge_length, gap_length_raw)
        gap_length_raw = torch.where(all_missing_gap, torch.full_like(gap_length_raw, float(steps)), gap_length_raw)

        relative_position = torch.zeros_like(mask)
        relative_position = torch.where(internal_gap, d_left_raw / (internal_length + eps), relative_position)
        relative_position = torch.where(right_edge_gap, torch.ones_like(relative_position), relative_position)

        distance_nearest_raw = torch.minimum(d_left_raw, d_right_raw)
        distance_nearest_raw = torch.where(
            left_exists | right_exists,
            distance_nearest_raw,
            torch.full_like(distance_nearest_raw, float(steps)),
        )
        distance_nearest = (distance_nearest_raw / float(max(steps, 1))).clamp(0.0, 1.0)

        missing_f = missing.to(dtype)
        internal_f = internal_gap.to(dtype)
        edge_f = edge_gap.to(dtype)
        miss_count = missing_f.sum(dim=1).clamp_min(1.0)
        internal_count = internal_f.sum(dim=1).clamp_min(1.0)

        observed_density = mask.mean(dim=1)
        anchor_coverage = internal_f.sum(dim=1) / miss_count
        gap_norm = gap_length_raw / float(max(steps, 1))
        mean_gap = (gap_norm * missing_f).sum(dim=1) / miss_count
        max_gap = gap_norm.masked_fill(~missing, 0.0).amax(dim=1)
        long_gap_ratio = ((gap_length_raw > 8.0) & missing).to(dtype).sum(dim=1) / miss_count
        burst_gap_ratio = ((gap_length_raw <= 8.0) & internal_gap).to(dtype).sum(dim=1) / miss_count
        internal_gap_ratio = internal_f.sum(dim=1) / miss_count
        edge_gap_ratio = edge_f.sum(dim=1) / miss_count

        hist_1_2 = (((gap_length_raw >= 1.0) & (gap_length_raw <= 2.0) & missing).to(dtype).sum(dim=1) / miss_count)
        hist_3_4 = (((gap_length_raw >= 3.0) & (gap_length_raw <= 4.0) & missing).to(dtype).sum(dim=1) / miss_count)
        hist_5_8 = (((gap_length_raw >= 5.0) & (gap_length_raw <= 8.0) & missing).to(dtype).sum(dim=1) / miss_count)
        hist_gt_8 = (((gap_length_raw > 8.0) & missing).to(dtype).sum(dim=1) / miss_count)

        gap_component = torch.log1p(gap_length_raw) / math.log1p(float(max(steps, 1)))
        sparse_component = (1.0 - observed_density).view(bsz, 1, flows).expand_as(mask)
        uncertainty_q = (
            gap_component.clamp(0.0, 1.0)
            + edge_f
            + distance_nearest
            + sparse_component.clamp(0.0, 1.0)
        ) / 4.0
        uncertainty_q = torch.where(missing, uncertainty_q.clamp(0.0, 1.0), torch.zeros_like(uncertainty_q))
        q_mean = (uncertainty_q * missing_f).sum(dim=1) / miss_count
        q_var = ((uncertainty_q - q_mean.view(bsz, 1, flows)).square() * missing_f).sum(dim=1) / miss_count
        q_std = torch.sqrt(q_var + eps)
        q_flow = uncertainty_q.permute(0, 2, 1).contiguous()
        missing_flow = missing.permute(0, 2, 1).contiguous()
        q_top10 = self._top_fraction_mean(q_flow, missing_flow, 0.10)
        q_top20 = self._top_fraction_mean(q_flow, missing_flow, 0.20)

        rel_0_25 = (((relative_position >= 0.0) & (relative_position < 0.25) & internal_gap).to(dtype).sum(dim=1) / internal_count)
        rel_25_50 = (((relative_position >= 0.25) & (relative_position < 0.50) & internal_gap).to(dtype).sum(dim=1) / internal_count)
        rel_50_75 = (((relative_position >= 0.50) & (relative_position < 0.75) & internal_gap).to(dtype).sum(dim=1) / internal_count)
        rel_75_100 = (((relative_position >= 0.75) & (relative_position <= 1.0) & internal_gap).to(dtype).sum(dim=1) / internal_count)

        features = [
            observed_density,
            anchor_coverage,
            mean_gap,
            max_gap,
            long_gap_ratio,
            burst_gap_ratio,
            internal_gap_ratio,
            edge_gap_ratio,
            hist_1_2,
            hist_3_4,
            hist_5_8,
            hist_gt_8,
        ]
        if self.descriptor_set == "full":
            features.extend([
                q_mean,
                q_std,
                q_top10,
                q_top20,
                rel_0_25,
                rel_25_50,
                rel_50_75,
                rel_75_100,
            ])
        descriptor = torch.stack(features, dim=-1)
        descriptor = torch.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize == "batch":
            mean = descriptor.mean(dim=1, keepdim=True)
            std = descriptor.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
            descriptor = (descriptor - mean) / std
            descriptor = torch.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)
        return descriptor


class PairwiseGeoBias(nn.Module):
    def __init__(self, bias_type="true", standardize=True, eps=1e-6):
        super().__init__()
        self.bias_type = str(bias_type or "true")
        self.standardize = bool(standardize)
        self.eps = float(eps)
        if self.bias_type not in {"true", "shuffled", "random", "zero"}:
            raise ValueError(f"Unsupported geoattn bias_type: {self.bias_type}")

    def _shuffle_descriptors(self, descriptors):
        if descriptors.shape[1] <= 1:
            return descriptors
        shuffled = []
        for batch in range(descriptors.shape[0]):
            perm = torch.randperm(descriptors.shape[1], device=descriptors.device)
            shuffled.append(descriptors[batch, perm])
        return torch.stack(shuffled, dim=0)

    def _fixed_random_bias(self, batch_size, flows, device, dtype):
        idx = torch.arange(flows, device=device, dtype=dtype)
        row = idx.view(flows, 1)
        col = idx.view(1, flows)
        values = torch.sin((row + 1.0) * 12.9898 + (col + 1.0) * 78.233) * 43758.5453
        values = values - torch.floor(values)
        values = values * 2.0 - 1.0
        values = 0.5 * (values + values.transpose(0, 1))
        return values.view(1, flows, flows).expand(batch_size, -1, -1)

    def _standardize(self, bias):
        if not self.standardize:
            return bias.clamp(-1.0, 1.0)
        mean = bias.mean(dim=(-2, -1), keepdim=True)
        std = bias.std(dim=(-2, -1), unbiased=False, keepdim=True).clamp_min(self.eps)
        return torch.nan_to_num((bias - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0)

    def forward(self, descriptors):
        if descriptors.dim() != 3:
            raise ValueError(f"GeoAttn descriptors must be [B,N,D], got {tuple(descriptors.shape)}")
        bsz, flows, _ = descriptors.shape
        if self.bias_type == "zero":
            sim = descriptors.new_zeros((bsz, flows, flows))
            return sim.unsqueeze(1), sim
        if self.bias_type == "random":
            sim = self._fixed_random_bias(bsz, flows, descriptors.device, descriptors.dtype)
            bias = self._standardize(sim)
            return bias.unsqueeze(1), sim
        source = self._shuffle_descriptors(descriptors) if self.bias_type == "shuffled" else descriptors
        normalized = F.normalize(source, dim=-1, eps=self.eps)
        sim = torch.matmul(normalized, normalized.transpose(-2, -1)).clamp(-1.0, 1.0)
        bias = self._standardize(sim)
        return bias.unsqueeze(1), sim


def _pearson_batch(x, y, mask, eps=1e-8):
    mask_f = mask.to(x.dtype)
    count = mask_f.sum(dim=-1).clamp_min(1.0)
    x_mean = (x * mask_f).sum(dim=-1) / count
    y_mean = (y * mask_f).sum(dim=-1) / count
    xc = (x - x_mean.unsqueeze(-1)) * mask_f
    yc = (y - y_mean.unsqueeze(-1)) * mask_f
    denom = torch.sqrt((xc.square().sum(dim=-1) * yc.square().sum(dim=-1)).clamp_min(eps))
    corr = (xc * yc).sum(dim=-1) / denom
    corr = torch.where(count > 1.0, corr, torch.zeros_like(corr))
    return corr.mean()


def geoattn_tensor_diagnostics(bias, sim, attn, gamma, descriptors, topk=5):
    out = {}
    if bias is None or sim is None:
        return out
    with torch.no_grad():
        b = bias.detach()
        if b.dim() == 4:
            b_flat = b.mean(dim=1)
        else:
            b_flat = b
        out["gamma_final"] = float(gamma.detach().item())
        out["mean_geo_bias"] = float(b_flat.mean().item())
        out["std_geo_bias"] = float(b_flat.std(unbiased=False).item())
        out["geo_bias_variance"] = float(b_flat.var(unbiased=False).item())
        probs = torch.softmax(b_flat.reshape(b_flat.shape[0], -1), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        max_entropy = math.log(max(1, probs.shape[-1]))
        out["geo_bias_entropy"] = float((entropy / max_entropy).mean().item()) if max_entropy > 0 else 0.0
        out["descriptor_mean"] = float(descriptors.detach().mean().item())
        out["descriptor_std"] = float(descriptors.detach().std(unbiased=False).item())
        if attn is not None:
            attn_mean = attn.detach().mean(dim=1)
            bsz, flows, _ = attn_mean.shape
            off_diag = ~torch.eye(flows, device=attn_mean.device, dtype=torch.bool).view(1, flows, flows)
            out["attention_geo_similarity_corr"] = float(
                _pearson_batch(attn_mean.reshape(bsz, -1), sim.detach().reshape(bsz, -1), off_diag.expand(bsz, -1, -1).reshape(bsz, -1)).item()
            )
            k = min(max(1, int(topk)), max(1, flows - 1))
            sim_for_topk = sim.detach().masked_fill(torch.eye(flows, device=sim.device, dtype=torch.bool).view(1, flows, flows), -float("inf"))
            indices = torch.topk(sim_for_topk, k=k, dim=-1).indices
            mass = torch.gather(attn_mean, dim=-1, index=indices).sum(dim=-1)
            out["topk_geo_similar_attention_mass"] = float(mass.mean().item())
            uniform_mass = float(k) / float(max(flows, 1))
            out["topk_geo_similar_attention_mass_over_uniform"] = float(mass.mean().item() - uniform_mass)
    return out


class GeometryBiasedAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, residual_scale=0.1, zero_init=False, use_ffn=True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by geoattn_num_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(d_model, d_model)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(d_ff, d_model),
            )
        else:
            self.norm2 = None
            self.ffn = None
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            if self.ffn is not None:
                nn.init.zeros_(self.ffn[-1].weight)
                nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x, geo_bias, gamma, collect_attention=False):
        bsz, flows, d_model = x.shape
        residual = x
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).view(bsz, flows, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if geo_bias is not None:
            scores = scores + gamma.to(dtype=scores.dtype, device=scores.device) * geo_bias.to(dtype=scores.dtype)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.attn_dropout(attn), v).transpose(1, 2).contiguous().view(bsz, flows, d_model)
        x = residual + self.residual_scale * self.out_proj(context)
        if self.ffn is not None:
            x = x + self.residual_scale * self.ffn(self.norm2(x))
        return x, attn.detach() if collect_attention else None


class GeoAttnBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.bias_type = str(getattr(configs, "geoattn_bias_type", "true"))
        self.same_param_control = bool(getattr(configs, "geoattn_same_param_control", 0))
        self.freeze_to_zero = bool(getattr(configs, "geoattn_freeze_to_zero", 0))
        self.save_diagnostics = bool(getattr(configs, "geoattn_save_diagnostics", 0))
        self.identity_when_gamma_zero = bool(getattr(configs, "geoattn_identity_when_gamma_zero", 1))
        self.topk = int(getattr(configs, "geoattn_diag_topk", 5))
        self.descriptor = ObservationGeometryDescriptor(
            descriptor_set=getattr(configs, "geoattn_descriptor_set", "full"),
            normalize=getattr(configs, "geoattn_descriptor_normalize", "batch"),
        )
        self.pairwise = PairwiseGeoBias(
            bias_type=self.bias_type,
            standardize=bool(getattr(configs, "geoattn_standardize_bias", 1)),
        )
        gamma_init = 0.0 if self.freeze_to_zero else float(getattr(configs, "geoattn_gamma_init", 0.1))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.gamma.requires_grad = bool(getattr(configs, "geoattn_gamma_learnable", 1)) and not self.freeze_to_zero
        layers = max(1, int(getattr(configs, "geoattn_num_layers", 1)))
        self.layers = nn.ModuleList([
            GeometryBiasedAttentionLayer(
                d_model=int(getattr(configs, "d_model")),
                n_heads=int(getattr(configs, "geoattn_num_heads", 4)),
                d_ff=int(getattr(configs, "d_ff")),
                dropout=float(getattr(configs, "geoattn_dropout", getattr(configs, "dropout", 0.1))),
                residual_scale=float(getattr(configs, "geoattn_residual_init", 0.1)),
                zero_init=bool(getattr(configs, "geoattn_zero_init", 0)),
                use_ffn=bool(getattr(configs, "geoattn_ffn", 1)),
            )
            for _ in range(layers)
        ])
        self._diag_totals = {}
        self._diag_count = 0

    def reset_diagnostics(self):
        self._diag_totals = {}
        self._diag_count = 0

    def record_diagnostics(self, diagnostics):
        if not diagnostics:
            return
        self._diag_count += 1
        for key, value in diagnostics.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                self._diag_totals[key] = self._diag_totals.get(key, 0.0) + value

    def get_diagnostics(self):
        out = {}
        if self._diag_count:
            for key, value in sorted(self._diag_totals.items()):
                out[key] = value / self._diag_count
        out["gamma_final"] = float(self.gamma.detach().item())
        out["geoattn_diagnostic_forwards"] = self._diag_count
        out["geoattn_bias_type"] = self.bias_type
        out["geoattn_same_param_control"] = int(self.same_param_control)
        return out

    def forward(self, x, observed_mask):
        if self.freeze_to_zero:
            return x
        if (
            self.identity_when_gamma_zero
            and not self.gamma.requires_grad
            and float(self.gamma.detach().cpu().item()) == 0.0
        ):
            return x
        descriptors = self.descriptor(observed_mask).to(device=x.device, dtype=x.dtype)
        bias, sim = self.pairwise(descriptors)
        bias = bias.to(device=x.device, dtype=x.dtype)
        sim = sim.to(device=x.device, dtype=x.dtype)
        if self.same_param_control:
            bias = torch.zeros_like(bias)
        last_attn = None
        collect_attention = self.save_diagnostics or not self.training
        for layer in self.layers:
            x, maybe_attn = layer(x, bias, self.gamma, collect_attention=collect_attention)
            if maybe_attn is not None:
                last_attn = maybe_attn
        if collect_attention:
            self.record_diagnostics(geoattn_tensor_diagnostics(bias, sim, last_attn, self.gamma, descriptors, self.topk))
        return x

