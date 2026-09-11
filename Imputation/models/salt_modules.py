import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def observed_indices_for_rate(length, rate, device):
    if float(rate) >= 100.0:
        return torch.arange(length, device=device)
    count = int(torch.ceil(torch.tensor(length * float(rate) / 100.0)).item())
    if length > 1:
        count = max(2, count)
    count = min(length, max(1, count))
    idx = torch.linspace(0, length - 1, count, device=device).round().long().unique(sorted=True)
    if idx[-1].item() != length - 1:
        idx = torch.cat([idx, torch.tensor([length - 1], device=device)])
    if idx[0].item() != 0:
        idx = torch.cat([torch.tensor([0], device=device), idx])
    return idx.unique(sorted=True)


def stage_observed_mask_like(x_current, observed_mask, known_rate):
    """Return a float mask with shape [B, T, N] aligned to x_current."""
    bsz, steps, flows = x_current.shape
    if observed_mask is None:
        mask = x_current.new_zeros((1, steps, 1))
        mask[:, observed_indices_for_rate(steps, known_rate, x_current.device), :] = 1.0
        return mask.expand(bsz, steps, flows)
    mask = observed_mask.to(device=x_current.device, dtype=x_current.dtype)
    if mask.shape[1] < steps:
        mask = F.pad(mask, (0, 0, 0, steps - mask.shape[1]))
    elif mask.shape[1] > steps:
        mask = mask[:, :steps, :]
    if mask.shape[2] < flows:
        mask = F.pad(mask, (0, flows - mask.shape[2]))
    elif mask.shape[2] > flows:
        mask = mask[:, :, :flows]
    return mask


def distance_to_observed(observed_mask):
    """Compute per-position distance to the nearest observed time point.

    Input is [B, T, N]. Output is [B, N, T], matching flow-token residual
    heads. The result is detached-style metadata; it is differentiable but is
    intended as a mask feature rather than a learned path.
    """
    mask = observed_mask > 0
    bsz, steps, flows = mask.shape
    flow_mask = mask.transpose(1, 2).contiguous()
    positions = torch.arange(steps, device=mask.device).view(1, 1, steps)
    large = torch.full_like(positions, steps + 1)

    left_seed = torch.where(flow_mask, positions, -large)
    left = torch.cummax(left_seed, dim=-1).values
    right_seed = torch.where(flow_mask, positions, large)
    right = torch.flip(torch.cummin(torch.flip(right_seed, dims=(-1,)), dim=-1).values, dims=(-1,))

    dist_left = positions - left
    dist_right = right - positions
    dist = torch.minimum(dist_left, dist_right).float()
    has_obs = flow_mask.any(dim=-1, keepdim=True)
    dist = torch.where(has_obs, dist, torch.full_like(dist, float(steps)))
    return dist


def _stage_features(batch_size, known_rate, target_rate, density, device, dtype):
    known = torch.full((batch_size, 1), float(known_rate) / 100.0, device=device, dtype=dtype)
    target = torch.full((batch_size, 1), float(target_rate) / 100.0, device=device, dtype=dtype)
    delta = (target - known).clamp_min(0.0)
    return torch.cat([known, target, delta, density.to(device=device, dtype=dtype).view(batch_size, 1)], dim=-1)


def _attention_entropy(attn):
    if attn is None:
        return None
    probs = attn.detach().clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    denom = math.log(max(probs.shape[-1], 2))
    return float((entropy / denom).mean().item())


class LatentTrafficCoupler(nn.Module):
    def __init__(self, d_model, latent_slots=16, num_heads=4, dropout=0.1, mixer_type="tiny_l1"):
        super().__init__()
        if d_model % int(num_heads) != 0:
            raise ValueError(f"d_model={d_model} must be divisible by salt_num_heads={num_heads}")
        self.latent_slots = int(latent_slots)
        self.d_model = int(d_model)
        self.mixer_type = str(mixer_type)
        self.slot_tokens = nn.Parameter(torch.randn(1, self.latent_slots, self.d_model) * 0.02)
        self.flow_norm = nn.LayerNorm(self.d_model)
        self.slot_norm = nn.LayerNorm(self.d_model)
        self.write_attn = nn.MultiheadAttention(
            self.d_model,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.read_attn = nn.MultiheadAttention(
            self.d_model,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.stage_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        if self.mixer_type == "none":
            self.mixer = nn.Identity()
        elif self.mixer_type in ("tiny_l1", "tiny_l2", "gpt2_l1"):
            layers = 1 if self.mixer_type in ("tiny_l1", "gpt2_l1") else 2
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=int(num_heads),
                dim_feedforward=self.d_model * 2,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mixer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        else:
            raise ValueError(f"Unsupported salt_latent_mixer: {self.mixer_type}")
        self.out_norm = nn.LayerNorm(self.d_model)

    def forward(self, h_flow, known_rate, target_rate, observed_density):
        if h_flow.dim() != 3:
            raise ValueError(f"LatentTrafficCoupler expects [B,N,d], got {tuple(h_flow.shape)}")
        bsz, _, d_model = h_flow.shape
        if d_model != self.d_model:
            raise ValueError(f"LatentTrafficCoupler d_model mismatch: got {d_model}, expected {self.d_model}")
        h_norm = self.flow_norm(h_flow)
        slots = self.slot_tokens.expand(bsz, -1, -1)
        stage = _stage_features(bsz, known_rate, target_rate, observed_density, h_flow.device, h_flow.dtype)
        slots = slots + self.stage_proj(stage).unsqueeze(1)
        written, write_attn = self.write_attn(
            query=self.slot_norm(slots),
            key=h_norm,
            value=h_norm,
            need_weights=True,
            average_attn_weights=False,
        )
        slots = slots + written
        slots = self.mixer(slots)
        read, read_attn = self.read_attn(
            query=h_norm,
            key=self.slot_norm(slots),
            value=self.slot_norm(slots),
            need_weights=True,
            average_attn_weights=False,
        )
        diagnostics = {
            "flow_to_latent_entropy": _attention_entropy(write_attn),
            "latent_to_flow_entropy": _attention_entropy(read_attn),
        }
        return self.out_norm(read), diagnostics


class SameParamMLPControl(nn.Module):
    def __init__(self, d_model, dropout=0.1, layers=2, expansion=2):
        super().__init__()
        blocks = []
        hidden = int(d_model) * int(expansion)
        for _ in range(max(1, int(layers))):
            blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden, d_model),
                ),
            }))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h_flow, known_rate=None, target_rate=None, observed_density=None):
        x = h_flow
        for block in self.blocks:
            x = x + block["ffn"](block["norm"](x))
        return self.out_norm(x), {
            "flow_to_latent_entropy": None,
            "latent_to_flow_entropy": None,
        }


class GlobalResidualHead(nn.Module):
    def __init__(self, d_model, t_step, dropout=0.1, zero_init=True):
        super().__init__()
        hidden = max(128, int(d_model) // 2)
        self.t_step = int(t_step)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.t_step),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, c_flow):
        if c_flow.dim() != 3:
            raise ValueError(f"GlobalResidualHead expects [B,N,d], got {tuple(c_flow.shape)}")
        bsz, flows, d_model = c_flow.shape
        out = self.net(c_flow.reshape(bsz * flows, d_model))
        return out.reshape(bsz, flows, self.t_step)


class StageAwareUncertaintyGate(nn.Module):
    def __init__(self, d_model, gate_bias=-2.0):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        hidden = max(32, int(d_model) // 2)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(gate_bias))

    def forward(self, h_flow, c_flow, x_current, observed_mask, known_rate, target_rate):
        if h_flow.dim() != 3 or c_flow.dim() != 3:
            raise ValueError("StageAwareUncertaintyGate expects h_flow and c_flow with shape [B,N,d]")
        bsz, steps, flows = x_current.shape
        if h_flow.shape[:2] != (bsz, flows):
            raise ValueError(
                f"Gate flow shape mismatch: h_flow={tuple(h_flow.shape)} x_current={tuple(x_current.shape)}"
            )
        mask = stage_observed_mask_like(x_current, observed_mask, known_rate)
        flow_mask = mask.transpose(1, 2).contiguous()
        current = x_current.transpose(1, 2).contiguous()
        reliability = flow_mask.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        missing = 1.0 - reliability
        distance = distance_to_observed(mask) / max(float(steps - 1), 1.0)
        dist_mean = distance.mean(dim=-1, keepdim=True)
        dist_max = distance.max(dim=-1, keepdim=True).values
        abs_mean = current.detach().abs().mean(dim=-1, keepdim=True)
        volatility = current.detach().std(dim=-1, unbiased=False, keepdim=True)
        density = mask.mean(dim=(1, 2), keepdim=True).view(bsz, 1)
        stage = _stage_features(bsz, known_rate, target_rate, density, x_current.device, x_current.dtype)
        stage = stage.unsqueeze(1).expand(-1, flows, -1)
        features = torch.cat(
            [reliability, missing, dist_mean, dist_max, abs_mean, volatility, stage],
            dim=-1,
        )
        gate_hidden = self.hidden_norm(h_flow) + self.context_norm(c_flow) + self.feature_proj(features)
        gate = torch.sigmoid(self.gate_head(gate_hidden))
        return gate, {
            "mask": mask,
            "distance": distance,
            "reliability": reliability,
        }
