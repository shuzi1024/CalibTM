import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.salt_modules import (
    GlobalResidualHead,
    LatentTrafficCoupler,
    SameParamMLPControl,
    distance_to_observed,
    stage_observed_mask_like,
)


def _safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class GlobalCouplingExpert(nn.Module):
    """Standalone CARE global expert: X_G = B + R_G(C)."""

    def __init__(
        self,
        d_model,
        t_step,
        latent_slots=16,
        num_heads=4,
        latent_mixer="tiny_l1",
        dropout=0.1,
        flow_drop_rate=0.15,
        same_param_control=False,
        anchor_observed=True,
    ):
        super().__init__()
        self.flow_drop_rate = float(flow_drop_rate)
        self.anchor_observed = bool(anchor_observed)
        self.same_param_control = bool(same_param_control)
        if self.same_param_control:
            self.coupler = SameParamMLPControl(d_model, dropout=dropout)
        else:
            self.coupler = LatentTrafficCoupler(
                d_model=d_model,
                latent_slots=latent_slots,
                num_heads=num_heads,
                dropout=dropout,
                mixer_type=latent_mixer,
            )
        self.global_head = GlobalResidualHead(d_model, t_step, dropout=dropout, zero_init=True)
        self._diag_totals = {}
        self._diag_count = 0

    def _drop_flow_tokens(self, h_flow):
        if not self.training or self.flow_drop_rate <= 0.0:
            return h_flow, 0.0
        keep_prob = max(1.0 - self.flow_drop_rate, 1e-3)
        keep = (torch.rand(h_flow.shape[:2], device=h_flow.device) < keep_prob).to(h_flow.dtype)
        # Avoid fully empty samples.
        empty = keep.sum(dim=1) < 1.0
        if bool(empty.any()):
            keep[empty, 0] = 1.0
        return h_flow * keep.unsqueeze(-1) / keep_prob, float((1.0 - keep).mean().detach().item())

    def _record(self, diagnostics):
        self._diag_count += 1
        for key, value in diagnostics.items():
            value = _safe_float(value)
            if value is None:
                continue
            self._diag_totals[key] = self._diag_totals.get(key, 0.0) + value

    def get_diagnostics(self):
        out = {}
        if self._diag_count:
            for key, value in sorted(self._diag_totals.items()):
                out[key] = value / self._diag_count
        out["care_diagnostic_forwards"] = self._diag_count
        out["care_same_param_control"] = float(self.same_param_control)
        return out

    def reset_diagnostics(self):
        self._diag_totals = {}
        self._diag_count = 0

    def forward(self, h_flow, x_current, observed_mask, known_rate, target_rate):
        mask = stage_observed_mask_like(x_current, observed_mask, known_rate)
        density = mask.mean(dim=(1, 2), keepdim=True).view(x_current.shape[0], 1)
        dropped_h, actual_drop = self._drop_flow_tokens(h_flow)
        c_flow, attn_diag = self.coupler(dropped_h, known_rate, target_rate, density)
        residual = self.global_head(c_flow)
        if self.anchor_observed:
            flow_missing = (1.0 - mask.transpose(1, 2).contiguous()).to(residual.dtype)
            residual = residual * flow_missing
        residual_norm = torch.sqrt(residual.detach().square().mean().clamp_min(1e-12))
        diagnostics = {
            "care_global_residual_norm": float(residual_norm.item()),
            "care_observed_density": float(density.detach().mean().item()),
            "care_actual_flow_drop_rate": actual_drop,
        }
        diagnostics.update({
            "care_flow_to_latent_entropy": attn_diag.get("flow_to_latent_entropy"),
            "care_latent_to_flow_entropy": attn_diag.get("latent_to_flow_entropy"),
        })
        self._record(diagnostics)
        return residual, diagnostics


class ErrorAwareRouter(nn.Module):
    """Per-flow or per-flow-time MLP router for non-additive expert mixing."""

    def __init__(self, feature_dim, hidden=64, dropout=0.1):
        super().__init__()
        hidden = max(8, int(hidden))
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, features):
        return torch.sigmoid(self.net(features))


def normalize_to_probabilities(arr):
    arr = np.asarray(arr, dtype=np.float64)
    total = np.sum(arr)
    if abs(total) < 1e-12:
        return arr
    return arr / total


def metric_np(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    abs_denom = np.sum(np.abs(true))
    sq_denom = np.sqrt(np.sum(true ** 2))
    nmae = np.sum(np.abs(pred - true)) / abs_denom if abs_denom > 0 else np.sum(np.abs(pred - true))
    nrmse = np.sqrt(np.sum((pred - true) ** 2)) / sq_denom if sq_denom > 0 else np.sqrt(np.sum((pred - true) ** 2))
    eps = 1e-10
    p = np.clip(normalize_to_probabilities(true), eps, 1.0)
    q = np.clip(normalize_to_probabilities(pred), eps, 1.0)
    kl = float(np.sum(p * np.log(p / q)))
    return float(nmae), float(nrmse), kl


def distance_features_from_mask(mask):
    """Numpy distance-to-observed helper for arrays shaped [B,T,N]."""
    tensor = torch.as_tensor(mask, dtype=torch.float32)
    dist = distance_to_observed(tensor).numpy()
    return np.transpose(dist, (0, 2, 1))


def build_per_flow_router_features(x_local, x_global, baseline, mask, true):
    """Return per-flow router features shaped [B,N,F]."""
    x_local = np.asarray(x_local, dtype=np.float32)
    x_global = np.asarray(x_global, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    diff = np.abs(x_local - x_global)
    missing = 1.0 - mask
    dist = distance_features_from_mask(mask)
    eps = 1e-6
    denom = np.maximum(missing.sum(axis=1, keepdims=False), eps)
    def miss_mean(values):
        return (values * missing).sum(axis=1) / denom

    features = [
        miss_mean(x_local),
        miss_mean(x_global),
        miss_mean(diff),
        miss_mean(baseline),
        mask.mean(axis=1),
        missing.mean(axis=1),
        dist.mean(axis=1),
        dist.max(axis=1),
        true.std(axis=1),
        baseline.std(axis=1),
        np.abs(x_local - baseline).mean(axis=1),
        np.abs(x_global - baseline).mean(axis=1),
    ]
    stacked = np.stack(features, axis=-1)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def per_flow_router_targets(x_local, x_global, true, mask, margin=0.0):
    x_local = np.asarray(x_local, dtype=np.float32)
    x_global = np.asarray(x_global, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    missing = 1.0 - mask
    denom = np.maximum(missing.sum(axis=1), 1e-6)
    e_local = (np.abs(x_local - true) * missing).sum(axis=1) / denom
    e_global = (np.abs(x_global - true) * missing).sum(axis=1) / denom
    valid = (missing.sum(axis=1) > 0).astype(np.float32)
    target = (e_global < (e_local - float(margin))).astype(np.float32) * valid
    return target[..., None].astype(np.float32), valid[..., None].astype(np.float32)


class OracleRouterEvaluator:
    @staticmethod
    def oracle_prediction(x_local, x_global, true):
        choose_global = np.abs(x_global - true) < np.abs(x_local - true)
        return np.where(choose_global, x_global, x_local), choose_global

    @staticmethod
    def evaluate(x_local, x_global, true, mask):
        missing = np.asarray(mask) == 0
        x_oracle, choose_global = OracleRouterEvaluator.oracle_prediction(x_local, x_global, true)
        local_metrics = metric_np(np.asarray(x_local)[missing], np.asarray(true)[missing])
        global_metrics = metric_np(np.asarray(x_global)[missing], np.asarray(true)[missing])
        oracle_metrics = metric_np(np.asarray(x_oracle)[missing], np.asarray(true)[missing])
        return {
            "local_NMAE": local_metrics[0],
            "local_NRMSE": local_metrics[1],
            "local_KL": local_metrics[2],
            "global_NMAE": global_metrics[0],
            "global_NRMSE": global_metrics[1],
            "global_KL": global_metrics[2],
            "oracle_NMAE": oracle_metrics[0],
            "oracle_NRMSE": oracle_metrics[1],
            "oracle_KL": oracle_metrics[2],
            "global_better_rate": float(np.mean(choose_global[missing])) if np.any(missing) else 0.0,
            "oracle_gain": local_metrics[0] - oracle_metrics[0],
        }
