"""Observation-only CalibTM API around the unmodified SPIN model architecture."""

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from .source_model import SPINModel


MODEL_CONFIG = dict(input_size=1, hidden_size=32, u_size=1, output_size=1,
                    temporal_self_attention=True, reweight="softmax", n_layers=4,
                    eta=3, message_layers=1)
TRAINING = dict(optimizer="Adam", lr=0.0008, weight_decay=0.0,
                betas=[0.9, 0.999], eps=1e-8, max_epochs=300,
                patience=40, effective_batch=8, physical_batch=1,
                gradient_clip=5.0, prediction_loss_weight=1.0,
                scheduler="CosineSchedulerWithRestarts", num_warmup_steps=12,
                min_factor=0.1, linear_decay=0.67, num_cycles=3,
                scheduler_interval="epoch", selection="mean_condition_nmae",
                precision="float32", loss="sum_of_four_scaled_missing_L1")


class SPINAdapter(nn.Module):
    variant = "spin_adapted"

    def __init__(self, num_flows):
        super().__init__()
        self.num_flows = int(num_flows)
        self.core = SPINModel(n_nodes=self.num_flows, **MODEL_CONFIG)
        self.register_buffer("fit_mean", torch.zeros(()))
        self.register_buffer("fit_scale", torch.ones(()))
        self.register_buffer("fit_configured", torch.tensor(False))

    def configure_fit_statistics(self, statistics: Mapping):
        """Global fit StandardScaler from registered per-flow sufficient stats.

        Original SPIN StandardScaler(axis=(0,1)) on [time,nodes,channels] uses
        one population mean/std. Here C supplies the exact global mean; the
        global second moment combines fit RMS std and dispersion of fit means.
        Only the latter summand inherits the parent mu array's FP32 precision.
        """
        mu = np.asarray(statistics["mu"], dtype=np.float64)
        mean = float(statistics.get("C", mu.mean()))
        within = float(statistics["global_scale"]) ** 2
        variance = within + float(np.mean((mu - mean) ** 2))
        scale = max(variance ** .5, 1e-8)
        if mu.shape != (self.num_flows,) or not np.isfinite([mean, scale]).all():
            raise ValueError("Invalid fit-only global statistics")
        self.fit_mean.fill_(mean)
        self.fit_scale.fill_(scale)
        self.fit_configured.fill_(True)
        return {"mean": mean, "scale": scale, "fit_only": True,
                "normalization": "global_population_std_axes_time_nodes",
                "mu_input_precision": "parent_fp32"}

    def configure_from_bundle(self, bundle):
        return self.configure_fit_statistics(dict(mu=bundle.mu, C=bundle.C,
            global_scale=bundle.metadata["global_scale"]))

    def set_execution(self, *, node_chunk=8, gradient_checkpointing=True):
        if node_chunk < 1:
            raise ValueError("node_chunk must be positive")
        for layer in self.core.encoder:
            layer.node_chunk = int(node_chunk)
            layer.gradient_checkpointing = bool(gradient_checkpointing)

    def _standardized(self, x_observed, mask, neighbors, target_ids):
        if not bool(self.fit_configured):
            raise RuntimeError("Configure fit-only global statistics before prediction")
        if x_observed.ndim != 3 or x_observed.shape != mask.shape:
            raise ValueError("x_observed and mask must have shape [B,T,F]")
        if x_observed.shape[2] != self.num_flows or x_observed.shape[1] != 50:
            raise ValueError("Frozen task requires full flow graph and native T50")
        if neighbors.shape[0] != self.num_flows:
            raise ValueError("Graph must include all flows")
        nodes = torch.arange(self.num_flows, device=neighbors.device)
        incoming = neighbors.masked_fill(neighbors == nodes[:, None], -1)
        # Keep original source->target edge orientation; same parent fit graph.
        clean = torch.where(mask.bool(), x_observed.float(), torch.zeros_like(x_observed).float())
        z = torch.where(mask.bool(), (clean - self.fit_mean) / self.fit_scale,
                        torch.zeros_like(clean)).unsqueeze(-1)
        u = torch.arange(50, device=z.device, dtype=z.dtype)[None, :, None] / 49.
        u = u.expand(z.shape[0], -1, -1)
        final, intermediate = self.core(z, u, mask.bool().unsqueeze(-1), incoming,
                                       target_nodes=target_ids)
        return final.squeeze(-1), [value.squeeze(-1) for value in intermediate]

    def predict_block(self, x_observed, mask, stats, neighbors, target_ids,
                      serial_order_context=None, *, return_aux=False, trace=False,
                      intervention=None):
        if intervention is not None:
            raise ValueError("SPIN baseline has no intervention variants")
        final, intermediate = self._standardized(x_observed, mask, neighbors, target_ids)
        raw = final * self.fit_scale + self.fit_mean
        if return_aux or trace:
            return raw, {"standardized_final": final,
                         "standardized_intermediate": intermediate}
        return raw

    def forward(self, x_observed, mask, stats, neighbors):
        ids = torch.arange(self.num_flows, device=x_observed.device)
        raw = self.predict_block(x_observed, mask, stats, neighbors, ids)
        return torch.where(mask.bool(), x_observed, raw.clamp_min(0))


def build_model(num_flows, init_seed=41001):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        return SPINAdapter(num_flows).float()


def make_optimizer_scheduler(model):
    from .scheduler import CosineSchedulerWithRestarts
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["lr"],
                                 weight_decay=TRAINING["weight_decay"])
    scheduler = CosineSchedulerWithRestarts(optimizer, num_warmup_steps=12,
        num_training_steps=300, min_factor=.1, linear_decay=.67, num_cycles=3)
    return optimizer, scheduler


def train_step(model, batch_values, batch_masks, window_starts, stats, neighbors,
               C, optimizer, *, dataset, epoch, physical_batch=1,
               target_block=None, order_seed=81001):
    """Official four-layer L1 supervision, accumulated over full effective batch.

    Run scheduler.step() ONCE after each completed epoch, as Lightning's default
    epoch interval in the upstream. Root runner owns stopping and checkpointing.
    """
    batch, length, flows = batch_values.shape
    if target_block is not None and target_block < flows:
        raise ValueError("SPIN must train the full graph once per microbatch")
    count = int((~batch_masks).sum())
    if length != 50 or count != batch * 40 * flows:
        raise ValueError("SPIN training must use unchanged 20% masks")
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    model.train()
    total = torch.zeros((), device=device)
    ids = torch.arange(flows, device=device)
    for first in range(0, batch, physical_batch):
        truth = torch.as_tensor(batch_values[first:first+physical_batch], device=device)
        mask = torch.as_tensor(batch_masks[first:first+physical_batch], dtype=torch.bool, device=device)
        visible = torch.where(mask, truth, torch.zeros_like(truth))
        final, intermediate = model._standardized(visible, mask, neighbors, ids)
        target = (truth - model.fit_mean) / model.fit_scale
        loss = sum((pred - target).abs().masked_select(~mask).sum()
                   for pred in [final, *intermediate]) / count
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SPIN deep-supervised loss")
        loss.backward()
        total += loss.detach()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(total), "gradient_norm": float(norm),
            "missing_targets": count, "supervised_readouts": 4}
