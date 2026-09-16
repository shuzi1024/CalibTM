"""Shared training mathematics, deterministic orders, and development scoring."""

from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Any

import numpy as np
import torch

from .data import CONDITIONS, DataBundle, linear_fill, make_mask_family


def seed_runtime(seed: int = 41001) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def build_serial_order(
    window_starts, target_ids, *, dataset: str, epoch: int | None,
    seed: int, slots: int = 9, length: int = 50,
) -> np.ndarray:
    """Stateless slot priorities: invariant to batching, target chunking, and init seed."""
    domain = int.from_bytes(hashlib.sha256(
        ("sync-delta-v1.1:serial:" + dataset).encode()).digest()[:8], "little")
    starts = np.asarray(window_starts, dtype=np.uint64)[:, None, None, None]
    targets = np.asarray(target_ids, dtype=np.uint64)[None, None, :, None]
    times = np.arange(length, dtype=np.uint64)[None, :, None, None]
    slot = np.arange(slots, dtype=np.uint64)[None, None, None, :]
    epoch_token = 0 if epoch is None else int(epoch) + 1
    with np.errstate(over="ignore"):
        z = (np.uint64(domain) ^ np.uint64(seed)
             ^ starts * np.uint64(0x9E3779B97F4A7C15)
             ^ targets * np.uint64(0xBF58476D1CE4E5B9)
             ^ times * np.uint64(0x94D049BB133111EB)
             ^ slot * np.uint64(0xD6E8FEB86659FD93)
             ^ np.uint64(epoch_token) * np.uint64(0xA0761D6478BD642F))
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return np.argsort(z, axis=-1, kind="stable").astype(np.int64)


def device_inputs(bundle: DataBundle, device: torch.device):
    stats = {name: torch.tensor(value, device=device, dtype=torch.float32)
             for name, value in bundle.stats.items()}
    neighbors = torch.tensor(bundle.neighbors, device=device, dtype=torch.long)
    return stats, neighbors


def train_step(
    model, batch_values: np.ndarray, batch_masks: np.ndarray, window_starts,
    stats, neighbors, C: float, optimizer, *, dataset: str, epoch: int,
    physical_batch: int = 2, target_block: int = 64, order_seed: int = 81001,
) -> dict[str, float]:
    """One full effective-batch update, with one denominator, clip, and step."""
    if batch_values.shape != batch_masks.shape or batch_values.ndim != 3:
        raise ValueError("values/masks must share [B,T,F]")
    batch, length, flows = batch_values.shape
    missing_count = int((~batch_masks).sum())
    if length != 50 or missing_count != batch * 40 * flows or C <= 0:
        raise ValueError("effective batch violates the frozen 20% mask budget")
    if physical_batch < 1 or target_block < 1:
        raise ValueError("chunk sizes must be positive")
    device = next(model.parameters()).device
    denominator = float(C) * missing_count
    optimizer.zero_grad(set_to_none=True)
    model.train()
    detached_loss = torch.zeros((), device=device)
    for b0 in range(0, batch, physical_batch):
        b1 = min(batch, b0 + physical_batch)
        truth = torch.as_tensor(batch_values[b0:b1], dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch_masks[b0:b1], dtype=torch.bool, device=device)
        observed = torch.where(mask, truth, torch.zeros_like(truth))
        for f0 in range(0, flows, target_block):
            ids = torch.arange(f0, min(flows, f0 + target_block), device=device)
            context = None
            if model.variant == "serial_delta":
                order = build_serial_order(window_starts[b0:b1], ids.cpu().numpy(),
                    dataset=dataset, epoch=epoch, seed=order_seed, slots=neighbors.shape[1])
                context = {"order": torch.as_tensor(order, device=device)}
            raw = model.predict_block(observed, mask, stats, neighbors, ids,
                                      serial_order_context=context)
            missing = ~mask.index_select(2, ids)
            error = (raw - truth.index_select(2, ids)).abs()
            block_loss = error.masked_select(missing).sum() / denominator
            if not torch.isfinite(block_loss).item():
                raise FloatingPointError("non-finite unclamped training loss")
            block_loss.backward()
            detached_loss += block_loss.detach()
            del raw, error, block_loss
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                               error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(detached_loss.item()), "gradient_norm": float(grad_norm.item()),
            "missing_targets": missing_count}


def make_dev_masks(bundle: DataBundle, seed: int):
    families = [make_mask_family(bundle.flows, int(start), seed, dataset=bundle.dataset)
                for start in bundle.dev_starts]
    masks = {condition: np.stack([family.masks[condition] for family in families])
             for condition in CONDITIONS}
    return masks, families


@torch.no_grad()
def predict_windows(
    model, values: np.ndarray, masks: np.ndarray, window_starts, stats, neighbors, *,
    dataset: str, physical_batch: int = 2, target_block: int = 64,
    order_seed: int = 81002, intervention: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    raw_all = np.empty_like(values, dtype=np.float32)
    for b0 in range(0, len(values), physical_batch):
        b1 = min(len(values), b0 + physical_batch)
        truth = torch.as_tensor(values[b0:b1], device=device, dtype=torch.float32)
        mask = torch.as_tensor(masks[b0:b1], device=device, dtype=torch.bool)
        observed = torch.where(mask, truth, torch.zeros_like(truth))
        for f0 in range(0, values.shape[2], target_block):
            ids = torch.arange(f0, min(values.shape[2], f0 + target_block), device=device)
            context = None
            if model.variant == "serial_delta":
                order = build_serial_order(window_starts[b0:b1], ids.cpu().numpy(),
                    dataset=dataset, epoch=None, seed=order_seed, slots=neighbors.shape[1])
                context = {"order": torch.as_tensor(order, device=device)}
            raw = model.predict_block(observed, mask, stats, neighbors, ids,
                serial_order_context=context, intervention=intervention)
            raw_all[b0:b1, :, f0:f0 + len(ids)] = raw.cpu().numpy()
    if not np.isfinite(raw_all).all():
        raise FloatingPointError("non-finite evaluation prediction")
    final = np.where(masks, values, np.maximum(raw_all, 0)).astype(np.float32)
    return final, raw_all


def error_sums(prediction, truth, target):
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    value = truth.astype(np.float64)
    return {"abs_error": float(np.abs(error[target]).sum()),
            "abs_truth": float(np.abs(value[target]).sum()),
            "sq_error": float(np.square(error[target]).sum()),
            "sq_truth": float(np.square(value[target]).sum()),
            "count": int(target.sum())}


def ratios(sums):
    return {**sums,
            "nmae": sums["abs_error"] / sums["abs_truth"] if sums["abs_truth"] > 0 else None,
            "nrmse": math.sqrt(sums["sq_error"] / sums["sq_truth"]) if sums["sq_truth"] > 0 else None}


def aggregate(rows):
    sums = {name: sum(row[name] for row in rows)
            for name in ("abs_error", "abs_truth", "sq_error", "sq_truth", "count")}
    return ratios(sums)


def score_predictions(bundle, condition, predictions, masks, families, raw=None):
    rows = []
    grid = np.arange(50)[:, None]
    for i, (prediction, truth, observed, family) in enumerate(
            zip(predictions, bundle.dev_windows, masks, families)):
        missing = ~observed
        first = np.where(observed, grid, 50).min(axis=0)
        last = np.where(observed, grid, -1).max(axis=0)
        boundary = missing & ((grid < first) | (grid > last))
        groups = {"internal": missing & ~boundary, "boundary": boundary}
        if condition != "uniform":
            groups.update(sparse=missing & family.sparse_group[None],
                          dense=missing & ~family.sparse_group[None])
        if condition == "unequal_gap":
            gap = (grid >= family.gap_start) & (grid < family.gap_start + 10) & family.gap_affected[None]
            groups.update(shared_gap=missing & gap, outside_shared_gap=missing & ~gap)
        rows.append({"window_start": int(bundle.dev_starts[i]),
            "cohort": str(bundle.dev_cohorts[i]), "condition": condition,
            **error_sums(prediction, truth, missing),
            "negative_raw_count": int(((raw[i] < 0) & missing).sum()) if raw is not None else 0,
            "groups": {name: error_sums(prediction, truth, target) for name, target in groups.items()}})
    return rows


def evaluate(
    model, bundle, stats, neighbors, *, seed=71001, physical_batch=2,
    target_block=64, order_seed=81002, intervention=None, mask_cache=None,
):
    masks, families = mask_cache if mask_cache is not None else make_dev_masks(bundle, seed)
    rows, metrics = [], {}
    begin = time.perf_counter()
    for condition in CONDITIONS:
        predictions, raw = predict_windows(model, bundle.dev_windows, masks[condition],
            bundle.dev_starts, stats, neighbors, dataset=bundle.dataset,
            physical_batch=physical_batch, target_block=target_block,
            order_seed=order_seed, intervention=intervention)
        condition_rows = score_predictions(bundle, condition, predictions, masks[condition], families, raw)
        rows.extend(condition_rows)
        metrics[condition] = aggregate(condition_rows)
        metrics[condition]["groups"] = {name: aggregate([row["groups"][name] for row in condition_rows])
            for name in condition_rows[0]["groups"]}
        metrics[condition]["negative_raw_fraction"] = sum(r["negative_raw_count"] for r in condition_rows) / metrics[condition]["count"]
    scores = [metrics[condition]["nmae"] for condition in CONDITIONS]
    if any(value is None or not math.isfinite(value) for value in scores):
        raise FloatingPointError("undefined aggregate development score")
    return {"role": "development_only", "mask_seed": seed, "serial_order_seed": order_seed,
        "selection_score": sum(scores) / len(scores), "metrics": metrics, "rows": rows,
        "intervention": intervention, "elapsed_seconds": time.perf_counter() - begin}


def evaluate_linear(bundle, seed=71001, mask_cache=None):
    masks, families = mask_cache if mask_cache is not None else make_dev_masks(bundle, seed)
    rows, metrics = [], {}
    for condition in CONDITIONS:
        prediction = np.stack([linear_fill(truth, mask, bundle.mu)
            for truth, mask in zip(bundle.dev_windows, masks[condition])])
        group_rows = score_predictions(bundle, condition, prediction, masks[condition], families)
        rows.extend(group_rows)
        metrics[condition] = aggregate(group_rows)
        metrics[condition]["groups"] = {name: aggregate([row["groups"][name] for row in group_rows])
                                        for name in group_rows[0]["groups"]}
    return {"method": "linear", "role": "development_only", "mask_seed": seed,
        "selection_score": sum(metrics[x]["nmae"] for x in CONDITIONS) / 3,
        "metrics": metrics, "rows": rows}


@torch.no_grad()
def fit_diagnostics(model, bundle, stats, neighbors, target_block=16):
    """Small fixed FIT subset: key spectra/projections and state/readout scale."""
    device = next(model.parameters()).device
    values = bundle.train_windows[:1]
    family = make_mask_family(bundle.flows, int(bundle.train_starts[0]), 61001,
                              dataset=bundle.dataset, epoch=0)
    masks = family.unequal_gap[None]
    truth = torch.as_tensor(values, device=device)
    mask = torch.as_tensor(masks, device=device)
    observed = torch.where(mask, truth, torch.zeros_like(truth))
    ids_np = np.unique(np.linspace(0, bundle.flows - 1, min(8, bundle.flows)).astype(np.int64))
    ids = torch.as_tensor(ids_np, device=device)
    context = None
    if model.variant == "serial_delta":
        order = build_serial_order(bundle.train_starts[:1], ids_np, dataset=bundle.dataset,
                                   epoch=None, seed=81002)
        context = {"order": torch.as_tensor(order, device=device)}
    _, aux = model.predict_block(observed, mask, stats, neighbors, ids,
        serial_order_context=context, return_aux=True, trace=model.variant == "attention")
    keys = aux["k"]
    queries = aux["q"].permute(0, 2, 1, 3).reshape(1, len(ids), 50, 4, 32)
    spectra = []
    # These loops inspect only eight fixed targets; never part of training.
    for g in range(len(ids)):
        for head in range(4):
            matrix = keys[0, g, :, head].reshape(-1, 32).double().T
            u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
            threshold = max(float(singular[0]) * 1e-5, 1e-10)
            rank = int((singular > threshold).sum())
            query = queries[0, g, :, head].double()
            projected = query @ u[:, :rank]
            fraction = projected.square().sum(dim=-1) / query.square().sum(dim=-1).clamp_min(1e-12)
            spectra.append({"target": int(ids_np[g]), "head": head, "effective_rank": rank,
                "singular_values": singular.cpu().tolist(),
                "mean_query_projection_fraction": float(fraction.mean())})
    summaries = {}
    for name, value in aux.items():
        if isinstance(value, torch.Tensor) and ("norm" in name or name.startswith("c") or "entropy" in name):
            summaries[name] = {"mean": float(value.float().mean()),
                               "max": float(value.float().max()), "min": float(value.float().min())}
    for name in ("hf", "hb"):
        norms = aux[name].float().norm(dim=-1)
        summaries[name + "_norm"] = {"mean": float(norms.mean()), "max": float(norms.max())}
    for name in ("weights_forward", "weights_backward"):
        if name in aux:
            weight = aux[name]
            entropy = -(weight * weight.clamp_min(1e-30).log()).sum(-1)
            summaries[name + "_entropy"] = {"mean": float(entropy.mean()),
                "empty_side_fraction": float((weight.sum(-1) == 0).float().mean())}
    return {"role": "fit_only_diagnostic", "window_start": int(bundle.train_starts[0]),
        "coordinate_key_rank_bound": 17 if model.variant != "attention" else None,
        "spectra": spectra, "scales": summaries}


@torch.no_grad()
def latency_probe(model, bundle, stats, neighbors, target_block=64, warmup=20, repeats=100):
    """CPU input -> observed construction -> GPU reader -> CPU final output."""
    model.eval()
    device = next(model.parameters()).device
    sample_count = min(5, len(bundle.dev_windows))
    families = [make_mask_family(bundle.flows, int(start), 71001, dataset=bundle.dataset)
                for start in bundle.dev_starts[:sample_count]]
    # Receive only the allowed observation tensor; truth is never copied to GPU here.
    inputs = []
    for index, family in enumerate(families):
        mask = family.unequal_gap[None]
        observed = np.where(mask, bundle.dev_windows[index:index + 1], 0).astype(np.float32)
        inputs.append((observed, mask, bundle.dev_starts[index:index + 1]))

    def once(index):
        observed_cpu, mask_cpu, starts = inputs[index % sample_count]
        x = torch.as_tensor(observed_cpu, device=device)
        m = torch.as_tensor(mask_cpu, device=device)
        output = []
        for f0 in range(0, bundle.flows, target_block):
            ids = torch.arange(f0, min(bundle.flows, f0 + target_block), device=device)
            context = None
            if model.variant == "serial_delta":
                order = build_serial_order(starts, ids.cpu().numpy(), dataset=bundle.dataset,
                                           epoch=None, seed=81002)
                context = {"order": torch.as_tensor(order, device=device)}
            raw = model.predict_block(x, m, stats, neighbors, ids, serial_order_context=context)
            final = torch.where(m.index_select(2, ids), x.index_select(2, ids), raw.clamp_min(0))
            output.append(final.cpu())
        return torch.cat(output, dim=2)

    for i in range(warmup):
        once(i)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for i in range(repeats):
        begin = time.perf_counter()
        once(i)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - begin)
    return {"batch_windows": 1, "input_device": "CPU observed tensor and mask",
        "output_device": "CPU completed window", "mask_generation_timed": False,
        "includes": ["CPU-to-GPU", "feature construction", "neighbor gather", "all reader operations",
                     "decoder", "clamp/copy", "GPU-to-CPU"],
        "warmup": warmup, "repeats": repeats, "target_block": target_block,
        "mean_ms": 1000 * float(np.mean(elapsed)), "median_ms": 1000 * float(np.median(elapsed)),
        "p95_ms": 1000 * float(np.percentile(elapsed, 95)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
