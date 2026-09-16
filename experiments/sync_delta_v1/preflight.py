"""Bounded correctness checks before the ten authorized Sync-Delta runs.

The synthetic checks deliberately replace the zero-initialized prediction head.
They test the real readers, without training or changing the shared initialization
used by the experiment.  ``run_geant_smoke`` reads fit windows only.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .models import VARIANTS, build_model, gram_denominator, sync_delta_update


def _runtime_settings() -> None:
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _close(a: torch.Tensor, b: torch.Tensor, label: str, *, exact: bool = False) -> float:
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError(f"{label}: nonfinite tensor")
    if a.dtype == torch.bool or not a.is_floating_point():
        if not torch.equal(a, b):
            raise AssertionError(f"{label}: integer/boolean tensors differ")
        return 0.0
    torch.testing.assert_close(a, b, rtol=0.0 if exact else 2e-4,
                               atol=0.0 if exact else 3e-6, msg=label)
    return float((a - b).abs().max().item()) if a.numel() else 0.0


def _nonzero_head(model: Any) -> None:
    with torch.no_grad():
        weight = model.decoder[-1].weight
        weight.copy_(0.03 * torch.sin(torch.arange(weight.numel(), device=weight.device)
                                      .reshape_as(weight) + 0.37))
        model.decoder[-1].bias.fill_(0.03)
    assert torch.count_nonzero(model.decoder[-1].weight).item() > 0


def _finite_gradients(model: Any, *, require_encoders: bool) -> dict[str, float]:
    norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"nonfinite gradient: {name}")
            norms[name] = float(parameter.grad.norm().item())
    if require_encoders:
        for name in ("flow_embedding.weight", "key.weight", "query.weight", "value.0.weight"):
            if norms.get(name, 0.0) <= 0.0:
                raise AssertionError(f"nonzero-head probe has no upstream gradient: {name}")
    return norms


def _synthetic_case(device: torch.device) -> dict[str, Any]:
    # Short native-grid prefix; the model continues to encode t / 49.
    batch, length, flows, slots = 2, 7, 6, 9
    b = torch.arange(batch, device=device)[:, None, None]
    t = torch.arange(length, device=device)[None, :, None]
    f = torch.arange(flows, device=device)[None, None, :]
    x = (2.0 + 0.13 * t + 0.17 * f + 0.11 * b + 0.09 * torch.cos(t + f)).float()
    mask = ((t + 2 * f + b) % 3 == 0).expand_as(x).clone()
    mask[:, 0] = False
    mask[:, -1] = False
    mask[:, 3, 0] = True
    neighbors = torch.full((flows, slots), -1, dtype=torch.long, device=device)
    for flow in range(flows):
        neighbors[flow, :flows] = torch.arange(flows, device=device).roll(-flow)
    stats = {"mu": torch.linspace(0.5, 0.75, flows, device=device),
             "scale": torch.linspace(0.3, 0.45, flows, device=device)}
    # Identity-derived orders with no mutable RNG; production generator is checked separately.
    order = torch.empty((batch, length, flows, slots), dtype=torch.long, device=device)
    for bi in range(batch):
        for ti in range(length):
            for fi in range(flows):
                order[bi, ti, fi] = torch.arange(slots, device=device).roll(bi + 2 * ti + fi)
    return dict(x=x, mask=mask, stats=stats, neighbors=neighbors,
                ids=torch.arange(flows, device=device), order=order, C=2.0)


def _predict(model: Any, case: dict[str, Any], *, x: torch.Tensor | None = None,
             ids: torch.Tensor | None = None, trace: bool = False,
             batch_slice: slice = slice(None), neighbors: torch.Tensor | None = None,
             order: torch.Tensor | None = None, final: bool = False) -> Any:
    targets = case["ids"] if ids is None else ids
    values = case["x"] if x is None else x
    orders = case["order"] if order is None else order
    selected_order = orders[batch_slice].index_select(2, targets)
    function = model if final else model.predict_block
    return function(values[batch_slice], case["mask"][batch_slice], case["stats"],
                    case["neighbors"] if neighbors is None else neighbors, targets,
                    serial_order_context={"order": selected_order},
                    return_aux=trace, trace=trace)


def check_tied_gram_gradients(device: torch.device) -> dict[str, Any]:
    """Exact tied-row counterexample; do not finite-difference a nonsmooth tie."""
    metrics = {}
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    inverse = torch.argsort(permutation)
    for dtype in (torch.float32, torch.float64):
        a = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=dtype, device=device)
        b = a.clone()
        b[-1] = -0.5
        keys = torch.stack((a, a, b, b))
        generator = torch.Generator().manual_seed(93001)
        values = torch.randn(4, 4, generator=generator, dtype=dtype).to(device)
        state = torch.randn(4, 4, generator=generator, dtype=dtype).to(device)
        probe = torch.randn(4, 4, generator=generator, dtype=dtype).to(device)

        def evaluate(reorder: bool, *, indexed_max: bool = False) -> tuple[Any, ...]:
            raw_keys = (keys[permutation] if reorder else keys).clone().requires_grad_()
            raw_values = (values[permutation] if reorder else values).clone().requires_grad_()
            raw_state = state.clone().requires_grad_()
            normalized = F.normalize(raw_keys, dim=-1)
            if indexed_max:
                c = (normalized @ normalized.T).abs().sum(-1).max(dim=-1).values.clamp_min(1)
                gradient, = torch.autograd.grad(c, raw_keys)
                return (gradient[inverse] if reorder else gradient,)
            result, c, _ = sync_delta_update(raw_state, normalized, raw_values, return_aux=True)
            denominator_gradient, = torch.autograd.grad(c, raw_keys, retain_graph=True)
            gradients = torch.autograd.grad((result * probe).sum(), (raw_keys, raw_values, raw_state))
            if reorder:
                denominator_gradient = denominator_gradient[inverse]
                gradients = (gradients[0][inverse], gradients[1][inverse], gradients[2])
            assert c.item() == 3.0, "the constructed row-sum maximum must be an exact tie"
            assert denominator_gradient.norm().item() > 0
            return result, c, denominator_gradient, *gradients

        reference, reordered = evaluate(False), evaluate(True)
        errors = [_close(a0, b0, f"amax tied {dtype} field {i}")
                  for i, (a0, b0) in enumerate(zip(reference, reordered))]
        bad_error = float((evaluate(False, indexed_max=True)[0]
                           - evaluate(True, indexed_max=True)[0]).abs().max().item())
        assert bad_error > 0.1, "counterexample must reject index-selecting max(dim)"
        metrics[str(dtype)] = {"maximum_equivariant_error": max(errors),
                               "indexed_max_counterexample_error": bad_error}

    # One smooth directional derivative, separate from the tied-max test.
    generator = torch.Generator().manual_seed(93002)
    keys = torch.randn(3, 5, generator=generator, dtype=torch.float64).to(device).requires_grad_()
    direction = torch.randn(3, 5, generator=generator, dtype=torch.float64).to(device)
    gradient, = torch.autograd.grad(gram_denominator(keys), keys)
    epsilon = 1e-5
    numerical = (gram_denominator(keys.detach() + epsilon * direction)
                 - gram_denominator(keys.detach() - epsilon * direction)) / (2 * epsilon)
    analytical = (gradient * direction).sum()
    torch.testing.assert_close(numerical, analytical, rtol=2e-6, atol=1e-7)
    metrics["smooth_directional_derivative_error"] = float((numerical - analytical).abs().item())
    return metrics


def _check_information(model: Any, case: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad():
        raw, aux = _predict(model, case, trace=True)
        assert aux["hf"].norm().item() > 0 and aux["hb"].norm().item() > 0
        assert aux["event_mask"].any().item()
        assert torch.count_nonzero(aux["hf"][:, 0]).item() == 0
        assert torch.count_nonzero(aux["hb"][:, -1]).item() == 0
        for prefix in ("states", "gram", "cross"):
            name = f"{prefix}_forward"
            if name in aux:
                _close(aux[name][:, :, -1], aux[name][:, :, -2], f"empty bucket / {name}", exact=True)
            name = f"{prefix}_backward"
            if name in aux:
                _close(aux[name][:, :, 0], aux[name][:, :, 1], f"empty bucket / {name}", exact=True)
        pred = _predict(model, case, final=True)
        for hidden_value in (float("nan"), 1e8):
            poisoned = torch.where(case["mask"], case["x"], hidden_value)
            other_raw, other_aux = _predict(model, case, x=poisoned, trace=True)
            _close(raw, other_raw, "hidden truth / raw", exact=True)
            for name in aux:
                _close(aux[name], other_aux[name], f"hidden truth / {name}", exact=True)
            _close(pred, _predict(model, case, x=poisoned, final=True),
                   "hidden truth / prediction", exact=True)

        cut = 3
        future = case["x"].clone()
        future[:, cut + 1:] += 3.0
        _, future_aux = _predict(model, case, x=future, trace=True)
        _close(aux["hf"][:, :cut + 1], future_aux["hf"][:, :cut + 1],
               "forward cannot see future", exact=True)
        past = case["x"].clone()
        past[:, :cut] += 3.0
        _, past_aux = _predict(model, case, x=past, trace=True)
        _close(aux["hb"][:, cut:], past_aux["hb"][:, cut:],
               "backward cannot see past", exact=True)
        for prefix in ("states", "gram", "cross"):
            name = f"{prefix}_forward"
            if name in aux:
                _close(aux[name][:, :, :cut + 1], future_aux[name][:, :, :cut + 1], name, exact=True)
            name = f"{prefix}_backward"
            if name in aux:
                _close(aux[name][:, :, cut:], past_aux[name][:, :, cut:], name, exact=True)

        same_time = case["x"].clone()
        same_time[:, cut, 0] += 3.0
        _, same_aux = _predict(model, case, x=same_time, trace=True)
        changes = {name: float((aux[name][:, cut] - same_aux[name][:, cut]).abs().max().item())
                   for name in ("hf", "hb")}
        assert min(changes.values()) > 1e-7, "both directions must read the complete current bucket"
        _close(pred[case["mask"]], case["x"][case["mask"]], "observed exact copy", exact=True)
        assert (pred[~case["mask"]] >= 0).all().item()

        clipped_model = copy.deepcopy(model)
        clipped_model.decoder[-1].weight.zero_()
        clipped_model.decoder[-1].bias.fill_(-1e4)
        clipped_raw = _predict(clipped_model, case)
        clipped_pred = _predict(clipped_model, case, final=True)
        assert (clipped_raw[~case["mask"]] < 0).all().item()
        assert torch.count_nonzero(clipped_pred[~case["mask"]]).item() == 0
        _close(clipped_pred[case["mask"]], case["x"][case["mask"]], "copy survives clamp", exact=True)

        if model.variant == "batch_ridge":
            q = aux["q"].permute(0, 2, 1, 3).reshape(*aux["k"].shape[:4], 32)
            for direction, read in (("forward", "hf"), ("backward", "hb")):
                gram, cross = aux[f"gram_{direction}"], aux[f"cross_{direction}"]
                solved_state = torch.linalg.solve(gram, cross.transpose(-2, -1)).transpose(-2, -1)
                expected = (solved_state @ q[..., None]).squeeze(-1)
                expected = expected.flatten(-2).permute(0, 2, 1, 3)
                _close(aux[read], expected, f"cholesky_ex versus direct solve / {direction}")
    return {"hidden_replacement": ["NaN", "1e8"], "trace_fields_checked": sorted(aux),
            "same_timestamp_readout_changes": changes}


def _loss(raw: torch.Tensor, case: dict[str, Any], ids: torch.Tensor) -> torch.Tensor:
    missing = ~case["mask"].index_select(2, ids)
    target = case["x"].index_select(2, ids)
    return (raw - target).abs().masked_select(missing).sum() / ((~case["mask"]).sum() * case["C"])


def _check_chunking_and_reload(model: Any, case: dict[str, Any]) -> dict[str, Any]:
    full, chunked = copy.deepcopy(model), copy.deepcopy(model)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
                  for m in (full, chunked)]
    raw = _predict(full, case)
    total_loss = _loss(raw, case, case["ids"])
    total_loss.backward()
    parts, part_losses = [], []
    for ids in case["ids"].split(2):
        part = _predict(chunked, case, ids=ids)
        loss = _loss(part, case, ids)
        loss.backward()
        parts.append(part.detach())
        part_losses.append(loss.detach())
    output_error = _close(raw.detach(), torch.cat(parts, 2), "target chunk output")
    _close(total_loss.detach(), torch.stack(part_losses).sum(), "target chunk loss")
    upstream = _finite_gradients(full, require_encoders=True)
    _finite_gradients(chunked, require_encoders=True)
    gradient_error = 0.0
    for (name, a), (_, b) in zip(full.named_parameters(), chunked.named_parameters()):
        if (a.grad is None) != (b.grad is None):
            raise AssertionError(f"target chunk disconnected gradient: {name}")
        if a.grad is not None:
            gradient_error = max(gradient_error, _close(a.grad, b.grad, f"target chunk gradient / {name}"))
    for m, optimizer in zip((full, chunked), optimizers):
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
    step_error = max(_close(a, b, f"one AdamW step / {name}")
                     for (name, a), (_, b) in zip(full.named_parameters(), chunked.named_parameters()))
    with torch.no_grad():
        separate_batches = torch.cat([_predict(model, case, batch_slice=slice(i, i + 1))
                                      for i in range(case["x"].shape[0])], dim=0)
        batch_error = _close(_predict(model, case), separate_batches, "physical batch output")
        expected = _predict(full, case)
    checkpoint = io.BytesIO()
    torch.save({"model": full.state_dict(), "optimizer": optimizers[0].state_dict(), "step": 1}, checkpoint)
    checkpoint.seek(0)
    payload = torch.load(checkpoint, map_location=case["x"].device)
    restored = copy.deepcopy(model)
    restored.load_state_dict(payload["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=1e-4)
    restored_optimizer.load_state_dict(payload["optimizer"])
    assert payload["step"] == 1
    assert len(restored_optimizer.state) == len(optimizers[0].state)
    with torch.no_grad():
        _close(expected, _predict(restored, case), "checkpoint reload", exact=True)
    return {"output_max_error": output_error, "gradient_max_error": gradient_error,
            "one_optimizer_step_max_error": step_error, "physical_batch_max_error": batch_error,
            "upstream_gradient_norms": {k: upstream[k] for k in
                ("flow_embedding.weight", "key.weight", "query.weight", "value.0.weight")},
            "checkpoint_model_and_optimizer_reloaded": True}


def _check_reordering(model: Any, case: dict[str, Any]) -> dict[str, Any]:
    if model.variant == "serial_delta":
        with torch.no_grad():
            baseline = _predict(model, case)
            differences = [float((baseline - _predict(model, case, order=alternate)).abs().max().item())
                           for alternate in (case["order"].flip(-1), case["order"].roll(2, -1))]
        return {"diagnostic_only": True, "alternative_order_max_changes": differences}
    alternate = copy.deepcopy(model)
    permutation = torch.tensor([4, 1, 7, 0, 8, 2, 5, 3, 6], device=case["x"].device)
    model.zero_grad(set_to_none=True)
    baseline = _predict(model, case)
    _loss(baseline, case, case["ids"]).backward()
    reordered = _predict(alternate, case, neighbors=case["neighbors"][:, permutation])
    _loss(reordered, case, case["ids"]).backward()
    output_error = _close(baseline.detach(), reordered.detach(), "same-time event permutation output")
    gradient_error = max(_close(a.grad, b.grad, f"event permutation gradient / {name}")
                         for (name, a), (_, b) in zip(model.named_parameters(), alternate.named_parameters())
                         if a.grad is not None)
    model.zero_grad(set_to_none=True)
    return {"output_max_error": output_error, "gradient_max_error": gradient_error}


def _check_empty_attention(device: torch.device) -> dict[str, Any]:
    case = _synthetic_case(device)
    case["mask"] = torch.zeros_like(case["mask"])
    case["x"] = torch.full_like(case["x"], float("nan"))
    model = build_model(6, "attention").to(device)
    _nonzero_head(model)
    raw, aux = _predict(model, case, trace=True)
    assert torch.isfinite(raw).all().item()
    for name in ("hf", "hb", "weights_forward", "weights_backward"):
        assert torch.count_nonzero(aux[name]).item() == 0, name
    raw.square().mean().backward()
    gradients = _finite_gradients(model, require_encoders=False)
    assert gradients.get("query.weight", 0.0) > 0
    return {"finite_backward": True, "both_empty_readouts_and_weights_exact_zero": True}


def check_serial_order_generator(generator: Callable[..., np.ndarray]) -> dict[str, Any]:
    starts = np.array([0, 37, 142, 511], dtype=np.int64)
    targets = np.array([0, 5, 41, 97], dtype=np.int64)
    options = dict(dataset="geant", epoch=3, seed=81001, slots=9)
    full = generator(starts, targets, **options)
    batch_parts = np.concatenate([generator(starts[i:i + 1], targets, **options)
                                  for i in range(len(starts))], axis=0)
    target_parts = np.concatenate([generator(starts, targets[i:i + 1], **options)
                                   for i in range(len(targets))], axis=2)
    assert full.shape == (4, 50, 4, 9)
    assert np.array_equal(full, batch_parts) and np.array_equal(full, target_parts)
    assert np.array_equal(np.sort(full, axis=-1), np.broadcast_to(np.arange(9), full.shape))
    assert np.array_equal(full, generator(starts, targets, **options))
    return {"stable_across_batch_target_chunks_and_repeated_calls": True}


def run_preflight(device: str | torch.device = "cpu", *,
                  serial_order_generator: Callable[..., np.ndarray] | None = None,
                  progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    device = torch.device(device)
    _runtime_settings()
    report: dict[str, Any] = {"torch_version": torch.__version__, "device": str(device),
                              "cuda_version": torch.version.cuda, "tf32": False,
                              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    report["gram_ties"] = check_tied_gram_gradients(device)
    if progress:
        progress("preflight: tied Gram gradients and smooth derivative passed")
    case = _synthetic_case(device)
    arms = {}
    for variant in VARIANTS:
        model = build_model(6, variant).to(device)
        assert torch.count_nonzero(model.decoder[-1].weight).item() == 0
        _nonzero_head(model)
        arms[variant] = {"information_and_copy": _check_information(model, case),
                         "chunking_and_reload": _check_chunking_and_reload(model, case),
                         "event_reordering": _check_reordering(model, case)}
        if progress:
            progress(f"preflight: {variant} information, chunk gradients/step, reload and order checks passed")
    report["arms"] = arms
    report["empty_attention"] = _check_empty_attention(device)
    if serial_order_generator is not None:
        report["serial_order_generator"] = check_serial_order_generator(serial_order_generator)
    else:
        report["serial_order_generator"] = {"status": "not_run", "reason": "no production generator supplied"}
    report["core_checks_passed"] = True
    report["passed"] = serial_order_generator is not None
    return report


def run_geant_smoke(device: str | torch.device, *, canonical_dir: str | Path | None = None,
                    serial_order_generator: Callable[..., np.ndarray] | None = None,
                    progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    from .data import load_fit_windows, make_mask_family
    device = torch.device(device)
    _runtime_settings()
    windows, starts, raw_stats, raw_neighbors = load_fit_windows("geant", indices=(0, 1),
                                                               canonical_dir=canonical_dir)
    masks = np.stack([make_mask_family(462, int(start), 61001, dataset="geant", epoch=0).uniform
                      for start in starts])
    x = torch.as_tensor(windows.copy(), dtype=torch.float32, device=device)
    mask = torch.as_tensor(masks, dtype=torch.bool, device=device)
    stats = {name: torch.as_tensor(raw_stats[name], device=device) for name in ("mu", "scale")}
    neighbors = torch.as_tensor(raw_neighbors, dtype=torch.long, device=device)
    ids = torch.arange(64, device=device)
    if serial_order_generator is None:
        from .engine import build_serial_order
        serial_order_generator = build_serial_order
    raw_order = serial_order_generator(starts, np.arange(64), dataset="geant", epoch=0,
                                       seed=81001, slots=9)
    context = {"order": torch.as_tensor(raw_order, dtype=torch.long, device=device)}
    report = {"scope": "fit only; first two registered windows; first 64 target flows",
              "window_starts": starts.tolist(), "physical_batch": 2, "target_block": 64,
              "full_effective_batch8": "must be checked with the production trainer", "arms": {}}
    for variant in VARIANTS:
        model = build_model(462, variant).to(device)
        _nonzero_head(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        raw = model.predict_block(x, mask, stats, neighbors, ids, serial_order_context=context)
        missing = ~mask.index_select(2, ids)
        loss = (raw - x.index_select(2, ids)).abs().masked_select(missing).sum()
        loss = loss / ((~mask).sum() * float(raw_stats["C"]))
        loss.backward()
        gradients = _finite_gradients(model, require_encoders=True)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        report["arms"][variant] = {"parameters": sum(p.numel() for p in model.parameters()),
            "partial_batch_loss": float(loss.detach().item()), "gradient_norm_before_clip": float(norm.item()),
            "seconds_forward_backward_step": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "nonzero_key_gradient_norm": gradients["key.weight"], "nonzero_head": True}
        if progress:
            progress(f"GEANT smoke: {variant} forward/backward/step passed in {elapsed:.3f}s")
        del model, optimizer, raw, loss
    report["passed"] = True
    return report


def run_effective_batch_smoke(device: str | torch.device, *,
                              canonical_dir: str | Path | None = None,
                              progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Exercise the production trainer's complete eight-window accumulation once per arm."""
    from .data import CONDITIONS, load_fit_windows, make_mask_family
    from .engine import train_step

    device = torch.device(device)
    _runtime_settings()
    values, starts, raw_stats, raw_neighbors = load_fit_windows(
        "geant", indices=tuple(range(8)), canonical_dir=canonical_dir)
    masks = np.stack([make_mask_family(462, int(start), 61001, dataset="geant", epoch=0)
                      .masks[CONDITIONS[i % len(CONDITIONS)]] for i, start in enumerate(starts)])
    stats = {name: torch.as_tensor(raw_stats[name], device=device) for name in ("mu", "scale")}
    neighbors = torch.as_tensor(raw_neighbors, dtype=torch.long, device=device)
    expected_missing = 8 * 40 * 462
    assert int((~masks).sum()) == expected_missing
    report: dict[str, Any] = {"scope": "eight registered GEANT fit windows, all 462 targets",
        "window_starts": starts.tolist(), "effective_batch": 8, "physical_batch": 2,
        "target_block": 64, "expected_missing_targets": expected_missing, "arms": {}}
    for variant in VARIANTS:
        model = build_model(462, variant).to(device)
        _nonzero_head(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        calls: list[dict[str, int]] = []
        error_numerators: list[float] = []
        original_predict = model.predict_block

        def traced_predict(observed, observed_mask, fit_stats, graph, targets, **kwargs):
            assert observed.shape[0] <= 2 and len(targets) <= 64
            assert torch.count_nonzero(observed[~observed_mask]).item() == 0
            window_offset = 2 * (len(calls) // 8)
            calls.append({"windows": observed.shape[0], "targets": len(targets),
                          "missing": int((~observed_mask.index_select(2, targets)).sum().item())})
            result = original_predict(observed, observed_mask, fit_stats, graph, targets, **kwargs)
            truth = torch.as_tensor(values[window_offset:window_offset + observed.shape[0]], device=device)
            errors = (result.detach() - truth.index_select(2, targets)).abs()
            error_numerators.append(float(errors.masked_select(~observed_mask.index_select(2, targets))
                                          .double().sum().item()))
            return result

        model.predict_block = traced_predict
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        clip = torch.nn.utils.clip_grad_norm_
        step = optimizer.step
        with patch("torch.nn.utils.clip_grad_norm_", wraps=clip) as clip_calls:
            with patch.object(optimizer, "step", wraps=step) as step_calls:
                result = train_step(model, values, masks, starts, stats, neighbors,
                    float(raw_stats["C"]), optimizer, dataset="geant", epoch=0,
                    physical_batch=2, target_block=64, order_seed=81001)
                assert clip_calls.call_count == 1, "clip must follow the whole effective batch"
                assert step_calls.call_count == 1, "optimizer must step once per effective batch"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        gradients = _finite_gradients(model, require_encoders=True)
        assert len(calls) == 32, "four physical batches times eight target chunks"
        assert sum(call["missing"] for call in calls) == expected_missing
        assert result["missing_targets"] == expected_missing
        expected_loss = sum(error_numerators) / (float(raw_stats["C"]) * expected_missing)
        assert np.isclose(result["loss"], expected_loss, rtol=1e-5, atol=1e-7)
        assert optimizer.state and all(float(state["step"].item()) == 1.0
                                       for state in optimizer.state.values())
        report["arms"][variant] = {**result, "reader_calls": len(calls), "clip_calls": 1,
            "optimizer_steps": 1, "seconds": elapsed, "nonzero_head": True,
            "loss_from_global_error_sum": expected_loss,
            "nonzero_key_gradient_norm": gradients["key.weight"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        if progress:
            progress(f"GEANT effective-batch smoke: {variant}, 32 blocks / one clip / one step, {elapsed:.3f}s")
        model.predict_block = original_predict
        del model, optimizer, original_predict
    report["passed"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--geant", action="store_true")
    parser.add_argument("--effective-batch-smoke", action="store_true")
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from .engine import build_serial_order
    report = run_preflight(args.device, serial_order_generator=build_serial_order)
    if args.geant:
        report["geant_smoke"] = run_geant_smoke(args.device, canonical_dir=args.canonical_dir,
                                               serial_order_generator=build_serial_order)
    if args.effective_batch_smoke:
        report["effective_batch_smoke"] = run_effective_batch_smoke(
            args.device, canonical_dir=args.canonical_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
