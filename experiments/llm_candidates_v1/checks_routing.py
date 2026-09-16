"""Bounded CPU correctness checks; no traffic data or training experiments."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from experiments.sync_delta_v1.engine import seed_runtime
from experiments.sync_delta_v1.models import build_model as build_base
from .routing import RoutingSyncModel


def _assert_close(a, b, *, atol=3e-5, rtol=3e-4):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return float((a - b).abs().max().item()) if a.numel() else 0.0


def _case(flows=12, batch=2, length=12):
    generator = torch.Generator().manual_seed(78401)
    values = torch.rand(batch, length, flows, generator=generator) * 5 + 1
    mask = torch.rand(batch, length, flows, generator=generator) < 0.3
    values[~mask] = torch.nan
    stats = {"mu": torch.linspace(2.0, 3.0, flows), "scale": torch.linspace(0.8, 1.2, flows)}
    slots = min(9, flows)
    neighbors = (torch.arange(flows)[:, None] + torch.arange(slots)[None]) % flows
    ids = torch.arange(flows)
    return values, mask, stats, neighbors, ids


def _model(flows, *, nonzero_head=True):
    model = RoutingSyncModel(flows)
    base = build_base(flows, init_seed=41001)
    result = model.load_state_dict(base.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    if nonzero_head:
        generator = torch.Generator().manual_seed(78402)
        with torch.no_grad():
            model.decoder[-1].weight.copy_(torch.randn(model.decoder[-1].weight.shape, generator=generator) * 0.03)
    return model


def _predict(model, case, *, aux=False):
    return model.predict_block(*case, return_aux=aux)


def check_initialization_and_reload():
    result = {}
    for flows in (144, 462):
        model, base = _model(flows, nonzero_head=False), build_base(flows)
        for name, value in model.state_dict().items():
            _assert_close(value, base.state_dict()[name], atol=0, rtol=0)
        result[str(flows)] = {"parameters": sum(p.numel() for p in model.parameters()),
                              "shared_tensors_exact": len(model.state_dict())}
    case = _case()
    model = _model(12, nonzero_head=False)
    _assert_close(_predict(model, case), case[2]["mu"][None, None].expand(2, 12, -1), atol=0, rtol=0)
    model = _model(12)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    reloaded = RoutingSyncModel(12)
    reloaded.load_state_dict(torch.load(buffer, map_location="cpu"))
    result["reload_prediction_max_error"] = _assert_close(_predict(model, case), _predict(reloaded, case), atol=0, rtol=0)
    return result


def check_visibility_and_inputs():
    case = _case()
    x, mask, stats, neighbors, ids = case
    model = _model(12)
    raw, aux = _predict(model, case, aux=True)
    changed = torch.where(mask, x, torch.full_like(x, 1e20))
    hidden_error = _assert_close(raw, _predict(model, (changed, mask, stats, neighbors, ids)), atol=0, rtol=0)
    final = model(*case)
    _assert_close(final[mask], x[mask], atol=0, rtol=0)
    assert torch.isfinite(final).all() and (final[~mask] >= 0).all()
    cut = 5
    later_x, later_m = x.clone(), mask.clone()
    later_x[:, cut+1:] = 10.0
    later_m[:, cut+1:] = True
    _, later = _predict(model, (later_x, later_m, stats, neighbors, ids), aux=True)
    earlier_x, earlier_m = x.clone(), mask.clone()
    earlier_x[:, :cut] = 10.0
    earlier_m[:, :cut] = True
    _, earlier = _predict(model, (earlier_x, earlier_m, stats, neighbors, ids), aux=True)
    forward_error = _assert_close(aux["hf"][:, :cut+1], later["hf"][:, :cut+1], atol=0, rtol=0)
    backward_error = _assert_close(aux["hb"][:, cut:], earlier["hb"][:, cut:], atol=0, rtol=0)
    _assert_close(aux["routes_forward"][:, :cut+1].float(), later["routes_forward"][:, :cut+1].float(), atol=0, rtol=0)
    _assert_close(aux["routes_backward"][:, cut:].float(), earlier["routes_backward"][:, cut:].float(), atol=0, rtol=0)
    z, clean_m, *_ = model._validate_and_clean(x.requires_grad_(), mask, stats, neighbors, ids)
    descriptors = model._directional_descriptors(z, clean_m, reverse=False)
    assert not any(d.requires_grad for d in descriptors)
    raw = _predict(model, case)
    raw.square().mean().backward()
    assert x.grad is not None and torch.count_nonzero(x.grad[~mask]).item() == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    return {"hidden_value_isolation_max_error": hidden_error,
            "forward_future_isolation_max_error": forward_error,
            "backward_past_isolation_max_error": backward_error,
            "routing_descriptors_detached": True, "hidden_input_gradient_exact_zero": True,
            "all_parameter_gradients_finite": True,
            "mean_valid_writes_forward": float(aux["selected_valid_count_forward"].float().mean()),
            "mean_nonstatic_fraction_forward": float(aux["nonstatic_valid_fraction_forward"].mean())}


def check_routing_behavior():
    model = _model(12)
    length = 12
    x = torch.zeros(1, length, 12)
    x[:, :, 0] = 2.0
    x[:, :, 9] = 2.0  # Matching observed source outside target 0's static 0..8.
    x[:, :, 1:9] = -3.0
    mask = torch.ones_like(x, dtype=torch.bool)
    stats = {"mu": torch.zeros(12), "scale": torch.ones(12)}
    neighbors = (torch.arange(12)[:, None] + torch.arange(9)[None]) % 12
    ids = torch.arange(12)
    z, mask, _, _, prior, ids = model._validate_and_clean(x, mask, stats, neighbors, ids)
    descriptors = model._directional_descriptors(z, mask, reverse=False)
    routes = model._route(descriptors, mask, prior, ids)
    assert 9 in routes[0, -1, 0].tolist()
    assert torch.equal(routes[..., 0], ids[None, None].expand(1, length, -1))
    # Every dynamic non-self slot is either actually observed or padding.
    source = routes.clamp_min(0)
    selected_observed = mask[torch.arange(1)[:, None, None, None], torch.arange(length)[None, :, None, None], source]
    assert ((routes[..., 1:] < 0) | selected_observed[..., 1:]).all()
    no_history = mask.clone()
    no_history[:, :, 0] = False
    x_nan = torch.where(no_history, x, torch.full_like(x, torch.nan))
    z, no_history, _, _, prior, ids = model._validate_and_clean(x_nan, no_history, stats, neighbors, ids)
    fallback = model._route(model._directional_descriptors(z, no_history, reverse=False), no_history, prior, ids)
    for t in range(length):
        assert set(fallback[0, t, 0].tolist()) == set(neighbors[0].tolist())
    # Padding remains padding in the no-target-history fallback.
    padded = neighbors.clone()
    padded[0, 3:] = -1
    fallback_padded = model._route(model._directional_descriptors(z, no_history, reverse=False), no_history, padded, ids)
    assert set(fallback_padded[0, -1, 0].tolist()) == {-1, 0, 1, 2}
    # Different sample counts explicitly produce different reliability.
    sparse = torch.zeros(1, length, 2, dtype=torch.bool)
    sparse[:, 0, 0] = True
    sparse[:, :8, 1] = True
    _, _, reliability = model._directional_descriptors(torch.zeros_like(sparse, dtype=torch.float32), sparse, reverse=False)
    _assert_close(reliability[0, -1], torch.tensor([1/5, 8/12]), atol=1e-7, rtol=0)
    return {"full_pool_source_9_selected": True, "self_always_slot_zero": True,
            "dynamic_slots_currently_observed_or_padding": True,
            "no_target_history_exact_static_set": True, "padded_fallback_preserved": True,
            "count_reliability_1_vs_8": reliability[0, -1].tolist()}


def check_permutation_and_chunking():
    case = _case()
    model = _model(12)
    raw, aux = _predict(model, case, aux=True)
    x, mask, stats, neighbors, ids = case
    reordered = _predict(model, (x, mask, stats, neighbors.flip(-1), ids))
    prior_error = _assert_close(raw, reordered, atol=0, rtol=0)
    z, clean_m, _, _, _, ids = model._validate_and_clean(*case)
    pool_k, pool_v, queries, coordinates = model._encode_pool(z, clean_m, ids, None)
    routes = aux["routes_forward"]
    k, v, _ = model._gather_events(pool_k, pool_v, coordinates, z, clean_m, routes.flip(-1))
    permuted_read, _ = model._memory_scan(k, v, queries, reverse=False, order=None, return_aux=False, trace=False)
    reference_read = aux["hf"].permute(0, 2, 1, 3).reshape_as(permuted_read)
    bucket_error = _assert_close(reference_read, permuted_read)
    generator = torch.Generator().manual_seed(78403)
    probe = torch.randn(reference_read.shape, generator=generator)
    parameters = tuple(model.parameters())
    ga = torch.autograd.grad((reference_read * probe).sum(), parameters, allow_unused=True, retain_graph=True)
    gb = torch.autograd.grad((permuted_read * probe).sum(), parameters, allow_unused=True)
    gradient_error = max(_assert_close(a, b, atol=8e-5, rtol=8e-4)
                         for a, b in zip(ga, gb) if a is not None)
    full, chunked = _model(12), _model(12)
    full_raw = _predict(full, case)
    full_raw.square().sum().div(full_raw.numel()).backward()
    outputs = []
    for start in range(0, 12, 5):
        raw_chunk = _predict(chunked, (x, mask, stats, neighbors, ids[start:start+5]))
        outputs.append(raw_chunk.detach())
        raw_chunk.square().sum().div(full_raw.numel()).backward()
    chunk_output_error = _assert_close(full_raw, torch.cat(outputs, dim=2))
    chunk_gradient_error = max(_assert_close(a.grad, b.grad, atol=5e-5, rtol=5e-4)
                              for a, b in zip(full.parameters(), chunked.parameters()))
    return {"static_slot_permutation_max_error": prior_error,
            "bucket_slot_permutation_max_error": bucket_error,
            "bucket_slot_gradient_max_error": gradient_error,
            "target_chunk_output_max_error": chunk_output_error,
            "target_chunk_gradient_max_error": chunk_gradient_error}


def check_empty_and_shared_encoding():
    rows = {}
    for flows in (1, 12):
        x, mask, stats, prior, ids = _case(flows=flows)
        mask.zero_()
        x.fill_(torch.nan)
        model = _model(flows)
        raw, aux = _predict(model, (x, mask, stats, prior, ids), aux=True)
        assert torch.isfinite(raw).all()
        assert torch.count_nonzero(aux["hf"]) == 0 and torch.count_nonzero(aux["hb"]) == 0
        raw.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        rows[str(flows)] = {"finite_empty_prediction_and_backward": True, "zero_memory_reads": True}
    model = _model(12)
    calls = []
    handle = model.value.register_forward_hook(lambda module, args, out: calls.append(tuple(args[0].shape)))
    _predict(model, _case())
    handle.remove()
    assert calls == [(2, 12, 12, 25)]
    rows["value_encoder_calls_per_predict_block"] = calls
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seed_runtime(41001)
    started = time.perf_counter()
    results = {"candidate": "routing_sync_v1", "device": "cpu", "synthetic_only": True,
               "torch_version": torch.__version__, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    for name, fn in (("initialization_reload", check_initialization_and_reload),
                     ("visibility_inputs", check_visibility_and_inputs),
                     ("routing_behavior", check_routing_behavior),
                     ("permutation_chunking", check_permutation_and_chunking),
                     ("empty_shared_encoding", check_empty_and_shared_encoding)):
        results[name] = fn()
    results.update(passed=True, elapsed_seconds=time.perf_counter() - started)
    text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
