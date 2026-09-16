"""CPU checks for the one retrained Direct-only component-removal ablation.

Only tiny synthetic data and synthetic optimizer steps are used. No dataset,
held-out targets, or GPU is accessed. The trained-full comparison removes only
the memory contribution algebraically; it does not use ``zero_readout``.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
from typing import Any

import torch

from experiments.llm_candidates_v1.direct import DirectSyncModel
from experiments.llm_candidates_v1.models import build_model as build_full_model
from experiments.sync_delta_v1.models import SyncDeltaModel
from experiments.sync_delta_v1.preflight import (
    _check_information,
    _check_reordering,
    _close,
    _loss,
    _nonzero_head,
    _predict,
    _runtime_settings,
    _synthetic_case,
)

from .models import DirectOnlyModel, build_model


def _gradients(model: DirectOnlyModel, *, require_upstream: bool) -> dict[str, float]:
    norms = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"disconnected retained parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"nonfinite gradient: {name}"
        norms[name] = float(parameter.grad.norm().item())
    if require_upstream:
        for name in ("flow_embedding.weight", "query.weight", "direct_query.weight",
                     "direct_key.weight", "direct_fusion.weight"):
            assert norms[name] > 0, f"upstream gradient is zero: {name}"
    return norms


def _check_initialization() -> dict[str, Any]:
    result = {}
    assert DirectOnlyModel._direct_reads is DirectSyncModel._direct_reads
    assert DirectOnlyModel.forward is SyncDeltaModel.forward
    for flows in (6, 144, 462):
        result[str(flows)] = {}
        for seed in (41001, 41002):
            before = torch.random.get_rng_state().clone()
            model = build_model(flows, seed)
            assert torch.equal(before, torch.random.get_rng_state())
            full = build_full_model(flows, "direct_sync", init_seed=seed)
            for name, tensor in model.state_dict().items():
                _close(tensor, full.state_dict()[name], f"shared initialization {name}", exact=True)
            expected_removed = {
                "key.weight", "key.bias", "value.0.weight", "value.0.bias",
                "value.2.weight", "value.2.bias",
            }
            assert set(full.state_dict()) - set(model.state_dict()) == expected_removed
            assert not hasattr(model, "key") and not hasattr(model, "value")
            assert not hasattr(model, "_memory_scan") and not hasattr(model, "_encode")
            count = sum(parameter.numel() for parameter in model.parameters())
            full_count = sum(parameter.numel() for parameter in full.parameters())
            assert full_count - count == 23040
            assert count == 53793 + 16 * flows
            result[str(flows)][str(seed)] = {
                "parameters": count, "full_parameters": full_count,
                "removed_parameters": full_count - count,
                "all_retained_initialization_tensors_bitwise_equal": True,
                "caller_rng_unchanged": True,
            }
    return result


@torch.no_grad()
def _check_selected_observations() -> dict[str, Any]:
    model = build_model(3)
    length = 9
    values = torch.arange(54, dtype=torch.float32).reshape(2, length, 3) / 10 + 1
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[:, [0, 2, 3, 5, 8], 0] = True
    mask[:, [1, 4, 7], 1] = True
    stats = {"mu": torch.tensor([0.4, 0.5, 0.6]), "scale": torch.tensor([0.2, 0.3, 0.4])}
    neighbors = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    ids = torch.tensor([2, 0, 1])
    raw, aux = model.predict_block(values, mask, stats, neighbors, ids, trace=True)
    assert torch.isfinite(raw).all()
    for label, reverse in (("forward", False), ("backward", True)):
        weights = aux[f"direct_weights_{label}"]
        expected = torch.zeros_like(weights, dtype=torch.bool)
        for b in range(2):
            for group, flow in enumerate(ids.tolist()):
                observed = mask[b, :, flow].nonzero().flatten().tolist()
                for t in range(length):
                    allowed = [s for s in observed if s >= t] if reverse else [s for s in observed if s <= t]
                    selected = sorted(allowed, key=lambda s: abs(t - s))[:2]
                    expected[b, group, t, selected] = True
        assert torch.equal(weights > 0, expected)
        assert int((weights > 0).sum(-1).max().item()) == 2
        _close(weights.sum(-1), expected.any(-1).float(), f"normalized support {label}")
        empty = ~expected.any(-1).permute(0, 2, 1)
        for prefix in ("direct_features", "direct_read"):
            assert torch.count_nonzero(aux[f"{prefix}_{label}"][empty]).item() == 0
        selected_values = values.index_select(2, ids).permute(0, 2, 1)
        selected_mask = mask.index_select(2, ids).permute(0, 2, 1)
        z = torch.where(
            selected_mask,
            (selected_values - stats["mu"][ids][None, :, None]) / stats["scale"][ids][None, :, None],
            torch.zeros_like(selected_values),
        )
        features = aux[f"direct_features_{label}"].permute(0, 2, 1, 3)
        _close(features[..., 0], (weights * z[:, :, None]).sum(-1), f"weighted observed value {label}")
        native = torch.arange(length)
        distance = (native[:, None] - native[None]).abs().float() / 49
        _close(features[..., 1], (weights * distance).sum(-1), f"native-slot distance {label}")
    # One-time-step windows include the current observation in both directions.
    one = model.predict_block(values[:, :1], mask[:, :1], stats, neighbors, ids, trace=True)[1]
    for label in ("forward", "backward"):
        _close(one[f"direct_weights_{label}"].flatten(),
               mask[:, :1].index_select(2, ids).permute(0, 2, 1).float().flatten(),
               f"length-one support {label}", exact=True)
    return {"nearest_two_support_matches_independent_enumeration": True,
            "weighted_values_and_native_distance_correct": True,
            "empty_directions_exact_zero": True, "length_one_supported": True}


def _check_own_flow_and_hidden_gradient() -> dict[str, Any]:
    case = _synthetic_case(torch.device("cpu"))
    model = build_model(6)
    _nonzero_head(model)
    ids = torch.tensor([0])
    baseline = _predict(model, case, ids=ids)
    changed = copy.deepcopy(case)
    changed["x"][:, :, 1:] = float("nan")
    changed["mask"][:, :, 1:] = ~changed["mask"][:, :, 1:]
    changed["neighbors"] = torch.full_like(changed["neighbors"], -1)
    _close(baseline, _predict(model, changed, ids=ids),
           "whole prediction independent of other-flow values, masks, neighbor table", exact=True)
    poisoned = torch.where(case["mask"], case["x"], float("nan")).requires_grad_()
    raw = _predict(model, case, ids=ids, x=poisoned)
    raw.square().mean().backward()
    assert poisoned.grad is not None and torch.isfinite(poisoned.grad).all()
    assert torch.count_nonzero(poisoned.grad[~case["mask"]]).item() == 0
    assert torch.count_nonzero(poisoned.grad[:, :, 1:]).item() == 0
    assert poisoned.grad[:, :, 0][case["mask"][:, :, 0]].norm().item() > 0
    # Empty input must still yield a finite query-only prediction and backward.
    model.zero_grad(set_to_none=True)
    empty = copy.deepcopy(case)
    empty["mask"] = torch.zeros_like(case["mask"])
    empty["x"] = torch.full_like(case["x"], float("nan"))
    raw, aux = _predict(model, empty, trace=True)
    assert torch.isfinite(raw).all()
    for label in ("forward", "backward"):
        for prefix in ("direct_weights", "direct_features", "direct_read"):
            assert torch.count_nonzero(aux[f"{prefix}_{label}"]).item() == 0
    raw.square().mean().backward()
    norms = _gradients(model, require_upstream=False)
    assert norms["query.weight"] > 0
    return {"whole_prediction_own_flow_only": True, "hidden_input_gradient_exact_zero": True,
            "other_flow_input_gradient_exact_zero": True, "observed_input_gradient_nonzero": True,
            "all_empty_forward_and_backward_finite": True}


def _check_full_without_memory() -> dict[str, Any]:
    case = _synthetic_case(torch.device("cpu"))
    full = build_full_model(6, "direct_sync", init_seed=41001)
    optimizer = torch.optim.AdamW(full.parameters(), lr=1e-3, weight_decay=1e-4)
    # Synthetic-only updates ensure a nontrivial trained decoder and memory.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _loss(_predict(full, case), case, case["ids"]).backward()
        optimizer.step()
    ablation = build_model(6, 41001)
    ablation.load_state_dict({name: full.state_dict()[name] for name in ablation.state_dict()}, strict=True)
    with torch.no_grad():
        raw_full, aux_full = _predict(full, case, trace=True)
        raw_direct, aux_direct = _predict(ablation, case, trace=True)
        assert aux_full["memory_hf"].norm().item() > 0
        assert aux_full["memory_hb"].norm().item() > 0
        # Manually omit only the two memory summands from the trained full model.
        normalized = full.decoder(torch.cat((aux_full["direct_read_forward"],
                                             aux_full["direct_read_backward"], aux_full["q"]), -1)).squeeze(-1)
        expected = case["stats"]["mu"][None, None] + case["stats"]["scale"][None, None] * normalized
        error = _close(raw_direct, expected, "trained full with memory contribution omitted", exact=True)
        for name in ("q", "direct_read_forward", "direct_read_backward"):
            _close(aux_direct[name], aux_full[name], f"retained computation {name}", exact=True)
        real_memory_effect = float((raw_full - raw_direct).abs().max().item())
        assert real_memory_effect > 1e-7, "comparison must exercise a nonzero memory contribution"
        # Full zero_readout would remove Direct too and must not be substituted.
        both_zero = full.predict_block(case["x"], case["mask"], case["stats"],
                                       case["neighbors"], case["ids"], intervention="zero_readout")
        assert float((both_zero - raw_direct).abs().max().item()) > 1e-7
    return {"synthetic_full_optimizer_steps": 3, "omitting_only_memory_max_error": error,
            "full_memory_effect_max": real_memory_effect,
            "direct_and_query_retained_bitwise": True,
            "zero_readout_is_distinct_and_not_the_ablation": True}


def _check_gradients_and_blocking() -> dict[str, Any]:
    case = _synthetic_case(torch.device("cpu"))
    model = build_model(6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    initial = _predict(model, case)
    _close(initial, case["stats"]["mu"][None, None].expand_as(initial),
           "initial common fit-mean prediction", exact=True)
    _loss(initial, case, case["ids"]).backward()
    first_norms = _gradients(model, require_upstream=False)
    for name in ("direct_query.weight", "direct_key.weight", "direct_fusion.weight"):
        assert first_norms[name] == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _loss(_predict(model, case), case, case["ids"]).backward()
    upstream = _gradients(model, require_upstream=True)
    model.zero_grad(set_to_none=True)

    full, chunked = copy.deepcopy(model), copy.deepcopy(model)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4) for m in (full, chunked)]
    raw = _predict(full, case)
    loss = _loss(raw, case, case["ids"])
    loss.backward()
    parts, losses = [], []
    for ids in case["ids"].split(2):
        part = _predict(chunked, case, ids=ids)
        part_loss = _loss(part, case, ids)
        part_loss.backward()
        parts.append(part.detach())
        losses.append(part_loss.detach())
    output_error = _close(raw.detach(), torch.cat(parts, 2), "target blocking output")
    _close(loss.detach(), torch.stack(losses).sum(), "target blocking total loss")
    _gradients(full, require_upstream=True)
    _gradients(chunked, require_upstream=True)
    gradient_error = max(_close(a.grad, b.grad, f"target blocking gradient {name}")
                         for (name, a), (_, b) in zip(full.named_parameters(), chunked.named_parameters()))
    for m, opt in zip((full, chunked), optimizers):
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    step_error = max(_close(a, b, f"target blocking AdamW step {name}")
                     for (name, a), (_, b) in zip(full.named_parameters(), chunked.named_parameters()))
    with torch.no_grad():
        batches = torch.cat([_predict(model, case, batch_slice=slice(b, b + 1))
                             for b in range(case["x"].shape[0])])
        batch_error = _close(_predict(model, case), batches, "physical batch output")
        reordered = torch.tensor([5, 2, 0, 4, 1, 3])
        _close(_predict(model, case).index_select(2, reordered),
               _predict(model, case, ids=reordered), "target ordering output")
        expected = _predict(full, case)
    checkpoint = io.BytesIO()
    torch.save({"model": full.state_dict(), "optimizer": optimizers[0].state_dict()}, checkpoint)
    checkpoint.seek(0)
    payload = torch.load(checkpoint, map_location="cpu")
    restored = build_model(6)
    restored.load_state_dict(payload["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=1e-4)
    restored_optimizer.load_state_dict(payload["optimizer"])
    assert len(restored_optimizer.state) == len(optimizers[0].state)
    _close(expected, _predict(restored, case), "checkpoint exact reload", exact=True)
    return {"initial_prediction_exact_fit_mean": True,
            "upstream_gradients_after_first_step": {name: upstream[name] for name in
                ("flow_embedding.weight", "query.weight", "direct_query.weight", "direct_key.weight", "direct_fusion.weight")},
            "target_block_output_max_error": output_error,
            "target_block_gradient_max_error": gradient_error,
            "target_block_optimizer_step_max_error": step_error,
            "physical_batch_output_max_error": batch_error,
            "target_reordering_correct": True, "checkpoint_model_optimizer_reloaded": True}


def run_checks() -> dict[str, Any]:
    _runtime_settings()
    case = _synthetic_case(torch.device("cpu"))
    model = build_model(6)
    _nonzero_head(model)
    return {
        "device": "cpu", "torch_version": torch.__version__, "variant": model.variant,
        "initialization": _check_initialization(),
        "selected_observations": _check_selected_observations(),
        "information_direction_copy_clamp": _check_information(model, case),
        "neighbor_order_independence": _check_reordering(model, case),
        "own_flow_and_hidden_gradient": _check_own_flow_and_hidden_gradient(),
        "trained_full_without_memory": _check_full_without_memory(),
        "gradients_and_blocking": _check_gradients_and_blocking(),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks()
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
