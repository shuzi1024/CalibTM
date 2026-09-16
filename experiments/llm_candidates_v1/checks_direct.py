"""CPU synthetic correctness checks for the single DirectSync candidate.

No real dataset is loaded and no training job is launched. Two tiny synthetic
optimizer updates only verify that the common zero decoder does not disconnect
the newly introduced direct attention parameters after its first update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.sync_delta_v1.models import build_model as build_sync_model
from experiments.sync_delta_v1.preflight import (
    _check_chunking_and_reload,
    _check_information,
    _check_reordering,
    _close,
    _finite_gradients,
    _loss,
    _nonzero_head,
    _predict,
    _runtime_settings,
    _synthetic_case,
)

from .direct import DirectSyncModel, DIRECT_NEAREST_PER_DIRECTION


def _build(flows: int, seed: int = 41001) -> DirectSyncModel:
    reference = build_sync_model(flows, init_seed=seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 42017)
        model = DirectSyncModel(flows)
    loaded = model.load_state_dict(reference.state_dict(), strict=False)
    assert not loaded.unexpected_keys
    assert loaded.missing_keys and all(name.startswith("direct_") for name in loaded.missing_keys)
    return model


def _check_initialization() -> dict[str, Any]:
    result = {}
    for flows in (6, 144, 462):
        reference = build_sync_model(flows)
        model = _build(flows)
        assert model.variant == "sync_delta"
        parameters = dict(model.named_parameters())
        for name, value in reference.named_parameters():
            torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)
        added = sum(p.numel() for name, p in parameters.items() if name.startswith("direct_"))
        assert added == 1184
        result[str(flows)] = {
            "parameters": sum(p.numel() for p in parameters.values()),
            "added_parameters": added,
            "all_original_parameters_exact": True,
        }
    return result


@torch.no_grad()
def _check_direct_visibility(model: DirectSyncModel, case: dict[str, Any]) -> dict[str, Any]:
    _, aux = _predict(model, case, trace=True)
    batch, length, flows = case["x"].shape
    for label, reverse in (("forward", False), ("backward", True)):
        weights = aux[f"direct_weights_{label}"]
        support = weights > 0
        assert (support.sum(-1) <= DIRECT_NEAREST_PER_DIRECTION).all().item()
        expected = torch.zeros_like(support)
        for b in range(batch):
            for f in range(flows):
                observed = case["mask"][b, :, f].nonzero().flatten().tolist()
                for t in range(length):
                    allowed = [s for s in observed if s >= t] if reverse else [s for s in observed if s <= t]
                    selected = sorted(allowed, key=lambda s: abs(t - s))[:DIRECT_NEAREST_PER_DIRECTION]
                    expected[b, f, t, selected] = True
        assert torch.equal(support, expected), f"wrong direct support: {label}"
        torch.testing.assert_close(weights.sum(-1), expected.any(-1).float(), rtol=1e-6, atol=1e-7)
        assert (weights.diagonal(dim1=-2, dim2=-1)[case["mask"].permute(0, 2, 1)] > 0).all().item()

    # The direct path cannot use another flow, even though the original memory can.
    changed = case["x"].clone()
    changed[:, :, 1:] += 123.0
    _, other = _predict(model, case, x=changed, trace=True)
    for label in ("forward", "backward"):
        name = f"direct_read_{label}"
        _close(aux[name][:, :, 0], other[name][:, :, 0], f"own-flow only / {label}", exact=True)
    cut = 3
    future = case["x"].clone()
    future[:, cut + 1:] += 3.0
    _, changed_future = _predict(model, case, x=future, trace=True)
    _close(aux["direct_read_forward"][:, :cut + 1],
           changed_future["direct_read_forward"][:, :cut + 1], "direct no future", exact=True)
    past = case["x"].clone()
    past[:, :cut] += 3.0
    _, changed_past = _predict(model, case, x=past, trace=True)
    _close(aux["direct_read_backward"][:, cut:],
           changed_past["direct_read_backward"][:, cut:], "direct no past", exact=True)
    current = case["x"].clone()
    current[:, cut, 0] += 3.0
    _, changed_current = _predict(model, case, x=current, trace=True)
    changes = {}
    for label in ("forward", "backward"):
        name = f"direct_read_{label}"
        changes[label] = float((aux[name][:, cut, 0] - changed_current[name][:, cut, 0]).abs().max())
        assert changes[label] > 1e-6
    return {"nearest_two_support_exact": True, "direction_and_own_flow_isolation_exact": True,
            "current_timestamp_read_changes": changes}


def _check_empty_and_hidden_gradient() -> dict[str, Any]:
    case = _synthetic_case(torch.device("cpu"))
    model = _build(6)
    _nonzero_head(model)
    poisoned = torch.where(case["mask"], case["x"], float("nan")).requires_grad_()
    raw = _predict(model, case, x=poisoned)
    raw.square().mean().backward()
    assert torch.isfinite(poisoned.grad).all()
    assert torch.count_nonzero(poisoned.grad[~case["mask"]]).item() == 0
    assert poisoned.grad[case["mask"]].norm().item() > 0
    model.zero_grad(set_to_none=True)
    empty = dict(case)
    empty["mask"] = torch.zeros_like(case["mask"])
    empty["x"] = torch.full_like(case["x"], float("nan"))
    raw, aux = _predict(model, empty, trace=True)
    assert torch.isfinite(raw).all()
    for label in ("forward", "backward"):
        for prefix in ("direct_weights", "direct_features", "direct_read"):
            assert torch.count_nonzero(aux[f"{prefix}_{label}"]).item() == 0
    assert torch.count_nonzero(aux["hf"]).item() == 0
    assert torch.count_nonzero(aux["hb"]).item() == 0
    raw.square().mean().backward()
    norms = _finite_gradients(model, require_encoders=False)
    assert norms.get("query.weight", 0) > 0
    return {"hidden_value_gradient_exact_zero": True,
            "empty_direct_weights_features_and_reads_exact_zero": True,
            "all_empty_backward_finite": True}


def _check_two_update_gradient() -> dict[str, Any]:
    case = _synthetic_case(torch.device("cpu"))
    model = _build(6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    first = _predict(model, case)
    expected = case["stats"]["mu"][None, None].expand_as(first)
    _close(first, expected, "common initialization fit mean", exact=True)
    _loss(first, case, case["ids"]).backward()
    first_norms = _finite_gradients(model, require_encoders=False)
    for name in ("direct_query.weight", "direct_key.weight", "direct_fusion.weight"):
        assert first_norms.get(name, 0) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _loss(_predict(model, case), case, case["ids"]).backward()
    norms = _finite_gradients(model, require_encoders=True)
    required = ("direct_query.weight", "direct_key.weight", "direct_fusion.weight")
    for name in required:
        assert norms.get(name, 0) > 0, f"no new-path gradient after decoder update: {name}"
    return {"initial_prediction_exact_fit_mean": True,
            "new_path_gradient_norms_after_first_update": {name: norms[name] for name in required}}


def run_checks() -> dict[str, Any]:
    _runtime_settings()
    case = _synthetic_case(torch.device("cpu"))
    model = _build(6)
    _nonzero_head(model)
    result = {
        "device": "cpu", "torch_version": torch.__version__,
        "candidate": "direct_sync_v1", "variant_for_engine": model.variant,
        "initialization": _check_initialization(),
        "information_and_copy": _check_information(model, case),
        "direct_visibility": _check_direct_visibility(model, case),
        "event_reordering": _check_reordering(model, case),
        "chunking_and_reload": _check_chunking_and_reload(model, case),
        "empty_and_hidden_gradient": _check_empty_and_hidden_gradient(),
        "two_synthetic_update_gradient": _check_two_update_gradient(),
        "status": "passed",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks()
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
