"""Synthetic correctness checks for FeatureSyncModel; never runs training jobs."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from experiments.sync_delta_v1.models import HEAD_DIM, HEADS, build_model as build_sync
from experiments.sync_delta_v1.preflight import (
    _check_chunking_and_reload,
    _check_information,
    _check_reordering,
    _finite_gradients,
    _loss,
    _nonzero_head,
    _predict,
    _runtime_settings,
    _synthetic_case,
)

from .feature import FeatureSyncModel


def _build(flows: int, seed: int = 41001) -> FeatureSyncModel:
    """Local check fixture; the production factory supplies its own identity."""
    reference = build_sync(flows, init_seed=seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 42017)
        model = FeatureSyncModel(flows)
    incompatible = model.load_state_dict(reference.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {"feature_weight", "feature_bias"}
    assert not incompatible.unexpected_keys
    return model


def check_identity() -> dict[str, Any]:
    rows = {}
    for flows in (144, 462):
        model, reference = _build(flows), build_sync(flows)
        assert model.variant == "sync_delta"
        original = dict(reference.named_parameters())
        actual = dict(model.named_parameters())
        for name, parameter in original.items():
            torch.testing.assert_close(actual[name], parameter, rtol=0, atol=0)
        added = {name: parameter.numel() for name, parameter in actual.items() if name not in original}
        assert added == {"feature_weight": 4096, "feature_bias": 128}
        torch.testing.assert_close(model.feature_weight, torch.eye(HEAD_DIM).repeat(HEADS, 1, 1), rtol=0, atol=0)
        assert torch.count_nonzero(model.feature_bias).item() == 0
        rows[str(flows)] = {"shared_parameter_tensors_exact": len(original),
                            "extra_parameters": sum(added.values()),
                            "total_parameters": sum(p.numel() for p in model.parameters())}
    return rows


def check_feature_map(model: FeatureSyncModel, case: dict[str, Any]) -> dict[str, Any]:
    encoded = model._encode(case["x"], case["mask"], case["stats"], case["neighbors"], case["ids"], None)
    keys, values, queries, mask = encoded[:4]
    assert keys.shape[-3:] == (HEADS, 9, HEAD_DIM)
    assert queries.shape[-2:] == (HEADS, HEAD_DIM)
    valid_keys = keys.transpose(-3, -2)[mask]
    hidden_keys = keys.transpose(-3, -2)[~mask]
    assert torch.count_nonzero(hidden_keys).item() == 0
    assert (valid_keys > 0).all().item() and (queries > 0).all().item()
    torch.testing.assert_close(valid_keys.norm(dim=-1), torch.ones_like(valid_keys[..., 0]),
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(queries.norm(dim=-1), torch.ones_like(queries[..., 0]),
                               rtol=1e-5, atol=1e-6)

    # Map the same addresses through event-shaped and query-shaped call sites.
    event_form = model._map_address(queries.unsqueeze(-3)).squeeze(-3)
    query_form = model._map_address(queries)
    torch.testing.assert_close(event_form, query_form, rtol=0, atol=0)

    # Observed magnitudes can change values but never the address coordinates.
    changed = case["x"] + case["mask"].to(case["x"].dtype) * 3.0
    changed_encoded = model._encode(changed, case["mask"], case["stats"], case["neighbors"], case["ids"], None)
    torch.testing.assert_close(keys, changed_encoded[0], rtol=0, atol=0)
    torch.testing.assert_close(queries, changed_encoded[2], rtol=0, atol=0)
    assert not torch.equal(values, changed_encoded[1])
    # Numerical records only: repeated appearances of the same source event
    # across target neighborhoods are retained, so these are not unique-event
    # estimates and neither similarity nor entropy is a task-success criterion.
    per_head = valid_keys.detach().transpose(0, 1)
    gram = per_head @ per_head.transpose(-1, -2)
    off_diagonal = ~torch.eye(gram.shape[-1], dtype=torch.bool, device=gram.device)
    average_cosine = gram[:, off_diagonal].mean(dim=-1)
    probabilities = per_head / per_head.sum(dim=-1, keepdim=True)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1).mean(dim=-1)
    return {"missing_and_padded_keys_exact_zero": True,
            "same_map_for_keys_and_queries": True,
            "positive_unit_norm_valid_addresses": True,
            "addresses_independent_of_observed_magnitudes": True,
            "memory_head_dimension": HEAD_DIM,
            "initial_diagnostics_not_success_criteria": {
                "valid_event_appearances": valid_keys.shape[0],
                "mean_off_diagonal_key_cosine_per_head": average_cosine.tolist(),
                "mean_key_feature_entropy_nats_per_head": entropy.tolist(),
                "repeated_source_events_across_neighborhoods_retained": True,
                "fixed_softmax_scale": HEAD_DIM ** 0.5}}


def check_new_gradients(model: FeatureSyncModel, case: dict[str, Any]) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    _loss(_predict(model, case), case, case["ids"]).backward()
    norms = _finite_gradients(model, require_encoders=True)
    rows = {}
    for name in ("feature_weight", "feature_bias"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None and norms[name] > 0
        per_head = parameter.grad.reshape(HEADS, -1).norm(dim=-1)
        assert (per_head > 0).all().item()
        rows[name] = per_head.tolist()
    model.zero_grad(set_to_none=True)

    # Hidden NaNs must also have exactly zero input derivatives.
    poisoned = torch.where(case["mask"], case["x"], float("nan")).requires_grad_()
    raw = _predict(model, case, x=poisoned)
    raw.square().mean().backward()
    assert poisoned.grad is not None and torch.isfinite(poisoned.grad).all().item()
    assert torch.count_nonzero(poisoned.grad[~case["mask"]]).item() == 0
    assert poisoned.grad[case["mask"]].norm().item() > 0
    _finite_gradients(model, require_encoders=True)
    model.zero_grad(set_to_none=True)
    return {"nonzero_gradient_norm_per_head": rows, "hidden_input_gradients_exact_zero": True}


def check_empty(model: FeatureSyncModel, case: dict[str, Any]) -> dict[str, Any]:
    model = copy.deepcopy(model)
    case = dict(case, x=torch.full_like(case["x"], float("nan")), mask=torch.zeros_like(case["mask"]))
    model.zero_grad(set_to_none=True)
    raw, aux = _predict(model, case, trace=True)
    assert torch.isfinite(raw).all().item()
    for name in ("k", "v", "hf", "hb", "states_forward", "states_backward"):
        assert torch.isfinite(aux[name]).all().item()
        assert torch.count_nonzero(aux[name]).item() == 0, name
    raw.square().mean().backward()
    norms = _finite_gradients(model, require_encoders=False)
    assert norms.get("query.weight", 0) > 0
    assert norms.get("feature_weight", 0) > 0
    return {"all_empty_finite_output_and_backward": True,
            "keys_values_states_and_readouts_exact_zero": True}


def run_checks(device: str | torch.device = "cpu") -> dict[str, Any]:
    device = torch.device(device)
    _runtime_settings()
    case = _synthetic_case(device)
    model = _build(6).to(device)
    # The production zero decoder initially predicts the frozen fit mean.
    raw = _predict(model, case)
    expected = case["stats"]["mu"][None, None].expand_as(raw)
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    _nonzero_head(model)
    report = {"device": str(device), "torch_version": torch.__version__,
              "initial_prediction_is_fit_mean": True,
              "identity": check_identity(),
              "feature_map": check_feature_map(model, case),
              "information_isolation": _check_information(model, case),
              "same_bucket_permutation": _check_reordering(model, case),
              "gradients": check_new_gradients(model, case),
              "all_empty": check_empty(model, case),
              "chunking_and_reload": _check_chunking_and_reload(model, case)}
    report["passed"] = True
    report["scope"] = "Synthetic correctness only; no task performance or feature-rank success claim."
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_checks(args.device)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
