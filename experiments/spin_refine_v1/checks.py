"""Synthetic CPU checks of refinement semantics, gradients and matched controls."""
from __future__ import annotations

import copy
import io
import json

import numpy as np
import torch

from experiments.spin_sync_unified_v1.checks import (
    check_finite_gradients, expected_direct_support, gradient_equivalence,
    missing_loss, nonzero_decoder, predict, synthetic_case,
)
from experiments.spin_sync_unified_v1.models import build_model as build_original
from .models import VARIANTS, build_model


def configured(variant, *, seed=41001):
    model = build_model(4, init_seed=seed, variant=variant)
    model.configure_fit_statistics(dict(mu=np.full(4, 1.5), C=1.5, global_scale=.7))
    model.set_execution(node_chunk=2, gradient_checkpointing=True)
    return model


def reject_path(*args, **kwargs):
    raise AssertionError("Refinement must not call the removed memory path")


def run_checks():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(170917)
    case = synthetic_case()
    report = dict(device="cpu", gpu_used=False, real_data_loaded=False,
                  torch_version=torch.__version__, variants=list(VARIANTS))

    # Matched skeletons have exactly equal states, parameter sets and order.
    first_weight = "h_enc.mlp.0.layer.0.weight"
    for seed in (41001, 41002):
        one, two = (configured(v, seed=seed) for v in VARIANTS)
        assert list(one.state_dict()) == list(two.state_dict())
        assert list(dict(one.named_parameters())) == list(dict(two.named_parameters()))
        for name, value in one.state_dict().items():
            torch.testing.assert_close(value, two.state_dict()[name], rtol=0, atol=0, msg=name)
        original = build_original(4, init_seed=seed, variant="spin_direct_no_memory")
        original.configure_fit_statistics(dict(mu=np.full(4, 1.5), C=1.5, global_scale=.7))
        assert list(original.state_dict()) == list(one.state_dict())
        for name, value in original.state_dict().items():
            extended = one.state_dict()[name]
            if name == first_weight:
                torch.testing.assert_close(extended[:, :1], value, rtol=0, atol=0)
                assert torch.count_nonzero(extended[:, 1]) == 0
            else:
                torch.testing.assert_close(extended, value, rtol=0, atol=0, msg=name)
        assert sum(p.numel() for p in one.parameters()) == 32 + sum(p.numel() for p in original.parameters())
    report.update(matched_state_and_parameter_order_exact_both_seeds=True,
                  original_parameters_preserved_with_32_zero_flag_weights=True)

    models = {v: configured(v) for v in VARIANTS}
    for model in models.values():
        nonzero_decoder(model)
        model._memory_scan = reject_path
        model._memory_tokens = reject_path
        assert not hasattr(model, "key") and not hasattr(model, "value")

    # Poisoned hidden truth cannot enter pass1, pseudo-observations, or Direct.
    for variant, model in models.items():
        baseline = predict(model, case)
        for poison in (float("nan"), float("inf"), float("-inf"), 1e8):
            values = torch.where(case["mask"], case["x"], poison)
            torch.testing.assert_close(predict(model, case, x=values), baseline, rtol=0, atol=0)
        values = torch.where(case["mask"], case["x"], float("nan")).requires_grad_()
        model.zero_grad(set_to_none=True)
        missing_loss(predict(model, case, x=values), case).backward()
        assert values.grad is not None and torch.isfinite(values.grad).all(), variant
        assert torch.count_nonzero(values.grad[~case["mask"]]) == 0, variant
        assert values.grad[case["mask"]].norm() > 0, variant
        check_finite_gradients(model)
        projected = model(values, case["mask"], case["stats"], case["neighbors"])
        assert torch.isfinite(projected).all()
        torch.testing.assert_close(projected[case["mask"]], case["x"][case["mask"]], rtol=0, atol=0)
    report.update(hidden_nan_infinity_truth_isolation=True,
                  hidden_input_gradients_zero=True, forward_restores_exact_observations=True)

    # Actual call count verifies weight reuse and fixed, same-resolution unroll.
    counts = {}
    for variant, model in models.items():
        calls = []
        hook = model.context_encoder.register_forward_hook(lambda *args: calls.append(1))
        raw, aux = predict(model, case, trace=True)
        hook.remove()
        counts[variant] = len(calls)
        assert len(calls) == model.refinement_steps
        assert torch.equal(aux["stage1_source_mask"], case["mask"])
        assert torch.equal(aux["true_observation_mask"], case["mask"])
        assert torch.equal(aux["direct_source_mask"], case["mask"])
        assert aux["stage1_raw"].shape == case["x"].shape
        if variant == "two_step":
            assert aux["final_source_mask"].all()
            torch.testing.assert_close(aux["refinement_input"][case["mask"]],
                                       case["x"][case["mask"]], rtol=0, atol=0)
            torch.testing.assert_close(aux["refinement_input"][~case["mask"]],
                                       aux["stage1_raw"].clamp_min(0)[~case["mask"]], rtol=0, atol=0)
        else:
            assert aux["refinement_input"] is None
            assert torch.equal(aux["final_source_mask"], case["mask"])
        for reverse, label in ((False, "forward"), (True, "backward")):
            support = expected_direct_support(case["mask"], case["ids"], reverse)
            assert torch.equal(aux["direct_weights_" + label] > 0, support)
    one_raw, one_aux = predict(models["one_step"], case, trace=True)
    two_raw, two_aux = predict(models["two_step"], case, trace=True)
    torch.testing.assert_close(one_raw, two_aux["stage1_raw"], rtol=0, atol=0)
    assert (two_raw - one_raw).abs().max() > 1e-7
    report.update(context_calls_per_prediction=counts,
                  fixed_shared_unroll_and_source_masks_verified=True,
                  direct_nearest_two_true_observations_preserved=True,
                  first_pass_matches_one_step_exactly=True)

    # Perturb only estimates between passes: context/output changes, Direct does not.
    changed = copy.deepcopy(models["two_step"])
    original_projection = changed._project_context
    def perturb_estimates(first_raw, observed, true_mask):
        result = original_projection(first_raw, observed, true_mask)
        offset = torch.linspace(.3, 3., result.shape[1])[None, :, None]
        return torch.where(true_mask, result, result + offset)
    changed._project_context = perturb_estimates
    changed_raw, changed_aux = predict(changed, case, trace=True)
    assert (changed_raw - two_raw).abs().max() > 1e-7
    assert (changed_aux["context"] - two_aux["context"]).abs().max() > 1e-7
    torch.testing.assert_close(changed_aux["stage1_raw"], two_aux["stage1_raw"], rtol=0, atol=0)
    for label in ("forward", "backward"):
        for category in ("weights", "features", "read"):
            key = "direct_" + category + "_" + label
            torch.testing.assert_close(changed_aux[key], two_aux[key], rtol=0, atol=0)
    # The final objective backpropagates through missing estimates, with exact
    # zero gradient to stage1 values replaced by original observations.
    stage_gradient, = torch.autograd.grad(two_raw[~case["mask"]].sum(),
                                          two_aux["stage1_raw"], retain_graph=True)
    assert torch.isfinite(stage_gradient).all()
    assert stage_gradient[~case["mask"]].norm() > 1e-8
    assert torch.count_nonzero(stage_gradient[case["mask"]]) == 0
    report.update(second_pass_depends_on_predicted_values=True,
                  estimated_values_cannot_change_direct_reads=True,
                  final_objective_backpropagates_through_estimates=True)

    # Chunking targets must not delete other flows' first-stage estimates.
    for variant, model in models.items():
        left, right = copy.deepcopy(model), copy.deepcopy(model)
        left.zero_grad(set_to_none=True)
        right.zero_grad(set_to_none=True)
        complete = predict(left, case)
        missing_loss(complete, case).backward()
        parts = []
        for ids in case["ids"].split(2):
            raw, aux = predict(right, case, ids=ids, trace=True)
            assert aux["stage1_raw"].shape == case["x"].shape
            missing_loss(raw, case, ids).backward()
            parts.append(raw.detach())
        torch.testing.assert_close(complete, torch.cat(parts, dim=2), rtol=3e-5, atol=3e-6)
        gradient_equivalence(left, right, variant + "/target_blocks")
        batch_parts = []
        for b in range(case["x"].shape[0]):
            batch_parts.append(predict(model, case, x=case["x"][b:b+1], mask=case["mask"][b:b+1]))
        torch.testing.assert_close(predict(model, case), torch.cat(batch_parts), rtol=3e-5, atol=3e-6)
        permuted = torch.tensor([3, 0, 2])
        torch.testing.assert_close(predict(model, case, ids=permuted),
                                   predict(model, case).index_select(2, permuted), rtol=3e-5, atol=3e-6)
    report.update(full_graph_target_chunk_output_gradient_invariance=True,
                  batch_chunk_and_target_order_invariance=True)

    # Checkpointing is only a memory optimization and preserves the unrolled graph.
    for variant, model in models.items():
        checkpointed, ordinary = copy.deepcopy(model), copy.deepcopy(model)
        checkpointed.set_execution(node_chunk=2, gradient_checkpointing=True)
        ordinary.set_execution(node_chunk=2, gradient_checkpointing=False)
        checkpointed.zero_grad(set_to_none=True)
        ordinary.zero_grad(set_to_none=True)
        a, b = predict(checkpointed, case), predict(ordinary, case)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        missing_loss(a, case).backward()
        missing_loss(b, case).backward()
        gradient_equivalence(checkpointed, ordinary, variant + "/checkpointing")
    report["checkpointing_forward_and_gradient_equivalence"] = True

    # Zero decoder initialization must not permanently disable either shared path.
    for variant in VARIANTS:
        model = configured(variant)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            raw = predict(model, case)
            # Same type of final-pass missing-only raw absolute objective as training.
            loss = (raw - case["x"]).abs().masked_select(~case["mask"]).mean() / 1.5
            loss.backward()
            norms = check_finite_gradients(model)
            assert model.decoder[-1].weight.grad.norm() > 0
            if step:
                for prefix in ("u_enc.", "h_enc.", "x_skip.", "context_encoder.",
                               "query.", "direct_query.", "direct_key.", "direct_fusion."):
                    assert sum(value for name, value in norms.items() if name.startswith(prefix)) > 0, (variant, prefix)
                assert model.h_enc.mlp[0].layer[0].weight.grad[:, 1].norm() > 0
            optimizer.step()
    report["all_retained_information_paths_and_provenance_trainable"] = True

    for variant, model in models.items():
        empty_mask = torch.zeros_like(case["mask"])
        empty_x = torch.full_like(case["x"], float("nan"))
        model.zero_grad(set_to_none=True)
        raw, aux = predict(model, case, x=empty_x, mask=empty_mask, trace=True)
        assert raw.shape == case["x"].shape and torch.isfinite(raw).all(), variant
        for key in ("hf", "hb", "direct_read_forward", "direct_read_backward"):
            assert torch.count_nonzero(aux[key]) == 0, (variant, key)
        raw.square().mean().backward()
        check_finite_gradients(model)
        payload = io.BytesIO()
        torch.save(model.state_dict(), payload)
        payload.seek(0)
        restored = configured(variant)
        restored.load_state_dict(torch.load(payload, map_location="cpu"), strict=True)
        torch.testing.assert_close(predict(restored, case), predict(model, case), rtol=0, atol=0)
        for kwargs in ({"intervention": "off"}, {"serial_order_context": [0]}):
            try:
                model.predict_block(case["x"], case["mask"], case["stats"], case["neighbors"], case["ids"], **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError("Unexpected inference intervention accepted")
    report.update(empty_observations_finite_with_zero_direct=True,
                  exact_state_reload=True, inference_interventions_rejected=True,
                  parameters_synthetic_F4={v: sum(p.numel() for p in m.parameters()) for v, m in models.items()},
                  passed=True)
    return report


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
