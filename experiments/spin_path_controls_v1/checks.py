"""Bounded synthetic CPU checks: no WAN data and no GPU execution."""
from __future__ import annotations

import copy
import io
import json

import numpy as np
import torch
from torch.nn import functional as F

from experiments.spin_sync_unified_v1.checks import (
    check_finite_gradients, expected_direct_support, missing_loss,
    nonzero_decoder, predict, synthetic_case,
)
from experiments.spin_sync_unified_v1.models import build_model as build_original
from experiments.sync_delta_v1.models import HEADS, HEAD_DIM
from .models import VARIANTS, build_model


def configured(variant, *, seed=41001, original=False):
    factory = build_original if original else build_model
    model = factory(4, init_seed=seed, variant=variant)
    model.configure_fit_statistics(dict(mu=np.full(4, 1.5), C=1.5, global_scale=.7))
    model.set_execution(node_chunk=2, gradient_checkpointing=True)
    return model


def exact_tensors(left, right, label):
    assert list(left) == list(right), label + "/keys_or_order"
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0,
                                   msg=label + "/" + name)


def exact_gradients(left, right):
    assert list(dict(left.named_parameters())) == list(dict(right.named_parameters()))
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


def reject_path(*args, **kwargs):
    raise AssertionError("A disabled information path was called")


def run_checks():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(170917)
    case = synthetic_case()
    report = {
        "device": "cpu", "gpu_used": False, "real_data_loaded": False,
        "torch_version": torch.__version__, "variants": list(VARIANTS),
    }

    # 1. Constructor order is important for historical optimizer-state resume.
    removed = {}
    for seed in (41001, 41002):
        old = configured("spin_direct_no_memory", seed=seed, original=True)
        old_state = old.state_dict()
        for variant in VARIANTS:
            model = configured(variant, seed=seed)
            state = model.state_dict()
            assert set(state).issubset(old_state), variant
            for name, value in state.items():
                torch.testing.assert_close(value, old_state[name], rtol=0, atol=0,
                                           msg=f"{seed}/{variant}/{name}")
            removed[variant] = sorted(set(old_state) - set(state))
            if variant == "spin_direct":
                exact_tensors(state, old_state, "original_state")
                assert list(dict(model.named_parameters())) == list(dict(old.named_parameters()))
            else:
                assert removed[variant], variant
    report.update(shared_initialization_exact_both_seeds=True,
                  removed_state_keys=removed,
                  original_state_and_parameter_order_exact=True)

    # 2. Compare nonconstant outputs, every gradient, and AdamW updates exactly.
    old = configured("spin_direct_no_memory", original=True)
    new = configured("spin_direct")
    new.load_state_dict(old.state_dict(), strict=True)
    for model in (old, new):
        torch.testing.assert_close(predict(model, case), torch.full_like(case['x'], 1.5),
                                   rtol=0, atol=0)
        nonzero_decoder(model)
    old_opt = torch.optim.AdamW(old.parameters(), lr=1e-3, weight_decay=1e-4)
    new_opt = torch.optim.AdamW(new.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(2):
        old_opt.zero_grad(set_to_none=True)
        new_opt.zero_grad(set_to_none=True)
        left, right = predict(old, case), predict(new, case)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        missing_loss(left, case).backward()
        missing_loss(right, case).backward()
        exact_gradients(old, new)
        old_opt.step()
        new_opt.step()
        exact_tensors(old.state_dict(), new.state_dict(), "adam_update")
    # Resume a populated historical optimizer to verify param-index mapping.
    restored = configured("spin_direct")
    restored.load_state_dict(old.state_dict(), strict=True)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=1e-4)
    restored_opt.load_state_dict(copy.deepcopy(old_opt.state_dict()))
    old_opt.zero_grad(set_to_none=True)
    restored_opt.zero_grad(set_to_none=True)
    missing_loss(predict(old, case), case).backward()
    missing_loss(predict(restored, case), case).backward()
    old_opt.step()
    restored_opt.step()
    exact_tensors(old.state_dict(), restored.state_dict(), "optimizer_resume")
    report['original_forward_gradients_updates_and_optimizer_resume_exact'] = True

    models = {v: configured(v) for v in VARIANTS}
    for model in models.values():
        nonzero_decoder(model)
        model._memory_scan = reject_path
        model._memory_tokens = reject_path

    # 3. Hidden raw truth is excluded before normalization and every read.
    for variant, model in models.items():
        baseline = predict(model, case)
        for poison in (float('nan'), float('inf'), 1e8):
            x = torch.where(case['mask'], case['x'], poison)
            torch.testing.assert_close(predict(model, case, x=x), baseline, rtol=0, atol=0)
        x = torch.where(case['mask'], case['x'], float('nan')).requires_grad_()
        model.zero_grad(set_to_none=True)
        missing_loss(predict(model, case, x=x), case).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), variant
        assert torch.count_nonzero(x.grad[~case['mask']]) == 0, variant
        assert x.grad[case['mask']].norm() > 0, variant
        check_finite_gradients(model)
    report['hidden_truth_invariance_and_zero_hidden_gradients'] = True
    report['all_variants_bypass_sync_scan'] = True

    # 4. Direct-only target output cannot use any other flow or the graph.
    model = models['direct_only']
    assert not any(hasattr(model, name) for name in
                   ('h_enc', 'h_norm', 'x_skip', 'context_encoder', 'key', 'value'))
    target_ids = torch.tensor([0])
    baseline, aux = predict(model, case, ids=target_ids, trace=True)
    x = case['x'].clone()
    x[:, :, 1:] = torch.linspace(-1e5, 1e5, x[:, :, 1:].numel()).reshape_as(x[:, :, 1:])
    mask = case['mask'].clone()
    mask[:, :, 1:] = ~mask[:, :, 1:]
    graph = torch.full_like(case['neighbors'], -1)
    changed, changed_aux = predict(model, case, x=x, ids=target_ids,
                                   mask=mask, neighbors=graph, trace=True)
    torch.testing.assert_close(changed, baseline, rtol=0, atol=0)
    torch.testing.assert_close(changed_aux['q'], aux['q'], rtol=0, atol=0)
    assert aux['query_source'] == 'position' and 'context' not in aux
    torch.testing.assert_close(aux['position_query_input'], aux['position'], rtol=0, atol=0)
    # The query must remain position-only even when own-flow values/mask change.
    _, own_changed = predict(model, case, x=case['x'] * 7.,
                             mask=~case['mask'], ids=target_ids, trace=True)
    torch.testing.assert_close(aux['q'], own_changed['q'], rtol=0, atol=0)
    batch, length = case['x'].shape[:2]
    expected_q = F.normalize(model.query(aux['position'].index_select(2, target_ids)).reshape(
        batch, length, 1, HEADS, HEAD_DIM), dim=-1, eps=1e-6).permute(0, 2, 1, 3, 4)
    torch.testing.assert_close(aux['q'], expected_q, rtol=0, atol=0)
    x = case['x'].clone().requires_grad_()
    model.zero_grad(set_to_none=True)
    missing_loss(predict(model, case, x=x, ids=target_ids), case, target_ids).backward()
    assert torch.count_nonzero(x.grad[:, :, 1:]) == 0
    assert x.grad[:, :, 0].norm() > 0
    for reverse, label in ((False, 'forward'), (True, 'backward')):
        support = expected_direct_support(case['mask'], target_ids, reverse)
        assert torch.equal(aux['direct_weights_' + label] > 0, support)
    report.update(direct_only_no_context_modules=True,
                  direct_only_other_flow_values_masks_graph_isolated=True,
                  direct_only_position_query_verified=True,
                  direct_only_other_flow_input_gradients_zero=True,
                  direct_nearest_two_support_preserved=True)

    # 5. Context-only genuinely deletes Direct, and passes [0,0,Q(H)].
    model = models['context_only']
    assert not any(hasattr(model, name) for name in
                   ('direct_query', 'direct_key', 'direct_fusion', 'key', 'value'))
    raw, aux = predict(model, case, trace=True)
    assert aux['query_source'] == 'spin_context'
    for key in ('hf', 'hb', 'direct_read_forward', 'direct_read_backward',
                'memory_hf', 'memory_hb'):
        assert torch.count_nonzero(aux[key]) == 0, key
    query_flat = aux['q'].permute(0, 2, 1, 3, 4).flatten(-2)
    zero = torch.zeros_like(query_flat)
    expected = model.fit_mean + model.fit_scale * model.decoder(
        torch.cat((zero, zero, query_flat), -1)).squeeze(-1)
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    report['context_only_direct_removed_and_zero_decoder_inputs_verified'] = True

    # 6. From zero output initialization, each retained path becomes trainable.
    trainable_groups = {
        'spin_direct': ('u_enc.', 'context_encoder.', 'query.', 'direct_fusion.'),
        'context_only': ('u_enc.', 'context_encoder.', 'query.'),
        'direct_only': ('u_enc.', 'query.', 'direct_fusion.', 'direct_query.', 'direct_key.'),
    }
    for variant in VARIANTS:
        model = configured(variant)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            missing_loss(predict(model, case), case).backward()
            norms = check_finite_gradients(model)
            assert model.decoder[-1].weight.grad.norm() > 0, variant
            if step:
                for prefix in trainable_groups[variant]:
                    assert sum(v for k, v in norms.items() if k.startswith(prefix)) > 0, (variant, prefix)
            opt.step()
    report['retained_information_paths_trainable_after_two_updates'] = True

    # 7. Empty observations are safe, including hidden NaN payloads.
    for variant, model in models.items():
        empty_mask = torch.zeros_like(case['mask'])
        empty_x = torch.full_like(case['x'], float('nan'))
        model.zero_grad(set_to_none=True)
        raw, aux = predict(model, case, x=empty_x, mask=empty_mask, trace=True)
        assert raw.shape == case['x'].shape and torch.isfinite(raw).all(), variant
        for key in ('hf', 'hb', 'direct_read_forward', 'direct_read_backward'):
            assert torch.count_nonzero(aux[key]) == 0, (variant, key)
        raw.square().mean().backward()
        check_finite_gradients(model)
        payload = io.BytesIO()
        torch.save(model.state_dict(), payload)
        payload.seek(0)
        restored = configured(variant)
        restored.load_state_dict(torch.load(payload, map_location='cpu'), strict=True)
        torch.testing.assert_close(predict(restored, case), predict(model, case), rtol=0, atol=0)
    report.update(all_missing_finite_with_zero_direct_reads=True,
                  state_reload_exact=True, passed=True)
    report['parameters_synthetic_F4'] = {
        variant: sum(p.numel() for p in model.parameters())
        for variant, model in models.items()
    }
    return report


if __name__ == '__main__':
    print(json.dumps(run_checks(), indent=2))
