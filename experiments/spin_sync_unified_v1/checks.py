"""Bounded CPU synthetic checks for the unified SPIN/Sync architecture.

No real datasets or GPU are used. Bidirectional SPIN context deliberately means
that Sync memory is not tested as a causal state. Only the Direct source support
is constrained to its respective direction.
"""
import copy
import io
import json

import torch


def synthetic_case():
    batch, length, flows = 2, 50, 4
    b = torch.arange(batch)[:, None, None]
    t = torch.arange(length)[None, :, None]
    f = torch.arange(flows)[None, None, :]
    values = (1.7 + .023*t + .13*f + .09*b + .11*torch.sin(t*.31+f)).float()
    mask = (((t+2*f+b) % 7) == 0).expand_as(values).clone()
    mask[:, 0] = False
    mask[:, -1] = False
    mask[:, 5, 1] = True
    neighbors = torch.full((flows, 6), -1, dtype=torch.long)
    for target in range(flows):
        neighbors[target, :flows] = torch.arange(flows).roll(-target)
    return dict(x=values, mask=mask, neighbors=neighbors,
                ids=torch.arange(flows), stats={"mu": torch.full((flows,), 1.5),
                "scale": torch.full((flows,), .7)})


def expected_direct_support(mask, target_ids, reverse):
    batch, length, _ = mask.shape
    support = torch.zeros(batch, len(target_ids), length, length, dtype=torch.bool)
    for b in range(batch):
        for g, flow in enumerate(target_ids.tolist()):
            observed = mask[b, :, flow].nonzero().flatten().tolist()
            for t in range(length):
                candidates = [s for s in observed if (s >= t if reverse else s <= t)]
                selected = sorted(candidates, key=lambda s: abs(t-s))[:2]
                support[b, g, t, selected] = True
    return support


def check_finite_gradients(model):
    norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            norms[name] = float(parameter.grad.norm())
    return norms


def predict(model, case, *, x=None, ids=None, neighbors=None, mask=None, trace=False):
    return model.predict_block(case['x'] if x is None else x,
        case['mask'] if mask is None else mask, case['stats'],
        case['neighbors'] if neighbors is None else neighbors,
        case['ids'] if ids is None else ids, return_aux=trace, trace=trace)


def nonzero_decoder(model):
    with torch.no_grad():
        last = model.decoder[-1]
        last.weight.copy_(torch.linspace(-.025,.035,last.weight.numel()).reshape_as(last.weight))
        last.bias.fill_(.03)


def missing_loss(raw, case, ids=None):
    ids = case['ids'] if ids is None else ids
    target = case['x'].index_select(2,ids)
    missing = ~case['mask'].index_select(2,ids)
    return (raw-target).square().masked_select(missing).sum() / (~case['mask']).sum()


def gradient_equivalence(left, right, label):
    assert dict(left.named_parameters()).keys() == dict(right.named_parameters()).keys()
    for (name,a),(_,b) in zip(left.named_parameters(),right.named_parameters()):
        assert (a.grad is None) == (b.grad is None), (label,name)
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-5,msg=label+'/'+name)


def run_checks(factory):
    """factory(mode) must return a fresh seeded model configured to mean=1.5,scale=.7.

    Factory mode names here are 'full' and 'no_memory'; the factory may translate
    these to the model's public variant names. Both must return trace keys k,v,q,
    event_mask,direct_weights_{forward,backward},direct_read_{forward,backward},
    memory_hf,memory_hb for the applicable paths. This explicit factory bridge
    avoids loading any dataset merely to configure the synthetic scaler.
    """
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(160916)
    case = synthetic_case()
    full, control = factory('full'), factory('no_memory')
    full_state, control_state = full.state_dict(), control.state_dict()
    assert set(control_state).issubset(full_state)
    removed = set(full_state)-set(control_state)
    assert removed, 'no-memory model must actually omit memory parameters'
    for name,value in control_state.items():
        torch.testing.assert_close(value,full_state[name],rtol=0,atol=0,msg=name)
    for model in (full,control):
        torch.testing.assert_close(predict(model,case),torch.full_like(case['x'],1.5),rtol=0,atol=0)
        nonzero_decoder(model)
    report = {'device':'cpu','real_data_loaded':False,'gpu_used':False,
              'torch_version':torch.__version__,'shared_initialization_exact':True,
              'no_memory_removed_state_keys':sorted(removed)}

    # 1. Hidden truth cannot affect either architecture, even through SPIN context.
    for label,model in (('full',full),('no_memory',control)):
        expected = predict(model,case)
        for poison in (float('nan'),float('inf'),1e8):
            x = torch.where(case['mask'],case['x'],poison)
            torch.testing.assert_close(predict(model,case,x=x),expected,rtol=0,atol=0)
        x = torch.where(case['mask'],case['x'],float('nan')).requires_grad_()
        model.zero_grad(set_to_none=True)
        missing_loss(predict(model,case,x=x),case).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert torch.count_nonzero(x.grad[~case['mask']]) == 0
        assert x.grad[case['mask']].norm() > 0
        check_finite_gradients(model)
    report['hidden_truth_invariance_and_zero_hidden_gradients'] = True

    # 2. SPIN query tokens at missing slots must never become Sync write events.
    _,aux = predict(full,case,trace=True)
    visible = aux['event_mask'][:, :, :, None, :, None].expand_as(aux['k'])
    assert torch.count_nonzero(aux['k'][~visible]) == 0
    assert torch.count_nonzero(aux['v'][~visible]) == 0
    assert aux['k'][visible].norm() > 0 and aux['v'][visible].norm() > 0
    source_ids = case['neighbors'].clamp_min(0)
    expected_events = (case['mask'][:,:,source_ids] & (case['neighbors']>=0)[None,None]).permute(0,2,1,3)
    assert torch.equal(aux['event_mask'],expected_events)
    report['masked_only_sync_writes'] = True

    # 3. Neighbor order is irrelevant to graph sums and simultaneous state writes.
    left,right = copy.deepcopy(full),copy.deepcopy(full)
    left.zero_grad(set_to_none=True)
    right.zero_grad(set_to_none=True)
    prediction = predict(left,case)
    permutation = torch.tensor([3,5,1,4,0,2])
    other = predict(right,case,neighbors=case['neighbors'][:,permutation])
    torch.testing.assert_close(prediction,other,rtol=3e-5,atol=3e-6)
    missing_loss(prediction,case).backward()
    missing_loss(other,case).backward()
    gradient_equivalence(left,right,'neighbor_permutation')
    report['neighbor_permutation_output_and_gradients'] = True

    # 4. Each target block must still encode the complete SPIN graph.
    for label,model in (('full',full),('no_memory',control)):
        left,right = copy.deepcopy(model),copy.deepcopy(model)
        left.zero_grad(set_to_none=True)
        right.zero_grad(set_to_none=True)
        complete = predict(left,case)
        missing_loss(complete,case).backward()
        blocks = []
        for ids in case['ids'].split(2):
            part = predict(right,case,ids=ids)
            missing_loss(part,case,ids).backward()
            blocks.append(part.detach())
        torch.testing.assert_close(complete,torch.cat(blocks,dim=2),rtol=3e-5,atol=3e-6)
        gradient_equivalence(left,right,label+'/target_blocks')
    report['full_graph_target_block_output_and_gradients'] = True

    # 5. Direct uses nearest-two own observations; SPIN-conditioned K/Q use values.
    for reverse,direction in ((False,'forward'),(True,'backward')):
        support = expected_direct_support(case['mask'],case['ids'],reverse)
        weights = aux['direct_weights_'+direction]
        assert torch.equal(weights>0,support), direction
        torch.testing.assert_close(weights.sum(-1),support.any(-1).float(),rtol=1e-6,atol=1e-7)
    changed = case['x'].clone()
    changed[:,:,1:] += 3.
    _,changed_aux = predict(full,case,x=changed,trace=True)
    for direction in ('forward','backward'):
        torch.testing.assert_close(aux['direct_read_'+direction][:,:,0],
            changed_aux['direct_read_'+direction][:,:,0],rtol=0,atol=0)
    assert (aux['k']-changed_aux['k']).abs().max() > 1e-7
    assert (aux['q']-changed_aux['q']).abs().max() > 1e-7
    report['direct_support_own_flow_and_value_conditioned_keys_queries'] = True

    # 6. Zero decoder initialization must still let contextual encoders train.
    fresh = factory('full')
    optimizer = torch.optim.AdamW(fresh.parameters(),lr=1e-3,weight_decay=0.)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        raw,trace = predict(fresh,case,trace=True)
        loss = missing_loss(raw,case)
        if step:
            gradients = torch.autograd.grad(loss,(trace['k'],trace['v'],trace['q']),retain_graph=True)
            for name,gradient in zip(('k','v','q'),gradients):
                assert torch.isfinite(gradient).all() and gradient.norm() > 0, name
        loss.backward()
        check_finite_gradients(fresh)
        assert fresh.decoder[-1].weight.grad.norm() > 0
        optimizer.step()
    report['two_update_contextual_kvq_trainability'] = True

    # 7. An entirely missing native T50 window has finite output and zero reads.
    for label,model in (('full',full),('no_memory',control)):
        empty_mask = torch.zeros_like(case['mask'])
        empty_x = torch.full_like(case['x'],float('nan'))
        model.zero_grad(set_to_none=True)
        raw,trace = predict(model,case,x=empty_x,mask=empty_mask,trace=True)
        assert raw.shape == case['x'].shape and torch.isfinite(raw).all()
        for name in ('direct_read_forward','direct_read_backward','memory_hf','memory_hb'):
            assert torch.count_nonzero(trace[name]) == 0, (label,name)
        raw.square().mean().backward()
        check_finite_gradients(model)
    report['all_missing_t50_finite_zero_reads'] = True

    # 8. Control never executes a matrix scan; state reload preserves predictions.
    def reject_scan(*args,**kwargs):
        raise AssertionError('no-memory model invoked the Sync scan')
    control._memory_scan = reject_scan
    assert torch.isfinite(predict(control,case)).all()
    for label,model in (('full',fresh),('no_memory',control)):
        payload = io.BytesIO()
        torch.save(model.state_dict(),payload)
        payload.seek(0)
        restored = factory(label)
        restored.load_state_dict(torch.load(payload,map_location='cpu'),strict=True)
        torch.testing.assert_close(predict(restored,case),predict(model,case),rtol=0,atol=0)
    report.update(no_memory_scan_bypassed=True,state_reload_exact=True,passed=True)
    return report


def synthetic_factory(mode):
    import numpy as np
    from .models import build_model
    variants = {'full':'spin_sync_direct','no_memory':'spin_direct_no_memory'}
    model = build_model(4, init_seed=41001, variant=variants[mode])
    model.configure_fit_statistics(dict(mu=np.full(4,1.5),C=1.5,global_scale=.7))
    model.set_execution(node_chunk=2,gradient_checkpointing=True)
    return model


if __name__ == '__main__':
    print(json.dumps(run_checks(synthetic_factory),indent=2))
