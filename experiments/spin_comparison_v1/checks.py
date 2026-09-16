"""CPU verification against the frozen upstream's actual model/layer source.

The oracle executes original SPIN source with tiny CPU gather/scatter shims in
place of compiled PyG/torch_scatter dispatch. The production implementation uses
chunked dense contractions; outputs AND all parameter gradients are compared.
"""

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor

from . import primitives
from .models import build_model, make_optimizer_scheduler, train_step


ROOT = Path(__file__).parent
SPIN = ROOT / "upstream/spin-7349ba31da7306e7e96c13668a3f1f0a4df90902"
TSL = ROOT / "upstream/tsl-0.1.1"


def _execute(path, replacements, scope):
    text = path.read_text()
    for before, after in replacements.items():
        if before not in text:
            raise AssertionError(f"Pinned source anchor disappeared: {before}")
        text = text.replace(before, after)
    exec(compile(text, str(path), "exec"), scope)
    return scope


def _scatter(src, index, dim, dim_size, reduce="sum"):
    if index.ndim == 1:
        shape = [1] * src.ndim
        shape[dim] = -1
        index = index.reshape(shape).expand_as(src)
    output_shape = list(src.shape)
    output_shape[dim] = dim_size
    if reduce == "max":
        out = src.new_full(output_shape, -torch.inf)
        return out.scatter_reduce(dim, index, src, reduce="amax", include_self=True)
    return src.new_zeros(output_shape).scatter_add(dim, index, src)


def _broadcast(index, weights, dim):
    shape = [1] * weights.ndim
    shape[dim] = -1
    return index.reshape(shape).expand_as(weights)


def _sparse_softmax(src, index, num_nodes, dim):
    expanded = _broadcast(index, src, dim)
    maximum = _scatter(src, expanded, dim, num_nodes, "max").index_select(dim, index)
    out = (src - maximum).exp()
    denominator = _scatter(out, expanded, dim, num_nodes).index_select(dim, index)
    return out / (denominator + 5e-8)


class _MessagePassing(nn.Module):
    def __init__(self, node_dim=-2, aggr="add", **kwargs):
        super().__init__()
        assert aggr == "add"
        self.node_dim = node_dim

    def propagate(self, edge_index, size, **kwargs):
        source, target = edge_index
        args = {}
        for name in inspect.signature(self.message).parameters:
            if name == "index":
                args[name] = target
            elif name == "size_i":
                args[name] = size[1]
            elif name.endswith("_j") or name.endswith("_i"):
                raw = kwargs[name[:-2]]
                is_source = name.endswith("_j")
                if isinstance(raw, tuple):
                    raw = raw[0 if is_source else 1]
                args[name] = None if raw is None else raw.index_select(self.node_dim,
                    source if is_source else target)
            else:
                args[name] = kwargs[name]
        messages = self.message(**args)
        return _scatter(messages, target, self.node_dim, size[1])


def official_model(flows):
    """Load official source, including actual TSL MLP/Dense/Positional/Norm."""
    from types import SimpleNamespace
    utils = SimpleNamespace(get_layer_activation=lambda name: {
        "relu": nn.ReLU, "prelu": nn.PReLU, "linear": nn.Identity}[name],
        maybe_cat_exog=lambda x, u: x if u is None else torch.cat((x, u), -1))
    dense_scope = _execute(TSL / "tsl/nn/base/dense.py", {
        "from tsl.nn.utils import utils": ""}, {"utils": utils})
    mlp_scope = _execute(TSL / "tsl/nn/blocks/encoders/mlp.py", {
        "from ...base.dense import Dense": "",
        "from ...utils import utils": ""}, {"Dense": dense_scope["Dense"], "utils": utils})
    positional_scope = _execute(TSL / "tsl/nn/layers/positional_encoding.py", {}, {})
    norm_scope = _execute(TSL / "tsl/nn/layers/norm/layer_norm.py", {
        "from torch_geometric.nn import inits": ""}, {
        "inits": SimpleNamespace(ones=nn.init.ones_, zeros=nn.init.zeros_)})
    inits = SimpleNamespace(uniform=lambda size, value: nn.init.uniform_(value,
        -1/math.sqrt(size), 1/math.sqrt(size)))
    embedding_scope = _execute(TSL / "tsl/nn/base/embedding.py", {
        "from torch_geometric.nn import inits": "",
        "from torch_geometric.typing import OptTensor": "OptTensor = Optional[Tensor]"}, {"inits": inits})
    common = {"MLP": mlp_scope["MLP"], "StaticGraphEmbedding": embedding_scope["StaticGraphEmbedding"],
              "MessagePassing": _MessagePassing, "Linear": primitives.PyGLinear,
              "sparse_softmax": _sparse_softmax, "scatter": _scatter, "broadcast": _broadcast}
    att_scope = _execute(SPIN / "spin/layers/additive_attention.py", {
        "from torch_geometric.nn.conv import MessagePassing": "",
        "from torch_geometric.nn.dense.linear import Linear": "",
        "from torch_geometric.typing import Adj, OptTensor, PairTensor":
        "Adj = Tensor\nOptTensor = Optional[Tensor]\nPairTensor = Tuple[Tensor, Tensor]",
        "from torch_scatter import scatter": "",
        "from torch_scatter.utils import broadcast": "",
        "from tsl.nn.blocks.encoders import MLP": "",
        "from tsl.nn.functional import sparse_softmax": ""}, dict(common))
    graph_scope = _execute(SPIN / "spin/layers/temporal_graph_additive_attention.py", {
        "from torch_geometric.nn.conv import MessagePassing": "",
        "from torch_geometric.nn.dense.linear import Linear": "",
        "from torch_geometric.typing import Adj, OptTensor, OptPairTensor":
        "Adj = Tensor\nOptTensor = Optional[Tensor]\nOptPairTensor = Optional[Tuple[Tensor, Tensor]]",
        "from tsl.nn.layers.norm import LayerNorm": "",
        "from .additive_attention import TemporalAdditiveAttention": ""}, {
        **common, "LayerNorm": norm_scope["LayerNorm"],
        "TemporalAdditiveAttention": att_scope["TemporalAdditiveAttention"]})
    pos_scope = _execute(SPIN / "spin/layers/postional_encoding.py", {
        "from tsl.nn.base import StaticGraphEmbedding": "",
        "from tsl.nn.blocks.encoders import MLP": "",
        "from tsl.nn.layers import PositionalEncoding": ""}, {
        **common, "PositionalEncoding": positional_scope["PositionalEncoding"]})
    model_scope = _execute(SPIN / "spin/models/spin.py", {
        "from torch_geometric.typing import OptTensor": "OptTensor = Optional[Tensor]",
        "from tsl.nn.base import StaticGraphEmbedding": "",
        "from tsl.nn.blocks.encoders import MLP": "",
        "from ..layers import PositionalEncoder, TemporalGraphAdditiveAttention": ""}, {
        **common, "PositionalEncoder": pos_scope["PositionalEncoder"],
        "TemporalGraphAdditiveAttention": graph_scope["TemporalGraphAdditiveAttention"]})
    return model_scope["SPINModel"](input_size=1, hidden_size=32, n_nodes=flows,
        u_size=1, output_size=1, temporal_self_attention=True, reweight="softmax",
        n_layers=4, eta=3, message_layers=1)


def run_checks():
    torch.set_num_threads(2)
    torch.manual_seed(77)
    model = build_model(4)
    model.configure_fit_statistics(dict(mu=np.ones(4), C=1., global_scale=2.))
    oracle = official_model(4)
    oracle.load_state_dict(model.core.state_dict(), strict=True)
    x = torch.randn(1, 50, 4, 1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, ::5] = True
    x = torch.where(mask, x, 0)
    u = torch.arange(50)[None, :, None].float() / 49.
    incoming = torch.tensor([[1, 2], [0, 3], [0, 3], [1, 2]])
    target = torch.arange(4).repeat_interleave(2)
    edges = torch.stack((incoming.flatten(), target))
    model.set_execution(node_chunk=2, gradient_checkpointing=False)
    actual = model.core(x, u, mask, incoming)
    expected = oracle(x, u, mask, edges)
    outputs_a, outputs_b = [actual[0], *actual[1]], [expected[0], *expected[1]]
    output_diff = max(float((a-b).abs().max()) for a, b in zip(outputs_a, outputs_b))
    for a, b in zip(outputs_a, outputs_b):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    sum(value.square().mean() for value in outputs_a).backward()
    sum(value.square().mean() for value in outputs_b).backward()
    gradient_diff = 0.
    for (name, a), (other, b) in zip(model.core.named_parameters(), oracle.named_parameters()):
        assert name == other
        assert a.grad is not None and b.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=5e-4, msg=name)
        gradient_diff = max(gradient_diff, float((a.grad-b.grad).abs().max()))
    reference_grad = {name: value.grad.clone() for name, value in model.core.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.set_execution(node_chunk=1, gradient_checkpointing=True)
    ckpt = model.core(x, u, mask, incoming)
    sum(value.square().mean() for value in [ckpt[0], *ckpt[1]]).backward()
    for name, value in model.core.named_parameters():
        torch.testing.assert_close(value.grad, reference_grad[name], atol=3e-6, rtol=5e-4)
    # Adapter hidden values, gradients and common output projection.
    neighbors = torch.cat((torch.arange(4)[:, None], incoming), 1)
    observed = x.squeeze(-1) * 2 + 1
    boolean_mask = mask.squeeze(-1)
    model.eval()
    ids = torch.arange(4)
    a = model.predict_block(observed, boolean_mask, {}, neighbors, ids)
    tampered = torch.where(boolean_mask, observed, torch.full_like(observed, float("nan")))
    b = model.predict_block(tampered, boolean_mask, {}, neighbors, ids)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    projection = model(observed, boolean_mask, {}, neighbors)
    assert torch.equal(projection[boolean_mask], observed[boolean_mask])
    assert torch.isfinite(projection).all() and (projection[~boolean_mask] >= 0).all()
    empty = model.predict_block(observed, torch.zeros_like(boolean_mask), {}, neighbors, ids)
    assert torch.isfinite(empty).all()
    optimizer, scheduler = make_optimizer_scheduler(model)
    result = train_step(model, observed.numpy(), boolean_mask.numpy(), [0], {}, neighbors,
        1., optimizer, dataset="synthetic", epoch=0, physical_batch=1, target_block=4)
    assert result["supervised_readouts"] == 4 and np.isfinite(result["loss"])
    assert abs(optimizer.param_groups[0]["lr"] - 8e-5) < 1e-12
    scheduler.step()
    # Effective-batch accumulation is independent of physical microbatching.
    import copy
    micro, full = copy.deepcopy(model), copy.deepcopy(model)
    micro.set_execution(node_chunk=1, gradient_checkpointing=True)
    full.set_execution(node_chunk=4, gradient_checkpointing=True)
    values2 = np.concatenate((observed.numpy(), observed.numpy() * 1.2), axis=0)
    masks2 = np.concatenate((boolean_mask.numpy(), boolean_mask.numpy()), axis=0)
    opt_micro, _ = make_optimizer_scheduler(micro)
    opt_full, _ = make_optimizer_scheduler(full)
    result_micro = train_step(micro, values2, masks2, [0, 50], {}, neighbors, 1., opt_micro,
        dataset="synthetic", epoch=0, physical_batch=1, target_block=4)
    result_full = train_step(full, values2, masks2, [0, 50], {}, neighbors, 1., opt_full,
        dataset="synthetic", epoch=0, physical_batch=2, target_block=4)
    assert abs(result_micro["loss"]-result_full["loss"]) < 2e-6
    for (name, a), (_, b) in zip(micro.named_parameters(), full.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=5e-4, msg=name)
        torch.testing.assert_close(a, b, atol=3e-6, rtol=5e-4, msg=name)
    # Stable identity and complete state reload.
    restored = build_model(4, 41002)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    model.eval()
    torch.testing.assert_close(restored(observed, boolean_mask, {}, neighbors),
                               model(observed, boolean_mask, {}, neighbors))
    counts = {str(flows): sum(p.numel() for p in build_model(flows).parameters())
              for flows in (144, 462)}
    return {"passed": True, "official_source_commit": "7349ba31da7306e7e96c13668a3f1f0a4df90902",
            "oracle": "executed_official_SPIN_and_TSL_source_with_CPU_PyG_dispatch_shim",
            "max_output_abs_diff": output_diff, "max_parameter_gradient_abs_diff": gradient_diff,
            "layer_readouts_compared": 4, "parameter_counts": counts,
            "checks": ["upstream_outputs", "all_parameter_gradients", "checkpoint_chunk_equivalence",
                       "hidden_nan_invariance", "empty_mask_finite", "observed_copy_nonnegative",
                       "four_readout_train_step", "physical_microbatch_equivalence",
                       "scheduler_initial_lr", "state_reload"]}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
