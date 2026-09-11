from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.models import (
    MODEL_METHODS,
    NonCausalGPT2SetBlock,
    QueryResidualModel,
    anchor_envelope,
)
from experiments.acil_innovation_v1.preprocessing import FitFallback


class _RepeatQKV(nn.Module):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.scales = nn.Parameter(torch.tensor([1.0, 0.75, 1.25]) + offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(tuple(x * scale for scale in self.scales), dim=-1)


class _RetainedBlock(nn.Module):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(768)
        self.c_attn = _RepeatQKV(offset)
        self.c_proj = nn.Identity()
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)
        self.ln_2 = nn.LayerNorm(768)
        self.mlp = nn.Identity()

    def forward(self, *args, **kwargs):  # pragma: no cover - forbidden path
        raise AssertionError("source GPT block forward must never be called")


def _blocks(offset: float = 0.0) -> tuple[nn.Module, ...]:
    return tuple(_RetainedBlock(offset + index * 1e-4) for index in range(6))


def _fixture() -> tuple[torch.Tensor, torch.Tensor, FitFallback]:
    observed = torch.zeros(1, 3, 7, dtype=torch.bool)
    observed[0, 0, [0, 3, 6]] = True
    observed[0, 1, [1, 2, 5]] = True
    observed[0, 2, [2, 4, 5]] = True
    values = torch.full((1, 3, 7), float("nan"))
    values[0, 0, [0, 3, 6]] = torch.tensor([2.0, 5.0, 11.0])
    values[0, 1, [1, 2, 5]] = torch.tensor([3.0, 7.0, 8.0])
    values[0, 2, [2, 4, 5]] = torch.tensor([4.0, 6.0, 9.0])
    return values, observed, FitFallback(mean=5.0, std=2.0)


def _parameter_schema(module: nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(value.shape)) for name, value in module.named_parameters())


def test_anchor_envelope_is_zero_at_anchors_and_bounded_on_internal_and_edges() -> None:
    torch.manual_seed(110)
    values, observed, fallback = _fixture()
    result = ACILBase().eval()(values, observed, fallback)

    envelope = anchor_envelope(result.geometry, observed)

    assert envelope.shape == values.shape
    assert torch.all((envelope >= 0) & (envelope <= 1))
    assert torch.equal(envelope.masked_select(observed), torch.zeros(int(observed.sum())))
    # Flow 2 has left edge [0,1], internal point 3, and right edge point 6.
    assert envelope[0, 2, 0].item() == pytest.approx(1.0)
    assert envelope[0, 2, 1].item() == pytest.approx(0.5)
    assert envelope[0, 2, 3].item() == pytest.approx(1.0)
    assert envelope[0, 2, 6].item() == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["local", "deepsets"])
def test_observation_only_models_are_flow_permutation_equivariant(method: str) -> None:
    torch.manual_seed(111)
    values, observed, fallback = _fixture()
    model = QueryResidualModel(method, ACILBase()).eval()
    permutation = torch.tensor([2, 0, 1])

    with torch.no_grad():
        baseline = model(values, observed, fallback)
        permuted = model(values[:, permutation], observed[:, permutation], fallback)

    assert torch.allclose(permuted, baseline[:, permutation], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("method", ["local", "deepsets", "gpt2_set"])
def test_models_are_nonnegative_hard_projected_and_hidden_payload_invariant(method: str) -> None:
    torch.manual_seed(112)
    values, observed, fallback = _fixture()
    kwargs = {"retained_blocks": _blocks()} if method == "gpt2_set" else {}
    model = QueryResidualModel(method, ACILBase(), **kwargs).eval()
    changed = values.clone()
    changed[~observed] = torch.linspace(-1e12, 1e12, int((~observed).sum()))

    with torch.no_grad():
        first = model(values, observed, fallback)
        second = model(changed, observed, fallback)

    assert first.shape == values.shape
    assert torch.isfinite(first).all()
    assert torch.all(first >= 0)
    assert torch.equal(first, second)
    assert torch.equal(
        first.contiguous().view(torch.int32).masked_select(observed),
        values.contiguous().view(torch.int32).masked_select(observed),
    )


def test_full_u0_zeros_only_the_innovation_channel() -> None:
    torch.manual_seed(113)
    values, observed, fallback = _fixture()
    common_acil = ACILBase().eval()
    full = QueryResidualModel("gpt2_set", copy.deepcopy(common_acil), retained_blocks=_blocks())
    u0 = QueryResidualModel("full_u0", copy.deepcopy(common_acil), retained_blocks=_blocks())

    full_features = full.prepare_features(values, observed, fallback)
    u0_features = u0.prepare_features(values, observed, fallback)
    full_tokens = full.token_inputs(full_features)
    u0_tokens = u0.token_inputs(u0_features)

    assert torch.count_nonzero(full_tokens[..., -2]).item() > 0
    assert torch.count_nonzero(u0_tokens[..., -2]).item() == 0
    assert torch.equal(full_tokens[..., :-2], u0_tokens[..., :-2])
    assert torch.equal(full_tokens[..., -1], u0_tokens[..., -1])


def test_explicit_innovation_override_reproduces_normal_forward() -> None:
    torch.manual_seed(115)
    values, observed, fallback = _fixture()
    model = QueryResidualModel("deepsets", ACILBase()).eval()
    with torch.no_grad():
        features = model.prepare_features(values, observed, fallback)
        ordinary = model(values, observed, fallback)
        explicit = model.forward_with_innovation(
            values,
            observed,
            fallback,
            innovation_normalized=features.loo.innovation_normalized,
        )
    assert torch.equal(explicit, ordinary)
    with pytest.raises(ValueError, match="innovation"):
        model.forward_with_innovation(
            values,
            observed,
            fallback,
            innovation_normalized=torch.zeros(1, 3),
        )


def test_noncausal_gpt_block_is_set_equivariant_and_has_no_forbidden_components() -> None:
    block = NonCausalGPT2SetBlock(_RetainedBlock()).eval()
    x = torch.randn(2, 4, 768)
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    permutation = torch.tensor([2, 0, 3, 1])

    output = block(x, valid)
    permuted = block(x[:, permutation], valid[:, permutation])

    assert output.shape == x.shape
    assert torch.all(output[~valid] == 0)
    assert torch.allclose(permuted, output[:, permutation], rtol=1e-5, atol=2e-5)
    forbidden = ("wte", "wpe", "tokenizer", "prompt", "causal", "position")
    assert not any(
        token in name.lower()
        for name, _ in block.named_modules()
        for token in forbidden
    )


def test_pretrained_and_scratch_set_models_are_structurally_identical() -> None:
    assert MODEL_METHODS == (
        "local",
        "deepsets",
        "full_u0",
        "gpt2_set",
        "gpt2_scratch",
    )
    torch.manual_seed(114)
    pretrained = QueryResidualModel(
        "gpt2_set", ACILBase(), retained_blocks=_blocks(offset=0.0)
    )
    torch.manual_seed(114)
    scratch = QueryResidualModel(
        "gpt2_scratch", ACILBase(), retained_blocks=_blocks(offset=0.2)
    )

    assert _parameter_schema(pretrained) == _parameter_schema(scratch)
    assert len(pretrained.set_blocks) == len(scratch.set_blocks) == 6
    assert not torch.equal(
        pretrained.set_blocks[0].c_attn.scales,
        scratch.set_blocks[0].c_attn.scales,
    )
    values, observed, fallback = _fixture()
    with torch.no_grad():
        output = scratch.eval()(values, observed, fallback)
    assert output.shape == values.shape
