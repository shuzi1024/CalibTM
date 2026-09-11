from __future__ import annotations

import copy

import torch

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.preprocessing import FitFallback


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


def test_full_and_u0_are_exact_matched_capacity_twins() -> None:
    from experiments.sc_acil_v1.model import new_sc_acil_model

    base = ACILBase().eval()
    full = new_sc_acil_model(
        "sc_acil", acil=copy.deepcopy(base), model_seed=41004
    )
    u0 = new_sc_acil_model(
        "sc_acil_u0", acil=copy.deepcopy(base), model_seed=41004
    )

    assert tuple((name, tuple(value.shape)) for name, value in full.named_parameters()) == tuple(
        (name, tuple(value.shape)) for name, value in u0.named_parameters()
    )
    assert sum(value.numel() for value in full.parameters()) == sum(
        value.numel() for value in u0.parameters()
    )
    for name, value in full.state_dict().items():
        assert torch.equal(value, u0.state_dict()[name]), name


def test_u0_zeros_only_the_scalar_loo_innovation_channel() -> None:
    from experiments.sc_acil_v1.model import new_sc_acil_model

    values, observed, fallback = _fixture()
    full = new_sc_acil_model("sc_acil", acil=ACILBase(), model_seed=41004)
    u0 = new_sc_acil_model("sc_acil_u0", acil=ACILBase(), model_seed=41004)
    full_features = full.prepare_features(values, observed, fallback)
    u0_features = u0.prepare_features(values, observed, fallback)
    full_tokens = full.token_inputs(full_features)
    u0_tokens = u0.token_inputs(u0_features)

    assert torch.count_nonzero(full_tokens[..., -2]).item() > 0
    assert torch.count_nonzero(u0_tokens[..., -2]).item() == 0
    assert torch.equal(full_tokens[..., :-2], u0_tokens[..., :-2])
    assert torch.equal(full_tokens[..., -1], u0_tokens[..., -1])


def test_sc_acil_is_observation_only_hard_projected_and_flow_equivariant() -> None:
    from experiments.sc_acil_v1.model import new_sc_acil_model

    values, observed, fallback = _fixture()
    model = new_sc_acil_model("sc_acil", acil=ACILBase(), model_seed=41004).eval()
    changed = values.clone()
    changed[~observed] = torch.linspace(-1e12, 1e12, int((~observed).sum()))
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        prediction = model(values, observed, fallback)
        hidden_changed = model(changed, observed, fallback)
        permuted = model(values[:, permutation], observed[:, permutation], fallback)

    assert torch.equal(prediction, hidden_changed)
    assert torch.equal(prediction.masked_select(observed), values.masked_select(observed))
    assert torch.allclose(permuted, prediction[:, permutation], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(prediction).all()
    assert torch.all(prediction >= 0)

