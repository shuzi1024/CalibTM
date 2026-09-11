from __future__ import annotations

import torch


def _states(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def test_three_arms_are_parameter_and_initialization_matched() -> None:
    from experiments.minimal_calibrator_v1.model import new_calibrator

    models = {
        method: new_calibrator(method, model_seed=41001)
        for method in ("static_zero", "blinear_only", "value_only")
    }
    assert {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    } == {5475}
    reference = _states(models["static_zero"])
    for model in models.values():
        assert reference.keys() == _states(model).keys()
        assert all(
            torch.equal(reference[name], _states(model)[name])
            for name in reference
        )


def test_feature_masks_are_exact_and_static_coefficients_are_constant() -> None:
    from experiments.minimal_calibrator_v1.model import new_calibrator

    torch.manual_seed(7)
    x_obs = torch.randn(2, 9, 3)
    observed = torch.rand(2, 9, 3) > 0.6
    b_linear = torch.randn(2, 9, 3)
    expected_nonzero = {
        "static_zero": set(),
        "blinear_only": {0},
        "value_only": {0, 1, 2, 13},
    }
    for method, expected in expected_nonzero.items():
        model = new_calibrator(method, model_seed=41001)
        geometry = model.extractor(x_obs, observed, b_linear)
        features = geometry["features"]
        for index in range(features.shape[-1]):
            if index not in expected:
                assert torch.count_nonzero(features[..., index]) == 0

    static = new_calibrator("static_zero", model_seed=41001)
    geometry = static.extractor(x_obs, observed, b_linear)
    hidden = static.interpolator.trunk(geometry["features"])
    for head in (
        static.interpolator.delta_r_head,
        static.interpolator.offset_head,
        static.interpolator.edge_offset_head,
    ):
        raw = head(hidden).squeeze(-1)
        assert torch.equal(raw, raw[0, 0, 0].expand_as(raw))


def test_value_only_is_bit_exact_with_previous_frozen_implementation() -> None:
    from experiments.acil_mechanism_v1.model import new_acil
    from experiments.acil_innovation_v1.preprocessing import FitFallback
    from experiments.minimal_calibrator_v1.model import new_calibrator

    previous = new_acil("value_only", model_seed=41001)
    current = new_calibrator("value_only", model_seed=41001)
    assert _states(previous).keys() == _states(current).keys()
    assert all(
        torch.equal(_states(previous)[name], _states(current)[name])
        for name in _states(previous)
    )
    values = torch.full((2, 3, 9), float("nan"))
    observed = torch.zeros_like(values, dtype=torch.bool)
    observed[..., (1, 4, 7)] = True
    values[..., 1] = 2.0
    values[..., 4] = 3.0
    values[..., 7] = 5.0
    fallback = FitFallback(mean=2.0, std=1.0)
    previous_prediction = previous(values, observed, fallback).prediction
    current_prediction = current(values, observed, fallback).prediction
    assert torch.equal(previous_prediction, current_prediction)
    previous_prediction.masked_select(~observed).sum().backward()
    current_prediction.masked_select(~observed).sum().backward()
    for (_, left), (_, right) in zip(
        previous.named_parameters(), current.named_parameters(), strict=True
    ):
        assert torch.equal(left.grad, right.grad)


def test_static_zero_head_biases_receive_gradient() -> None:
    from experiments.minimal_calibrator_v1.model import new_calibrator

    model = new_calibrator("static_zero", model_seed=41001)
    x_obs = torch.zeros(2, 9, 3)
    observed = torch.zeros_like(x_obs, dtype=torch.bool)
    observed[:, (2, 4, 6), :] = True
    x_obs[:, 2, :] = 1.0
    x_obs[:, 4, :] = 2.0
    x_obs[:, 6, :] = 4.0
    b_linear = x_obs.clone()
    geometry = model.extractor(x_obs, observed, b_linear)
    _, auxiliary = model.interpolator(
        b_linear, x_obs, observed, geometry
    )
    loss = (
        auxiliary["delta_r"].sum()
        + auxiliary["offset"].sum()
        + auxiliary["edge_offset"].sum()
    )
    loss.backward()
    for head in (
        model.interpolator.delta_r_head,
        model.interpolator.offset_head,
        model.interpolator.edge_offset_head,
    ):
        assert head.bias.grad is not None
        assert torch.count_nonzero(head.bias.grad) > 0


def test_invalid_method_or_seed_is_rejected() -> None:
    from experiments.minimal_calibrator_v1.model import new_calibrator

    for method in ("full", "bias", ""):
        try:
            new_calibrator(method, model_seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError("unregistered method was accepted")
    for seed in (-1, True, 1.5):
        try:
            new_calibrator("static_zero", model_seed=seed)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid model seed was accepted")
