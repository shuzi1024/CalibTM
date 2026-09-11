from __future__ import annotations

import torch


def _state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _synthetic_inputs():
    from experiments.acil_innovation_v1.preprocessing import FitFallback

    values = torch.full((2, 3, 9), float("nan"), dtype=torch.float32)
    observed = torch.zeros_like(values, dtype=torch.bool)
    observed[..., (2, 4, 6)] = True
    for batch in range(values.shape[0]):
        for flow in range(values.shape[1]):
            values[batch, flow, 2] = 1.25 + batch + 0.25 * flow
            values[batch, flow, 4] = 2.75 + batch + 0.50 * flow
            values[batch, flow, 6] = 5.50 + batch + 0.75 * flow
    return values, observed, FitFallback(mean=2.0, std=1.5)


def test_registered_parameter_counts_and_exact_mlp_control() -> None:
    from experiments.minimal_calibrator_v1.model import new_calibrator as incumbent
    from experiments.tiny_kan_calibrator_v1.model import (
        count_parameters,
        new_calibrator,
    )

    expected = incumbent("value_only", model_seed=41001)
    control = new_calibrator("mlp_value", model_seed=41001)
    kan_value = new_calibrator("kan_value", model_seed=41001)
    kan_static = new_calibrator("kan_static", model_seed=41001)

    assert count_parameters(expected) == 5475
    assert count_parameters(control) == 5475
    assert count_parameters(kan_value) == 5539
    assert count_parameters(kan_static) == 5539
    assert abs(count_parameters(kan_value) / count_parameters(control) - 1) < 0.05
    assert _state(expected).keys() == _state(control).keys()
    assert all(
        torch.equal(_state(expected)[name], _state(control)[name])
        for name in _state(expected)
    )
    assert _state(kan_value).keys() == _state(kan_static).keys()
    assert all(
        torch.equal(_state(kan_value)[name], _state(kan_static)[name])
        for name in _state(kan_value)
    )


def test_cubic_basis_is_finite_nonnegative_and_partitions_unity() -> None:
    from experiments.tiny_kan_calibrator_v1.model import TinyKANLayer

    layer = TinyKANLayer(2, 3)
    inputs = torch.tensor(
        [[-8.0, -1.0], [-0.5, 0.0], [0.5, 1.0], [8.0, 0.25]],
        dtype=torch.float64,
    )
    layer = layer.to(dtype=torch.float64)
    basis = layer.b_spline_basis(inputs)
    assert basis.shape == (4, 2, 8)
    assert torch.isfinite(basis).all()
    assert (basis >= 0).all()
    assert torch.allclose(
        basis.sum(dim=-1), torch.ones_like(inputs), atol=1e-12, rtol=1e-12
    )


def test_cpu_bfloat16_autocast_keeps_kan_forward_and_backward_finite() -> None:
    from experiments.tiny_kan_calibrator_v1.model import TinyKANLogitNetwork

    torch.manual_seed(7)
    network = TinyKANLogitNetwork()
    features = torch.randn(4, 7, 5, 16).to(torch.bfloat16).requires_grad_()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        basis = network.first.b_spline_basis(features)
        logits = network(features)
        loss = logits.square().mean()
    # Explicit FP32 is the registered low-precision arithmetic path, rather
    # than an incidental outcome of today's CPU autocast allow-list.
    assert basis.dtype is torch.float32
    assert logits.dtype is torch.float32
    assert torch.isfinite(basis).all()
    assert torch.isfinite(logits).all()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    for parameter in network.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_architecture_record_exactly_describes_registered_kan() -> None:
    from experiments.tiny_kan_calibrator_v1.model import (
        KAN_GRID_SIZE,
        KAN_SPLINE_ORDER,
        count_parameters,
        new_calibrator,
    )

    model = new_calibrator("kan_value", model_seed=41001)
    record = model.architecture_record()
    assert record == {
        "arithmetic": "fp32-for-fp16-or-bf16",
        "base_branch": "silu",
        "beta_e": 0.10,
        "beta_o": 0.10,
        "beta_r": 0.25,
        "feature_dim": 16,
        "feature_set": "value_only",
        "grid_range": [-1.0, 1.0],
        "grid_size": 5,
        "hidden_dim": 32,
        "knot_vector": "uniform-extended-by-spline-order",
        "layer_widths": [16, 32, 3],
        "logit_dim": 3,
        "output_layer_initialization": "base-and-spline-normal-1e-4-bias-zero",
        "spline_basis_count_per_edge": 8,
        "spline_input_map": "tanh(x/2)",
        "spline_order": 3,
        "trunk": "layernorm-kan-kan",
    }
    assert record["grid_size"] == KAN_GRID_SIZE
    assert record["spline_order"] == KAN_SPLINE_ORDER
    assert model.interpolator.trunk.first.num_basis == 8
    assert model.interpolator.trunk.second.num_basis == 8
    assert tuple(model.interpolator.trunk.first.base_weight.shape) == (32, 16)
    assert tuple(model.interpolator.trunk.second.base_weight.shape) == (3, 32)
    assert count_parameters(model) == 5539
    second = model.interpolator.trunk.second
    assert torch.count_nonzero(second.bias) == 0
    assert second.base_weight.std() < 2.0e-4
    assert second.spline_weight.std() < 2.0e-4


def test_kan_features_are_exactly_value_or_static_and_payload_is_hidden() -> None:
    from experiments.tiny_kan_calibrator_v1.model import new_calibrator

    values, observed, fallback = _synthetic_inputs()
    value_model = new_calibrator("kan_value", model_seed=41001)
    static_model = new_calibrator("kan_static", model_seed=41001)
    value_features = value_model(values, observed, fallback).features
    static_features = static_model(values, observed, fallback).features
    active = {0, 1, 2, 13}
    for index in range(value_features.shape[-1]):
        if index not in active:
            assert torch.count_nonzero(value_features[..., index]) == 0
    assert torch.count_nonzero(static_features) == 0

    alternate = values.clone()
    alternate[~observed] = torch.linspace(
        -1.0e6,
        1.0e6,
        int((~observed).sum()),
        dtype=alternate.dtype,
    )
    for model in (value_model, static_model):
        first = model(values, observed, fallback)
        second = model(alternate, observed, fallback)
        assert torch.equal(first.linear_fill, second.linear_fill)
        assert torch.equal(first.features, second.features)
        assert torch.equal(first.prediction, second.prediction)


def test_k2_leave_one_out_payload_is_algebraically_hidden() -> None:
    from experiments.tiny_kan_calibrator_v1.model import new_calibrator

    values, observed, fallback = _synthetic_inputs()
    submask = observed.clone()
    submask[..., 4] = False
    alternate = values.clone()
    alternate[..., 4] = torch.linspace(
        -1.0e7,
        1.0e7,
        values.shape[0] * values.shape[1],
        dtype=values.dtype,
    ).reshape(values.shape[0], values.shape[1])
    for method in ("mlp_value", "kan_value", "kan_static"):
        model = new_calibrator(method, model_seed=41001)
        original = model(values, submask, fallback)
        perturbed = model(alternate, submask, fallback)
        assert torch.equal(original.linear_fill, perturbed.linear_fill)
        assert torch.equal(original.features, perturbed.features)
        assert torch.equal(original.prediction, perturbed.prediction)


def test_hard_copy_nonnegativity_and_registered_bounds() -> None:
    from experiments.acil_innovation_v1.preprocessing import prepare_observations
    from experiments.tiny_kan_calibrator_v1.model import new_calibrator

    values, observed, fallback = _synthetic_inputs()
    for method in ("mlp_value", "kan_value", "kan_static"):
        model = new_calibrator(method, model_seed=41001)
        result = model(values, observed, fallback)
        assert torch.equal(
            result.prediction.masked_select(observed),
            values.masked_select(observed),
        )
        assert (result.prediction.masked_select(~observed) >= 0).all()

        prepared = prepare_observations(values, observed, fallback)
        b_linear = prepared.normalized_fill.permute(0, 2, 1).contiguous()
        normalized_observed = torch.where(
            observed,
            (values - prepared.statistics.mean) / prepared.statistics.std,
            torch.zeros_like(values),
        )
        x_obs = normalized_observed.permute(0, 2, 1).contiguous()
        mask_btf = observed.permute(0, 2, 1).contiguous()
        geometry = model.extractor(x_obs, mask_btf, b_linear)
        normalized, auxiliary = model.interpolator(
            b_linear, x_obs, mask_btf, geometry
        )

        tolerance = 2e-6
        assert auxiliary["delta_r"].abs().max() <= 0.25 + tolerance
        assert auxiliary["r_hat"].min() >= 0.0
        assert auxiliary["r_hat"].max() <= 1.0
        scale = geometry["anchor_delta"] + geometry["local_volatility"] + 1e-6
        assert torch.all(auxiliary["offset"].abs() <= 0.10 * scale + tolerance)
        assert torch.all(
            auxiliary["edge_offset"].abs()
            <= 0.10 * geometry["local_volatility"] + tolerance
        )

        internal = geometry["internal_gap"]
        deviation_bound = (
            0.25 * geometry["anchor_delta"] + 0.10 * scale + tolerance
        )
        assert torch.all(
            (normalized - b_linear).abs().masked_select(internal)
            <= deviation_bound.masked_select(internal)
        )


def test_kan_gradients_reach_both_spline_layers_and_all_three_logits() -> None:
    from experiments.tiny_kan_calibrator_v1.model import new_calibrator

    values, observed, fallback = _synthetic_inputs()
    model = new_calibrator("kan_value", model_seed=41001)
    result = model(values, observed, fallback)
    loss = result.prediction.masked_select(~observed).square().mean()
    loss.backward()

    trunk = model.interpolator.trunk
    for layer in (trunk.first, trunk.second):
        for parameter in (
            layer.base_weight,
            layer.spline_weight,
            layer.bias,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0
    # Each output row is one of the three constrained correction logits.
    assert torch.all(
        trunk.second.base_weight.grad.abs().sum(dim=1) > 0
    )
    assert torch.all(
        trunk.second.spline_weight.grad.abs().sum(dim=(1, 2)) > 0
    )


def test_seed_determinism_rng_isolation_and_input_validation() -> None:
    from experiments.tiny_kan_calibrator_v1.model import new_calibrator

    torch.manual_seed(9123)
    before = torch.random.get_rng_state().clone()
    first = new_calibrator("kan_value", model_seed=41001)
    after = torch.random.get_rng_state().clone()
    second = new_calibrator("kan_value", model_seed=41001)
    different = new_calibrator("kan_value", model_seed=41002)
    assert torch.equal(before, after)
    assert _state(first).keys() == _state(second).keys()
    assert all(
        torch.equal(_state(first)[name], _state(second)[name])
        for name in _state(first)
    )
    assert any(
        not torch.equal(_state(first)[name], _state(different)[name])
        for name in _state(first)
        if _state(first)[name].is_floating_point()
    )
    values, observed, fallback = _synthetic_inputs()
    assert torch.equal(
        first(values, observed, fallback).prediction,
        second(values, observed, fallback).prediction,
    )

    for method in ("value_only", "kan", ""):
        try:
            new_calibrator(method, model_seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError("unregistered method was accepted")
    for seed in (-1, True, 1.5):
        try:
            new_calibrator("kan_value", model_seed=seed)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid seed was accepted")
