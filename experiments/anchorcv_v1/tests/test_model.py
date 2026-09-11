from __future__ import annotations

import inspect

import pytest
import torch

from experiments.anchorcv_v1.model import (
    PriorFreeMaskNativeExpert,
    hard_project_observations,
)


def _tiny_model(*, dropout: float = 0.0) -> PriorFreeMaskNativeExpert:
    return PriorFreeMaskNativeExpert(
        num_flows=5,
        time_steps=7,
        temporal_hidden=4,
        d_model=16,
        num_heads=4,
        num_flow_layers=2,
        dim_feedforward=32,
        dropout=dropout,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.randn(2, 5, 7)
    mask = torch.zeros(2, 5, 7, dtype=torch.bool)
    mask[..., [0, 3, 6]] = True
    observed_values = torch.where(mask, values, torch.zeros_like(values))
    time = torch.linspace(-1.0, 1.0, 7)
    return observed_values, mask, time


def test_forward_shape_and_finite_values() -> None:
    model = _tiny_model().eval()
    observed_values, mask, time = _inputs()

    output = model(observed_values, mask, time)

    assert output.shape == observed_values.shape
    assert torch.isfinite(output).all()


def test_hard_projection_preserves_every_observation() -> None:
    observed_values, mask, _ = _inputs()
    prediction = torch.randn_like(observed_values)

    projected = hard_project_observations(prediction, observed_values, mask)

    torch.testing.assert_close(projected[mask], observed_values[mask])
    torch.testing.assert_close(projected[~mask], prediction[~mask])
    assert projected.data_ptr() != prediction.data_ptr()


def test_unobserved_value_perturbations_cannot_reach_model() -> None:
    model = _tiny_model().eval()
    observed_values, mask, time = _inputs()
    perturbed = observed_values.clone()
    perturbed[~mask] = torch.randn_like(perturbed[~mask]) * 1_000_000.0

    clean_output = model(observed_values, mask, time)
    perturbed_output = model(perturbed, mask, time)

    torch.testing.assert_close(clean_output, perturbed_output, rtol=0.0, atol=0.0)


def test_public_forward_contract_has_no_prior_or_interpolation_channel() -> None:
    parameters = list(inspect.signature(PriorFreeMaskNativeExpert.forward).parameters)

    assert parameters == [
        "self",
        "observed_values",
        "observed_mask",
        "normalized_time",
    ]
    forbidden = ("interp", "interpolation", "prior", "acil", "coarse", "residual")
    all_names = " ".join(
        name.lower()
        for name, _ in _tiny_model().named_parameters()
    )
    assert not any(token in all_names for token in forbidden)


def test_backward_produces_only_finite_gradients() -> None:
    model = _tiny_model().train()
    observed_values, mask, time = _inputs()

    loss = model(observed_values, mask, time).square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_fixed_seed_recreates_identical_model_and_output() -> None:
    observed_values, mask, time = _inputs()
    torch.manual_seed(271828)
    first = _tiny_model(dropout=0.0).eval()
    first_output = first(observed_values, mask, time)

    torch.manual_seed(271828)
    second = _tiny_model(dropout=0.0).eval()
    second_output = second(observed_values, mask, time)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(
            first_parameter, second_parameter, rtol=0.0, atol=0.0
        )
    torch.testing.assert_close(first_output, second_output, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("values_shape", "mask_shape", "time_shape", "message"),
    [
        ((2, 4, 7), (2, 4, 7), (7,), "num_flows"),
        ((2, 5, 6), (2, 5, 6), (6,), "time_steps"),
        ((2, 5, 7), (2, 5, 6), (7,), "observed_mask"),
        ((2, 5, 7), (2, 5, 7), (6,), "normalized_time"),
    ],
)
def test_invalid_shapes_fail_closed(
    values_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    time_shape: tuple[int, ...],
    message: str,
) -> None:
    model = _tiny_model()
    values = torch.zeros(values_shape)
    mask = torch.zeros(mask_shape, dtype=torch.bool)
    time = torch.zeros(time_shape)

    with pytest.raises(ValueError, match=message):
        model(values, mask, time)


def test_mask_must_be_boolean() -> None:
    model = _tiny_model()
    observed_values, mask, time = _inputs()

    with pytest.raises(TypeError, match="boolean"):
        model(observed_values, mask.float(), time)
