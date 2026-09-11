from __future__ import annotations

import pytest
import torch

from experiments.acil_innovation_v1.preprocessing import (
    FitFallback,
    observation_only_linear_fill,
    prepare_observations,
)


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32)


def test_linear_fill_uses_only_observed_values_and_extends_edges() -> None:
    observed = torch.tensor([[[False, True, False, False, True, False]]])
    values = torch.tensor([[[9e8, 2.0, -7e8, 8e8, 8.0, -6e8]]])
    fallback = FitFallback(mean=5.0, std=3.0)

    filled = observation_only_linear_fill(values, observed, fallback)

    assert torch.equal(filled, torch.tensor([[[2.0, 2.0, 4.0, 6.0, 8.0, 8.0]]]))
    perturbed = values.clone()
    perturbed[~observed] = torch.tensor([float("nan"), -1e20, 1e20, 12345.0])
    assert torch.equal(
        filled,
        observation_only_linear_fill(perturbed, observed, fallback),
    )


def test_prepare_observations_uses_fit_fallback_for_empty_or_degenerate_flows() -> None:
    observed = torch.tensor(
        [
            [
                [False, False, False, False],
                [True, False, True, False],
                [True, False, True, False],
            ]
        ]
    )
    values = torch.tensor(
        [
            [
                [float("nan"), float("nan"), float("nan"), float("nan")],
                [3.0, -999.0, 3.0, 999.0],
                [2.0, -999.0, 6.0, 999.0],
            ]
        ]
    )
    fallback = FitFallback(mean=7.0, std=2.0)

    prepared = prepare_observations(values, observed, fallback)

    assert torch.equal(prepared.linear_fill[0, 0], torch.full((4,), 7.0))
    assert prepared.statistics.mean[0, 0, 0].item() == 7.0
    assert prepared.statistics.std[0, 0, 0].item() == 2.0
    assert prepared.statistics.mean[0, 1, 0].item() == 3.0
    assert prepared.statistics.std[0, 1, 0].item() == 2.0
    assert prepared.statistics.mean[0, 2, 0].item() == 4.0
    assert prepared.statistics.std[0, 2, 0].item() == 2.0
    assert torch.isfinite(prepared.normalized_fill).all()


def test_observed_projection_is_bitwise_and_missing_payload_is_not_retained() -> None:
    observed = torch.tensor([[[True, False, True, False, True]]])
    values = torch.tensor([[[1.25, float("nan"), 4.5, float("nan"), 9.75]]])
    prepared = prepare_observations(values, observed, FitFallback(mean=3.0, std=2.0))

    assert torch.equal(
        _bits(prepared.linear_fill).masked_select(observed),
        _bits(values).masked_select(observed),
    )
    assert torch.equal(prepared.observed, observed)
    assert prepared.observed.data_ptr() != observed.data_ptr()
    assert not torch.isnan(prepared.linear_fill).any()


@pytest.mark.parametrize(
    ("values", "observed", "message"),
    [
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), "bool"),
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 4, dtype=torch.bool), "shape"),
        (torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool), "rank"),
    ],
)
def test_preprocessing_fails_closed(values, observed, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        prepare_observations(values, observed, FitFallback(mean=1.0, std=1.0))
