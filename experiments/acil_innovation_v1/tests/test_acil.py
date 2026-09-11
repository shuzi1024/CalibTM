from __future__ import annotations

import pytest
import torch

from experiments.acil_innovation_v1.acil import (
    ACILBase,
    middle_anchor_leave_one_out,
)
from experiments.acil_innovation_v1.preprocessing import FitFallback


def _fixture() -> tuple[torch.Tensor, torch.Tensor, FitFallback]:
    observed = torch.zeros(1, 2, 7, dtype=torch.bool)
    observed[0, 0, [0, 3, 6]] = True
    observed[0, 1, [1, 2, 5]] = True
    values = torch.full((1, 2, 7), float("nan"))
    values[0, 0, [0, 3, 6]] = torch.tensor([2.0, 5.0, 11.0])
    values[0, 1, [1, 2, 5]] = torch.tensor([3.0, 7.0, 8.0])
    return values, observed, FitFallback(mean=5.0, std=2.0)


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32)


def test_acil_runs_in_normalized_coordinates_and_projects_raw_observations() -> None:
    torch.manual_seed(100)
    values, observed, fallback = _fixture()
    acil = ACILBase()

    result = acil(values, observed, fallback)

    assert result.prediction.shape == values.shape
    assert result.normalized_prediction.shape == values.shape
    assert result.features.shape == (*values.shape, 16)
    assert torch.isfinite(result.prediction).all()
    assert torch.all(result.prediction >= 0)
    assert torch.equal(
        _bits(result.prediction).masked_select(observed),
        _bits(values).masked_select(observed),
    )
    observed_normalized = (
        values - result.statistics.mean
    ) / result.statistics.std
    assert torch.allclose(
        result.normalized_prediction.masked_select(observed),
        observed_normalized.masked_select(observed),
    )


def test_middle_anchor_loo_removes_exactly_the_sorted_middle_and_is_unique() -> None:
    torch.manual_seed(101)
    values, observed, fallback = _fixture()
    acil = ACILBase()

    loo = middle_anchor_leave_one_out(acil, values, observed, fallback)

    assert torch.equal(loo.middle_index, torch.tensor([[3, 2]]))
    expected_submask = observed.clone()
    expected_submask[0, 0, 3] = False
    expected_submask[0, 1, 2] = False
    assert torch.equal(loo.submask, expected_submask)
    assert torch.equal(loo.submask.sum(dim=-1), torch.full((1, 2), 2))
    middle_truth = values.gather(-1, loo.middle_index.unsqueeze(-1))
    middle_prediction = loo.submask_result.prediction.gather(
        -1, loo.middle_index.unsqueeze(-1)
    )
    assert torch.equal(loo.innovation_raw, middle_truth - middle_prediction)
    assert torch.allclose(
        loo.innovation_normalized,
        loo.innovation_raw / loo.full_result.statistics.std,
    )


def test_acil_and_loo_are_invariant_to_every_unobserved_payload() -> None:
    torch.manual_seed(102)
    values, observed, fallback = _fixture()
    changed = values.clone()
    changed[~observed] = torch.linspace(-1e9, 1e9, int((~observed).sum()))
    acil = ACILBase().eval()

    first = middle_anchor_leave_one_out(acil, values, observed, fallback)
    second = middle_anchor_leave_one_out(acil, changed, observed, fallback)

    assert torch.equal(first.full_result.prediction, second.full_result.prediction)
    assert torch.equal(first.submask_result.prediction, second.submask_result.prediction)
    assert torch.equal(first.innovation_raw, second.innovation_raw)
    assert torch.equal(first.innovation_normalized, second.innovation_normalized)
    assert torch.equal(first.submask, second.submask)


@pytest.mark.parametrize("count", [2, 4])
def test_middle_anchor_loo_requires_exactly_three_observations_per_flow(count) -> None:
    values = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
    observed = torch.zeros_like(values, dtype=torch.bool)
    observed[..., :count] = True
    with pytest.raises(ValueError, match="exactly three"):
        middle_anchor_leave_one_out(
            ACILBase(), values, observed, FitFallback(mean=1.0, std=1.0)
        )
