from __future__ import annotations

import pytest
import torch

from experiments.acil_innovation_v1.preprocessing import FitFallback
from experiments.anchorcv_v1.model import PriorFreeMaskNativeExpert
from experiments.anchorcv_v1.objective import (
    drop_observed_anchor,
    paired_mask_native_objective,
    predict_raw,
    prepare_mask_native_input,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, FitFallback]:
    truth = torch.tensor(
        [
            [
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
                [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0],
            ],
            [
                [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            ],
        ]
    )
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., (0, 3, 6)] = True
    return truth, observed, FitFallback(mean=8.0, std=4.0)


def _model() -> PriorFreeMaskNativeExpert:
    return PriorFreeMaskNativeExpert(
        num_flows=2,
        time_steps=7,
        temporal_hidden=4,
        d_model=16,
        num_heads=4,
        num_flow_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )


def test_prepare_is_observation_only_and_has_no_interpolated_payload() -> None:
    truth, observed, fallback = _inputs()
    sealed = torch.where(observed, truth, torch.full_like(truth, float("nan")))
    perturbed = sealed.clone()
    perturbed[~observed] = 1_000_000.0

    first = prepare_mask_native_input(sealed, observed, fallback)
    second = prepare_mask_native_input(perturbed, observed, fallback)

    torch.testing.assert_close(first.visible_normalized, second.visible_normalized)
    assert torch.equal(first.visible_normalized[~observed], torch.zeros_like(first.visible_normalized[~observed]))
    torch.testing.assert_close(first.mean, second.mean)
    torch.testing.assert_close(first.std, second.std)
    torch.testing.assert_close(first.normalized_time, torch.linspace(-1.0, 1.0, 7))


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_drop_observed_anchor_removes_exact_sorted_rank(rank: int) -> None:
    _, observed, _ = _inputs()

    dropped = drop_observed_anchor(observed, rank=rank)

    assert torch.equal(dropped.submask.sum(dim=-1), torch.full((2, 2), 2))
    assert torch.equal(dropped.anchor_index, torch.full((2, 2), (0, 3, 6)[rank]))
    assert not dropped.submask.gather(-1, dropped.anchor_index.unsqueeze(-1)).any()
    assert torch.equal(dropped.anchor_mask.sum(dim=-1), torch.ones((2, 2), dtype=torch.long))


def test_drop_observed_anchor_fails_closed_outside_k3() -> None:
    _, observed, _ = _inputs()
    observed[..., 1] = True

    with pytest.raises(ValueError, match="exactly three"):
        drop_observed_anchor(observed, rank=1)


def test_middle_rank_drop_is_simultaneous_but_each_flow_keeps_its_own_time() -> None:
    observed = torch.zeros((1, 2, 7), dtype=torch.bool)
    observed[0, 0, (0, 2, 6)] = True
    observed[0, 1, (1, 4, 5)] = True

    dropped = drop_observed_anchor(observed, rank=1)

    assert dropped.anchor_index.tolist() == [[2, 4]]
    assert dropped.submask.sum(dim=-1).tolist() == [[2, 2]]
    assert not dropped.submask.gather(
        -1, dropped.anchor_index.unsqueeze(-1)
    ).any()


def test_predict_raw_projects_observations_and_does_not_read_missing_truth() -> None:
    torch.manual_seed(11)
    truth, observed, fallback = _inputs()
    model_input = torch.where(observed, truth, torch.full_like(truth, float("nan")))
    altered = model_input.clone()
    altered[~observed] = -999_999.0
    model = _model().eval()

    first = predict_raw(model, model_input, observed, fallback)
    second = predict_raw(model, altered, observed, fallback)

    torch.testing.assert_close(first.prediction, second.prediction, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.prediction[observed], truth[observed])
    assert torch.isfinite(first.prediction).all()
    assert (first.prediction[~observed] >= 0.0).all()


def test_paired_objective_matches_acil_middle_anchor_training_and_backpropagates() -> None:
    torch.manual_seed(17)
    truth, observed, fallback = _inputs()
    model_input = torch.where(observed, truth, torch.full_like(truth, float("nan")))
    model = _model().train()

    loss, details = paired_mask_native_objective(
        model,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=fallback,
        drop_rank=1,
    )

    expected = 0.50 * details.k3_missing_loss + 0.50 * details.k2_all_missing_loss
    torch.testing.assert_close(loss, expected)
    assert details.drop_rank == 1
    assert details.k3_target_count == int((~observed).sum())
    assert details.k2_target_count == int((~details.submask).sum())
    assert not hasattr(details, "k2_anchor_loss")
    assert not hasattr(details, "anchor_target_count")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_loss_truth_is_separate_from_model_input() -> None:
    torch.manual_seed(19)
    truth, observed, fallback = _inputs()
    model_input = torch.where(observed, truth, torch.full_like(truth, float("nan")))
    model = _model().eval()

    loss, _ = paired_mask_native_objective(
        model,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=fallback,
        drop_rank=1,
    )

    changed_truth = truth.clone()
    changed_truth[~observed] += 100.0
    changed_loss, _ = paired_mask_native_objective(
        model,
        model_input=model_input,
        truth=changed_truth,
        observed=observed,
        fit_fallback=fallback,
        drop_rank=1,
    )
    assert not torch.equal(loss, changed_loss)
