from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.preprocessing import FitFallback
from experiments.anchorcv_v1.evaluation import (
    CaseEvidence,
    compute_anchor_decision,
    evaluate_tensor_batch,
    summarize_case_evidence,
)
from experiments.anchorcv_v1.model import PriorFreeMaskNativeExpert


def _tiny_models() -> tuple[ACILBase, PriorFreeMaskNativeExpert]:
    torch.manual_seed(7)
    prior = ACILBase(hidden=8).eval()
    neural = PriorFreeMaskNativeExpert(
        num_flows=2,
        time_steps=7,
        temporal_hidden=4,
        d_model=16,
        num_heads=4,
        num_flow_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    ).eval()
    return prior, neural


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    truth = torch.tensor(
        [
            [
                [2.0, 3.0, 7.0, 4.0, 9.0, 8.0, 5.0],
                [5.0, 7.0, 6.0, 10.0, 12.0, 11.0, 13.0],
            ],
            [
                [3.0, 9.0, 2.0, 8.0, 4.0, 7.0, 6.0],
                [9.0, 3.0, 5.0, 7.0, 8.0, 4.0, 6.0],
            ],
        ]
    )
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., (0, 3, 6)] = True
    model_input = torch.where(observed, truth, torch.full_like(truth, float("nan")))
    return truth, observed, model_input


def test_tensor_evaluation_uses_one_fair_middle_anchor_and_original_k3_scale() -> None:
    prior, neural = _tiny_models()
    truth, observed, model_input = _batch()

    evidence = evaluate_tensor_batch(
        prior=prior,
        neural=neural,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=FitFallback(mean=7.0, std=3.0),
        window_offset=11,
    )

    assert evidence.case_count == 4
    assert evidence.middle_anchor_index.shape == (4,)
    assert evidence.p_anchor_absolute_error.shape == (4,)
    assert evidence.n_anchor_absolute_error.shape == (4,)
    assert evidence.p_anchor_normalized_error.shape == (4,)
    assert evidence.n_anchor_normalized_error.shape == (4,)
    assert evidence.middle_anchor_index.tolist() == [3, 3, 3, 3]
    np.testing.assert_allclose(
        evidence.p_anchor_normalized_error,
        evidence.p_anchor_absolute_error / evidence.k3_scale,
    )
    np.testing.assert_allclose(
        evidence.n_anchor_normalized_error,
        evidence.n_anchor_absolute_error / evidence.k3_scale,
    )
    np.testing.assert_allclose(
        evidence.loo_score,
        evidence.p_anchor_normalized_error
        - evidence.n_anchor_normalized_error,
    )
    assert evidence.window_index.tolist() == [11, 11, 12, 12]
    assert evidence.flow_index.tolist() == [0, 1, 0, 1]
    assert np.all(evidence.target_count == 4)
    assert np.all(np.isfinite(evidence.loo_score))
    assert np.all(evidence.p_error_sum >= 0.0)
    assert np.all(evidence.n_error_sum >= 0.0)
    np.testing.assert_array_equal(evidence.hard_winner_is_n, evidence.loo_score > 0.0)
    np.testing.assert_array_equal(evidence.oracle_winner_is_n, evidence.target_regret > 0.0)


def test_tensor_evaluation_model_path_cannot_read_missing_payload() -> None:
    prior, neural = _tiny_models()
    truth, observed, model_input = _batch()
    altered = model_input.clone()
    altered[~observed] = -123_456.0
    fallback = FitFallback(mean=7.0, std=3.0)

    first = evaluate_tensor_batch(
        prior=prior,
        neural=neural,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=fallback,
    )
    second = evaluate_tensor_batch(
        prior=prior,
        neural=neural,
        model_input=altered,
        truth=truth,
        observed=observed,
        fit_fallback=fallback,
    )

    for field in (
        "p_error_sum",
        "n_error_sum",
        "hard_error_sum",
        "oracle_error_sum",
        "p_anchor_absolute_error",
        "n_anchor_absolute_error",
        "p_anchor_normalized_error",
        "n_anchor_normalized_error",
        "loo_score",
        "target_regret",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_deployable_anchor_decision_is_bitwise_independent_of_missing_truth() -> None:
    prior, neural = _tiny_models()
    truth, observed, model_input = _batch()
    fallback = FitFallback(mean=7.0, std=3.0)

    decision = compute_anchor_decision(
        prior=prior,
        neural=neural,
        model_input=model_input,
        observed=observed,
        fit_fallback=fallback,
    )
    changed_truth = truth.clone()
    changed_truth[~observed] = torch.flip(
        changed_truth[~observed], dims=(0,)
    ) + 10_000.0
    first = evaluate_tensor_batch(
        prior=prior,
        neural=neural,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=fallback,
    )
    second = evaluate_tensor_batch(
        prior=prior,
        neural=neural,
        model_input=model_input,
        truth=changed_truth,
        observed=observed,
        fit_fallback=fallback,
    )

    for field in (
        "middle_anchor_index",
        "k3_scale",
        "p_anchor_absolute_error",
        "n_anchor_absolute_error",
        "p_anchor_normalized_error",
        "n_anchor_normalized_error",
        "loo_score",
        "hard_winner_is_n",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    np.testing.assert_array_equal(decision.loo_score, first.loo_score)
    assert not np.array_equal(first.p_error_sum, second.p_error_sum)


def _manual_evidence() -> CaseEvidence:
    return CaseEvidence(
        window_index=np.array([0, 0, 1, 1], dtype=np.int64),
        flow_index=np.array([0, 1, 0, 1], dtype=np.int64),
        middle_anchor_index=np.array([3, 3, 3, 3], dtype=np.int64),
        p_error_sum=np.array([4.0, 8.0, 3.0, 7.0]),
        n_error_sum=np.array([2.0, 10.0, 2.0, 9.0]),
        hard_error_sum=np.array([2.0, 8.0, 2.0, 7.0]),
        oracle_error_sum=np.array([2.0, 8.0, 2.0, 7.0]),
        truth_sum=np.array([20.0, 20.0, 10.0, 10.0]),
        target_count=np.array([4, 4, 4, 4], dtype=np.int64),
        k3_scale=np.array([1.0, 1.0, 0.5, 1.0]),
        p_anchor_absolute_error=np.array([2.0, 1.0, 1.0, 1.0]),
        n_anchor_absolute_error=np.array([1.0, 2.0, 0.5, 2.0]),
        p_anchor_normalized_error=np.array([2.0, 1.0, 2.0, 1.0]),
        n_anchor_normalized_error=np.array([1.0, 2.0, 1.0, 2.0]),
        loo_score=np.array([1.0, -1.0, 1.0, -1.0]),
        target_regret=np.array([0.5, -0.5, 0.5, -0.5]),
        hard_winner_is_n=np.array([True, False, True, False]),
        oracle_winner_is_n=np.array([True, False, True, False]),
    )


def test_summary_uses_ratio_of_sums_and_reports_oracle_and_capture() -> None:
    summary = summarize_case_evidence(_manual_evidence())

    assert summary["p_nmae"] == pytest.approx(22.0 / 60.0)
    assert summary["n_nmae"] == pytest.approx(23.0 / 60.0)
    assert summary["hard_nmae"] == pytest.approx(19.0 / 60.0)
    assert summary["oracle_nmae"] == pytest.approx(19.0 / 60.0)
    assert summary["best_single_expert"] == "p"
    assert summary["oracle_relative_improvement"] == pytest.approx(3.0 / 22.0)
    assert summary["hard_relative_improvement"] == pytest.approx(3.0 / 22.0)
    assert summary["oracle_gap_capture"] == pytest.approx(1.0)
    assert summary["loo_winner_auroc"] == pytest.approx(1.0)
    assert summary["loo_target_regret_spearman"] == pytest.approx(1.0)


def test_undefined_discrimination_metric_is_retained_as_null_not_fabricated() -> None:
    evidence = _manual_evidence()
    single_class = CaseEvidence(
        **{
            field: getattr(evidence, field)
            for field in evidence.__dataclass_fields__
            if field
            not in {
                "n_error_sum",
                "hard_error_sum",
                "oracle_error_sum",
                "target_regret",
                "oracle_winner_is_n",
            }
        },
        n_error_sum=np.array([2.0, 6.0, 2.5, 5.0]),
        hard_error_sum=np.array([2.0, 8.0, 2.5, 7.0]),
        oracle_error_sum=np.array([2.0, 6.0, 2.5, 5.0]),
        target_regret=np.array([0.5, 0.5, 0.25, 0.5]),
        oracle_winner_is_n=np.ones(4, dtype=np.bool_),
    )

    summary = summarize_case_evidence(single_class)

    assert summary["loo_winner_auroc"] is None
    assert "both classes" in summary["loo_winner_auroc_undefined_reason"]
    assert math.isfinite(summary["loo_target_regret_spearman"])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("target_count", np.array([4, 4, 0, 4]), "target_count"),
        ("truth_sum", np.array([20.0, 20.0, np.nan, 10.0]), "finite"),
        ("p_anchor_absolute_error", np.ones((4, 2)), "one-dimensional"),
        ("hard_winner_is_n", np.array([1, 0, 1, 0]), "boolean"),
    ],
)
def test_case_evidence_fails_closed_on_invalid_arrays(field, replacement, message) -> None:
    payload = {
        name: getattr(_manual_evidence(), name)
        for name in _manual_evidence().__dataclass_fields__
    }
    payload[field] = replacement

    with pytest.raises(ValueError, match=message):
        CaseEvidence(**payload)
