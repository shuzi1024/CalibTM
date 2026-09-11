from __future__ import annotations

import numpy as np
import pytest

from experiments.anchorcv_v1.metrics import (
    AnchorDecision,
    binary_auroc,
    flow_target_regret,
    hard_anchorcv_decision,
    oracle_gap_capture,
    ratio_of_sums_nmae,
    relative_improvement,
    select_expert_predictions,
    spearman_correlation,
)


def test_ratio_of_sums_nmae_pools_operands_instead_of_averaging_rows() -> None:
    # Row NMAEs are 1/1 and 3/9.  Their mean is 2/3, but the frozen pooled
    # ratio of sums is (1 + 3) / (1 + 9) = 0.4.
    assert ratio_of_sums_nmae([1.0, 3.0], [1.0, 9.0]) == pytest.approx(0.4)


def test_ratio_of_sums_allows_zero_rows_but_rejects_zero_pooled_truth() -> None:
    assert ratio_of_sums_nmae([2.0, 1.0], [0.0, 3.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="pooled truth"):
        ratio_of_sums_nmae([0.0, 0.0], [0.0, 0.0])


@pytest.mark.parametrize(
    ("errors", "truth"),
    [
        ([np.nan], [1.0]),
        ([1.0], [np.inf]),
        ([-1.0], [1.0]),
        ([1.0], [-1.0]),
    ],
)
def test_ratio_of_sums_rejects_nonfinite_or_negative_operands(errors, truth) -> None:
    with pytest.raises(ValueError):
        ratio_of_sums_nmae(errors, truth)


def test_flow_regret_and_oracle_are_per_flow_and_ties_prefer_p() -> None:
    truth = np.array([[1.0, 3.0], [2.0, 4.0], [5.0, 5.0]])
    p = np.array([[0.0, 1.0], [0.0, 4.0], [4.0, 5.0]])
    n = np.array([[1.0, 2.0], [1.0, 3.0], [5.0, 4.0]])
    target = np.ones_like(truth, dtype=bool)

    evidence = flow_target_regret(p, n, truth, target)

    np.testing.assert_allclose(evidence.p_error_sum, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(evidence.n_error_sum, [1.0, 2.0, 1.0])
    np.testing.assert_allclose(evidence.regret, [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(evidence.oracle_winner_is_n, [True, False, False])
    np.testing.assert_allclose(evidence.truth_sum, [4.0, 6.0, 10.0])
    np.testing.assert_array_equal(evidence.target_count, [2, 2, 2])

    chosen = select_expert_predictions(p, n, evidence.oracle_winner_is_n)
    np.testing.assert_allclose(chosen, [n[0], p[1], p[2]])


def test_flow_regret_only_checks_selected_entries_and_requires_each_flow_target() -> None:
    truth = np.array([[2.0, np.nan], [np.nan, 4.0]])
    p = np.array([[1.0, np.nan], [np.nan, 6.0]])
    n = np.array([[3.0, np.nan], [np.nan, 4.0]])
    target = np.array([[True, False], [False, True]])
    evidence = flow_target_regret(p, n, truth, target)
    np.testing.assert_allclose(evidence.regret, [0.0, 2.0])

    target[1] = False
    with pytest.raises(ValueError, match="every flow"):
        flow_target_regret(p, n, truth, target)


def test_flow_regret_rejects_selected_nan_and_shape_mismatch() -> None:
    values = np.ones((2, 2))
    target = np.ones((2, 2), dtype=bool)
    bad = values.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        flow_target_regret(bad, values, values, target)
    with pytest.raises(ValueError, match="aligned"):
        flow_target_regret(values[:, :1], values, values, target)


def test_hard_anchorcv_direction_positive_score_selects_n_and_tie_selects_p() -> None:
    # score = mean(e_P - e_N), so flow 0 selects N, flow 1 selects P,
    # and flow 2 is a deterministic tie resolved to P.
    p_anchor_error = np.array([[3.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    n_anchor_error = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])

    decision = hard_anchorcv_decision(p_anchor_error, n_anchor_error)

    assert isinstance(decision, AnchorDecision)
    np.testing.assert_allclose(decision.score, [1.0, -1.0, 0.0])
    np.testing.assert_array_equal(decision.winner_is_n, [True, False, False])
    np.testing.assert_allclose(decision.p_mean_error, [2.0, 1.0, 1.0])
    np.testing.assert_allclose(decision.n_mean_error, [1.0, 2.0, 1.0])


def test_hard_anchorcv_rejects_nan_negative_shape_and_no_anchor() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        hard_anchorcv_decision([[np.nan]], [[0.0]])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        hard_anchorcv_decision([[-1.0]], [[0.0]])
    with pytest.raises(ValueError, match="aligned"):
        hard_anchorcv_decision([[1.0, 2.0]], [[1.0]])
    with pytest.raises(ValueError, match="two-dimensional"):
        hard_anchorcv_decision([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least one anchor"):
        hard_anchorcv_decision(np.empty((2, 0)), np.empty((2, 0)))


def test_binary_auroc_uses_average_tie_credit() -> None:
    labels = np.array([0, 0, 1, 1])
    assert binary_auroc(labels, [0.0, 0.5, 0.5, 1.0]) == pytest.approx(0.875)
    assert binary_auroc(labels, [1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.5)
    assert binary_auroc(labels, [1.0, 0.0, -1.0, -2.0]) == pytest.approx(0.0)


def test_binary_auroc_rejects_single_class_invalid_labels_and_nan() -> None:
    with pytest.raises(ValueError, match="both classes"):
        binary_auroc([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="binary"):
        binary_auroc([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        binary_auroc([0, 1], [0.1, np.nan])


def test_spearman_handles_ties_with_average_ranks() -> None:
    # Average ranks are [1, 2.5, 2.5, 4] and [4, 1.5, 1.5, 3].
    assert spearman_correlation([1, 2, 2, 4], [4, 1, 1, 2]) == pytest.approx(
        -1.0 / 3.0
    )
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_spearman_rejects_constant_empty_shape_mismatch_and_nan() -> None:
    with pytest.raises(ValueError, match="non-constant"):
        spearman_correlation([1, 1], [1, 2])
    with pytest.raises(ValueError, match="nonempty"):
        spearman_correlation([], [])
    with pytest.raises(ValueError, match="aligned"):
        spearman_correlation([1, 2], [1])
    with pytest.raises(ValueError, match="finite"):
        spearman_correlation([1, np.nan], [1, 2])


def test_relative_improvement_has_lower_is_better_direction() -> None:
    assert relative_improvement(candidate=0.8, baseline=1.0) == pytest.approx(0.2)
    assert relative_improvement(candidate=1.2, baseline=1.0) == pytest.approx(-0.2)
    assert relative_improvement(candidate=0.0, baseline=1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        relative_improvement(candidate=0.0, baseline=0.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        relative_improvement(candidate=np.nan, baseline=1.0)


def test_oracle_gap_capture_endpoints_and_out_of_range_evidence() -> None:
    assert oracle_gap_capture(candidate=0.8, best_single=0.8, oracle=0.6) == 0.0
    assert oracle_gap_capture(candidate=0.6, best_single=0.8, oracle=0.6) == 1.0
    # Do not clip contradictions: negative means worse than the best expert and
    # >1 means the measured candidate beat the declared oracle.
    assert oracle_gap_capture(candidate=0.9, best_single=0.8, oracle=0.6) == pytest.approx(
        -0.5
    )
    assert oracle_gap_capture(candidate=0.5, best_single=0.8, oracle=0.6) == pytest.approx(
        1.5
    )


def test_oracle_gap_capture_rejects_missing_gap_invalid_order_and_nan() -> None:
    with pytest.raises(ValueError, match="strictly better"):
        oracle_gap_capture(candidate=0.8, best_single=0.8, oracle=0.8)
    with pytest.raises(ValueError, match="cannot be worse"):
        oracle_gap_capture(candidate=0.8, best_single=0.8, oracle=0.9)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        oracle_gap_capture(candidate=np.nan, best_single=0.8, oracle=0.6)
