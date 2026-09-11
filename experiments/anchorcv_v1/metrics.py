"""Pure NumPy evidence metrics for the AnchorCV prototype.

All lower-is-better expert comparisons use the same sign convention:

``regret_or_score = P_error - N_error``.

Consequently, a strictly positive value selects the mask-native neural expert
(``N``); zero is a deterministic tie resolved in favour of the interpolation
expert (``P``).  The hidden-truth oracle functions in this module are
diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def _readonly(values: ArrayLike, *, dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FlowTargetEvidence:
    """Per-flow hidden-target errors and the diagnostic oracle decision.

    ``regret`` is ``p_error_sum - n_error_sum``.  Positive regret therefore
    means that N has lower hidden-target absolute error.  Oracle ties select P.
    Individual ``truth_sum`` values may be zero; only a pooled NMAE
    denominator is required to be positive.
    """

    p_error_sum: FloatArray
    n_error_sum: FloatArray
    truth_sum: FloatArray
    target_count: IntArray
    regret: FloatArray
    oracle_winner_is_n: BoolArray

    @property
    def mean_error_regret(self) -> FloatArray:
        """Return ``(P_error - N_error) / target_count`` for each flow."""

        values = self.regret / self.target_count
        return _readonly(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AnchorDecision:
    """Deployable per-flow hard decision derived only from anchor errors."""

    p_mean_error: FloatArray
    n_mean_error: FloatArray
    score: FloatArray
    winner_is_n: BoolArray


def _numeric_array(values: ArrayLike, label: str) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric array") from error
    return result


def ratio_of_sums_nmae(
    absolute_error_sums: ArrayLike,
    absolute_truth_sums: ArrayLike,
) -> float:
    """Pool NMAE operands as ``sum(error) / sum(abs(truth))``.

    Zero-denominator *rows* are retained because other rows can make the
    pooled denominator valid.  Empty inputs, negative operands, non-finite
    operands, and a non-positive final pooled truth denominator are rejected.
    """

    errors = _numeric_array(absolute_error_sums, "absolute error sums")
    truths = _numeric_array(absolute_truth_sums, "absolute truth sums")
    if errors.shape != truths.shape:
        raise ValueError("absolute error and truth sum arrays must be aligned")
    if errors.size == 0:
        raise ValueError("ratio-of-sums inputs must be nonempty")
    if (
        not np.isfinite(errors).all()
        or not np.isfinite(truths).all()
        or np.any(errors < 0.0)
        or np.any(truths < 0.0)
    ):
        raise ValueError("ratio-of-sums operands must be finite and nonnegative")
    numerator = math.fsum(float(value) for value in errors.ravel())
    denominator = math.fsum(float(value) for value in truths.ravel())
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError("pooled ratio-of-sums operands must be finite")
    if denominator <= 0.0:
        raise ValueError("pooled truth denominator must be strictly positive")
    return numerator / denominator


def _aligned_flow_inputs(
    p_prediction: ArrayLike,
    n_prediction: ArrayLike,
    truth: ArrayLike,
    target: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    p_values = _numeric_array(p_prediction, "P prediction")
    n_values = _numeric_array(n_prediction, "N prediction")
    actual = _numeric_array(truth, "truth")
    mask = np.asarray(target)
    if (
        p_values.ndim != 2
        or n_values.shape != p_values.shape
        or actual.shape != p_values.shape
        or mask.shape != p_values.shape
    ):
        raise ValueError(
            "P prediction, N prediction, truth, and target must be aligned "
            "two-dimensional flow-by-position arrays"
        )
    if mask.dtype != np.dtype(np.bool_):
        raise ValueError("target must be a boolean array")
    if p_values.shape[0] == 0:
        raise ValueError("at least one flow is required")
    counts = mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("every flow must contain at least one target")
    if (
        not np.isfinite(p_values[mask]).all()
        or not np.isfinite(n_values[mask]).all()
        or not np.isfinite(actual[mask]).all()
    ):
        raise ValueError("selected predictions and truth must be finite")
    return p_values, n_values, actual, mask


def flow_target_regret(
    p_prediction: ArrayLike,
    n_prediction: ArrayLike,
    truth: ArrayLike,
    target: ArrayLike,
) -> FlowTargetEvidence:
    """Compute per-flow target errors and the hidden-truth expert oracle.

    Inputs have shape ``(flow, position)``.  Values outside ``target`` are not
    inspected and may remain NaN-sealed.  This function deliberately exposes
    hidden target truth and must not be used as a deployable routing policy.
    """

    p_values, n_values, actual, mask = _aligned_flow_inputs(
        p_prediction, n_prediction, truth, target
    )
    p_difference = np.zeros(p_values.shape, dtype=np.float64)
    n_difference = np.zeros(n_values.shape, dtype=np.float64)
    selected_truth = np.zeros(actual.shape, dtype=np.float64)
    np.subtract(p_values, actual, out=p_difference, where=mask)
    np.subtract(n_values, actual, out=n_difference, where=mask)
    np.copyto(selected_truth, actual, where=mask)

    p_error = np.abs(p_difference).sum(axis=1, dtype=np.float64)
    n_error = np.abs(n_difference).sum(axis=1, dtype=np.float64)
    truth_sum = np.abs(selected_truth).sum(axis=1, dtype=np.float64)
    target_count = mask.sum(axis=1, dtype=np.int64)
    regret = p_error - n_error
    winner_is_n = regret > 0.0
    return FlowTargetEvidence(
        p_error_sum=_readonly(p_error, dtype=np.float64),
        n_error_sum=_readonly(n_error, dtype=np.float64),
        truth_sum=_readonly(truth_sum, dtype=np.float64),
        target_count=_readonly(target_count, dtype=np.int64),
        regret=_readonly(regret, dtype=np.float64),
        oracle_winner_is_n=_readonly(winner_is_n, dtype=np.bool_),
    )


def select_expert_predictions(
    p_prediction: ArrayLike,
    n_prediction: ArrayLike,
    winner_is_n: ArrayLike,
) -> FloatArray:
    """Select complete per-flow predictions using a boolean winner vector."""

    p_values = _numeric_array(p_prediction, "P prediction")
    n_values = _numeric_array(n_prediction, "N prediction")
    winners = np.asarray(winner_is_n)
    if p_values.shape != n_values.shape or p_values.ndim < 1:
        raise ValueError("P and N predictions must be aligned non-scalar arrays")
    if (
        winners.dtype != np.dtype(np.bool_)
        or winners.ndim != 1
        or winners.shape[0] != p_values.shape[0]
    ):
        raise ValueError("winner_is_n must be a boolean vector aligned by flow")
    broadcast_shape = (winners.shape[0],) + (1,) * (p_values.ndim - 1)
    return np.where(winners.reshape(broadcast_shape), n_values, p_values)


def hard_anchorcv_decision(
    p_anchor_errors: ArrayLike,
    n_anchor_errors: ArrayLike,
) -> AnchorDecision:
    """Route each flow from its mean cross-validated anchor error.

    Inputs have shape ``(flow, held_out_anchor)`` and contain normalized
    nonnegative absolute errors.  The score is ``mean(e_P - e_N)``.  Positive
    scores select N; non-positive scores select P, giving deterministic P
    tie-breaking.
    """

    p_errors = _numeric_array(p_anchor_errors, "P anchor errors")
    n_errors = _numeric_array(n_anchor_errors, "N anchor errors")
    if p_errors.ndim != 2 or n_errors.ndim != 2:
        raise ValueError("anchor error arrays must be two-dimensional")
    if p_errors.shape != n_errors.shape:
        raise ValueError("P and N anchor error arrays must be aligned")
    if p_errors.shape[0] == 0:
        raise ValueError("at least one flow is required")
    if p_errors.shape[1] == 0:
        raise ValueError("at least one anchor error is required per flow")
    if (
        not np.isfinite(p_errors).all()
        or not np.isfinite(n_errors).all()
        or np.any(p_errors < 0.0)
        or np.any(n_errors < 0.0)
    ):
        raise ValueError("anchor errors must be finite and nonnegative")
    p_mean = p_errors.mean(axis=1, dtype=np.float64)
    n_mean = n_errors.mean(axis=1, dtype=np.float64)
    score = p_mean - n_mean
    return AnchorDecision(
        p_mean_error=_readonly(p_mean, dtype=np.float64),
        n_mean_error=_readonly(n_mean, dtype=np.float64),
        score=_readonly(score, dtype=np.float64),
        winner_is_n=_readonly(score > 0.0, dtype=np.bool_),
    )


def _average_ranks(values: FloatArray) -> FloatArray:
    """Return one-based ascending ranks with average rank for exact ties."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        # One-based ranks [start + 1, ..., end] have this average.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Compute binary AUROC with 0.5 credit for tied positive/negative scores."""

    label_values = np.asarray(labels)
    score_values = _numeric_array(scores, "scores")
    if label_values.ndim != 1 or score_values.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if label_values.shape != score_values.shape:
        raise ValueError("labels and scores must be aligned")
    if label_values.size == 0:
        raise ValueError("labels and scores must be nonempty")
    if not np.isfinite(score_values).all():
        raise ValueError("scores must be finite")
    if not np.all(np.logical_or(label_values == 0, label_values == 1)):
        raise ValueError("labels must be binary values 0 or 1")
    positive = label_values.astype(np.bool_)
    positive_count = int(positive.sum())
    negative_count = int(positive.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both classes")
    ranks = _average_ranks(score_values)
    rank_sum_positive = math.fsum(float(rank) for rank in ranks[positive])
    auc = (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def spearman_correlation(left: ArrayLike, right: ArrayLike) -> float:
    """Compute Spearman's rho using average ranks for ties."""

    left_values = _numeric_array(left, "left values")
    right_values = _numeric_array(right, "right values")
    if left_values.ndim != 1 or right_values.ndim != 1:
        raise ValueError("Spearman inputs must be one-dimensional")
    if left_values.shape != right_values.shape:
        raise ValueError("Spearman inputs must be aligned")
    if left_values.size == 0:
        raise ValueError("Spearman inputs must be nonempty")
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("Spearman inputs must be finite")
    left_rank = _average_ranks(left_values)
    right_rank = _average_ranks(right_values)
    left_centered = left_rank - left_rank.mean(dtype=np.float64)
    right_centered = right_rank - right_rank.mean(dtype=np.float64)
    left_square_sum = float(np.dot(left_centered, left_centered))
    right_square_sum = float(np.dot(right_centered, right_centered))
    if left_square_sum <= 0.0 or right_square_sum <= 0.0:
        raise ValueError("Spearman inputs must both be non-constant")
    numerator = float(np.dot(left_centered, right_centered))
    return numerator / math.sqrt(left_square_sum * right_square_sum)


def relative_improvement(*, candidate: float, baseline: float) -> float:
    """Return ``1 - candidate / baseline`` for lower-is-better metrics."""

    candidate_value = float(candidate)
    baseline_value = float(baseline)
    if (
        not math.isfinite(candidate_value)
        or not math.isfinite(baseline_value)
        or candidate_value < 0.0
        or baseline_value < 0.0
    ):
        raise ValueError("candidate and baseline must be finite and nonnegative")
    if baseline_value <= 0.0:
        raise ValueError("baseline must be strictly positive")
    return 1.0 - candidate_value / baseline_value


def oracle_gap_capture(
    *,
    candidate: float,
    best_single: float,
    oracle: float,
) -> float:
    """Return the fraction of the expert-oracle gap captured by a candidate.

    The formula is ``(best_single - candidate) / (best_single - oracle)``.
    It is intentionally not clipped: 0 denotes the best single expert, 1 the
    oracle, a negative value reports harm, and a value above 1 exposes an
    inconsistent or noisy measured ordering.
    """

    values = tuple(float(value) for value in (candidate, best_single, oracle))
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("candidate, best_single, and oracle must be finite and nonnegative")
    candidate_value, best_value, oracle_value = values
    if oracle_value > best_value:
        raise ValueError("oracle cannot be worse than the best single expert")
    if oracle_value == best_value:
        raise ValueError("oracle must be strictly better to define a capture gap")
    return (best_value - candidate_value) / (best_value - oracle_value)


__all__ = [
    "AnchorDecision",
    "FlowTargetEvidence",
    "binary_auroc",
    "flow_target_regret",
    "hard_anchorcv_decision",
    "oracle_gap_capture",
    "ratio_of_sums_nmae",
    "relative_improvement",
    "select_expert_predictions",
    "spearman_correlation",
]
