from __future__ import annotations

import numpy as np
import pytest


def test_metric_record_scores_exact_registered_target() -> None:
    from experiments.acil_innovation_v1.metrics import metric_record

    truth = np.arange(1.0, 51.0, dtype=np.float64)
    prediction = truth.copy()
    prediction[3] += 2.0
    observed = np.zeros(50, dtype=np.bool_)
    observed[[0, 20, 49]] = True
    target = ~observed
    record = metric_record(
        identity={"dataset": "geant", "flow": 0, "window_start": 7572},
        truth=truth,
        prediction=prediction,
        observed=observed,
        target=target,
    )
    assert record.absolute_error_sum == 2.0
    assert record.absolute_truth_sum == float(truth[target].sum())
    assert record.target_count == 47


def test_metric_record_rejects_observed_projection_drift() -> None:
    from experiments.acil_innovation_v1.metrics import metric_record

    truth = np.ones(50, dtype=np.float64)
    prediction = truth.copy()
    observed = np.zeros(50, dtype=np.bool_)
    observed[[0, 20, 49]] = True
    prediction[20] = 2.0
    with pytest.raises(ValueError, match="projection"):
        metric_record(
            identity={"dataset": "geant"},
            truth=truth,
            prediction=prediction,
            observed=observed,
            target=~observed,
        )


def test_ratio_of_sums_is_not_mean_of_record_nmae() -> None:
    from experiments.acil_innovation_v1.metrics import aggregate_records

    records = [
        {"absolute_error_sum": 1.0, "absolute_truth_sum": 1.0,
         "squared_error_sum": 1.0, "squared_truth_sum": 1.0,
         "target_count": 1},
        {"absolute_error_sum": 9.0, "absolute_truth_sum": 99.0,
         "squared_error_sum": 81.0, "squared_truth_sum": 9801.0,
         "target_count": 1},
    ]
    result = aggregate_records(records)
    assert result.nmae == pytest.approx(0.1)
    assert result.nmae != pytest.approx((1.0 + 9.0 / 99.0) / 2.0)


def test_oracle_target_requires_q_e_partition() -> None:
    from experiments.acil_innovation_v1.metrics import validate_qe_partition

    observed = np.zeros(50, dtype=np.bool_)
    observed[[0, 20, 49]] = True
    q = np.zeros(50, dtype=np.bool_)
    q[10] = True
    e = (~observed) & (~q)
    validate_qe_partition(observed, q, e)
    bad = e.copy()
    bad[10] = True
    with pytest.raises(ValueError, match="disjoint"):
        validate_qe_partition(observed, q, bad)

