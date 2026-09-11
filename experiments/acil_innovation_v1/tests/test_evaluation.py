from __future__ import annotations

import numpy as np
import torch


def test_evaluate_prediction_records_all_flows_on_deployable_u() -> None:
    from experiments.acil_innovation_v1.evaluation import records_for_window
    from experiments.acil_innovation_v1.masks import build_evaluation_mask
    from experiments.acil_innovation_v1.registries import ordered_window_starts

    start = ordered_window_starts("abilene", "tune")[0]
    truth = np.ones((2, 50), dtype=np.float32)
    prediction = truth.copy()
    bundles = tuple(
        build_evaluation_mask(
            dataset="abilene", cohort="tune", window_start=start,
            flow_index=flow, family="random", seed_bundle=1,
        ) for flow in range(2)
    )
    rows = records_for_window(
        method="acil", seed_bundle=1, dataset="abilene", mask_family="random",
        window_start=start, truth=truth, prediction=prediction, bundles=bundles,
        oracle=False,
    )
    assert len(rows) == 2
    assert all(row.target_count == 47 for row in rows)


def test_oracle_records_score_e_not_q() -> None:
    from experiments.acil_innovation_v1.evaluation import records_for_window
    from experiments.acil_innovation_v1.masks import build_evaluation_mask, oracle_partition
    from experiments.acil_innovation_v1.registries import ordered_window_starts

    start = ordered_window_starts("geant", "tune")[0]
    bundle = build_evaluation_mask(
        dataset="geant", cohort="tune", window_start=start, flow_index=0,
        family="internal_block", seed_bundle=1,
    )
    truth = np.ones((1, 50), dtype=np.float32)
    prediction = truth.copy()
    q = oracle_partition(bundle).support
    prediction[0, q] = 999.0
    rows = records_for_window(
        method="truth_q_deepsets", seed_bundle=1, dataset="geant",
        mask_family="internal_block", window_start=start, truth=truth,
        prediction=prediction, bundles=(bundle,), oracle=True,
    )
    assert rows[0].target_count == 46
    assert rows[0].absolute_error_sum == 0.0

