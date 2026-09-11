from __future__ import annotations

import pytest


def _records(drop=None):
    rows = []
    for method, multiplier in (("base", 1.0), ("candidate", 0.8)):
        for seed in (1, 2, 3):
            for dataset, starts in (("a", (0, 50, 100, 150)), ("g", (0, 50, 100, 150))):
                for mask in ("internal_block", "two_burst"):
                    for start in starts:
                        identity = (method, seed, dataset, mask, start)
                        if identity == drop:
                            continue
                        rows.append(
                            {
                                "identity": {
                                    "method": method,
                                    "seed_bundle": seed,
                                    "dataset": dataset,
                                    "mask_family": mask,
                                    "window_start": start,
                                    "flow": 0,
                                },
                                "absolute_error_sum": multiplier * (10.0 + seed + start / 100),
                                "absolute_truth_sum": 100.0,
                                "squared_error_sum": 1.0,
                                "squared_truth_sum": 100.0,
                                "target_count": 47,
                            }
                        )
    return rows


def test_paired_bootstrap_is_deterministic_ratio_of_sums() -> None:
    from experiments.acil_innovation_v1.statistics import paired_bootstrap

    kwargs = dict(
        records=_records(),
        candidate="candidate",
        comparator="base",
        seed_bundles=(1, 2, 3),
        datasets={"a": (0, 50, 100, 150), "g": (0, 50, 100, 150)},
        mask_families=("internal_block", "two_burst"),
        draws=500,
        bootstrap_seed=81001,
        block_length=4,
    )
    first = paired_bootstrap(**kwargs)
    second = paired_bootstrap(**kwargs)
    assert first == second
    assert first.point_estimate == pytest.approx(0.2)
    assert first.ci_lower == pytest.approx(0.2)
    assert first.ci_upper == pytest.approx(0.2)
    assert len(first.draws_sha256) == 64
    assert len(first.draw_plan_sha256) == 64


def test_paired_bootstrap_rejects_missing_window_pair() -> None:
    from experiments.acil_innovation_v1.statistics import paired_bootstrap

    with pytest.raises(ValueError, match="missing paired window"):
        paired_bootstrap(
            records=_records(drop=("candidate", 2, "g", "two_burst", 100)),
            candidate="candidate",
            comparator="base",
            seed_bundles=(1, 2, 3),
            datasets={"a": (0, 50, 100, 150), "g": (0, 50, 100, 150)},
            mask_families=("internal_block", "two_burst"),
            draws=20,
            bootstrap_seed=81001,
            block_length=4,
        )

