from __future__ import annotations

import numpy as np
import pytest


def _table(multiplier: float = 0.8):
    rows = {}
    for method, factor in (
        ("acil_only", 1.0),
        ("sc_acil_u0", 0.9),
        ("sc_acil", multiplier),
        ("linear_interpolation", 1.1),
    ):
        for seed in (4, 5, 6):
            for dataset, count in (("abilene", 8), ("geant", 4)):
                for mask in ("random", "internal_block", "two_burst"):
                    error = np.full(count, factor * (10.0 + seed), dtype="<f8")
                    truth = np.full(count, 100.0, dtype="<f8")
                    rows[(method, seed, dataset, mask)] = (error, truth)
    return rows


def test_dataset_equal_paired_bootstrap_is_deterministic_and_ratio_of_sums() -> None:
    from experiments.sc_acil_v1.statistics import dataset_equal_bootstrap

    first = dataset_equal_bootstrap(
        _table(),
        candidate="sc_acil",
        comparator="acil_only",
        masks=("internal_block", "two_burst"),
        draws=500,
        bootstrap_seed=91001,
        block_length=4,
    )
    second = dataset_equal_bootstrap(
        _table(),
        candidate="sc_acil",
        comparator="acil_only",
        masks=("internal_block", "two_burst"),
        draws=500,
        bootstrap_seed=91001,
        block_length=4,
    )
    assert first == second
    assert first.point_estimate == pytest.approx(0.2)
    assert first.ci_lower == pytest.approx(0.2)
    assert first.ci_upper == pytest.approx(0.2)
    assert len(first.draws_sha256) == 64
    assert len(first.draw_plan_sha256) == 64


def test_adjudication_applies_capacity_attribution_and_no_harm_clauses() -> None:
    from experiments.sc_acil_v1.adjudication import decide

    passing = {
        "main_point": 0.02,
        "main_ci_lower": 0.001,
        "innovation_point": 0.01,
        "innovation_ci_lower": 0.001,
        "random_effect": 0.0,
        "per_dataset_main": {"abilene": 0.03, "geant": 0.01},
        "dataset_mask_main": {
            "abilene/internal_block": 0.03,
            "abilene/two_burst": 0.02,
            "geant/internal_block": 0.01,
            "geant/two_burst": -0.002,
        },
        "per_seed_main": {"4": 0.02, "5": 0.01, "6": -0.001},
    }
    assert decide(**passing).verdict == "proceed"
    assert decide(**{**passing, "innovation_point": 0.004}).verdict == "kill"
    assert decide(**{**passing, "innovation_ci_lower": 0.0}).verdict == "kill"
    assert decide(**{**passing, "random_effect": -0.006}).verdict == "kill"
    assert decide(
        **{
            **passing,
            "dataset_mask_main": {
                "abilene/internal_block": 0.03,
                "abilene/two_burst": -0.002,
                "geant/internal_block": -0.003,
                "geant/two_burst": -0.004,
            },
        }
    ).verdict == "kill"


def test_formal_table_summary_keeps_internal_and_external_roles_separate() -> None:
    from experiments.sc_acil_v1.adjudication import summarize_table

    summary = summarize_table(_table())
    assert summary["main_full_over_acil"] == pytest.approx(0.2)
    assert summary["innovation_full_over_u0"] == pytest.approx(1.0 - 0.8 / 0.9)
    assert summary["random_full_over_acil"] == pytest.approx(0.2)
    assert summary["full_over_linear"] == pytest.approx(1.0 - 0.8 / 1.1)
    assert set(summary["per_dataset_main"]) == {"abilene", "geant"}
    assert set(summary["per_seed_main"]) == {"4", "5", "6"}
    assert len(summary["dataset_mask_main"]) == 4
