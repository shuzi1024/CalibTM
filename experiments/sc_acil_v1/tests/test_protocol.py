from __future__ import annotations

import inspect

import pytest


def test_frozen_protocol_names_acil_as_internal_component_not_external_baseline() -> None:
    from experiments.sc_acil_v1.protocol import load_protocol

    config = load_protocol()
    assert config["protocol"] == {
        "branch_sealed": True,
        "git_available": False,
        "git_commit": None,
        "id": "sc-acil-v1",
        "status": "frozen_before_gate",
        "window_length": 50,
    }
    assert config["method_roles"] == {
        "sc_acil": "proposed_full_method",
        "acil_only": "internal_ablation_without_self_calibration",
        "sc_acil_u0": "matched_capacity_internal_ablation",
        "linear_interpolation": "external_classical_baseline",
    }
    assert config["evidence_boundary"]["sealed_test_access"] is False
    assert config["evidence_boundary"]["legacy_test_cache_access"] is False
    assert config["formal_gate"]["seed_bundles"] == [4, 5, 6]
    assert config["formal_gate"]["evaluation_cohort"] == "gate"
    assert config["formal_gate"]["observation_count"] == 3


def test_stage_a_gate_and_stop_rule_are_exactly_frozen() -> None:
    from experiments.sc_acil_v1.protocol import load_protocol

    gate = load_protocol()["gate"]
    assert gate["main_full_over_acil"] == {
        "bootstrap_ci_lower_strictly_above": 0.0,
        "candidate": "sc_acil",
        "comparator": "acil_only",
        "dataset_equal_structured_improvement_minimum": 0.015,
        "per_dataset_minimum": -0.005,
        "positive_dataset_mask_cells_required": 3,
        "positive_dataset_mask_cells_total": 4,
        "positive_seed_bundles_required": 2,
        "positive_seed_bundles_total": 3,
        "worst_dataset_mask_cell_minimum": -0.01,
    }
    assert gate["innovation_attribution"] == {
        "bootstrap_ci_lower_strictly_above": 0.0,
        "candidate": "sc_acil",
        "comparator": "sc_acil_u0",
        "dataset_equal_structured_improvement_minimum": 0.005,
    }
    assert gate["random_no_harm"] == {
        "candidate": "sc_acil",
        "comparator": "acil_only",
        "dataset_equal_improvement_minimum": -0.005,
    }
    assert gate["failure"] == "kill_sc_acil_method_claim_no_gate_revision"
    assert gate["success"] == "proceed_to_rate_and_external_baseline_stage"


def test_job_registry_is_complete_unique_and_outcome_independent() -> None:
    from experiments.sc_acil_v1.jobs import planned_jobs, resolve_job

    acil_jobs = planned_jobs("fit_acil")
    gate_jobs = planned_jobs("formal_gate")
    assert len(acil_jobs) == 6
    assert len(gate_jobs) == 12
    assert {job.method for job in acil_jobs} == {"acil_only"}
    assert {job.method for job in gate_jobs} == {"sc_acil_u0", "sc_acil"}
    assert {job.dataset for job in (*acil_jobs, *gate_jobs)} == {"abilene", "geant"}
    assert {job.seed_bundle for job in (*acil_jobs, *gate_jobs)} == {4, 5, 6}
    assert len({job.job_id for job in (*acil_jobs, *gate_jobs)}) == 18
    for job in (*acil_jobs, *gate_jobs):
        assert resolve_job(job.stage, job.job_id) == job
        assert len(job.job_id) == 64

    with pytest.raises(ValueError, match="planned"):
        resolve_job("formal_gate", "0" * 64)


def test_public_worker_has_no_data_split_or_method_override_surface() -> None:
    from experiments.sc_acil_v1.run_job import run_planned_job

    assert tuple(inspect.signature(run_planned_job).parameters) == (
        "stage",
        "job_id",
        "output_root",
    )

