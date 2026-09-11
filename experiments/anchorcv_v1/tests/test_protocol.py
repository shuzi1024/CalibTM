from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from experiments.anchorcv_v1 import protocol


def test_research_card_is_canonical_json_compatible_yaml_and_fingerprinted():
    spec = protocol.load_protocol()
    first = spec.to_dict()
    encoded = protocol.canonical_json_bytes(spec)

    assert json.loads(encoded) == first
    assert encoded == json.dumps(
        first, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert protocol.fingerprint(spec) == hashlib.sha256(encoded).hexdigest()
    assert len(protocol.fingerprint(spec)) == 64

    first["protocol"]["id"] = "caller-mutation"
    assert spec.protocol_id == "anchorcv-v1"
    assert spec.to_dict()["protocol"]["id"] == "anchorcv-v1"


def test_frozen_protocol_records_question_grid_experts_and_stage_bundles():
    spec = protocol.load_protocol()
    payload = spec.to_dict()

    assert spec.status == "frozen_full_pipeline"
    assert spec.datasets == ("abilene", "geant")
    assert spec.permitted_splits == ("fit", "source_dev", "tune")
    assert spec.mask_families == ("random", "internal_block", "two_burst")
    assert spec.observed_points_per_flow == 3
    assert spec.prototype_bundles == (1,)
    assert spec.extension_bundles == (2, 3)
    assert spec.final_gate_bundles == (1, 2, 3)
    assert spec.final_gate_job_count == 6
    assert spec.bootstrap_seed == 81001
    assert spec.bootstrap_replicates == 10_000
    assert spec.positive_bundle_count_min == 2

    assert payload["experts"]["p"] == {
        "identity": "seed_matched_frozen_acil",
        "checkpoint_selection": "source_dev_only",
        "bundle_matching": "exact",
        "training": "forbidden_in_anchorcv_stage",
        "hard_project_observations": True,
    }
    neural = payload["experts"]["n"]
    assert neural["identity"] == "prior_free_mask_native_neural"
    assert neural["inputs"] == [
        "observed_values",
        "observation_mask",
        "normalized_time_coordinate",
        "flow_id",
    ]
    assert neural["forbidden_inputs"] == [
        "linear_interpolation",
        "acil_output",
        "missing_truth",
    ]
    assert neural["architecture"] == {
        "temporal_encoder": "one_layer_bidirectional_gru",
        "temporal_hidden_per_direction": 128,
        "flow_token_dim": 256,
        "flow_id_embedding": "learned_dataset_specific",
        "cross_flow_encoder": "four_layer_noncausal_transformer_encoder",
        "attention_heads": 8,
        "feedforward_dim": 1024,
        "dropout": 0.1,
        "decoder": "linear_256_128_gelu_dropout_linear_128_50",
        "output": "direct_normalized_trajectory_not_residual",
    }
    assert neural["normalization"] == {
        "location_scale": "per_window_flow_observation_only_mean_population_std",
        "degenerate_fallback": "fit_only_dataset_mean_std",
        "missing_payload_before_model": 0,
        "normalized_time_range": [-1.0, 1.0],
    }
    assert neural["training"] == {
        "epochs": 20,
        "physical_batches_per_epoch": 64,
        "physical_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 32,
        "optimizer_updates": 320,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "lr_schedule": "constant",
        "betas": [0.9, 0.999],
        "epsilon": 1e-08,
        "weight_decay": 0.01,
        "no_decay_rules": [
            "parameter_name_endswith_bias",
            "parameter_belongs_to_layernorm_and_name_endswith_weight",
        ],
        "weight_decay_applies_to": "all_other_trainable_parameters",
        "gradient_clip_norm": 1.0,
        "cuda_precision": "bf16_with_fp32_gru_frontend",
        "attention_backend": "math_sdp_flash_and_mem_efficient_disabled",
        "outcome_dependent_early_stop": False,
        "checkpoint_selection": "source_dev_ratio_of_sums_all_three_masks",
        "objective": {
            "k3_missing_weight": 0.5,
            "k2_all_missing_weight": 0.5,
        },
        "anchor_drop_schedule": "fixed_sorted_middle_rank_1",
    }
    assert neural["training_masks"] == ["k3", "k2_anchor_drop"]
    assert neural["anchor_drop_ranks"] == [1]
    assert neural["hard_project_observations"] is True
    assert payload["evaluation"]["anchor_cross_validation"] == {
        "held_out_anchors": "sorted_middle_rank_1_only",
        "anchor_index_synchronization": (
            "within_window_all_flows_drop_their_own_sorted_middle_rank_simultaneously"
        ),
        "expert_error": "normalized_absolute_error",
        "routing_unit": "per_window_flow",
        "anchor_error_scale": "original_k3_observation_only_std",
        "anchor_error_aggregation": "single_middle_anchor",
        "tie_break": "expert_p",
        "target_regret_scale": "k3_observation_only_std",
        "target_scope": "fixed_k3_missing_complement",
        "hidden_truth_oracle_is_deployable": False,
    }


def test_branch_sealed_boundary_and_final_only_cohort_are_explicit():
    payload = protocol.load_protocol().to_dict()

    assert payload["evidence_boundary"] == {
        "branch_sealed": True,
        "pristine_project_wide_holdout": False,
        "physical_gate_isolation_from_upstream_loader": False,
        "upstream_loader_may_read_validation_parent_before_cohort_slice": True,
        "anchorcv_discovery_api_returns_final_gate": False,
        "anchorcv_discovery_models_final_gate": False,
        "permitted_splits": ["fit", "source_dev", "tune"],
        "test_access": False,
        "legacy_test_cache_access": False,
        "whole_raw_csv_access": False,
        "runner_exposes_test_path": False,
        "runner_exposes_split_override": False,
    }
    assert payload["cohorts"] == {
        "fit": "neural_expert_parameter_fitting_only",
        "source_dev": "checkpoint_selection_only",
        "tune": "prototype_extension_and_multibundle_adjudication_only",
        "final_gate": (
            "final_only_confirmation_after_verified_method_freeze_authority"
        ),
    }
    assert "final_gate" not in payload["evidence_boundary"]["permitted_splits"]


def test_stage_order_and_one_shot_final_gate_are_frozen():
    payload = protocol.load_protocol().to_dict()

    assert payload["stages"] == {
        "stage_order": [
            "prototype_bundle_1",
            "conditional_extension_bundles_2_and_3",
            "complete_six_tune_job_multibundle_gate",
            "method_freeze_authority",
            "one_shot_six_job_final_gate",
        ],
        "prototype_bundles": [1],
        "prototype_job_count": 2,
        "extension_bundles": [2, 3],
        "extension_job_count": 4,
        "extension_is_conditional": True,
        "extension_condition": "prototype_verdict_proceed",
        "multibundle_input": (
            "complete_six_tune_jobs_two_datasets_by_three_bundles"
        ),
        "multibundle_condition": "all_six_tune_jobs_verified_complete",
        "method_freeze_authority_condition": "multibundle_verdict_proceed",
        "final_gate_bundles": [1, 2, 3],
        "final_gate_job_count": 6,
        "final_gate_condition": "verified_method_freeze_authority",
    }
    assert payload["final_gate"] == {
        "cohort": "gate",
        "datasets": ["abilene", "geant"],
        "bundles": [1, 2, 3],
        "job_count": 6,
        "checkpoint_policy": "reuse_exact_authority_bound_tune_checkpoint",
        "training": "forbidden",
        "launch_policy": "single_complete_six_job_grid",
        "mid_grid_metric_inspection": "forbidden",
        "method_or_threshold_change_after_access": "forbidden",
        "outcome_selected_seed_addition": "forbidden",
        "infrastructure_retry": "identical_job_identity_only",
        "adjudication": "same_seven_method_gates_plus_uncertainty_gates",
    }


def test_two_level_bootstrap_is_exact_and_shared_across_contrasts():
    payload = protocol.load_protocol().to_dict()

    assert payload["uncertainty"] == {
        "bit_generator": "PCG64DXSM",
        "seed": 81001,
        "replicates": 10_000,
        "outer": {
            "unit": "seed_bundle",
            "draws_per_replicate": 3,
            "replacement": True,
        },
        "inner": {
            "method": "circular_block_bootstrap",
            "block_length_windows": 4,
            "sampling_unit": "whole_window_all_flows",
            "independent_per": (
                "sampled_bundle_occurrence_by_dataset"
            ),
            "shared_across": ["mask_families", "method_contrasts"],
        },
        "interval": {
            "confidence": 0.95,
            "quantile_method": "linear_percentile",
        },
        "hard_improvement_statistic": (
            "dataset_equal_structured_masks_relative_improvement"
        ),
        "hard_improvement_ci_lower_bound": "strictly_greater_than_zero",
        "positive_bundle_count_min": 2,
    }


def test_frozen_metrics_gates_stop_rules_and_integrity_are_exact():
    spec = protocol.load_protocol()
    payload = spec.to_dict()

    assert spec.gates == protocol.GateSpec(
        structured_per_flow_oracle_dataset_equal_min=0.03,
        loo_winner_auroc_min=0.60,
        loo_target_regret_spearman_min=0.15,
        hard_anchorcv_oracle_gap_capture_min=0.30,
        hard_anchorcv_actual_improvement_min=0.005,
        worst_structured_oracle_cell_relative_improvement_min=0.015,
        worst_structured_cell_relative_improvement_min=-0.01,
    )
    assert payload["evaluation"]["metrics"] == [
        "ratio_of_sums_nmae",
        "loo_winner_auroc",
        "loo_target_regret_spearman",
        "oracle_gap_capture",
        "paired_per_run_relative_improvement",
    ]
    assert payload["stop_rules"] == {
        "oracle_gate_failure": "kill_routing_paper",
        "loo_gate_failure": "kill_anchorcv",
        "prototype_gate_pass": "run_extension_bundles_2_and_3",
        "allowed_verdicts": ["revise", "kill", "proceed"],
        "prototype_verdict_rule": (
            "revise_if_incomplete_or_invalid_else_kill_if_any_seven_gate_fails"
            "_else_proceed"
        ),
        "extension_verdict_rule": (
            "revise_if_incomplete_or_invalid_else_kill_if_any_seven_or"
            "_uncertainty_gate_fails_else_proceed"
        ),
        "final_gate_verdict_rule": (
            "revise_if_incomplete_or_invalid_else_kill_if_any_seven_or"
            "_uncertainty_gate_fails_else_proceed"
        ),
        "undefined_metric_on_complete_evidence": (
            "kill_scientific_gate_failure"
        ),
        "replacement_seeds_from_outcomes": "forbidden",
        "failed_or_missing_jobs": "retain_and_report",
        "post_final_gate_adaptation": "forbidden",
    }
    assert payload["integrity"] == {
        "hash_algorithm": "sha256",
        "required_manifest_fields": [
            "source_tree_sha256",
            "config_sha256",
            "data_sha256",
            "result_sha256",
        ],
        "source_tree_sha256_required": True,
        "config_sha256_required": True,
        "data_sha256_required": True,
        "result_sha256_required": True,
        "data_hash_scope": "canonical_parsed_permitted_split_arrays_only",
        "whole_raw_csv_hashing": False,
        "source_change_after_freeze": "new_protocol_version_and_rerun_affected_pairs",
        "git_available_required": False,
        "git_commit_required": None,
        "method_freeze_authority_issuance": (
            "only_after_verified_multibundle_proceed"
        ),
        "method_freeze_authority_binds": [
            "source_tree_sha256",
            "config_sha256",
            "six_tune_result_file_sha256",
            "six_tune_manifest_file_sha256",
            "six_neural_checkpoint_file_sha256",
            "six_neural_checkpoint_tensor_sha256",
            "six_acil_checkpoint_sha256",
        ],
        "method_freeze_authority_immutable": True,
    }
    assert payload["protocol"]["git_available"] is False
    assert payload["protocol"]["git_commit"] is None


def test_protocol_exposes_no_discovery_test_or_split_override_authority():
    payload = protocol.load_protocol().to_dict()

    assert payload["evidence_boundary"]["branch_sealed"] is True
    assert payload["evidence_boundary"]["pristine_project_wide_holdout"] is False
    assert payload["evidence_boundary"]["test_access"] is False
    assert payload["evidence_boundary"]["legacy_test_cache_access"] is False
    assert payload["evidence_boundary"]["runner_exposes_test_path"] is False
    assert payload["evidence_boundary"]["runner_exposes_split_override"] is False
    assert "test" not in payload["evidence_boundary"]["permitted_splits"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.__setitem__("surprise", 1),
            "protocol config keys drifted",
        ),
        (
            lambda value: value["experts"]["n"].__setitem__("test_path", "x"),
            "forbidden discovery authority",
        ),
        (
            lambda value: value["evidence_boundary"].__setitem__(
                "split_override", "test"
            ),
            "forbidden discovery authority",
        ),
        (
            lambda value: value["evidence_boundary"].__setitem__(
                "test_access", True
            ),
            "test_access",
        ),
        (
            lambda value: value["evidence_boundary"]["permitted_splits"].append(
                "test"
            ),
            "permitted_splits",
        ),
        (
            lambda value: value.__setitem__("datasets", ["abilene"]),
            "datasets",
        ),
        (
            lambda value: value["masks"].__setitem__(
                "observed_points_per_flow", 4
            ),
            "observed_points_per_flow",
        ),
        (
            lambda value: value["experts"]["n"]["inputs"].append("acil_output"),
            "prior-free",
        ),
        (
            lambda value: value["stages"].__setitem__(
                "prototype_bundles", [1, 2]
            ),
            "prototype_bundles",
        ),
        (
            lambda value: value["gates"].__setitem__(
                "loo_winner_auroc_min", 0.59
            ),
            "gate",
        ),
        (
            lambda value: value["gates"].__setitem__(
                "hard_anchorcv_actual_improvement_min", math.nan
            ),
            "gate",
        ),
        (
            lambda value: value["integrity"].__setitem__(
                "result_sha256_required", False
            ),
            "integrity",
        ),
        (
            lambda value: value["protocol"].__setitem__(
                "git_available", True
            ),
            "git",
        ),
        (
            lambda value: value["uncertainty"].__setitem__(
                "replicates", 9999
            ),
            "uncertainty",
        ),
        (
            lambda value: value["final_gate"].__setitem__(
                "training", "allowed"
            ),
            "final_gate",
        ),
    ],
)
def test_validation_rejects_drift_unknown_keys_and_test_authority(
    mutate, message
):
    payload = copy.deepcopy(protocol.load_protocol().to_dict())
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        protocol.validate_protocol(payload)


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "../sealed_test.csv",
        "/tmp/test/cache.npz",
        "results/old_test_metrics.json",
    ],
)
def test_validation_rejects_test_paths_even_under_an_unknown_neutral_key(
    unsafe_value,
):
    payload = protocol.load_protocol().to_dict()
    payload["experts"]["n"]["artifact"] = unsafe_value

    with pytest.raises(ValueError, match="forbidden discovery authority"):
        protocol.validate_protocol(payload)


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "../artifacts/model.pt",
        "/tmp/artifacts/model.pt",
        r"C:\artifacts\model.pt",
        "results/model.pt",
    ],
)
def test_validation_is_path_free_even_for_non_test_paths(unsafe_value):
    payload = protocol.load_protocol().to_dict()
    payload["experts"]["n"]["artifact"] = unsafe_value

    with pytest.raises(ValueError, match="path-free"):
        protocol.validate_protocol(payload)
