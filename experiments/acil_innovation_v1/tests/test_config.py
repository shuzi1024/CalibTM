from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.acil_innovation_v1.config import (
    load_protocol_config,
    protocol_config_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_protocol_identity_and_freeze_boundary_are_exact():
    config = load_protocol_config()
    legacy_absolute_key = "no_" + "overwrite"

    assert config["schema_version"] == 1
    assert config["protocol"] == {
        "branch_sealed": True,
        "id": "acil-innovation-v1",
        "status": "draft_protocol",
        "window_length": 50,
    }
    assert config["freeze"]["git_available"] is False
    assert config["freeze"]["git_commit"] is None
    assert config["freeze"]["allow_latest"] is False
    assert legacy_absolute_key not in config["freeze"]
    assert config["freeze"]["publication_contract"] == (
        "protocol_controlled_single_publisher_no_replace"
    )
    assert config["evidence_boundary"]["project_wide_pristine_holdout"] is False
    assert config["evidence_boundary"]["discovery_test_access"] is False


def test_publication_contract_rejects_absolute_or_ambiguous_variants():
    from experiments.acil_innovation_v1.config import _validate_config

    source = load_protocol_config()
    legacy_absolute_key = "no_" + "overwrite"
    source["freeze"].pop(legacy_absolute_key, None)
    source["freeze"]["publication_contract"] = (
        "protocol_controlled_single_publisher_no_replace"
    )
    validated = _validate_config(source)
    assert validated["freeze"]["publication_contract"] == (
        "protocol_controlled_single_publisher_no_replace"
    )

    for invalid in (
        "unscoped_no_replace",
        "single_publisher",
        True,
        None,
    ):
        mutated = json.loads(json.dumps(source))
        mutated["freeze"]["publication_contract"] = invalid
        with pytest.raises(ValueError, match="publication contract"):
            _validate_config(mutated)

    legacy = json.loads(json.dumps(source))
    legacy["freeze"][legacy_absolute_key] = True
    with pytest.raises(ValueError, match="publication contract|freeze registry"):
        _validate_config(legacy)


def test_config_registers_only_k3_and_fixed_mixed_mask_semantics():
    config = load_protocol_config()
    masks = config["masks"]

    assert masks["observed_count"] == 3
    assert masks["families"] == ["random", "internal_block", "two_burst"]
    assert masks["evaluation_replica"] == 0
    assert masks["k5_registered"] is False
    assert config["innovation"]["loo_anchor_rule"] == "sorted_observed_middle"
    assert config["innovation"]["loo_anchor_count"] == 1
    assert config["innovation"]["temporary_observed_count"] == 2

    mixed = config["training"]["mixed_masks"]
    assert mixed["training_job_has_mask_axis"] is False
    assert mixed["one_checkpoint_per_dataset_seed_method"] is True
    assert mixed["family_formula"] == (
        "families[(epoch*512+epoch_order_position+bundle-1)%3]"
    )
    assert config["training"]["checkpoint"] == {
        "metric": "source_dev_three_mask_raw_ratio_of_sums_nmae",
        "tie_break": "earlier_epoch",
    }


def test_optimizer_and_residual_target_semantics_are_complete():
    config = load_protocol_config()
    optimizer = config["training"]["optimizer"]

    assert optimizer["type"] == "AdamW"
    assert optimizer["betas"] == [0.9, 0.999]
    assert optimizer["epsilon"] == 1e-8
    assert optimizer["lr_schedule"] == "constant"
    assert optimizer["learning_rates"] == {
        "new_modules": 3e-4,
        "scratch_gpt_blocks": 3e-4,
        "pretrained_gpt_blocks": 1e-5,
    }
    assert optimizer["no_decay_rules"] == [
        "parameter_name_endswith_bias",
        "parameter_belongs_to_layernorm_and_name_endswith_weight",
    ]
    assert optimizer["weight_decay_applies_to"] == "all_other_trainable_parameters"

    targets = config["training"]["residual_target_contracts"]
    assert targets["stage_h_truth_q_deepsets"] == {
        "loss_target": "E=unobserved_complement_minus_support",
        "checkpoint_target": "E=unobserved_complement_minus_support",
    }
    assert targets["deployable_residual_methods"] == {
        "methods": [
            "local_loo",
            "global_loo",
            "full_u0",
            "loo_deepsets",
            "full_scratch",
            "full_gpt2",
        ],
        "loss_target": "U=unobserved_complement",
        "checkpoint_target": "U=unobserved_complement",
    }


def test_oracle_and_deployable_target_contracts_are_separate():
    config = load_protocol_config()
    oracle = config["oracle_partition"]

    assert oracle["support_count_per_flow"] == 1
    assert oracle["support_pool"] == "unobserved_complement"
    assert oracle["structured_preference"] == "outside_controlled_gap_then_all_missing"
    assert oracle["evaluation_set"] == "unobserved_complement_minus_support"
    assert oracle["support_truth_role"] == "diagnostic_only"
    assert oracle["method_dependent_identity"] is False
    assert config["deployable_partition"] == {
        "oracle_support_count": 0,
        "evaluation_set": "unobserved_complement",
        "target_count_per_flow": 47,
    }


def test_stage_methods_and_numeric_gates_are_frozen():
    config = load_protocol_config()

    assert config["stages"]["stage_h"]["methods"] == [
        "acil",
        "truth_q_deepsets",
    ]
    structured = ["internal_block", "two_burst"]
    assert config["gates"]["stage_h"]["effect_contrasts"]["main"] == {
        "candidate": "truth_q_deepsets",
        "comparator": "acil",
        "pooling_masks": structured,
        "minimum": 0.05,
    }

    assert config["stages"]["stage_i"]["methods"] == [
        "acil",
        "local_loo",
        "global_loo",
    ]
    assert config["gates"]["stage_i"]["effect_contrasts"]["main"][
        "minimum"
    ] == 0.015
    assert config["gates"]["stage_i"]["effect_contrasts"]["global_over_local"] == {
        "candidate": "global_loo",
        "comparator": "local_loo",
        "pooling_masks": structured,
        "minimum": 0.005,
    }

    assert config["stages"]["full_tune"]["methods"] == [
        "acil",
        "full_u0",
        "loo_deepsets",
        "full_scratch",
        "full_gpt2",
    ]
    assert config["gates"]["full_tune"]["effect_contrasts"]["main"][
        "minimum"
    ] == 0.02
    assert config["gates"]["full_tune"]["effect_contrasts"]["over_deepsets"][
        "minimum"
    ] == 0.005
    assert config["stages"]["formal_gate"]["seed_bundles"] == [4, 5, 6]


def test_every_gate_effect_contrast_has_an_explicit_mask_scope():
    config = load_protocol_config()
    structured = ["internal_block", "two_burst"]

    for gate in config["gates"].values():
        for contrast in gate["effect_contrasts"].values():
            assert contrast["pooling_masks"] in (
                structured,
                ["random"],
                ["random", "internal_block", "two_burst"],
            )

    for stage in (
        "stage0_acil_tune",
        "stage_h",
        "stage_i",
        "full_tune",
        "formal_gate",
    ):
        assert config["gates"][stage]["effect_contrasts"]["main"][
            "pooling_masks"
        ] == structured
        assert config["gates"][stage]["bootstrap_ci"]["effect_contrast"] == "main"
        assert config["gates"][stage]["positive_seed_rule"][
            "effect_contrast"
        ] == "main"
        assert config["gates"][stage]["positive_seed_rule"]["required"] == 2

    assert config["metrics"]["positive_seed_definition"] == (
        "paired_improvement_of_ratio_of_sums_pooled_over_both_datasets_and_"
        "effect_pooling_masks_for_one_seed_bundle_strictly_above_zero_before_rounding"
    )


def test_stage_i_derangement_is_fixed_outcome_independent_and_tune_only():
    config = load_protocol_config()
    derangement = config["stages"]["stage_i"]["derangement"]

    assert derangement["cohort"] == "tune"
    assert derangement["retraining"] is False
    assert derangement["rng"] == "PCG64DXSM"
    assert derangement["domain"] == "acil-innovation-v1:stage-i-derangement:v1"
    assert derangement["seed_derivation"] == (
        "sha256_domain_nul_canonical_identity_first_128_bits_big_endian"
    )
    assert derangement["identity_fields"] == [
        "protocol",
        "dataset",
        "cohort",
        "window_schedule_sha256",
        "seed_bundle",
        "mask_family",
        "evaluation_mask_seed",
        "registered_window_count",
        "flow_count",
    ]
    assert derangement["joint_permutation_axis"] == (
        "all_registered_window_flow_scalar_loo_innovations"
    )
    assert derangement["fixed_fields"] == [
        "query_geometry",
        "middle_indicator",
        "truth",
        "targets",
    ]
    assert derangement["method_outcome_in_identity"] is False

    contrast = config["gates"]["stage_i"]["effect_contrasts"][
        "derangement_gain_loss"
    ]
    assert contrast["candidate"] == "global_loo"
    assert contrast["comparator"] == "global_loo_deranged"
    assert contrast["pooling_masks"] == ["internal_block", "two_burst"]
    assert contrast["gain_loss_formula"] == (
        "unshuffled_improvement_over_acil_minus_deranged_improvement_over_acil"
    )
    assert contrast["fraction_formula"] == (
        "gain_loss_divided_by_unshuffled_improvement_over_acil"
    )
    assert contrast["requires_unshuffled_main_gain_strictly_above"] == 0.0
    assert contrast["minimum"] == 0.005
    assert contrast["fraction_minimum"] == 0.30


def test_formal_external_no_harm_pools_all_masks_and_each_dataset():
    config = load_protocol_config()
    external = config["gates"]["formal_gate"]["effect_contrasts"][
        "external_no_harm"
    ]

    assert external["candidate"] == "full_gpt2"
    assert external["comparator_candidates"] == ["ari_llm", "imputeformer"]
    assert external["comparator_selection"] == (
        "lowest_ratio_of_sums_nmae_on_same_pooling_masks"
    )
    assert external["pooling_masks"] == [
        "random",
        "internal_block",
        "two_burst",
    ]
    assert external["minimum"] == -0.005
    assert external["per_dataset_minimum"] == -0.005


def test_config_contains_no_raw_test_or_split_override_authority():
    config = load_protocol_config()
    keys = tuple(_walk_keys(config))

    forbidden_exact = {
        "raw_path",
        "raw_csv",
        "test_path",
        "test_split",
        "split_override",
        "data_path",
    }
    assert forbidden_exact.isdisjoint(keys)
    encoded = json.dumps(config, sort_keys=True).lower()
    assert "_test.npz" not in encoded
    assert "test_cache" not in encoded
    assert "ARI-LLM-E7A3" not in encoded


def test_config_semantic_hash_is_stable_and_source_pinned():
    first = load_protocol_config()
    first["protocol"]["status"] = "mutated-by-caller"
    second = load_protocol_config()

    assert second["protocol"]["status"] == "draft_protocol"
    assert len(protocol_config_sha256()) == 64
    assert protocol_config_sha256() == protocol_config_sha256()
    assert (ROOT / "configs" / "protocol_v1.json").is_file()
