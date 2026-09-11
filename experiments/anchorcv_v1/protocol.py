"""Strict, path-free protocol identity for the AnchorCV v1 experiment.

The adjacent ``research_card.yaml`` deliberately uses the JSON subset of YAML.
That keeps the frozen card human-readable while allowing strict parsing without
an optional YAML dependency.  This module validates the complete card before
returning an immutable dataclass and exposes a deterministic configuration
fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "anchorcv-v1"
_CARD_PATH = Path(__file__).with_name("research_card.yaml")
_ROOT_KEYS = {
    "schema_version",
    "protocol",
    "question",
    "evidence_boundary",
    "datasets",
    "cohorts",
    "masks",
    "stages",
    "experts",
    "evaluation",
    "uncertainty",
    "gates",
    "final_gate",
    "stop_rules",
    "integrity",
}
_SAFE_TEST_SENTINELS = {
    "test_access",
    "legacy_test_cache_access",
    "runner_exposes_test_path",
    "runner_exposes_split_override",
}
_TEST_PATH_TOKEN = re.compile(r"(^|[/_.-])test([/_.-]|$)", re.IGNORECASE)
_ARTIFACT_FILENAME_TOKEN = re.compile(
    r"\.(?:bin|ckpt|csv|json|npz|npy|pt|pth|yaml|yml)$",
    re.IGNORECASE,
)

_PROTOCOL = {
    "id": PROTOCOL_ID,
    "status": "frozen_full_pipeline",
    "claim_scope": (
        "within_instance_anchor_cross_validation_for_traffic_matrix_completion"
    ),
    "git_available": False,
    "git_commit": None,
}
_QUESTION = (
    "Can within-instance held-out-anchor errors identify, per flow, whether a "
    "frozen ACIL prior or a prior-free mask-native neural expert will have "
    "lower error on the fixed missing targets?"
)
_EVIDENCE_BOUNDARY = {
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
_DATASETS = ["abilene", "geant"]
_COHORTS = {
    "fit": "neural_expert_parameter_fitting_only",
    "source_dev": "checkpoint_selection_only",
    "tune": "prototype_extension_and_multibundle_adjudication_only",
    "final_gate": (
        "final_only_confirmation_after_verified_method_freeze_authority"
    ),
}
_MASKS = {
    "families": ["random", "internal_block", "two_burst"],
    "structured_families": ["internal_block", "two_burst"],
    "observed_points_per_flow": 3,
    "headline_target_scope": "fixed_k3_missing_complement",
    "mask_identity": "dataset_cohort_window_family_bundle_deterministic",
}
_STAGES = {
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
_EXPERT_P = {
    "identity": "seed_matched_frozen_acil",
    "checkpoint_selection": "source_dev_only",
    "bundle_matching": "exact",
    "training": "forbidden_in_anchorcv_stage",
    "hard_project_observations": True,
}
_EXPERT_N = {
    "identity": "prior_free_mask_native_neural",
    "inputs": [
        "observed_values",
        "observation_mask",
        "normalized_time_coordinate",
        "flow_id",
    ],
    "forbidden_inputs": [
        "linear_interpolation",
        "acil_output",
        "missing_truth",
    ],
    "architecture": {
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
    },
    "normalization": {
        "location_scale": "per_window_flow_observation_only_mean_population_std",
        "degenerate_fallback": "fit_only_dataset_mean_std",
        "missing_payload_before_model": 0,
        "normalized_time_range": [-1.0, 1.0],
    },
    "training": {
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
        "epsilon": 1e-8,
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
            "k3_missing_weight": 0.50,
            "k2_all_missing_weight": 0.50,
        },
        "anchor_drop_schedule": "fixed_sorted_middle_rank_1",
    },
    "training_masks": ["k3", "k2_anchor_drop"],
    "anchor_drop_ranks": [1],
    "hard_project_observations": True,
}
_EVALUATION = {
    "anchor_cross_validation": {
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
    },
    "metrics": [
        "ratio_of_sums_nmae",
        "loo_winner_auroc",
        "loo_target_regret_spearman",
        "oracle_gap_capture",
        "paired_per_run_relative_improvement",
    ],
    "primary_aggregation": "dataset_equal_structured_masks",
    "error_aggregation": "ratio_of_sums",
    "uncertainty": "frozen_two_level_bundle_and_window_bootstrap",
    "cell_definition": "dataset_by_structured_mask_family",
}
_UNCERTAINTY = {
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
        "independent_per": "sampled_bundle_occurrence_by_dataset",
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
_GATES = {
    "structured_per_flow_oracle_dataset_equal_min": 0.03,
    "loo_winner_auroc_min": 0.60,
    "loo_target_regret_spearman_min": 0.15,
    "hard_anchorcv_oracle_gap_capture_min": 0.30,
    "hard_anchorcv_actual_improvement_min": 0.005,
    "worst_structured_oracle_cell_relative_improvement_min": 0.015,
    "worst_structured_cell_relative_improvement_min": -0.01,
}
_FINAL_GATE = {
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
_STOP_RULES = {
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
_INTEGRITY = {
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
    "source_change_after_freeze": (
        "new_protocol_version_and_rerun_affected_pairs"
    ),
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


@dataclass(frozen=True, slots=True)
class GateSpec:
    """Frozen numerical method-level gates for the prototype decision."""

    structured_per_flow_oracle_dataset_equal_min: float
    loo_winner_auroc_min: float
    loo_target_regret_spearman_min: float
    hard_anchorcv_oracle_gap_capture_min: float
    hard_anchorcv_actual_improvement_min: float
    worst_structured_oracle_cell_relative_improvement_min: float
    worst_structured_cell_relative_improvement_min: float


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """Immutable high-use view backed by the complete canonical research card."""

    protocol_id: str
    status: str
    question: str
    datasets: tuple[str, ...]
    permitted_splits: tuple[str, ...]
    mask_families: tuple[str, ...]
    structured_mask_families: tuple[str, ...]
    observed_points_per_flow: int
    prototype_bundles: tuple[int, ...]
    extension_bundles: tuple[int, ...]
    final_gate_bundles: tuple[int, ...]
    final_gate_job_count: int
    bootstrap_seed: int
    bootstrap_replicates: int
    positive_bundle_count_min: int
    gates: GateSpec
    required_hash_fields: tuple[str, ...]
    _canonical_payload: bytes = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated JSON object; callers cannot mutate this spec."""

        decoded = json.loads(self._canonical_payload)
        if not isinstance(decoded, dict):  # defensive; validation guarantees it
            raise RuntimeError("validated protocol payload ceased to be an object")
        return decoded


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} keys drifted; missing={missing!r}, extra={extra!r}"
        )


def _reject_forbidden_discovery_authority(
    value: object, *, location: str = "protocol config"
) -> None:
    """Reject hidden path/split authority before ordinary schema validation."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{location} keys must be strings")
            key = raw_key.casefold()
            if key not in _SAFE_TEST_SENTINELS and (
                "test" in key
                or "split_override" in key
                or key in {"raw_path", "data_path", "cache_path"}
            ):
                raise ValueError(
                    f"forbidden discovery authority at {location}.{raw_key}"
                )
            _reject_forbidden_discovery_authority(
                child, location=f"{location}.{raw_key}"
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_discovery_authority(
                child, location=f"{location}[{index}]"
            )
        return
    if isinstance(value, str):
        path_like = "/" in value or "\\" in value or "." in value
        if path_like and _TEST_PATH_TOKEN.search(value):
            raise ValueError(
                f"forbidden discovery authority: test-like path at {location}"
            )


def _reject_paths(
    value: object, *, location: str = "protocol config"
) -> None:
    """Keep the research identity declarative and independent of host paths."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            _reject_paths(child, location=f"{location}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_paths(child, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and (
        "/" in value
        or "\\" in value
        or value in {".", ".."}
        or _ARTIFACT_FILENAME_TOKEN.search(value) is not None
    ):
        raise ValueError(
            f"protocol config must be path-free; path-like value at {location}"
        )


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("protocol config must contain finite JSON values") from exc
    return encoded.encode("utf-8")


def _require_frozen(value: object, expected: object, label: str) -> None:
    if _canonical_json(value) != _canonical_json(expected):
        raise ValueError(f"{label} drifted from the frozen AnchorCV v1 protocol")


def _validate_numeric_gates(gates: Mapping[str, Any]) -> None:
    _require_exact_keys(gates, set(_GATES), "gate")
    for name, value in gates.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"gate {name!r} must be a finite number")
    _require_frozen(gates, _GATES, "gate")


def validate_protocol(payload: object) -> ProtocolSpec:
    """Validate the complete frozen card without reading data or model files."""

    root = _require_mapping(payload, "protocol config")
    _reject_forbidden_discovery_authority(root)
    _reject_paths(root)
    _require_exact_keys(root, _ROOT_KEYS, "protocol config")

    if root.get("schema_version") != 1 or isinstance(
        root.get("schema_version"), bool
    ):
        raise ValueError("schema_version must be exactly integer 1")

    identity = _require_mapping(root.get("protocol"), "protocol")
    _require_exact_keys(identity, set(_PROTOCOL), "protocol")
    if identity.get("git_available") is not False:
        raise ValueError("git_available must be false")
    if identity.get("git_commit") is not None:
        raise ValueError("git_commit must be null")
    _require_frozen(identity, _PROTOCOL, "protocol")

    if root.get("question") != _QUESTION:
        raise ValueError("question drifted from the frozen falsifiable question")

    boundary = _require_mapping(
        root.get("evidence_boundary"), "evidence_boundary"
    )
    _require_exact_keys(
        boundary, set(_EVIDENCE_BOUNDARY), "evidence_boundary"
    )
    for field_name in (
        "test_access",
        "legacy_test_cache_access",
        "whole_raw_csv_access",
        "runner_exposes_test_path",
        "runner_exposes_split_override",
    ):
        if boundary.get(field_name) is not False:
            raise ValueError(f"{field_name} must be false")
    if boundary.get("permitted_splits") != ["fit", "source_dev", "tune"]:
        raise ValueError(
            "permitted_splits must be exactly fit/source_dev/tune"
        )
    _require_frozen(
        boundary, _EVIDENCE_BOUNDARY, "evidence_boundary"
    )

    if root.get("datasets") != _DATASETS:
        raise ValueError("datasets must be exactly abilene and geant")
    _require_frozen(root.get("cohorts"), _COHORTS, "cohorts")

    masks = _require_mapping(root.get("masks"), "masks")
    _require_exact_keys(masks, set(_MASKS), "masks")
    if masks.get("observed_points_per_flow") != 3 or isinstance(
        masks.get("observed_points_per_flow"), bool
    ):
        raise ValueError("observed_points_per_flow must be exactly integer 3")
    _require_frozen(masks, _MASKS, "masks")

    stages = _require_mapping(root.get("stages"), "stages")
    _require_exact_keys(stages, set(_STAGES), "stages")
    if stages.get("prototype_bundles") != [1]:
        raise ValueError("prototype_bundles must be exactly [1]")
    if stages.get("extension_bundles") != [2, 3]:
        raise ValueError("extension_bundles must be exactly [2, 3]")
    _require_frozen(stages, _STAGES, "stages")

    experts = _require_mapping(root.get("experts"), "experts")
    _require_exact_keys(experts, {"p", "n"}, "experts")
    expert_p = _require_mapping(experts.get("p"), "experts.p")
    expert_n = _require_mapping(experts.get("n"), "experts.n")
    _require_exact_keys(expert_p, set(_EXPERT_P), "experts.p")
    _require_exact_keys(expert_n, set(_EXPERT_N), "experts.n")
    if expert_n.get("inputs") != _EXPERT_N["inputs"]:
        raise ValueError("prior-free expert inputs drifted")
    if expert_n.get("forbidden_inputs") != _EXPERT_N["forbidden_inputs"]:
        raise ValueError("prior-free expert forbidden inputs drifted")
    _require_frozen(expert_p, _EXPERT_P, "expert P")
    _require_frozen(expert_n, _EXPERT_N, "prior-free expert N")

    _require_frozen(root.get("evaluation"), _EVALUATION, "evaluation")

    uncertainty = _require_mapping(root.get("uncertainty"), "uncertainty")
    _require_exact_keys(uncertainty, set(_UNCERTAINTY), "uncertainty")
    _require_frozen(uncertainty, _UNCERTAINTY, "uncertainty")

    gates = _require_mapping(root.get("gates"), "gates")
    _validate_numeric_gates(gates)

    final_gate = _require_mapping(root.get("final_gate"), "final_gate")
    _require_exact_keys(final_gate, set(_FINAL_GATE), "final_gate")
    _require_frozen(final_gate, _FINAL_GATE, "final_gate")

    _require_frozen(root.get("stop_rules"), _STOP_RULES, "stop_rules")

    integrity = _require_mapping(root.get("integrity"), "integrity")
    _require_exact_keys(integrity, set(_INTEGRITY), "integrity")
    _require_frozen(integrity, _INTEGRITY, "integrity")

    canonical = _canonical_json(root)
    return ProtocolSpec(
        protocol_id=str(identity["id"]),
        status=str(identity["status"]),
        question=str(root["question"]),
        datasets=tuple(root["datasets"]),
        permitted_splits=tuple(boundary["permitted_splits"]),
        mask_families=tuple(masks["families"]),
        structured_mask_families=tuple(masks["structured_families"]),
        observed_points_per_flow=int(masks["observed_points_per_flow"]),
        prototype_bundles=tuple(stages["prototype_bundles"]),
        extension_bundles=tuple(stages["extension_bundles"]),
        final_gate_bundles=tuple(stages["final_gate_bundles"]),
        final_gate_job_count=int(stages["final_gate_job_count"]),
        bootstrap_seed=int(uncertainty["seed"]),
        bootstrap_replicates=int(uncertainty["replicates"]),
        positive_bundle_count_min=int(
            uncertainty["positive_bundle_count_min"]
        ),
        gates=GateSpec(**{name: float(gates[name]) for name in _GATES}),
        required_hash_fields=tuple(integrity["required_manifest_fields"]),
        _canonical_payload=canonical,
    )


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token!r} is forbidden")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate protocol key {key!r}")
        result[key] = value
    return result


def load_protocol(path: str | Path | None = None) -> ProtocolSpec:
    """Load and validate the JSON-compatible YAML research card."""

    card_path = _CARD_PATH if path is None else Path(path)
    try:
        text = card_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read protocol card {card_path}") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "research_card.yaml must remain in the strict JSON subset of YAML"
        ) from exc
    return validate_protocol(payload)


def canonical_json_bytes(
    spec: ProtocolSpec | Mapping[str, Any] | None = None,
) -> bytes:
    """Return canonical bytes for hashing and immutable manifest identity."""

    if spec is None:
        return load_protocol()._canonical_payload
    if isinstance(spec, ProtocolSpec):
        return spec._canonical_payload
    return validate_protocol(spec)._canonical_payload


def fingerprint(
    spec: ProtocolSpec | Mapping[str, Any] | None = None,
) -> str:
    """Return the frozen configuration SHA-256."""

    return hashlib.sha256(canonical_json_bytes(spec)).hexdigest()


def protocol_sha256() -> str:
    """Compatibility-style explicit name for manifest builders."""

    return fingerprint()
