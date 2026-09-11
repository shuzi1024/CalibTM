"""Pure-payload grid audit and frozen-gate adjudication for AnchorCV v1.

No directory is searched and no evidence/data array is opened here.  Each
input item must contain the already-decoded ``result.json`` and
``manifest.json`` payload for one job.  The manifest, result, checkpoints,
three per-mask evidence identities, and recomputable summary fields are bound
before any method gate is interpreted.

Headline values are arithmetic means over the two structured cells within
each dataset and then over the two datasets.  Thus every dataset and every
structured dataset/mask cell has equal weight.  ``random`` is reported but
does not enter a gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from .protocol import fingerprint, load_protocol


_REPORT_SCHEMA = "anchorcv-v1:prototype-adjudication:v1"
_PROTOCOL_ID = "anchorcv-v1"
_DATASETS = ("abilene", "geant")
_BUNDLE = 1
_MASKS = ("random", "internal_block", "two_burst")
_STRUCTURED_MASKS = ("internal_block", "two_burst")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_SCIENTIFIC_IDENTITY_FIELDS = {
    "acil_checkpoint_file_sha256",
    "config_sha256",
    "compute_lane",
    "data_sha256",
    "dataset",
    "evaluation_cohort",
    "flows",
    "git_available",
    "git_commit",
    "mask_families",
    "model_seed",
    "parsed_array_sha256",
    "protocol",
    "seed_bundle",
    "source_tree_sha256",
    "stage",
    "test_access",
    "training_cohort",
    "checkpoint_selection_cohort",
}
_RESULT_FIELDS = {
    "job_id",
    "status",
    "scientific_identity",
    "neural_parameter_count",
    "neural_checkpoint",
    "prior_checkpoint",
    "training",
    "cells",
}
_MANIFEST_FIELDS = _SCIENTIFIC_IDENTITY_FIELDS | {
    "job_id",
    "status",
    "result_file",
    "result_sha256",
    "neural_checkpoint_file_sha256",
    "neural_checkpoint_tensor_sha256",
}
_CELL_FIELDS = {
    "summary",
    "evidence_file",
    "evidence_file_sha256",
    "evidence_content_sha256",
    "mask_sha256",
    "oracle_q_sha256",
    "oracle_e_sha256",
}
_CELL_HASH_FIELDS = (
    "evidence_file_sha256",
    "evidence_content_sha256",
    "mask_sha256",
    "oracle_q_sha256",
    "oracle_e_sha256",
)
_SUMMARY_METRICS = (
    "oracle_relative_improvement",
    "loo_winner_auroc",
    "loo_target_regret_spearman",
    "oracle_gap_capture",
    "hard_relative_improvement",
)
_SUMMARY_BASE_FIELDS = {
    "case_count",
    "target_count",
    "absolute_truth_sum",
    "p_absolute_error_sum",
    "n_absolute_error_sum",
    "hard_absolute_error_sum",
    "oracle_absolute_error_sum",
    "p_nmae",
    "n_nmae",
    "hard_nmae",
    "oracle_nmae",
    "best_single_expert",
    "best_single_nmae",
    "oracle_relative_improvement",
    "hard_relative_improvement",
    "neural_selection_rate",
    "oracle_neural_rate",
    "loo_winner_auroc",
    "loo_target_regret_spearman",
    "oracle_gap_capture",
}
_UNDEFINED_REASON = {
    "loo_winner_auroc": "loo_winner_auroc_undefined_reason",
    "loo_target_regret_spearman": (
        "loo_target_regret_spearman_undefined_reason"
    ),
    "oracle_gap_capture": "oracle_gap_capture_undefined_reason",
}


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _issue(issues: list[str], message: str) -> None:
    if message not in issues:
        issues.append(message)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _exact_fields(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
    issues: list[str],
) -> bool:
    actual = set(payload)
    if actual == expected:
        return True
    _issue(
        issues,
        f"{label}: fields drifted; "
        f"missing={sorted(expected - actual)!r}, "
        f"extra={sorted(actual - expected)!r}",
    )
    return False


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _positive_int(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value > 0
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _validate_scientific_identity(
    identity: Mapping[str, object],
    *,
    expected_config_sha256: str,
    label: str,
    issues: list[str],
) -> tuple[object, object]:
    _exact_fields(
        identity,
        _SCIENTIFIC_IDENTITY_FIELDS,
        label=label,
        issues=issues,
    )
    for field in (
        "acil_checkpoint_file_sha256",
        "config_sha256",
        "data_sha256",
        "source_tree_sha256",
    ):
        if not _is_sha256(identity.get(field)):
            _issue(issues, f"{label}: {field} is not a lowercase SHA-256")
    parsed = _as_mapping(identity.get("parsed_array_sha256"))
    if parsed is None or set(parsed) != {"train", "val"}:
        _issue(issues, f"{label}: parsed-array identity drifted")
    else:
        for split in ("train", "val"):
            if not _is_sha256(parsed.get(split)):
                _issue(
                    issues,
                    f"{label}: parsed {split} identity is not a lowercase SHA-256",
                )
    if identity.get("config_sha256") != expected_config_sha256:
        _issue(issues, f"{label}: config hash differs from frozen protocol")
    fixed = {
        "protocol": _PROTOCOL_ID,
        "seed_bundle": _BUNDLE,
        "stage": "prototype",
        "evaluation_cohort": "tune",
        "training_cohort": "fit",
        "checkpoint_selection_cohort": "source_dev",
        "compute_lane": "cuda_bf16_math_sdp",
        "test_access": False,
        "git_available": False,
        "git_commit": None,
        "mask_families": list(_MASKS),
    }
    for field, expected in fixed.items():
        if identity.get(field) != expected:
            _issue(issues, f"{label}: {field} identity drifted")
    if identity.get("dataset") not in _DATASETS:
        _issue(issues, f"{label}: dataset is outside the frozen registry")
    if not _positive_int(identity.get("flows")):
        _issue(issues, f"{label}: flow count must be a positive integer")
    model_seed = identity.get("model_seed")
    if (
        isinstance(model_seed, bool)
        or not isinstance(model_seed, int)
        or model_seed < 0
    ):
        _issue(issues, f"{label}: model seed must be a nonnegative integer")
    return identity.get("dataset"), identity.get("seed_bundle")


def _validate_summary(
    summary: Mapping[str, object],
    *,
    cell: str,
    issues: list[str],
) -> dict[str, float | None]:
    allowed = _SUMMARY_BASE_FIELDS | set(_UNDEFINED_REASON.values())
    missing = _SUMMARY_BASE_FIELDS - set(summary)
    extra = set(summary) - allowed
    if missing or extra:
        _issue(
            issues,
            f"{cell}: summary fields drifted; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}",
        )

    counts = {}
    for name in ("case_count", "target_count"):
        value = summary.get(name)
        counts[name] = value
        if not _positive_int(value):
            _issue(issues, f"{cell}: {name} must be a positive integer")

    numeric_names = (
        "absolute_truth_sum",
        "p_absolute_error_sum",
        "n_absolute_error_sum",
        "hard_absolute_error_sum",
        "oracle_absolute_error_sum",
        "p_nmae",
        "n_nmae",
        "hard_nmae",
        "oracle_nmae",
        "best_single_nmae",
        "oracle_relative_improvement",
        "hard_relative_improvement",
        "neural_selection_rate",
        "oracle_neural_rate",
    )
    numeric: dict[str, float | None] = {}
    for name in numeric_names:
        value = _finite_number(summary.get(name))
        numeric[name] = value
        if value is None:
            _issue(issues, f"{cell}: {name} must be finite numeric")

    for name in (
        "absolute_truth_sum",
        "p_absolute_error_sum",
        "n_absolute_error_sum",
        "hard_absolute_error_sum",
        "oracle_absolute_error_sum",
        "p_nmae",
        "n_nmae",
        "hard_nmae",
        "oracle_nmae",
        "best_single_nmae",
    ):
        value = numeric[name]
        if value is not None and value < 0.0:
            _issue(issues, f"{cell}: {name} must be nonnegative")
    truth_sum = numeric["absolute_truth_sum"]
    if truth_sum is not None and truth_sum <= 0.0:
        _issue(issues, f"{cell}: absolute truth sum must be positive")

    if truth_sum is not None and truth_sum > 0.0:
        for prefix in ("p", "n", "hard", "oracle"):
            error = numeric[f"{prefix}_absolute_error_sum"]
            nmae = numeric[f"{prefix}_nmae"]
            if (
                error is not None
                and nmae is not None
                and not _close(nmae, error / truth_sum)
            ):
                _issue(
                    issues,
                    f"{cell}: {prefix} NMAE differs from ratio of sums",
                )

    p_nmae = numeric["p_nmae"]
    n_nmae = numeric["n_nmae"]
    hard_nmae = numeric["hard_nmae"]
    oracle_nmae = numeric["oracle_nmae"]
    best_nmae = numeric["best_single_nmae"]
    if p_nmae is not None and n_nmae is not None:
        expected_name = "n" if n_nmae < p_nmae else "p"
        expected_best = min(p_nmae, n_nmae)
        if summary.get("best_single_expert") != expected_name:
            _issue(issues, f"{cell}: best-single expert identity drifted")
        if best_nmae is not None and not _close(best_nmae, expected_best):
            _issue(issues, f"{cell}: best-single NMAE drifted")
    else:
        expected_best = None

    if (
        oracle_nmae is not None
        and expected_best is not None
        and oracle_nmae > expected_best + 1e-12
    ):
        _issue(issues, f"{cell}: oracle is worse than the best single expert")
    if (
        oracle_nmae is not None
        and hard_nmae is not None
        and hard_nmae + 1e-12 < oracle_nmae
    ):
        _issue(issues, f"{cell}: hard method is better than diagnostic oracle")

    for metric, candidate_name in (
        ("oracle_relative_improvement", "oracle_nmae"),
        ("hard_relative_improvement", "hard_nmae"),
    ):
        reported = numeric[metric]
        candidate = numeric[candidate_name]
        if (
            reported is not None
            and candidate is not None
            and expected_best is not None
            and expected_best > 0.0
            and not _close(reported, 1.0 - candidate / expected_best)
        ):
            _issue(issues, f"{cell}: {metric} is not recomputable from NMAE")

    for rate in ("neural_selection_rate", "oracle_neural_rate"):
        value = numeric[rate]
        if value is not None and not 0.0 <= value <= 1.0:
            _issue(issues, f"{cell}: {rate} must lie in [0, 1]")

    parsed: dict[str, float | None] = {
        "oracle_relative_improvement": numeric[
            "oracle_relative_improvement"
        ],
        "hard_relative_improvement": numeric["hard_relative_improvement"],
    }
    for metric in (
        "loo_winner_auroc",
        "loo_target_regret_spearman",
        "oracle_gap_capture",
    ):
        raw = summary.get(metric)
        reason_name = _UNDEFINED_REASON[metric]
        reason = summary.get(reason_name)
        if raw is None:
            parsed[metric] = None
            if not isinstance(reason, str) or not reason:
                _issue(
                    issues,
                    f"{cell}: undefined {metric} lacks its reason",
                )
        else:
            value = _finite_number(raw)
            parsed[metric] = value
            if value is None:
                _issue(
                    issues,
                    f"{cell}: {metric} must be finite numeric or null",
                )
            if reason_name in summary:
                _issue(
                    issues,
                    f"{cell}: defined {metric} has an undefined reason",
                )
            if (
                metric == "loo_winner_auroc"
                and value is not None
                and not 0.0 <= value <= 1.0
            ):
                _issue(issues, f"{cell}: AUROC must lie in [0, 1]")
            if (
                metric == "loo_target_regret_spearman"
                and value is not None
                and not -1.0 <= value <= 1.0
            ):
                _issue(issues, f"{cell}: Spearman must lie in [-1, 1]")

    capture = parsed["oracle_gap_capture"]
    if (
        expected_best is not None
        and oracle_nmae is not None
        and hard_nmae is not None
    ):
        gap = expected_best - oracle_nmae
        if gap > 0.0:
            expected_capture = (expected_best - hard_nmae) / gap
            if capture is None or not _close(capture, expected_capture):
                _issue(
                    issues,
                    f"{cell}: oracle-gap capture is not recomputable from NMAE",
                )
        elif capture is not None:
            _issue(
                issues,
                f"{cell}: oracle-gap capture must be undefined without a gap",
            )
    return parsed


def _metric_aggregate(
    cells: Mapping[tuple[str, str], Mapping[str, float | None]],
    metric: str,
) -> dict[str, object]:
    per_dataset: dict[str, float | None] = {}
    cell_values: dict[str, float | None] = {}
    undefined: list[str] = []
    for dataset in _DATASETS:
        values: list[float] = []
        for mask in _STRUCTURED_MASKS:
            label = f"{dataset}/{mask}"
            row = cells.get((dataset, mask))
            value = None if row is None else row.get(metric)
            cell_values[label] = value
            if value is None:
                undefined.append(label)
            else:
                values.append(value)
        per_dataset[dataset] = (
            math.fsum(values) / len(_STRUCTURED_MASKS)
            if len(values) == len(_STRUCTURED_MASKS)
            else None
        )
    available = [
        value for value in per_dataset.values() if value is not None
    ]
    actual = (
        math.fsum(available) / len(_DATASETS)
        if len(available) == len(_DATASETS)
        else None
    )
    return {
        "actual": actual,
        "per_dataset": per_dataset,
        "cell_values": cell_values,
        "undefined_cells": undefined,
        "aggregation": "structured_cell_equal_within_dataset_then_dataset_equal",
    }


def _worst_cell(metric: Mapping[str, object]) -> dict[str, object]:
    undefined = list(metric["undefined_cells"])
    values = metric["cell_values"]
    if undefined:
        return {
            "actual": None,
            "cell": None,
            "undefined_cells": undefined,
        }
    pairs = [
        (float(value), str(cell))
        for cell, value in values.items()
        if value is not None
    ]
    if len(pairs) != len(_DATASETS) * len(_STRUCTURED_MASKS):
        return {
            "actual": None,
            "cell": None,
            "undefined_cells": sorted(str(cell) for cell in values),
        }
    actual, cell = min(pairs, key=lambda item: (item[0], item[1]))
    return {"actual": actual, "cell": cell, "undefined_cells": []}


def _gate(
    source: Mapping[str, object], threshold: float
) -> dict[str, object]:
    raw = source["actual"]
    actual = _finite_number(raw)
    return {
        "actual": actual,
        "threshold": float(threshold),
        "operator": ">=",
        "pass": bool(actual is not None and actual >= float(threshold)),
        "undefined_cells": list(source["undefined_cells"]),
    }


def adjudicate_prototype(
    job_artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Audit two ``{result, manifest}`` payload pairs and apply frozen gates.

    Missing, failed, duplicate, extra, or identity-invalid evidence yields
    ``revise``.  With a complete valid grid, an undefined metric is an
    explicit scientific gate failure and yields ``kill``.  ``proceed`` means
    every frozen numerical gate passed on the exact prototype grid.
    """

    protocol = load_protocol()
    expected_config_sha256 = fingerprint(protocol)
    if (
        protocol.protocol_id != _PROTOCOL_ID
        or protocol.datasets != _DATASETS
        or protocol.prototype_bundles != (_BUNDLE,)
        or protocol.mask_families != _MASKS
        or protocol.structured_mask_families != _STRUCTURED_MASKS
    ):
        raise RuntimeError("active AnchorCV prototype protocol drifted")

    grid_issues: list[str] = []
    integrity_issues: list[str] = []
    if not _is_sequence(job_artifacts):
        artifacts: list[object] = []
        _issue(grid_issues, "job artifacts must be a sequence")
    else:
        artifacts = list(job_artifacts)

    expected = {(dataset, _BUNDLE) for dataset in _DATASETS}
    key_counts: dict[tuple[object, object], int] = {}
    successful: set[tuple[str, int]] = set()
    cells: dict[tuple[str, str], dict[str, float | None]] = {}
    random_rows: dict[str, dict[str, float | None]] = {}
    manifest_records: list[dict[str, object]] = []

    for index, raw_artifact in enumerate(artifacts):
        artifact_label = f"artifact[{index}]"
        artifact = _as_mapping(raw_artifact)
        if artifact is None:
            _issue(
                integrity_issues,
                f"{artifact_label}: artifact must be an object",
            )
            continue
        _exact_fields(
            artifact,
            {"result", "manifest"},
            label=artifact_label,
            issues=integrity_issues,
        )
        result = _as_mapping(artifact.get("result"))
        manifest = _as_mapping(artifact.get("manifest"))
        if result is None or manifest is None:
            _issue(
                integrity_issues,
                f"{artifact_label}: result and manifest objects are required",
            )
            continue
        _exact_fields(
            result,
            _RESULT_FIELDS,
            label=f"{artifact_label}/result",
            issues=integrity_issues,
        )
        _exact_fields(
            manifest,
            _MANIFEST_FIELDS,
            label=f"{artifact_label}/manifest",
            issues=integrity_issues,
        )

        result_identity = _as_mapping(result.get("scientific_identity"))
        if result_identity is None:
            _issue(
                integrity_issues,
                f"{artifact_label}: result scientific identity is absent",
            )
            result_key = (None, None)
        else:
            result_key = _validate_scientific_identity(
                result_identity,
                expected_config_sha256=expected_config_sha256,
                label=f"{artifact_label}/result identity",
                issues=integrity_issues,
            )

        manifest_identity = {
            field: manifest.get(field)
            for field in _SCIENTIFIC_IDENTITY_FIELDS
        }
        manifest_key = _validate_scientific_identity(
            manifest_identity,
            expected_config_sha256=expected_config_sha256,
            label=f"{artifact_label}/manifest identity",
            issues=integrity_issues,
        )
        key = manifest_key
        key_counts[key] = key_counts.get(key, 0) + 1
        display = f"{key[0]}/bundle{key[1]}"
        if result_key != manifest_key:
            _issue(
                integrity_issues,
                f"{display}: result and manifest job identities differ",
            )
        if result_identity is not None:
            for field in _SCIENTIFIC_IDENTITY_FIELDS:
                if result_identity.get(field) != manifest.get(field):
                    _issue(
                        integrity_issues,
                        f"{display}: result/manifest {field} drifted",
                    )
        if key not in expected:
            _issue(grid_issues, f"unexpected prototype job {display}")

        result_status = result.get("status")
        manifest_status = manifest.get("status")
        if result_status != "succeeded" or manifest_status != "succeeded":
            _issue(
                grid_issues,
                f"{display}: retained status result={result_status!r}, "
                f"manifest={manifest_status!r}",
            )
        result_job_id = result.get("job_id")
        manifest_job_id = manifest.get("job_id")
        if not _is_sha256(result_job_id) or not _is_sha256(manifest_job_id):
            _issue(integrity_issues, f"{display}: job ID is not a SHA-256")
        if result_job_id != manifest_job_id:
            _issue(integrity_issues, f"{display}: result/manifest job ID drifted")
        if manifest.get("result_file") != "result.json":
            _issue(integrity_issues, f"{display}: result filename drifted")
        if not _is_sha256(manifest.get("result_sha256")):
            _issue(integrity_issues, f"{display}: result hash is invalid")

        neural = _as_mapping(result.get("neural_checkpoint"))
        expected_neural_fields = {
            "file",
            "file_sha256",
            "tensor_sha256",
            "best_epoch",
            "best_source_dev_nmae",
        }
        if neural is None:
            _issue(integrity_issues, f"{display}: neural checkpoint is absent")
        else:
            _exact_fields(
                neural,
                expected_neural_fields,
                label=f"{display}/neural checkpoint",
                issues=integrity_issues,
            )
            if neural.get("file") != "neural_best.safetensors":
                _issue(integrity_issues, f"{display}: neural checkpoint filename drifted")
            for result_field, manifest_field in (
                ("file_sha256", "neural_checkpoint_file_sha256"),
                ("tensor_sha256", "neural_checkpoint_tensor_sha256"),
            ):
                value = neural.get(result_field)
                if not _is_sha256(value):
                    _issue(
                        integrity_issues,
                        f"{display}: neural {result_field} is invalid",
                    )
                if value != manifest.get(manifest_field):
                    _issue(
                        integrity_issues,
                        f"{display}: neural {result_field} differs from manifest",
                    )
        for field in (
            "neural_checkpoint_file_sha256",
            "neural_checkpoint_tensor_sha256",
        ):
            if not _is_sha256(manifest.get(field)):
                _issue(integrity_issues, f"{display}: manifest {field} is invalid")

        prior = _as_mapping(result.get("prior_checkpoint"))
        if prior is None or set(prior) != {"file_sha256"}:
            _issue(integrity_issues, f"{display}: prior checkpoint fields drifted")
        elif (
            not _is_sha256(prior.get("file_sha256"))
            or prior.get("file_sha256")
            != manifest.get("acil_checkpoint_file_sha256")
        ):
            _issue(integrity_issues, f"{display}: ACIL checkpoint identity drifted")

        parameter_count = result.get("neural_parameter_count")
        if not _positive_int(parameter_count):
            _issue(integrity_issues, f"{display}: neural parameter count is invalid")
        training = _as_mapping(result.get("training"))
        training_fields = {
            "epochs_completed",
            "optimizer_updates",
            "best_epoch",
            "best_source_dev_nmae",
            "epoch_records",
        }
        if training is None:
            _issue(integrity_issues, f"{display}: training record is absent")
        else:
            _exact_fields(
                training,
                training_fields,
                label=f"{display}/training",
                issues=integrity_issues,
            )
            if training.get("epochs_completed") != 20:
                _issue(integrity_issues, f"{display}: epoch budget drifted")
            if training.get("optimizer_updates") != 320:
                _issue(integrity_issues, f"{display}: optimizer-update budget drifted")
            epoch_records = training.get("epoch_records")
            if not _is_sequence(epoch_records) or len(epoch_records) != 20:
                _issue(integrity_issues, f"{display}: epoch records are incomplete")
            if neural is not None:
                for field in ("best_epoch", "best_source_dev_nmae"):
                    if training.get(field) != neural.get(field):
                        _issue(
                            integrity_issues,
                            f"{display}: training/checkpoint {field} drifted",
                        )

        raw_cells = _as_mapping(result.get("cells"))
        parsed_cells: dict[str, dict[str, float | None]] = {}
        per_mask_identities: dict[str, dict[str, object]] = {}
        if raw_cells is None:
            _issue(integrity_issues, f"{display}: cells object is absent")
        else:
            if set(raw_cells) != set(_MASKS):
                _issue(
                    integrity_issues,
                    f"{display}: mask grid drifted; "
                    f"missing={sorted(set(_MASKS) - set(raw_cells))!r}, "
                    f"extra={sorted(set(raw_cells) - set(_MASKS))!r}",
                )
            for mask in _MASKS:
                raw_cell = _as_mapping(raw_cells.get(mask))
                cell_label = f"{key[0]}/{mask}"
                if raw_cell is None:
                    continue
                _exact_fields(
                    raw_cell,
                    _CELL_FIELDS,
                    label=cell_label,
                    issues=integrity_issues,
                )
                if raw_cell.get("evidence_file") != f"evidence_{mask}.npz":
                    _issue(
                        integrity_issues,
                        f"{cell_label}: evidence filename drifted",
                    )
                identity_record: dict[str, object] = {}
                for field in _CELL_HASH_FIELDS:
                    value = raw_cell.get(field)
                    identity_record[field] = value
                    if not _is_sha256(value):
                        _issue(
                            integrity_issues,
                            f"{cell_label}: {field} is not a lowercase SHA-256",
                        )
                per_mask_identities[mask] = identity_record
                summary = _as_mapping(raw_cell.get("summary"))
                if summary is None:
                    _issue(
                        integrity_issues,
                        f"{cell_label}: summary object is absent",
                    )
                else:
                    parsed_cells[mask] = _validate_summary(
                        summary,
                        cell=cell_label,
                        issues=integrity_issues,
                    )
        for field in _CELL_HASH_FIELDS:
            values = [
                identity.get(field)
                for identity in per_mask_identities.values()
                if _is_sha256(identity.get(field))
            ]
            if len(values) == len(_MASKS) and len(set(values)) != len(values):
                _issue(
                    integrity_issues,
                    f"{display}: per-mask {field} identities are not unique",
                )

        if (
            key in expected
            and result_status == "succeeded"
            and manifest_status == "succeeded"
            and key_counts[key] == 1
        ):
            successful.add((str(key[0]), int(key[1])))
            for mask in _STRUCTURED_MASKS:
                if mask in parsed_cells:
                    cells[(str(key[0]), mask)] = parsed_cells[mask]
            if "random" in parsed_cells:
                random_rows[str(key[0])] = parsed_cells["random"]
        manifest_records.append(
            {
                "dataset": manifest.get("dataset"),
                "seed_bundle": manifest.get("seed_bundle"),
                "job_id": manifest.get("job_id"),
                "source_tree_sha256": manifest.get("source_tree_sha256"),
                "config_sha256": manifest.get("config_sha256"),
                "data_sha256": manifest.get("data_sha256"),
                "acil_checkpoint_file_sha256": manifest.get(
                    "acil_checkpoint_file_sha256"
                ),
                "neural_checkpoint_file_sha256": manifest.get(
                    "neural_checkpoint_file_sha256"
                ),
                "neural_checkpoint_tensor_sha256": manifest.get(
                    "neural_checkpoint_tensor_sha256"
                ),
                "result_sha256": manifest.get("result_sha256"),
                "per_mask_identities": per_mask_identities,
            }
        )

    for key, count in sorted(key_counts.items(), key=lambda item: str(item[0])):
        if count != 1:
            _issue(
                grid_issues,
                f"prototype job {key[0]}/bundle{key[1]} occurs {count} times",
            )
    for dataset, bundle in sorted(expected):
        if key_counts.get((dataset, bundle), 0) == 0:
            _issue(
                grid_issues,
                f"missing prototype job {dataset}/bundle{bundle}",
            )
    for field in ("source_tree_sha256", "config_sha256"):
        values = {
            record.get(field)
            for record in manifest_records
            if _is_sha256(record.get(field))
        }
        if len(values) > 1:
            _issue(integrity_issues, f"cross-job {field} drifted")

    exact_counts = (
        set(key_counts) == expected
        and all(key_counts[key] == 1 for key in expected)
    )
    grid_complete = (
        len(artifacts) == len(expected)
        and exact_counts
        and successful == expected
        and not grid_issues
    )
    integrity_valid = not integrity_issues

    metric_reports = {
        metric: _metric_aggregate(cells, metric)
        for metric in _SUMMARY_METRICS
    }
    worst_oracle = _worst_cell(
        metric_reports["oracle_relative_improvement"]
    )
    worst_actual = _worst_cell(
        metric_reports["hard_relative_improvement"]
    )
    frozen = protocol.gates
    gate_sources = {
        "structured_per_flow_oracle_dataset_equal_min": (
            metric_reports["oracle_relative_improvement"],
            frozen.structured_per_flow_oracle_dataset_equal_min,
        ),
        "loo_winner_auroc_min": (
            metric_reports["loo_winner_auroc"],
            frozen.loo_winner_auroc_min,
        ),
        "loo_target_regret_spearman_min": (
            metric_reports["loo_target_regret_spearman"],
            frozen.loo_target_regret_spearman_min,
        ),
        "hard_anchorcv_oracle_gap_capture_min": (
            metric_reports["oracle_gap_capture"],
            frozen.hard_anchorcv_oracle_gap_capture_min,
        ),
        "hard_anchorcv_actual_improvement_min": (
            metric_reports["hard_relative_improvement"],
            frozen.hard_anchorcv_actual_improvement_min,
        ),
        "worst_structured_oracle_cell_relative_improvement_min": (
            worst_oracle,
            frozen.worst_structured_oracle_cell_relative_improvement_min,
        ),
        "worst_structured_cell_relative_improvement_min": (
            worst_actual,
            frozen.worst_structured_cell_relative_improvement_min,
        ),
    }
    gate_report = {
        name: _gate(source, threshold)
        for name, (source, threshold) in gate_sources.items()
    }

    if not grid_complete or not integrity_valid:
        verdict = "revise"
        reason = "incomplete_or_invalid_evidence"
    elif all(item["pass"] for item in gate_report.values()):
        verdict = "proceed"
        reason = "all_scientific_gates_pass"
    else:
        verdict = "kill"
        reason = "scientific_gate_failure"

    common_identity: dict[str, object] = {}
    for field in ("source_tree_sha256", "config_sha256"):
        values = sorted(
            {
                str(record[field])
                for record in manifest_records
                if _is_sha256(record.get(field))
            }
        )
        common_identity[field] = values[0] if len(values) == 1 else None

    return {
        "schema": _REPORT_SCHEMA,
        "protocol_id": _PROTOCOL_ID,
        "prototype_bundle": _BUNDLE,
        "verdict": verdict,
        "reason": reason,
        "grid": {
            "required_jobs": [
                {"dataset": dataset, "seed_bundle": _BUNDLE}
                for dataset in _DATASETS
            ],
            "expected_job_count": len(expected),
            "received_job_count": len(artifacts),
            "received_successful_jobs": len(successful),
            "complete": grid_complete,
            "issues": grid_issues,
        },
        "integrity": {
            "valid": integrity_valid,
            "required_manifest_hashes": [
                "source_tree_sha256",
                "config_sha256",
                "data_sha256",
                "acil_checkpoint_file_sha256",
                "neural_checkpoint_file_sha256",
                "neural_checkpoint_tensor_sha256",
                "result_sha256",
            ],
            "common_identity": common_identity,
            "job_manifest_identities": manifest_records,
            "issues": integrity_issues,
        },
        "headline": {
            "datasets": list(_DATASETS),
            "masks": list(_STRUCTURED_MASKS),
            "cell_count": len(_DATASETS) * len(_STRUCTURED_MASKS),
            "aggregation": (
                "structured_cell_equal_within_dataset_then_dataset_equal"
            ),
            "metrics": metric_reports,
            "worst_oracle_cell": worst_oracle,
            "worst_actual_cell": worst_actual,
        },
        "gates": gate_report,
        "non_headline_random": {
            dataset: random_rows.get(dataset) for dataset in _DATASETS
        },
    }


__all__ = ["adjudicate_prototype"]
