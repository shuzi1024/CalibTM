"""Pure multi-bundle adjudication for the AnchorCV extension stage.

The caller supplies the complete, already review-verified cell summaries and
the frozen uncertainty report.  This module has no path, data-loader, or file
authority.  It validates both payloads, recomputes every point gate with seed
bundles averaged before the four structured dataset/mask cells, and treats the
random mask as report-only.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import re

from .protocol import _GATES


GridKey = tuple[int, str, str]

_SCHEMA = "anchorcv-v1:multibundle-adjudication:v1"
_UNCERTAINTY_SCHEMA = "anchorcv-v1:extension-uncertainty:v1"
_BUNDLES = (1, 2, 3)
_DATASETS = ("abilene", "geant")
_MASKS = ("random", "internal_block", "two_burst")
_STRUCTURED_MASKS = ("internal_block", "two_burst")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

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
_METRICS = (
    "oracle_relative_improvement",
    "loo_winner_auroc",
    "loo_target_regret_spearman",
    "oracle_gap_capture",
    "hard_relative_improvement",
)
_UNCERTAINTY_FIELDS = {
    "schema_version",
    "headline",
    "diagnostic_oracle_headline",
    "per_bundle_paired_effect",
    "per_bundle_oracle_paired_effect",
    "per_bundle_cell_effect",
    "positive_bundle_count",
    "bundle_count",
    "draw_plan",
    "draw_plan_sha256",
    "bootstrap_distribution_sha256",
}
_INTERVAL_FIELDS = {
    "contrast",
    "aggregation",
    "point_estimate",
    "ci_lower",
    "ci_upper",
    "ci_method",
    "bootstrap_draws",
}
_DRAW_PLAN_FIELDS = {
    "bit_generator",
    "seed",
    "bootstrap_draws",
    "bundle_draws_per_replicate",
    "window_resampling",
    "window_block_length",
    "window_draw_unit",
    "window_draw_scope",
    "window_draw_shared_across",
    "ci_quantiles",
    "ci_quantile_method",
    "bundles",
    "datasets",
    "masks",
    "structured_masks",
    "window_counts",
    "flow_counts",
}
_EXPECTED_PROTOCOL_GATES = {
    "structured_per_flow_oracle_dataset_equal_min",
    "loo_winner_auroc_min",
    "loo_target_regret_spearman_min",
    "hard_anchorcv_oracle_gap_capture_min",
    "hard_anchorcv_actual_improvement_min",
    "worst_structured_oracle_cell_relative_improvement_min",
    "worst_structured_cell_relative_improvement_min",
}
_AGGREGATION = (
    "bundle_equal_within_dataset_mask_then_four_structured_cells_equal"
)


def _issue(issues: list[str], message: str) -> None:
    if message not in issues:
        issues.append(message)


def _mapping(value: object) -> Mapping[object, object] | None:
    return value if isinstance(value, Mapping) else None


def _finite(value: object) -> float | None:
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


def _nonnegative_int(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _exact_fields(
    payload: Mapping[object, object],
    expected: set[str],
    *,
    label: str,
    issues: list[str],
) -> bool:
    actual = set(payload)
    if actual == expected:
        return True
    missing = sorted(expected - actual)
    extra = sorted(repr(item) for item in actual - expected)
    _issue(
        issues,
        f"{label}: fields drifted; missing={missing!r}, extra={extra!r}",
    )
    return False


def _validate_summary(
    raw: object,
    *,
    label: str,
    issues: list[str],
) -> dict[str, float | None] | None:
    issue_count = len(issues)
    summary = _mapping(raw)
    if summary is None:
        _issue(issues, f"{label}: summary must be a mapping")
        return None
    allowed = _SUMMARY_BASE_FIELDS | set(_UNDEFINED_REASON.values())
    missing = _SUMMARY_BASE_FIELDS - set(summary)
    extra = set(summary) - allowed
    if missing or extra:
        _issue(
            issues,
            f"{label}: fields drifted; missing={sorted(missing)!r}, "
            f"extra={sorted(repr(item) for item in extra)!r}",
        )

    for name in ("case_count", "target_count"):
        if not _positive_int(summary.get(name)):
            _issue(issues, f"{label}: {name} must be a positive integer")

    numeric_names = (
        _SUMMARY_BASE_FIELDS
        - {"case_count", "target_count", "best_single_expert"}
        - set(_UNDEFINED_REASON)
    )
    numeric: dict[str, float] = {}
    for name in sorted(numeric_names):
        value = _finite(summary.get(name))
        if value is None:
            _issue(
                issues,
                f"{label}: {name} is missing, undefined, or non-finite",
            )
        else:
            numeric[name] = value

    nullable: dict[str, float | None] = {}
    for metric, reason_name in _UNDEFINED_REASON.items():
        raw_metric = summary.get(metric)
        reason = summary.get(reason_name)
        if raw_metric is None:
            nullable[metric] = None
            if not isinstance(reason, str) or not reason:
                _issue(
                    issues,
                    f"{label}: undefined {metric} lacks its reason",
                )
        else:
            value = _finite(raw_metric)
            nullable[metric] = value
            if value is None:
                _issue(
                    issues,
                    f"{label}: {metric} must be finite numeric or null",
                )
            if reason_name in summary:
                _issue(
                    issues,
                    f"{label}: defined {metric} has an undefined reason",
                )

    nonnegative = {
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
    }
    for name in sorted(nonnegative):
        value = numeric.get(name)
        if value is not None and value < 0.0:
            _issue(issues, f"{label}: {name} must be nonnegative")
    truth = numeric.get("absolute_truth_sum")
    if truth is not None and truth <= 0.0:
        _issue(issues, f"{label}: absolute_truth_sum must be positive")

    if truth is not None and truth > 0.0:
        for prefix in ("p", "n", "hard", "oracle"):
            error = numeric.get(f"{prefix}_absolute_error_sum")
            nmae = numeric.get(f"{prefix}_nmae")
            if (
                error is not None
                and nmae is not None
                and not _close(nmae, error / truth)
            ):
                _issue(
                    issues,
                    f"{label}: {prefix}_nmae differs from ratio of sums",
                )

    p_nmae = numeric.get("p_nmae")
    n_nmae = numeric.get("n_nmae")
    best = numeric.get("best_single_nmae")
    expected_best: float | None = None
    if p_nmae is not None and n_nmae is not None:
        expected_best = min(p_nmae, n_nmae)
        expected_expert = "n" if n_nmae < p_nmae else "p"
        if summary.get("best_single_expert") != expected_expert:
            _issue(issues, f"{label}: best_single_expert drifted")
        if best is not None and not _close(best, expected_best):
            _issue(issues, f"{label}: best_single_nmae drifted")
    elif summary.get("best_single_expert") not in {"p", "n"}:
        _issue(issues, f"{label}: best_single_expert must be 'p' or 'n'")

    oracle_nmae = numeric.get("oracle_nmae")
    hard_nmae = numeric.get("hard_nmae")
    if (
        expected_best is not None
        and oracle_nmae is not None
        and oracle_nmae > expected_best + 1e-12
    ):
        _issue(issues, f"{label}: oracle is worse than best single")
    if (
        oracle_nmae is not None
        and hard_nmae is not None
        and hard_nmae + 1e-12 < oracle_nmae
    ):
        _issue(issues, f"{label}: hard method is better than oracle")

    if expected_best is not None:
        if expected_best <= 0.0:
            _issue(issues, f"{label}: best-single denominator must be positive")
        else:
            for metric, candidate in (
                ("oracle_relative_improvement", oracle_nmae),
                ("hard_relative_improvement", hard_nmae),
            ):
                reported = numeric.get(metric)
                if (
                    reported is not None
                    and candidate is not None
                    and not _close(reported, 1.0 - candidate / expected_best)
                ):
                    _issue(issues, f"{label}: {metric} is inconsistent")

    for rate in ("neural_selection_rate", "oracle_neural_rate"):
        value = numeric.get(rate)
        if value is not None and not 0.0 <= value <= 1.0:
            _issue(issues, f"{label}: {rate} must lie in [0, 1]")
    auroc = nullable["loo_winner_auroc"]
    if auroc is not None and not 0.0 <= auroc <= 1.0:
        _issue(issues, f"{label}: loo_winner_auroc must lie in [0, 1]")
    spearman = nullable["loo_target_regret_spearman"]
    if spearman is not None and not -1.0 <= spearman <= 1.0:
        _issue(
            issues,
            f"{label}: loo_target_regret_spearman must lie in [-1, 1]",
        )

    capture = nullable["oracle_gap_capture"]
    if (
        expected_best is not None
        and oracle_nmae is not None
        and hard_nmae is not None
    ):
        gap = expected_best - oracle_nmae
        if gap <= 0.0:
            if capture is not None:
                _issue(
                    issues,
                    f"{label}: oracle_gap_capture must be undefined "
                    "without headroom",
                )
        else:
            expected_capture = (expected_best - hard_nmae) / gap
            if capture is None or not _close(capture, expected_capture):
                _issue(
                    issues,
                    f"{label}: oracle_gap_capture is inconsistent",
                )

    if (
        len(issues) != issue_count
        or any(name not in numeric for name in numeric_names)
    ):
        return None
    parsed: dict[str, float | None] = {
        "oracle_relative_improvement": numeric[
            "oracle_relative_improvement"
        ],
        "hard_relative_improvement": numeric[
            "hard_relative_improvement"
        ],
    }
    parsed.update(nullable)
    return parsed


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _metric_aggregate(
    cells: Mapping[GridKey, Mapping[str, float | None]],
    metric: str,
) -> dict[str, object]:
    bundle_values: dict[str, float | None] = {}
    for bundle in _BUNDLES:
        values = [
            float(value)
            for dataset in _DATASETS
            for mask in _STRUCTURED_MASKS
            if (bundle, dataset, mask) in cells
            and (
                value := cells[(bundle, dataset, mask)].get(metric)
            ) is not None
        ]
        bundle_values[str(bundle)] = (
            _mean(values) if len(values) == 4 else None
        )

    cell_values: dict[str, float | None] = {}
    undefined: list[str] = []
    for dataset in _DATASETS:
        for mask in _STRUCTURED_MASKS:
            label = f"{dataset}/{mask}"
            values = [
                float(value)
                for bundle in _BUNDLES
                if (bundle, dataset, mask) in cells
                and (
                    value := cells[(bundle, dataset, mask)].get(metric)
                ) is not None
            ]
            if len(values) == len(_BUNDLES):
                cell_values[label] = _mean(values)
            else:
                cell_values[label] = None
                undefined.append(label)
    complete = [
        float(value) for value in cell_values.values() if value is not None
    ]
    actual = _mean(complete) if len(complete) == 4 else None
    return {
        "actual": actual,
        "per_bundle": bundle_values,
        "cell_values_after_bundle_average": cell_values,
        "undefined_cells": undefined,
        "aggregation": _AGGREGATION,
    }


def _worst_cell(metric: Mapping[str, object]) -> dict[str, object]:
    undefined = list(metric["undefined_cells"])
    values = metric["cell_values_after_bundle_average"]
    if undefined or not isinstance(values, Mapping):
        return {
            "actual": None,
            "cell": None,
            "undefined_cells": undefined,
        }
    pairs = [
        (float(value), str(cell))
        for cell, value in values.items()
        if _finite(value) is not None
    ]
    if len(pairs) != 4:
        return {
            "actual": None,
            "cell": None,
            "undefined_cells": sorted(str(cell) for cell in values),
        }
    actual, cell = min(pairs, key=lambda item: (item[0], item[1]))
    return {"actual": actual, "cell": cell, "undefined_cells": []}


def _validate_interval(
    raw: object,
    *,
    label: str,
    contrast: str,
    issues: list[str],
) -> dict[str, float] | None:
    interval = _mapping(raw)
    if interval is None:
        _issue(issues, f"{label}: interval must be a mapping")
        return None
    fields_ok = _exact_fields(
        interval, _INTERVAL_FIELDS, label=label, issues=issues
    )
    fixed = {
        "contrast": contrast,
        "aggregation": (
            "equal_seed_bundle_then_dataset_equal_structured_mask"
        ),
        "ci_method": "linear_percentile_2.5_97.5",
        "bootstrap_draws": 10_000,
    }
    for field, expected in fixed.items():
        if interval.get(field) != expected:
            _issue(issues, f"{label}: {field} drifted")
    parsed: dict[str, float] = {}
    for field in ("point_estimate", "ci_lower", "ci_upper"):
        value = _finite(interval.get(field))
        if value is None:
            _issue(issues, f"{label}: {field} must be finite")
        else:
            parsed[field] = value
    if (
        "ci_lower" in parsed
        and "ci_upper" in parsed
        and parsed["ci_lower"] > parsed["ci_upper"]
    ):
        _issue(issues, f"{label}: confidence interval bounds are reversed")
    return (
        parsed
        if fields_ok and len(parsed) == 3
        and all(interval.get(k) == v for k, v in fixed.items())
        else None
    )


def _validate_number_map(
    raw: object,
    *,
    label: str,
    issues: list[str],
) -> dict[str, float] | None:
    payload = _mapping(raw)
    if payload is None:
        _issue(issues, f"{label}: must be a mapping")
        return None
    expected = {str(bundle) for bundle in _BUNDLES}
    fields_ok = _exact_fields(payload, expected, label=label, issues=issues)
    parsed: dict[str, float] = {}
    for key in sorted(expected):
        value = _finite(payload.get(key))
        if value is None:
            _issue(issues, f"{label}/{key}: must be finite")
        else:
            parsed[key] = value
    return parsed if fields_ok and len(parsed) == len(expected) else None


def _validate_uncertainty_cells(
    raw: object,
    *,
    issues: list[str],
) -> dict[GridKey, float]:
    result: dict[GridKey, float] = {}
    bundles = _mapping(raw)
    if bundles is None:
        _issue(issues, "uncertainty/per_bundle_cell_effect: must be a mapping")
        return result
    _exact_fields(
        bundles,
        {str(bundle) for bundle in _BUNDLES},
        label="uncertainty/per_bundle_cell_effect",
        issues=issues,
    )
    for bundle in _BUNDLES:
        datasets = _mapping(bundles.get(str(bundle)))
        label = f"uncertainty/per_bundle_cell_effect/{bundle}"
        if datasets is None:
            _issue(issues, f"{label}: must be a mapping")
            continue
        _exact_fields(
            datasets, set(_DATASETS), label=label, issues=issues
        )
        for dataset in _DATASETS:
            masks = _mapping(datasets.get(dataset))
            cell_label = f"{label}/{dataset}"
            if masks is None:
                _issue(issues, f"{cell_label}: must be a mapping")
                continue
            _exact_fields(
                masks, set(_MASKS), label=cell_label, issues=issues
            )
            for mask in _MASKS:
                value = _finite(masks.get(mask))
                if value is None:
                    _issue(
                        issues, f"{cell_label}/{mask}: must be finite"
                    )
                else:
                    result[(bundle, dataset, mask)] = value
    return result


def _validate_draw_plan(raw: object, *, issues: list[str]) -> None:
    plan = _mapping(raw)
    label = "uncertainty/draw_plan"
    if plan is None:
        _issue(issues, f"{label}: must be a mapping")
        return
    _exact_fields(plan, _DRAW_PLAN_FIELDS, label=label, issues=issues)
    fixed: dict[str, object] = {
        "bit_generator": "PCG64DXSM",
        "seed": 81_001,
        "bootstrap_draws": 10_000,
        "bundle_draws_per_replicate": 3,
        "window_resampling": "circular_block",
        "window_block_length": 4,
        "window_draw_unit": "whole_window_all_flows",
        "window_draw_scope": (
            "independent_per_sampled_bundle_occurrence_and_dataset"
        ),
        "window_draw_shared_across": "all_masks_and_contrasts",
        "ci_quantiles": [0.025, 0.975],
        "ci_quantile_method": "linear",
        "bundles": list(_BUNDLES),
        "datasets": list(_DATASETS),
        "masks": list(_MASKS),
        "structured_masks": list(_STRUCTURED_MASKS),
        "flow_counts": {"abilene": 144, "geant": 462},
    }
    for field, expected in fixed.items():
        if plan.get(field) != expected:
            _issue(issues, f"{label}: {field} drifted")
    windows = _mapping(plan.get("window_counts"))
    if windows is None or set(windows) != set(_DATASETS):
        _issue(issues, f"{label}: window_counts schema drifted")
    else:
        for dataset in _DATASETS:
            if not _positive_int(windows.get(dataset)):
                _issue(
                    issues,
                    f"{label}: {dataset} window count must be positive",
                )


def _validate_uncertainty(
    raw: object,
    *,
    cells: Mapping[GridKey, Mapping[str, float | None]],
    issues: list[str],
) -> dict[str, object]:
    report = _mapping(raw)
    if report is None:
        _issue(issues, "uncertainty report must be a mapping")
        return {}
    _exact_fields(
        report,
        _UNCERTAINTY_FIELDS,
        label="uncertainty",
        issues=issues,
    )
    if report.get("schema_version") != _UNCERTAINTY_SCHEMA:
        _issue(issues, "uncertainty: schema_version drifted")

    headline = _validate_interval(
        report.get("headline"),
        label="uncertainty/headline",
        contrast="hard_vs_best_single",
        issues=issues,
    )
    oracle_headline = _validate_interval(
        report.get("diagnostic_oracle_headline"),
        label="uncertainty/diagnostic_oracle_headline",
        contrast="oracle_vs_best_single",
        issues=issues,
    )
    hard_bundles = _validate_number_map(
        report.get("per_bundle_paired_effect"),
        label="uncertainty/per_bundle_paired_effect",
        issues=issues,
    )
    oracle_bundles = _validate_number_map(
        report.get("per_bundle_oracle_paired_effect"),
        label="uncertainty/per_bundle_oracle_paired_effect",
        issues=issues,
    )
    uncertainty_cells = _validate_uncertainty_cells(
        report.get("per_bundle_cell_effect"), issues=issues
    )

    derived_hard: dict[str, float] = {}
    derived_oracle: dict[str, float] = {}
    for bundle in _BUNDLES:
        structured = [
            cells[(bundle, dataset, mask)]
            for dataset in _DATASETS
            for mask in _STRUCTURED_MASKS
            if (bundle, dataset, mask) in cells
        ]
        if len(structured) == 4:
            derived_hard[str(bundle)] = _mean(
                [cell["hard_relative_improvement"] for cell in structured]
            )
            derived_oracle[str(bundle)] = _mean(
                [cell["oracle_relative_improvement"] for cell in structured]
            )

    for key, reported in uncertainty_cells.items():
        cell = cells.get(key)
        if (
            cell is not None
            and not _close(
                reported, cell["hard_relative_improvement"]
            )
        ):
            _issue(
                issues,
                "uncertainty/per_bundle_cell_effect/"
                f"{key[0]}/{key[1]}/{key[2]} differs from cell summary",
            )
    for label, reported, derived in (
        ("per_bundle_paired_effect", hard_bundles, derived_hard),
        (
            "per_bundle_oracle_paired_effect",
            oracle_bundles,
            derived_oracle,
        ),
    ):
        if reported is not None:
            for bundle in map(str, _BUNDLES):
                if (
                    bundle in derived
                    and not _close(reported[bundle], derived[bundle])
                ):
                    _issue(
                        issues,
                        f"uncertainty/{label}/{bundle} differs from summaries",
                    )

    hard_point = (
        _mean(list(derived_hard.values()))
        if len(derived_hard) == len(_BUNDLES)
        else None
    )
    oracle_point = (
        _mean(list(derived_oracle.values()))
        if len(derived_oracle) == len(_BUNDLES)
        else None
    )
    if (
        headline is not None
        and hard_point is not None
        and not _close(headline["point_estimate"], hard_point)
    ):
        _issue(
            issues,
            "uncertainty/headline point_estimate differs from summaries",
        )
    if (
        oracle_headline is not None
        and oracle_point is not None
        and not _close(oracle_headline["point_estimate"], oracle_point)
    ):
        _issue(
            issues,
            "uncertainty/diagnostic oracle point differs from summaries",
        )

    positive_count = report.get("positive_bundle_count")
    if (
        not _nonnegative_int(positive_count)
        or int(positive_count) > len(_BUNDLES)
    ):
        _issue(
            issues,
            "uncertainty: positive_bundle_count must be an integer in [0, 3]",
        )
    elif len(derived_hard) == len(_BUNDLES):
        expected_positive = sum(value > 0.0 for value in derived_hard.values())
        if positive_count != expected_positive:
            _issue(
                issues,
                "uncertainty: positive_bundle_count differs from summaries",
            )
    if report.get("bundle_count") != len(_BUNDLES):
        _issue(issues, "uncertainty: bundle_count drifted")

    _validate_draw_plan(report.get("draw_plan"), issues=issues)
    for field in (
        "draw_plan_sha256",
        "bootstrap_distribution_sha256",
    ):
        value = report.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            _issue(issues, f"uncertainty: {field} is not a SHA-256")

    return {
        "headline": headline,
        "positive_bundle_count": (
            int(positive_count)
            if _nonnegative_int(positive_count)
            and int(positive_count) <= len(_BUNDLES)
            else None
        ),
        "per_bundle_paired_effect": derived_hard,
    }


def _gate(
    actual: object,
    *,
    threshold: float,
    operator: str = ">=",
    undefined_cells: list[str] | None = None,
) -> dict[str, object]:
    parsed = _finite(actual)
    if operator == ">=":
        passed = parsed is not None and parsed >= threshold
    elif operator == ">":
        passed = parsed is not None and parsed > threshold
    else:  # pragma: no cover - internal construction only
        raise RuntimeError(f"unsupported gate operator {operator!r}")
    return {
        "actual": parsed,
        "threshold": float(threshold),
        "operator": operator,
        "pass": bool(passed),
        "undefined_cells": list(undefined_cells or []),
    }


def _scientific_reason(name: str, gate: Mapping[str, object]) -> str:
    return (
        f"{name} failed: actual={gate['actual']!r} "
        f"{gate['operator']} threshold={gate['threshold']!r}"
    )


def adjudicate_multibundle(
    cell_summaries: Mapping[GridKey, Mapping[str, object]],
    uncertainty_report: Mapping[str, object],
) -> dict[str, object]:
    """Validate and adjudicate the exact 3-bundle extension evidence grid.

    Invalid, missing, non-finite, schema-drifted, or malformed-hash evidence
    returns ``revise``.  A schema-valid diagnostic metric may be undefined only
    with its matching reason; it remains valid evidence but makes the
    corresponding scientific gate fail.  A complete valid grid that misses any
    scientific threshold returns ``kill``.  Only complete evidence passing all
    nine gates (the seven frozen protocol gates plus two replication gates)
    returns ``proceed``.
    """

    if set(_GATES) != _EXPECTED_PROTOCOL_GATES:
        raise RuntimeError("AnchorCV protocol gate registry drifted")
    protocol_gates = {name: float(value) for name, value in _GATES.items()}

    expected_keys = {
        (bundle, dataset, mask)
        for bundle in _BUNDLES
        for dataset in _DATASETS
        for mask in _MASKS
    }
    grid_issues: list[str] = []
    integrity_issues: list[str] = []
    raw_grid = _mapping(cell_summaries)
    if raw_grid is None:
        actual_keys: set[object] = set()
        _issue(grid_issues, "cell_summaries must be a mapping")
    else:
        actual_keys = set(raw_grid)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        _issue(
            grid_issues,
            "cell grid drifted; "
            f"missing={sorted(missing)!r}, "
            f"extra={sorted(repr(item) for item in extra)!r}",
        )

    cells: dict[GridKey, dict[str, float | None]] = {}
    random_rows: dict[str, dict[str, float | None]] = {}
    if raw_grid is not None:
        for key in sorted(expected_keys):
            if key not in raw_grid:
                continue
            bundle, dataset, mask = key
            parsed = _validate_summary(
                raw_grid[key],
                label=f"bundle{bundle}/{dataset}/{mask}",
                issues=integrity_issues,
            )
            if parsed is None:
                continue
            cells[key] = parsed
            if mask == "random":
                random_rows[f"bundle{bundle}/{dataset}"] = dict(parsed)

    uncertainty = _validate_uncertainty(
        uncertainty_report,
        cells=cells,
        issues=integrity_issues,
    )
    for issue in grid_issues:
        _issue(integrity_issues, issue)

    metrics = {
        metric: _metric_aggregate(cells, metric) for metric in _METRICS
    }
    worst_oracle = _worst_cell(
        metrics["oracle_relative_improvement"]
    )
    worst_actual = _worst_cell(metrics["hard_relative_improvement"])

    gates = {
        "structured_per_flow_oracle_dataset_equal_min": _gate(
            metrics["oracle_relative_improvement"]["actual"],
            threshold=protocol_gates[
                "structured_per_flow_oracle_dataset_equal_min"
            ],
            undefined_cells=metrics["oracle_relative_improvement"][
                "undefined_cells"
            ],
        ),
        "loo_winner_auroc_min": _gate(
            metrics["loo_winner_auroc"]["actual"],
            threshold=protocol_gates["loo_winner_auroc_min"],
            undefined_cells=metrics["loo_winner_auroc"]["undefined_cells"],
        ),
        "loo_target_regret_spearman_min": _gate(
            metrics["loo_target_regret_spearman"]["actual"],
            threshold=protocol_gates["loo_target_regret_spearman_min"],
            undefined_cells=metrics["loo_target_regret_spearman"][
                "undefined_cells"
            ],
        ),
        "hard_anchorcv_oracle_gap_capture_min": _gate(
            metrics["oracle_gap_capture"]["actual"],
            threshold=protocol_gates[
                "hard_anchorcv_oracle_gap_capture_min"
            ],
            undefined_cells=metrics["oracle_gap_capture"][
                "undefined_cells"
            ],
        ),
        "hard_anchorcv_actual_improvement_min": _gate(
            metrics["hard_relative_improvement"]["actual"],
            threshold=protocol_gates[
                "hard_anchorcv_actual_improvement_min"
            ],
            undefined_cells=metrics["hard_relative_improvement"][
                "undefined_cells"
            ],
        ),
        "worst_structured_oracle_cell_relative_improvement_min": _gate(
            worst_oracle["actual"],
            threshold=protocol_gates[
                "worst_structured_oracle_cell_relative_improvement_min"
            ],
            undefined_cells=worst_oracle["undefined_cells"],
        ),
        "worst_structured_cell_relative_improvement_min": _gate(
            worst_actual["actual"],
            threshold=protocol_gates[
                "worst_structured_cell_relative_improvement_min"
            ],
            undefined_cells=worst_actual["undefined_cells"],
        ),
        "uncertainty_headline_ci_lower_gt_zero": _gate(
            (
                uncertainty["headline"]["ci_lower"]
                if isinstance(uncertainty.get("headline"), Mapping)
                else None
            ),
            threshold=0.0,
            operator=">",
        ),
        "positive_bundle_count_min": _gate(
            uncertainty.get("positive_bundle_count"),
            threshold=2.0,
        ),
    }

    valid = not integrity_issues
    scientific_failures = (
        [
            _scientific_reason(name, gate)
            for name, gate in gates.items()
            if not gate["pass"]
        ]
        if valid
        else []
    )
    if not valid:
        verdict = "revise"
        reason = "incomplete_or_invalid_evidence"
        failure_reasons = list(integrity_issues)
    elif scientific_failures:
        verdict = "kill"
        reason = "scientific_gate_failure"
        failure_reasons = list(scientific_failures)
    else:
        verdict = "proceed"
        reason = "all_scientific_gates_pass"
        failure_reasons = []

    return {
        "schema": _SCHEMA,
        "verdict": verdict,
        "reason": reason,
        "grid": {
            "complete": actual_keys == expected_keys,
            "expected_cell_count": len(expected_keys),
            "received_cell_count": len(actual_keys),
            "bundles": list(_BUNDLES),
            "datasets": list(_DATASETS),
            "masks": list(_MASKS),
            "issues": grid_issues,
        },
        "integrity": {
            "valid": valid,
            "issues": list(integrity_issues),
        },
        "aggregation": _AGGREGATION,
        "headline": {
            "masks": list(_STRUCTURED_MASKS),
            "cell_count_after_bundle_average": 4,
            "metrics": metrics,
            "worst_oracle_cell": worst_oracle,
            "worst_actual_cell": worst_actual,
        },
        "gates": gates,
        "per_bundle_paired_effect": dict(
            uncertainty.get("per_bundle_paired_effect", {})
        ),
        "uncertainty": {
            "headline": uncertainty.get("headline"),
            "positive_bundle_count": uncertainty.get(
                "positive_bundle_count"
            ),
        },
        "non_headline_random": random_rows,
        "scientific_failure_reasons": scientific_failures,
        "failure_reasons": failure_reasons,
    }


__all__ = ["adjudicate_multibundle"]
