"""Strict paired evidence primitives used by every frozen stage gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Iterable, Mapping, Sequence

from .config import canonical_json_bytes, load_protocol_config
from .derangement import build_derangement
from .jobs import planned_jobs
from .registries import ordered_window_starts, stage_spec
from .result_io import (
    _read_canonical_json,
    _read_canonical_jsonl,
    _read_regular_bytes,
    _validate_metric_payload,
    load_job_result,
    regular_file_sha256,
    write_canonical_json_exclusive,
)
from .statistics import paired_bootstrap


@dataclass(frozen=True, slots=True)
class ContrastResult:
    candidate: str
    comparator: str
    pooling_masks: tuple[str, ...]
    candidate_nmae: float
    comparator_nmae: float
    improvement: float
    paired_record_count: int


@dataclass(frozen=True, slots=True)
class GateDecision:
    stage: str
    verdict: str
    checks: Mapping[str, bool]


_GATE_STAGES = (
    "stage0_acil_tune",
    "stage_h",
    "stage_i",
    "full_tune",
)
_LEGAL_MASKS = ("random", "internal_block", "two_burst")
_DIRECT_UPSTREAM = {
    "stage_h": "stage0_acil_tune",
    "stage_i": "stage_h",
    "full_tune": "stage_i",
}


def _value(record: object, name: str):
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _identity(record: object) -> dict[str, object]:
    value = _value(record, "identity")
    if not isinstance(value, Mapping):
        raise TypeError("record identity must be a mapping")
    return dict(value)


def _pair_key(identity: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    if "method" not in identity:
        raise ValueError("record identity has no method")
    return tuple(sorted((str(key), value) for key, value in identity.items() if key != "method"))


def _method_table(
    records: Iterable[object],
    *,
    methods: set[str],
    pooling_masks: tuple[str, ...],
    seed_bundle: int | None = None,
) -> dict[str, dict[tuple[tuple[str, object], ...], object]]:
    if not pooling_masks or len(set(pooling_masks)) != len(pooling_masks):
        raise ValueError("pooling masks must be nonempty and unique")
    legal_masks = {"random", "internal_block", "two_burst"}
    if not set(pooling_masks) <= legal_masks:
        raise ValueError("pooling mask registry is invalid")
    tables = {method: {} for method in methods}
    for record in records:
        identity = _identity(record)
        method = str(identity.get("method", ""))
        if method not in methods:
            continue
        if identity.get("mask_family") not in pooling_masks:
            continue
        if seed_bundle is not None and identity.get("seed_bundle") != seed_bundle:
            continue
        key = _pair_key(identity)
        if key in tables[method]:
            raise ValueError(f"duplicate paired record for method {method!r}")
        tables[method][key] = record
    return tables


def _finite_nonnegative(record: object, name: str) -> float:
    value = float(_value(record, name))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"record {name} must be finite and nonnegative")
    return value


def paired_contrast(
    records: Sequence[object] | Iterable[object],
    *,
    candidate: str,
    comparator: str,
    pooling_masks: tuple[str, ...],
    seed_bundle: int | None = None,
) -> ContrastResult:
    """Compute one strict ratio-of-sums contrast over identical target records."""

    if not candidate or not comparator or candidate == comparator:
        raise ValueError("paired contrast requires two distinct methods")
    tables = _method_table(
        records,
        methods={candidate, comparator},
        pooling_masks=tuple(pooling_masks),
        seed_bundle=seed_bundle,
    )
    candidate_keys = set(tables[candidate])
    comparator_keys = set(tables[comparator])
    if not candidate_keys or candidate_keys != comparator_keys:
        raise ValueError("paired methods have missing or extra records")

    candidate_errors: list[float] = []
    candidate_truths: list[float] = []
    comparator_errors: list[float] = []
    comparator_truths: list[float] = []
    for key in sorted(candidate_keys, key=repr):
        candidate_record = tables[candidate][key]
        comparator_record = tables[comparator][key]
        if _value(candidate_record, "target_set_sha256") != _value(
            comparator_record, "target_set_sha256"
        ):
            raise ValueError("paired target SHA-256 drifted")
        candidate_truth = _finite_nonnegative(candidate_record, "absolute_truth_sum")
        comparator_truth = _finite_nonnegative(comparator_record, "absolute_truth_sum")
        if candidate_truth != comparator_truth:
            raise ValueError("paired absolute-truth denominator drifted")
        if _finite_nonnegative(candidate_record, "squared_truth_sum") != _finite_nonnegative(
            comparator_record, "squared_truth_sum"
        ):
            raise ValueError("paired squared-truth denominator drifted")
        if int(_value(candidate_record, "target_count")) != int(
            _value(comparator_record, "target_count")
        ):
            raise ValueError("paired target-count denominator drifted")
        candidate_errors.append(_finite_nonnegative(candidate_record, "absolute_error_sum"))
        comparator_errors.append(_finite_nonnegative(comparator_record, "absolute_error_sum"))
        candidate_truths.append(candidate_truth)
        comparator_truths.append(comparator_truth)
    candidate_denominator = math.fsum(candidate_truths)
    comparator_denominator = math.fsum(comparator_truths)
    if candidate_denominator <= 0.0 or comparator_denominator <= 0.0:
        raise ValueError("paired pooled truth denominator must be positive")
    candidate_nmae = math.fsum(candidate_errors) / candidate_denominator
    comparator_nmae = math.fsum(comparator_errors) / comparator_denominator
    if comparator_nmae <= 0.0:
        raise ValueError("paired comparator NMAE must be positive")
    return ContrastResult(
        candidate=candidate,
        comparator=comparator,
        pooling_masks=tuple(pooling_masks),
        candidate_nmae=candidate_nmae,
        comparator_nmae=comparator_nmae,
        improvement=1.0 - candidate_nmae / comparator_nmae,
        paired_record_count=len(candidate_keys),
    )


def strongest_comparator(
    records: Sequence[object] | Iterable[object],
    *,
    candidates: tuple[str, ...],
    pooling_masks: tuple[str, ...],
) -> str:
    """Select the registered comparator with the lowest pooled NMAE."""

    if len(candidates) < 1 or len(set(candidates)) != len(candidates):
        raise ValueError("comparator candidates must be nonempty and unique")
    if len(candidates) == 1:
        return candidates[0]
    reference = candidates[0]
    scores = {reference: None}
    for candidate in candidates[1:]:
        result = paired_contrast(
            records,
            candidate=candidate,
            comparator=reference,
            pooling_masks=pooling_masks,
        )
        scores[candidate] = result.candidate_nmae
        if scores[reference] is None:
            scores[reference] = result.comparator_nmae
    return min(candidates, key=lambda method: (float(scores[method]), candidates.index(method)))


def positive_seed_count(
    records: Sequence[object] | Iterable[object],
    *,
    candidate: str,
    comparator: str,
    pooling_masks: tuple[str, ...],
    seed_bundles: tuple[int, ...],
) -> tuple[int, Mapping[int, float]]:
    """Apply the pre-rounding seed-level pooled positive-effect definition."""

    if not seed_bundles or len(set(seed_bundles)) != len(seed_bundles):
        raise ValueError("seed bundles must be nonempty and unique")
    effects = {
        seed: paired_contrast(
            records,
            candidate=candidate,
            comparator=comparator,
            pooling_masks=pooling_masks,
            seed_bundle=seed,
        ).improvement
        for seed in seed_bundles
    }
    return sum(value > 0.0 for value in effects.values()), effects


def decide_discovery_gate(
    stage: str,
    *,
    main: float,
    per_mask: Mapping[str, float],
    ci_lower: float,
    positive_seed_count: int,
    random_effect: float | None = None,
) -> GateDecision:
    """Apply every numeric Stage-0/H clause without rounding."""

    if stage not in {"stage0_acil_tune", "stage_h"}:
        raise ValueError("discovery gate must be stage0_acil_tune or stage_h")
    numeric = [main, ci_lower, *per_mask.values()]
    if random_effect is not None:
        numeric.append(random_effect)
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("gate operands must be finite")
    config = load_protocol_config()["gates"][stage]
    main_spec = config["effect_contrasts"]["main"]
    mask_spec = config["per_mask_rule"]
    required_masks = tuple(mask_spec["masks"])
    if set(per_mask) != set(required_masks):
        raise ValueError("per-mask gate operands differ from the registry")
    checks = {
        "main_minimum": float(main) >= float(main_spec["minimum"]),
        **{
            f"{mask}_strictly_positive": float(per_mask[mask])
            > float(mask_spec["strictly_above"])
            for mask in required_masks
        },
        "bootstrap_ci_lower": float(ci_lower)
        > float(config["bootstrap_ci"]["lower_strictly_above"]),
        "positive_seed_count": int(positive_seed_count)
        >= int(config["positive_seed_rule"]["required"]),
    }
    if stage == "stage_h":
        if random_effect is None:
            raise ValueError("stage_h requires the registered random no-harm operand")
        checks["random_no_harm"] = float(random_effect) >= float(
            config["effect_contrasts"]["random_no_harm"]["minimum"]
        )
    elif random_effect is not None:
        raise ValueError("stage0 has no registered random no-harm contrast")
    return GateDecision(
        stage=stage,
        verdict="proceed" if all(checks.values()) else str(config["failure"]),
        checks=checks,
    )


def _require_finite_operands(values: Iterable[object]) -> None:
    for value in values:
        if isinstance(value, bool):
            raise ValueError("gate operands must be finite numeric values")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("gate operands must be finite numeric values") from exc
        if not math.isfinite(numeric):
            raise ValueError("gate operands must be finite numeric values")


def _registered_dataset_operands(
    stage: str, values: Mapping[str, float]
) -> tuple[str, ...]:
    datasets = stage_spec(stage).datasets
    if set(values) != set(datasets):
        raise ValueError("per-dataset operands differ from the frozen registry")
    _require_finite_operands(values.values())
    return datasets


def _decide_stage_i_gate(
    *,
    main: float,
    global_over_local: float,
    per_dataset: Mapping[str, float],
    ci_lower: float,
    positive_seed_count: int,
    unshuffled_main_gain: float,
    derangement_gain_loss: float,
    derangement_fraction: float | None,
) -> GateDecision:
    """Apply the frozen Stage-I gate, including its sole revision region."""

    _require_finite_operands(
        (
            main,
            global_over_local,
            ci_lower,
            unshuffled_main_gain,
            derangement_gain_loss,
        )
    )
    if derangement_fraction is not None:
        _require_finite_operands((derangement_fraction,))
    datasets = _registered_dataset_operands("stage_i", per_dataset)
    if isinstance(positive_seed_count, bool) or not isinstance(positive_seed_count, int):
        raise ValueError("positive seed count must be an integer")
    config = load_protocol_config()["gates"]["stage_i"]
    main_spec = config["effect_contrasts"]["main"]
    local_spec = config["effect_contrasts"]["global_over_local"]
    deranged_spec = config["effect_contrasts"]["derangement_gain_loss"]
    dataset_floor = float(config["per_dataset_rule"]["minimum"])
    checks = {
        "main_minimum": float(main) >= float(main_spec["minimum"]),
        "global_over_local_minimum": float(global_over_local)
        >= float(local_spec["minimum"]),
        **{
            f"{dataset}_no_harm": float(per_dataset[dataset]) >= dataset_floor
            for dataset in datasets
        },
        "bootstrap_ci_lower": float(ci_lower)
        > float(config["bootstrap_ci"]["lower_strictly_above"]),
        "positive_seed_count": positive_seed_count
        >= int(config["positive_seed_rule"]["required"]),
        "unshuffled_main_gain_positive": float(unshuffled_main_gain)
        > float(deranged_spec["requires_unshuffled_main_gain_strictly_above"]),
        "derangement_gain_loss": float(derangement_gain_loss)
        >= float(deranged_spec["minimum"]),
        "derangement_fraction": derangement_fraction is not None
        and float(derangement_fraction) >= float(deranged_spec["fraction_minimum"]),
    }
    if all(checks.values()):
        verdict = "proceed"
    elif (
        float(config["revision_primary_min"])
        <= float(main)
        < float(main_spec["minimum"])
    ):
        verdict = "revise"
    else:
        verdict = "kill"
    return GateDecision(stage="stage_i", verdict=verdict, checks=checks)


def _decide_full_tune_gate(
    *,
    main: float,
    over_deepsets: float,
    pretraining_attribution: float,
    random_effect: float,
    ci_lower: float,
    positive_seed_count: int,
) -> GateDecision:
    """Apply the method gate while leaving the pretraining claim separate."""

    _require_finite_operands(
        (
            main,
            over_deepsets,
            pretraining_attribution,
            random_effect,
            ci_lower,
        )
    )
    if isinstance(positive_seed_count, bool) or not isinstance(positive_seed_count, int):
        raise ValueError("positive seed count must be an integer")
    config = load_protocol_config()["gates"]["full_tune"]
    effects = config["effect_contrasts"]
    checks = {
        "main_minimum": float(main) >= float(effects["main"]["minimum"]),
        "over_deepsets_minimum": float(over_deepsets)
        >= float(effects["over_deepsets"]["minimum"]),
        "pretraining_no_harm_floor": float(pretraining_attribution)
        >= float(effects["pretraining_attribution"]["no_claim_floor"]),
        "random_no_harm": float(random_effect)
        >= float(effects["random_no_harm"]["minimum"]),
        "bootstrap_ci_lower": float(ci_lower)
        > float(config["bootstrap_ci"]["lower_strictly_above"]),
        "positive_seed_count": positive_seed_count
        >= int(config["positive_seed_rule"]["required"]),
    }
    return GateDecision(
        stage="full_tune",
        verdict="proceed" if all(checks.values()) else "kill",
        checks=checks,
    )


def _pretraining_claim(*, effect: float, ci_lower: float) -> str:
    """Return the only two frozen attribution claim states."""

    _require_finite_operands((effect, ci_lower))
    config = load_protocol_config()["gates"]["full_tune"]
    threshold = float(
        config["effect_contrasts"]["pretraining_attribution"]["claim_minimum"]
    )
    ci_threshold = float(config["pretraining_claim_ci"]["lower_strictly_above"])
    return (
        "claim_supported"
        if float(effect) >= threshold and float(ci_lower) > ci_threshold
        else "no_claim"
    )


def _decide_formal_gate(
    *,
    main: float,
    per_dataset: Mapping[str, float],
    ci_lower: float,
    positive_seed_count: int,
    random_effect: float,
    cell_wins: int,
    external_effect: float,
    external_per_dataset: Mapping[str, float],
) -> GateDecision:
    """Apply all frozen formal clauses; there is no gate-set revision path."""

    _require_finite_operands((main, ci_lower, random_effect, external_effect))
    datasets = _registered_dataset_operands("formal_gate", per_dataset)
    if set(external_per_dataset) != set(datasets):
        raise ValueError("external per-dataset operands differ from the frozen registry")
    _require_finite_operands(external_per_dataset.values())
    for label, value in (
        ("positive seed count", positive_seed_count),
        ("cell win count", cell_wins),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be an integer")
    config = load_protocol_config()["gates"]["formal_gate"]
    effects = config["effect_contrasts"]
    dataset_floor = float(config["per_dataset_rule"]["minimum"])
    external_floor = float(effects["external_no_harm"]["per_dataset_minimum"])
    checks = {
        "main_minimum": float(main) >= float(effects["main"]["minimum"]),
        **{
            f"{dataset}_no_harm": float(per_dataset[dataset]) >= dataset_floor
            for dataset in datasets
        },
        "bootstrap_ci_lower": float(ci_lower)
        > float(config["bootstrap_ci"]["lower_strictly_above"]),
        "positive_seed_count": positive_seed_count
        >= int(config["positive_seed_rule"]["required"]),
        "random_no_harm": float(random_effect)
        >= float(effects["random_no_harm"]["minimum"]),
        "dataset_mask_cell_wins": cell_wins
        >= int(effects["dataset_mask_cell_wins"]["required"]),
        "external_no_harm": float(external_effect)
        >= float(effects["external_no_harm"]["minimum"]),
        **{
            f"external_{dataset}_no_harm": float(external_per_dataset[dataset])
            >= external_floor
            for dataset in datasets
        },
    }
    return GateDecision(
        stage="formal_gate",
        verdict="proceed" if all(checks.values()) else "kill",
        checks=checks,
    )


def _summary_table(records: Sequence[object], methods: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str, str], list[tuple[float, float, float, float, int]]] = {}
    for record in records:
        identity = _identity(record)
        method = str(identity["method"])
        if method not in methods:
            continue
        key = (
            method,
            int(identity["seed_bundle"]),
            str(identity["dataset"]),
            str(identity["mask_family"]),
        )
        groups.setdefault(key, []).append(
            (
                _finite_nonnegative(record, "absolute_error_sum"),
                _finite_nonnegative(record, "absolute_truth_sum"),
                _finite_nonnegative(record, "squared_error_sum"),
                _finite_nonnegative(record, "squared_truth_sum"),
                int(_value(record, "target_count")),
            )
        )
    rows = []
    for key in sorted(groups):
        values = groups[key]
        ae = math.fsum(value[0] for value in values)
        truth = math.fsum(value[1] for value in values)
        se = math.fsum(value[2] for value in values)
        squared_truth = math.fsum(value[3] for value in values)
        if truth <= 0.0 or squared_truth <= 0.0:
            raise ValueError("summary denominator must be positive")
        rows.append(
            {
                "method": key[0],
                "seed_bundle": key[1],
                "dataset": key[2],
                "mask_family": key[3],
                "absolute_error_sum": ae,
                "absolute_truth_sum": truth,
                "squared_error_sum": se,
                "squared_truth_sum": squared_truth,
                "target_count": sum(value[4] for value in values),
                "record_count": len(values),
                "nmae": ae / truth,
                "nrmse": math.sqrt(se / squared_truth),
            }
        )
    return rows


def _canonical_report(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def _records_for_dataset(
    records: Sequence[object], dataset: str
) -> tuple[object, ...]:
    return tuple(
        record
        for record in records
        if str(_identity(record).get("dataset")) == dataset
    )


def _bootstrap_contrast(
    *,
    stage: str,
    records: Sequence[object],
    candidate: str,
    comparator: str,
    pooling_masks: tuple[str, ...],
):
    config = load_protocol_config()
    registry = stage_spec(stage)
    if registry.eval_cohort is None:
        raise ValueError("a training-only stage has no bootstrap cohort")
    present = {
        str(_identity(record)["dataset"])
        for record in records
        if str(_identity(record).get("method")) in {candidate, comparator}
    }
    datasets = tuple(dataset for dataset in registry.datasets if dataset in present)
    if not datasets or present != set(datasets):
        raise ValueError("bootstrap records contain an invalid dataset registry")
    result = paired_bootstrap(
        records=records,
        candidate=candidate,
        comparator=comparator,
        seed_bundles=registry.seed_bundles,
        datasets={
            dataset: ordered_window_starts(dataset, registry.eval_cohort)
            for dataset in datasets
        },
        mask_families=pooling_masks,
        draws=int(config["bootstrap"]["draws"]),
        bootstrap_seed=int(config["bootstrap"]["seed"]),
        block_length=int(config["bootstrap"]["block_length"]),
    )
    exact = paired_contrast(
        records,
        candidate=candidate,
        comparator=comparator,
        pooling_masks=pooling_masks,
    )
    if not math.isclose(
        result.point_estimate, exact.improvement, rel_tol=0.0, abs_tol=1e-15
    ):
        raise RuntimeError("bootstrap point estimate differs from exact contrast")
    return result


def _active_freeze_hashes() -> tuple[str, str, str]:
    """Resolve the write-once freeze before inspecting any result number."""

    from .scripts.build_freeze import _FREEZE_DIRECTORY

    root = Path(__file__).resolve().parent / "generated" / _FREEZE_DIRECTORY
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"active generated/{_FREEZE_DIRECTORY} is missing"
        ) from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"active generated/{_FREEZE_DIRECTORY} is invalid")
    from .execution import _active_freeze, _read_json_exact

    manifest_sha256, provenance_sha256 = _active_freeze()
    anchor = _read_json_exact(root / "anchor.json")
    if not isinstance(anchor, Mapping):
        raise ValueError("active freeze anchor must be an object")
    freeze_sha256 = anchor.get("freeze_sha256")
    if (
        not isinstance(freeze_sha256, str)
        or len(freeze_sha256) != 64
        or any(character not in "0123456789abcdef" for character in freeze_sha256)
    ):
        raise ValueError("active freeze hash is invalid")
    return manifest_sha256, provenance_sha256, freeze_sha256


def _read_derangement_artifacts(
    *,
    output_root: Path,
    job,
    payload: Mapping[str, object],
    primary_rows: Sequence[object],
) -> tuple[dict[str, object], ...]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Stage-I job has no named artifact registry")
    required = ("derangement_records", "derangement_plans")
    if any(name not in artifacts for name in required):
        raise ValueError("Stage-I global job is missing a derangement artifact")
    root = output_root / job.stage / job.job_id
    rows_descriptor = artifacts["derangement_records"]
    plans_descriptor = artifacts["derangement_plans"]
    if not isinstance(rows_descriptor, Mapping) or not isinstance(plans_descriptor, Mapping):
        raise ValueError("Stage-I derangement artifact descriptor is invalid")

    rows, rows_artifact = _read_canonical_jsonl(root / str(rows_descriptor["path"]))
    if (
        rows_artifact.sha256 != rows_descriptor.get("sha256")
        or rows_artifact.bytes != rows_descriptor.get("bytes")
    ):
        raise ValueError("Stage-I derangement record descriptor drifted")
    validated = tuple(_validate_metric_payload(row) for row in rows)

    config = load_protocol_config()
    families = tuple(config["masks"]["families"])
    starts = ordered_window_starts(job.dataset, "tune")
    plans = tuple(build_derangement(job.dataset, job.seed_bundle, family) for family in families)
    expected_identities = tuple(
        {
            "method": "global_loo_deranged",
            "seed_bundle": job.seed_bundle,
            "dataset": job.dataset,
            "mask_family": family,
            "window_start": start,
            "flow": flow,
            "oracle": False,
        }
        for family, plan in zip(families, plans)
        for start in starts
        for flow in range(plan.identity.flow_count)
    )
    if tuple(row["identity"] for row in validated) != expected_identities:
        raise ValueError("Stage-I derangement record grid/order drifted")

    plan_payload, plan_artifact = _read_canonical_json(
        root / str(plans_descriptor["path"])
    )
    if (
        plan_artifact.sha256 != plans_descriptor.get("sha256")
        or plan_artifact.bytes != plans_descriptor.get("bytes")
    ):
        raise ValueError("Stage-I derangement plan descriptor drifted")
    expected_plan_payload = {
        "schema": "acil-innovation-v1:stage-i-derangement-plans:v1",
        "job": job.to_json(),
        "plans": [
            {
                "identity": asdict(plan.identity),
                "permutation_sha256": plan.permutation_sha256,
            }
            for plan in plans
        ],
    }
    if plan_payload != expected_plan_payload:
        raise ValueError("Stage-I derangement plans differ from the frozen registry")

    paired_contrast(
        tuple(primary_rows) + validated,
        candidate="global_loo",
        comparator="global_loo_deranged",
        pooling_masks=families,
    )
    return validated


def _stage_directory_entries(
    *, stage: str, output_root: Path, jobs: Sequence[object]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Inventory extra entries and unfinished registered jobs without following links."""

    stage_root = output_root / stage
    missing: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    inprogress_jobs: list[dict[str, object]] = []
    try:
        stage_root.lstat()
    except FileNotFoundError:
        return (
            [
                {"job": job.to_json(), "error": "missing_job_directory"}
                for job in jobs
            ],
            failures,
            inprogress_jobs,
        )
    if stage_root.is_symlink() or not stage_root.is_dir():
        failures.append(
            {"status": "integrity_failure", "error": "stage_output_is_not_a_plain_directory"}
        )
        return missing, failures, inprogress_jobs
    allowed = {job.job_id for job in jobs} | {
        f".{job.job_id}.inprogress" for job in jobs
    }
    extras = sorted(path.name for path in stage_root.iterdir() if path.name not in allowed)
    if extras:
        failures.append(
            {
                "status": "integrity_failure",
                "error": "unregistered_stage_output_entries",
                "entries": extras,
            }
        )
    for job in jobs:
        final = stage_root / job.job_id
        inprogress = stage_root / f".{job.job_id}.inprogress"
        try:
            final.lstat()
            final_exists = True
        except FileNotFoundError:
            final_exists = False
        try:
            metadata = inprogress.lstat()
            inprogress_exists = True
        except FileNotFoundError:
            metadata = None
            inprogress_exists = False
        if inprogress_exists:
            state = (
                "in_progress_alongside_final"
                if final_exists
                else "in_progress_without_final"
            )
            inprogress_jobs.append(
                {
                    "job": job.to_json(),
                    "path": f"{stage}/.{job.job_id}.inprogress",
                    "state": state,
                }
            )
            if metadata is None or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
                metadata.st_mode
            ):
                failures.append(
                    {
                        "job": job.to_json(),
                        "status": "integrity_failure",
                        "error": "inprogress_entry_is_not_a_plain_directory",
                    }
                )
            elif final_exists:
                failures.append(
                    {
                        "job": job.to_json(),
                        "status": "integrity_failure",
                        "error": "final_and_inprogress_both_exist",
                    }
                )
            else:
                missing.append(
                    {"job": job.to_json(), "error": "job_still_in_progress"}
                )
        elif not final_exists:
            missing.append(
                {"job": job.to_json(), "error": "missing_job_directory"}
            )
    return missing, failures, inprogress_jobs


def _attempt_file_inventory(
    attempt_root: Path, *, control_hashes: Mapping[str, str]
) -> tuple[dict[str, object], ...]:
    """Inventory opaque partials by name/size and hash only known control files."""

    files: list[dict[str, object]] = []
    pending = [attempt_root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir(), key=lambda value: value.name, reverse=True):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("infrastructure attempt contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("infrastructure attempt contains a non-regular entry")
            relative = path.relative_to(attempt_root).as_posix()
            files.append(
                {
                    "bytes": metadata.st_size,
                    "path": relative,
                    "sha256": control_hashes.get(relative),
                }
            )
    return tuple(sorted(files, key=lambda item: str(item["path"])))


def _inventory_infrastructure_attempts(
    *, stage: str, output_root: Path, jobs: Sequence[object]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Semantically validate retry incidents without opening opaque partial metrics."""

    stage_root = output_root / "_infrastructure_attempts" / stage
    try:
        metadata = stage_root.lstat()
    except FileNotFoundError:
        return [], []
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        return [], [
            {
                "status": "integrity_failure",
                "error": "infrastructure_attempt_stage_is_not_a_plain_directory",
            }
        ]
    registered = {job.job_id: job for job in jobs}
    entries = sorted(stage_root.iterdir(), key=lambda value: value.name)
    extras = [entry.name for entry in entries if entry.name not in registered]
    failures: list[dict[str, object]] = []
    if extras:
        failures.append(
            {
                "status": "integrity_failure",
                "error": "unregistered_infrastructure_attempt_job",
                "entries": extras,
            }
        )
    try:
        manifest_sha256, provenance_sha256, freeze_sha256 = _active_freeze_hashes()
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
        failures.append(
            {
                "status": "integrity_failure",
                "error": f"infrastructure active freeze is invalid: {exc}",
            }
        )
        return [], failures
    # Local import avoids the recovery -> gpu_queue -> adjudication import cycle.
    from .recovery import _validate_audited_retry_chain

    attempts: list[dict[str, object]] = []
    for job_id, job in registered.items():
        job_root = stage_root / job_id
        try:
            job_metadata = job_root.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(job_metadata.st_mode) or not stat.S_ISDIR(job_metadata.st_mode):
            failures.append(
                {
                    "job": job.to_json(),
                    "status": "integrity_failure",
                    "error": "infrastructure_attempt_job_is_not_a_plain_directory",
                }
            )
            continue
        try:
            semantic_chain, retry_tail = _validate_audited_retry_chain(
                stage=stage,
                job=job,
                output_root=output_root,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
                freeze_sha256=freeze_sha256,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
            failures.append(
                {
                    "job": job.to_json(),
                    "status": "integrity_failure",
                    "error": str(exc),
                }
            )
            continue
        for record in semantic_chain:
            attempt = job_root / str(record["attempt"])
            try:
                files = _attempt_file_inventory(
                    attempt,
                    control_hashes={
                        "incident.json": str(record["incident_sha256"]),
                        "started.json": str(record["started_sha256"]),
                    },
                )
                digest = hashlib.sha256(
                    b"acil-innovation-v1:infrastructure-attempt:v1\x00"
                    + canonical_json_bytes(
                        {
                            "attempt": attempt.name,
                            "files": list(files),
                            "incident": record["incident"],
                            "job": job.to_json(),
                        }
                    )
                ).hexdigest()
                inventory_row = {
                    "attempt": attempt.name,
                    "attempt_sha256": digest,
                    "files": list(files),
                    "incident": record["incident"],
                    "job": job.to_json(),
                }
                if (
                    retry_tail is not None
                    and attempt.name == semantic_chain[-1]["attempt"]
                ):
                    inventory_row["retry_tail"] = retry_tail
                attempts.append(inventory_row)
            except (OSError, ValueError) as exc:
                failures.append(
                    {
                        "job": job.to_json(),
                        "status": "integrity_failure",
                        "attempt": attempt.name,
                        "error": str(exc),
                    }
                )
    return attempts, failures


def _collect_stage_evidence(
    *,
    stage: str,
    output_root: Path,
    manifest_sha256: str,
    provenance_sha256: str,
) -> dict[str, object]:
    jobs = planned_jobs(stage)
    missing, failures, inprogress_jobs = _stage_directory_entries(
        stage=stage, output_root=output_root, jobs=jobs
    )
    infrastructure_attempts, attempt_failures = _inventory_infrastructure_attempts(
        stage=stage, output_root=output_root, jobs=jobs
    )
    failures.extend(attempt_failures)
    records: list[dict[str, object]] = []
    deranged_records: list[dict[str, object]] = []
    job_results: list[dict[str, object]] = []
    successful_jobs = 0
    stage_root = output_root / stage
    if failures and not stage_root.is_dir():
        return {
            "records": tuple(records),
            "deranged_records": tuple(deranged_records),
            "missing": missing,
            "failures": failures,
            "job_results": tuple(job_results),
            "infrastructure_attempts": tuple(infrastructure_attempts),
            "inprogress_jobs": tuple(inprogress_jobs),
            "successful_jobs": successful_jobs,
        }
    if missing and not stage_root.exists():
        return {
            "records": tuple(records),
            "deranged_records": tuple(deranged_records),
            "missing": missing,
            "failures": failures,
            "job_results": tuple(job_results),
            "infrastructure_attempts": tuple(infrastructure_attempts),
            "inprogress_jobs": tuple(inprogress_jobs),
            "successful_jobs": successful_jobs,
        }
    for job in jobs:
        if any(
            item.get("job", {}).get("job_id") == job.job_id
            for item in (*missing, *failures)
            if isinstance(item.get("job"), Mapping)
        ):
            continue
        job_root = stage_root / job.job_id
        try:
            payload = load_job_result(
                stage=stage, job_id=job.job_id, output_root=output_root
            )
            if (
                payload.get("manifest_sha256") != manifest_sha256
                or payload.get("provenance_sha256") != provenance_sha256
            ):
                raise ValueError("job result is not bound to the active freeze")
            binding: dict[str, object] = {
                "job": job.to_json(),
                "result_sha256": regular_file_sha256(job_root / "result.json"),
                "status": payload["status"],
            }
            if payload["status"] == "succeeded":
                binding.update(
                    {
                        "records_sha256": payload["records"]["sha256"],
                        "checkpoint_identity_sha256": payload["checkpoint"]["identity_sha256"],
                        "checkpoint_file_sha256": payload["checkpoint"]["file_sha256"],
                        "checkpoint_tensor_sha256": payload["checkpoint"]["tensor_sha256"],
                        "named_artifact_sha256": {
                            name: descriptor["sha256"]
                            for name, descriptor in sorted(payload["artifacts"].items())
                        },
                    }
                )
            job_results.append(binding)
            if payload["status"] != "succeeded":
                failures.append(
                    {
                        "job": job.to_json(),
                        "status": payload["status"],
                        "failure": payload.get("failure"),
                    }
                )
                continue
            descriptor = payload["records"]
            rows, artifact = _read_canonical_jsonl(
                job_root / str(descriptor["jsonl"])
            )
            if (
                artifact.sha256 != descriptor["sha256"]
                or artifact.count != descriptor["count"]
            ):
                raise ValueError("validated result record descriptor drifted")
            if stage == "stage_i" and job.method == "global_loo":
                deranged_records.extend(
                    _read_derangement_artifacts(
                        output_root=output_root,
                        job=job,
                        payload=payload,
                        primary_rows=rows,
                    )
                )
            records.extend(rows)
            successful_jobs += 1
        except (FileNotFoundError, OSError) as exc:
            missing.append(
                {"job": job.to_json(), "error": type(exc).__name__}
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            failures.append(
                {
                    "job": job.to_json(),
                    "status": "integrity_failure",
                    "error": str(exc),
                }
            )
    return {
        "records": tuple(records),
        "deranged_records": tuple(deranged_records),
        "missing": missing,
        "failures": failures,
        "job_results": tuple(job_results),
        "infrastructure_attempts": tuple(infrastructure_attempts),
        "inprogress_jobs": tuple(inprogress_jobs),
        "successful_jobs": successful_jobs,
    }


def _adjudicate_discovery_records(
    stage: str, records: Sequence[object]
) -> dict[str, object]:
    config = load_protocol_config()
    specification = config["gates"][stage]
    main_spec = specification["effect_contrasts"]["main"]
    candidate = str(main_spec["candidate"])
    comparator = str(main_spec["comparator"])
    masks = tuple(main_spec["pooling_masks"])
    main = paired_contrast(
        records, candidate=candidate, comparator=comparator, pooling_masks=masks
    )
    per_mask = {
        mask: paired_contrast(
            records,
            candidate=candidate,
            comparator=comparator,
            pooling_masks=(mask,),
        ).improvement
        for mask in specification["per_mask_rule"]["masks"]
    }
    registry = stage_spec(stage)
    positive_count, seed_effects = positive_seed_count(
        records,
        candidate=candidate,
        comparator=comparator,
        pooling_masks=masks,
        seed_bundles=registry.seed_bundles,
    )
    bootstrap = _bootstrap_contrast(
        stage=stage,
        records=records,
        candidate=candidate,
        comparator=comparator,
        pooling_masks=masks,
    )
    random_effect = None
    if stage == "stage_h":
        random_spec = specification["effect_contrasts"]["random_no_harm"]
        random_effect = paired_contrast(
            records,
            candidate=str(random_spec["candidate"]),
            comparator=str(random_spec["comparator"]),
            pooling_masks=tuple(random_spec["pooling_masks"]),
        ).improvement
    decision = decide_discovery_gate(
        stage,
        main=main.improvement,
        per_mask=per_mask,
        ci_lower=bootstrap.ci_lower,
        positive_seed_count=positive_count,
        random_effect=random_effect,
    )
    return {
        "verdict": decision.verdict,
        "reason": (
            "all_registered_gate_clauses"
            if decision.verdict == "proceed"
            else "numeric_gate_failed"
        ),
        "evidence_type": "diagnostic" if stage == "stage_h" else "method",
        "main_contrast": asdict(main),
        "per_mask_improvement": per_mask,
        "random_improvement": random_effect,
        "positive_seed_count": positive_count,
        "seed_improvements": {str(key): value for key, value in seed_effects.items()},
        "bootstrap": asdict(bootstrap),
        "checks": dict(decision.checks),
        "cell_summaries": _summary_table(records, (comparator, candidate)),
    }


def _adjudicate_stage_i_records(
    records: Sequence[object], deranged_records: Sequence[object]
) -> dict[str, object]:
    config = load_protocol_config()
    specification = config["gates"]["stage_i"]
    effects = specification["effect_contrasts"]
    main_spec = effects["main"]
    candidate = str(main_spec["candidate"])
    comparator = str(main_spec["comparator"])
    masks = tuple(main_spec["pooling_masks"])
    combined = tuple(records) + tuple(deranged_records)
    main = paired_contrast(
        records, candidate=candidate, comparator=comparator, pooling_masks=masks
    )
    local_spec = effects["global_over_local"]
    global_over_local = paired_contrast(
        records,
        candidate=str(local_spec["candidate"]),
        comparator=str(local_spec["comparator"]),
        pooling_masks=tuple(local_spec["pooling_masks"]),
    )
    registry = stage_spec("stage_i")
    per_dataset_results = {
        dataset: paired_contrast(
            _records_for_dataset(records, dataset),
            candidate=candidate,
            comparator=comparator,
            pooling_masks=tuple(specification["per_dataset_rule"]["pooling_masks"]),
        )
        for dataset in registry.datasets
    }
    positive_count, seed_effects = positive_seed_count(
        records,
        candidate=candidate,
        comparator=comparator,
        pooling_masks=masks,
        seed_bundles=registry.seed_bundles,
    )
    bootstrap = _bootstrap_contrast(
        stage="stage_i",
        records=records,
        candidate=candidate,
        comparator=comparator,
        pooling_masks=masks,
    )
    deranged_spec = effects["derangement_gain_loss"]
    deranged = paired_contrast(
        combined,
        candidate=str(deranged_spec["comparator"]),
        comparator=comparator,
        pooling_masks=tuple(deranged_spec["pooling_masks"]),
    )
    gain_loss = main.improvement - deranged.improvement
    fraction = gain_loss / main.improvement if main.improvement > 0.0 else None
    per_dataset = {
        dataset: result.improvement
        for dataset, result in per_dataset_results.items()
    }
    decision = _decide_stage_i_gate(
        main=main.improvement,
        global_over_local=global_over_local.improvement,
        per_dataset=per_dataset,
        ci_lower=bootstrap.ci_lower,
        positive_seed_count=positive_count,
        unshuffled_main_gain=main.improvement,
        derangement_gain_loss=gain_loss,
        derangement_fraction=fraction,
    )
    return {
        "verdict": decision.verdict,
        "reason": {
            "proceed": "all_registered_gate_clauses",
            "revise": "primary_effect_in_single_revision_region",
            "kill": "numeric_gate_failed_outside_revision_region",
        }[decision.verdict],
        "evidence_type": "method",
        "main_comparator": comparator,
        "main_contrast": asdict(main),
        "global_over_local": asdict(global_over_local),
        "per_dataset": {
            dataset: asdict(result)
            for dataset, result in per_dataset_results.items()
        },
        "positive_seed_count": positive_count,
        "seed_improvements": {str(key): value for key, value in seed_effects.items()},
        "bootstrap": asdict(bootstrap),
        "derangement": {
            "unshuffled_improvement_over_acil": main.improvement,
            "deranged_improvement_over_acil": deranged.improvement,
            "gain_loss": gain_loss,
            "fraction": fraction,
            "record_count": len(deranged_records),
        },
        "revision_limit": int(specification["revision_limit"]),
        "checks": dict(decision.checks),
        "cell_summaries": _summary_table(
            combined, (comparator, "local_loo", candidate, "global_loo_deranged")
        ),
    }


def _adjudicate_full_tune_records(records: Sequence[object]) -> dict[str, object]:
    config = load_protocol_config()
    specification = config["gates"]["full_tune"]
    effects = specification["effect_contrasts"]
    main_spec = effects["main"]
    candidate = str(main_spec["candidate"])
    masks = tuple(main_spec["pooling_masks"])
    comparator = strongest_comparator(
        records,
        candidates=tuple(main_spec["comparator_candidates"]),
        pooling_masks=masks,
    )
    main = paired_contrast(
        records, candidate=candidate, comparator=comparator, pooling_masks=masks
    )
    deepsets_spec = effects["over_deepsets"]
    over_deepsets = paired_contrast(
        records,
        candidate=str(deepsets_spec["candidate"]),
        comparator=str(deepsets_spec["comparator"]),
        pooling_masks=tuple(deepsets_spec["pooling_masks"]),
    )
    pretraining_spec = effects["pretraining_attribution"]
    pretraining = paired_contrast(
        records,
        candidate=str(pretraining_spec["candidate"]),
        comparator=str(pretraining_spec["comparator"]),
        pooling_masks=tuple(pretraining_spec["pooling_masks"]),
    )
    random_spec = effects["random_no_harm"]
    random_comparator = strongest_comparator(
        records,
        candidates=tuple(random_spec["comparator_candidates"]),
        pooling_masks=tuple(random_spec["pooling_masks"]),
    )
    random_result = paired_contrast(
        records,
        candidate=str(random_spec["candidate"]),
        comparator=random_comparator,
        pooling_masks=tuple(random_spec["pooling_masks"]),
    )
    registry = stage_spec("full_tune")
    positive_count, seed_effects = positive_seed_count(
        records,
        candidate=candidate,
        comparator=comparator,
        pooling_masks=masks,
        seed_bundles=registry.seed_bundles,
    )
    bootstrap = _bootstrap_contrast(
        stage="full_tune",
        records=records,
        candidate=candidate,
        comparator=comparator,
        pooling_masks=masks,
    )
    pretraining_bootstrap = _bootstrap_contrast(
        stage="full_tune",
        records=records,
        candidate=str(pretraining_spec["candidate"]),
        comparator=str(pretraining_spec["comparator"]),
        pooling_masks=tuple(pretraining_spec["pooling_masks"]),
    )
    decision = _decide_full_tune_gate(
        main=main.improvement,
        over_deepsets=over_deepsets.improvement,
        pretraining_attribution=pretraining.improvement,
        random_effect=random_result.improvement,
        ci_lower=bootstrap.ci_lower,
        positive_seed_count=positive_count,
    )
    claim = _pretraining_claim(
        effect=pretraining.improvement, ci_lower=pretraining_bootstrap.ci_lower
    )
    return {
        "verdict": decision.verdict,
        "reason": (
            "all_registered_method_gate_clauses"
            if decision.verdict == "proceed"
            else "current_full_gpt_candidate_failed"
        ),
        "evidence_type": "method",
        "main_comparator": comparator,
        "random_comparator": random_comparator,
        "main_contrast": asdict(main),
        "over_deepsets": asdict(over_deepsets),
        "pretraining_attribution": {
            "contrast": asdict(pretraining),
            "bootstrap": asdict(pretraining_bootstrap),
            "claim": claim,
        },
        "random_no_harm": asdict(random_result),
        "positive_seed_count": positive_count,
        "seed_improvements": {str(key): value for key, value in seed_effects.items()},
        "bootstrap": asdict(bootstrap),
        "checks": dict(decision.checks),
        "cell_summaries": _summary_table(records, registry.methods),
    }


def _adjudicate_formal_records(records: Sequence[object]) -> dict[str, object]:
    config = load_protocol_config()
    specification = config["gates"]["formal_gate"]
    effects = specification["effect_contrasts"]
    registry = stage_spec("formal_gate")
    main_spec = effects["main"]
    candidate = str(main_spec["candidate"])
    masks = tuple(main_spec["pooling_masks"])
    comparator_candidates = tuple(main_spec["comparator_candidates"])
    comparator = strongest_comparator(
        records, candidates=comparator_candidates, pooling_masks=masks
    )
    main = paired_contrast(
        records, candidate=candidate, comparator=comparator, pooling_masks=masks
    )
    per_dataset: dict[str, ContrastResult] = {}
    per_dataset_comparators: dict[str, str] = {}
    for dataset in registry.datasets:
        subset = _records_for_dataset(records, dataset)
        selected = strongest_comparator(
            subset, candidates=comparator_candidates, pooling_masks=masks
        )
        per_dataset_comparators[dataset] = selected
        per_dataset[dataset] = paired_contrast(
            subset, candidate=candidate, comparator=selected, pooling_masks=masks
        )

    random_spec = effects["random_no_harm"]
    random_masks = tuple(random_spec["pooling_masks"])
    random_comparator = strongest_comparator(
        records,
        candidates=tuple(random_spec["comparator_candidates"]),
        pooling_masks=random_masks,
    )
    random_result = paired_contrast(
        records,
        candidate=str(random_spec["candidate"]),
        comparator=random_comparator,
        pooling_masks=random_masks,
    )

    cell_spec = effects["dataset_mask_cell_wins"]
    cell_results: list[dict[str, object]] = []
    for dataset in registry.datasets:
        subset = _records_for_dataset(records, dataset)
        for mask in tuple(cell_spec["pooling_masks"]):
            selected = strongest_comparator(
                subset,
                candidates=tuple(cell_spec["comparator_candidates"]),
                pooling_masks=(mask,),
            )
            contrast = paired_contrast(
                subset,
                candidate=str(cell_spec["candidate"]),
                comparator=selected,
                pooling_masks=(mask,),
            )
            cell_results.append(
                {
                    "dataset": dataset,
                    "mask_family": mask,
                    "comparator": selected,
                    "improvement": contrast.improvement,
                    "win": contrast.improvement > float(cell_spec["strictly_above"]),
                }
            )
    if len(cell_results) != int(cell_spec["total"]):
        raise RuntimeError("formal cell registry total drifted")
    cell_wins = sum(bool(item["win"]) for item in cell_results)

    external_spec = effects["external_no_harm"]
    external_masks = tuple(external_spec["pooling_masks"])
    external_candidates = tuple(external_spec["comparator_candidates"])
    external_comparator = strongest_comparator(
        records, candidates=external_candidates, pooling_masks=external_masks
    )
    external = paired_contrast(
        records,
        candidate=str(external_spec["candidate"]),
        comparator=external_comparator,
        pooling_masks=external_masks,
    )
    external_per_dataset: dict[str, ContrastResult] = {}
    external_per_dataset_comparators: dict[str, str] = {}
    for dataset in registry.datasets:
        subset = _records_for_dataset(records, dataset)
        selected = strongest_comparator(
            subset, candidates=external_candidates, pooling_masks=external_masks
        )
        external_per_dataset_comparators[dataset] = selected
        external_per_dataset[dataset] = paired_contrast(
            subset,
            candidate=str(external_spec["candidate"]),
            comparator=selected,
            pooling_masks=external_masks,
        )

    positive_count, seed_effects = positive_seed_count(
        records,
        candidate=candidate,
        comparator=comparator,
        pooling_masks=masks,
        seed_bundles=registry.seed_bundles,
    )
    bootstrap = _bootstrap_contrast(
        stage="formal_gate",
        records=records,
        candidate=candidate,
        comparator=comparator,
        pooling_masks=masks,
    )
    decision = _decide_formal_gate(
        main=main.improvement,
        per_dataset={key: value.improvement for key, value in per_dataset.items()},
        ci_lower=bootstrap.ci_lower,
        positive_seed_count=positive_count,
        random_effect=random_result.improvement,
        cell_wins=cell_wins,
        external_effect=external.improvement,
        external_per_dataset={
            key: value.improvement for key, value in external_per_dataset.items()
        },
    )
    return {
        "verdict": decision.verdict,
        "reason": (
            "all_registered_formal_gate_clauses"
            if decision.verdict == "proceed"
            else "formal_gate_failed_no_revision"
        ),
        "evidence_type": "method",
        "main_comparator": comparator,
        "main_contrast": asdict(main),
        "per_dataset_comparators": per_dataset_comparators,
        "per_dataset": {
            key: asdict(value) for key, value in per_dataset.items()
        },
        "random_comparator": random_comparator,
        "random_no_harm": asdict(random_result),
        "dataset_mask_cells": cell_results,
        "dataset_mask_cell_wins": cell_wins,
        "external_comparator": external_comparator,
        "external_no_harm": asdict(external),
        "external_per_dataset_comparators": external_per_dataset_comparators,
        "external_per_dataset": {
            key: asdict(value) for key, value in external_per_dataset.items()
        },
        "positive_seed_count": positive_count,
        "seed_improvements": {str(key): value for key, value in seed_effects.items()},
        "bootstrap": asdict(bootstrap),
        "checks": dict(decision.checks),
        "cell_summaries": _summary_table(records, registry.methods),
    }


def _base_adjudication(stage: str) -> dict[str, object]:
    return {
        "schema": "acil-innovation-v1:stage-adjudication:v2",
        "protocol": "acil-innovation-v1",
        "stage": stage,
        "planned_job_count": len(planned_jobs(stage)),
        "direct_upstream": None,
    }


def _direct_upstream_binding(
    stage: str, output_root: Path
) -> dict[str, object] | None:
    upstream = _DIRECT_UPSTREAM.get(stage)
    if upstream is None:
        return None
    payload = load_stage_adjudication(upstream, output_root)
    path = _adjudication_artifact_path(upstream, output_root, create=False)
    reread, artifact = _read_canonical_json(path)
    if reread != payload:
        raise ValueError("direct-upstream artifact changed after revalidation")
    result = {
        "stage": upstream,
        "path": f"_adjudication/{upstream}/adjudication.json",
        "sha256": artifact.sha256,
        "verdict": payload.get("verdict"),
        "freeze_sha256": payload.get("freeze_sha256"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "provenance_sha256": payload.get("provenance_sha256"),
    }
    for name in ("freeze_sha256", "manifest_sha256", "provenance_sha256"):
        value = result[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"direct-upstream {name} is invalid")
    return result


def adjudicate_stage(stage: str, output_root: Path) -> dict[str, object]:
    """Audit and adjudicate one exact frozen gate stage with no override surface."""

    if not isinstance(stage, str) or stage not in _GATE_STAGES:
        raise ValueError(f"unsupported frozen gate stage {stage!r}")
    output_root = Path(output_root)
    base = _base_adjudication(stage)
    try:
        manifest_sha256, provenance_sha256, freeze_sha256 = _active_freeze_hashes()
    except FileNotFoundError as exc:
        base.update(
            {
                "verdict": "blocked",
                "reason": "active_freeze_missing",
                "missing_jobs": [],
                "failures": [
                    {"status": "missing_artifact", "artifact": "active_freeze", "error": str(exc)}
                ],
                "inprogress_jobs": [],
                "successful_job_count": 0,
            }
        )
        return _canonical_report(base)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        base.update(
            {
                "verdict": "kill",
                "reason": "active_freeze_integrity_failure",
                "missing_jobs": [],
                "failures": [
                    {"status": "integrity_failure", "artifact": "active_freeze", "error": str(exc)}
                ],
                "inprogress_jobs": [],
                "successful_job_count": 0,
            }
        )
        return _canonical_report(base)

    base.update(
        {
            "manifest_sha256": manifest_sha256,
            "provenance_sha256": provenance_sha256,
            "freeze_sha256": freeze_sha256,
        }
    )
    try:
        direct_upstream = _direct_upstream_binding(stage, output_root)
    except FileNotFoundError as exc:
        base.update(
            {
                "verdict": "blocked",
                "reason": "direct_upstream_adjudication_missing",
                "missing_jobs": [],
                "failures": [
                    {
                        "status": "missing_artifact",
                        "artifact": "direct_upstream_adjudication",
                        "error": str(exc),
                    }
                ],
                "inprogress_jobs": [],
                "successful_job_count": 0,
            }
        )
        return _canonical_report(base)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        base.update(
            {
                "verdict": "kill",
                "reason": "direct_upstream_integrity_failure",
                "missing_jobs": [],
                "failures": [
                    {
                        "status": "integrity_failure",
                        "artifact": "direct_upstream_adjudication",
                        "error": str(exc),
                    }
                ],
                "inprogress_jobs": [],
                "successful_job_count": 0,
            }
        )
        return _canonical_report(base)
    base["direct_upstream"] = direct_upstream
    if direct_upstream is not None and direct_upstream.get("verdict") != "proceed":
        base.update(
            {
                "verdict": "blocked",
                "reason": "direct_upstream_did_not_proceed",
                "missing_jobs": [],
                "failures": [],
                "inprogress_jobs": [],
                "successful_job_count": 0,
            }
        )
        return _canonical_report(base)
    evidence = _collect_stage_evidence(
        stage=stage,
        output_root=output_root,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )
    failures = list(evidence["failures"])
    missing = list(evidence["missing"])
    base.update(
        {
            "missing_jobs": missing,
            "failures": failures,
            "successful_job_count": int(evidence["successful_jobs"]),
            "record_count": len(evidence["records"]),
            "job_results": list(evidence["job_results"]),
            "infrastructure_attempts": list(evidence["infrastructure_attempts"]),
            "inprogress_jobs": list(evidence.get("inprogress_jobs", ())),
        }
    )
    if failures:
        base.update(
            {"verdict": "kill", "reason": "failed_or_invalid_registered_job"}
        )
        return _canonical_report(base)
    if base["inprogress_jobs"]:
        base.update(
            {"verdict": "blocked", "reason": "registered_job_in_progress"}
        )
        return _canonical_report(base)
    if missing:
        base.update({"verdict": "blocked", "reason": "incomplete_job_grid"})
        return _canonical_report(base)
    try:
        if stage in {"stage0_acil_tune", "stage_h"}:
            result = _adjudicate_discovery_records(stage, evidence["records"])
        elif stage == "stage_i":
            result = _adjudicate_stage_i_records(
                evidence["records"], evidence["deranged_records"]
            )
        elif stage == "full_tune":
            result = _adjudicate_full_tune_records(evidence["records"])
        else:
            result = _adjudicate_formal_records(evidence["records"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        base["failures"] = [
            {
                "status": "integrity_failure",
                "artifact": "complete_stage_evidence",
                "error": str(exc),
            }
        ]
        base.update(
            {"verdict": "kill", "reason": "adjudication_integrity_failure"}
        )
        return _canonical_report(base)
    base.update(result)
    return _canonical_report(base)


def adjudicate_discovery_stage(stage: str, output_root: Path) -> dict[str, object]:
    """Backward-compatible Stage-0/H spelling bound to the generic auditor."""

    if stage not in {"stage0_acil_tune", "stage_h"}:
        raise ValueError("discovery adjudication supports only stage0_acil_tune or stage_h")
    return adjudicate_stage(stage, output_root)


def _adjudication_artifact_path(
    stage: str, output_root: Path, *, create: bool
) -> Path:
    output_root = Path(output_root)
    if create and not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
    for directory in (
        output_root,
        output_root / "_adjudication",
        output_root / "_adjudication" / stage,
    ):
        try:
            metadata = directory.lstat()
        except FileNotFoundError:
            if not create or directory == output_root:
                raise
            directory.mkdir(exist_ok=False)
            metadata = directory.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("adjudication artifact ancestry must use plain directories")
    return output_root / "_adjudication" / stage / "adjudication.json"


def write_stage_adjudication(stage: str, output_root: Path) -> Path:
    """Write the sole immutable canonical artifact for a completed stage."""

    report = adjudicate_stage(stage, output_root)
    if report["verdict"] == "blocked":
        raise RuntimeError("a blocked stage cannot publish a final adjudication artifact")
    path = _adjudication_artifact_path(stage, output_root, create=True)
    try:
        write_canonical_json_exclusive(path, report)
    except FileExistsError:
        value = load_stage_adjudication(stage, output_root)
        if value != report:
            raise ValueError("existing adjudication artifact content drifted")
    return path


def load_stage_adjudication(stage: str, output_root: Path) -> dict[str, object]:
    """Load and recompute-verify one upstream stage decision for queue gating."""

    if not isinstance(stage, str) or stage not in _GATE_STAGES:
        raise ValueError(f"unsupported active-manifest stage {stage!r}")
    path = _adjudication_artifact_path(stage, output_root, create=False)
    payload, _ = _read_canonical_json(path)
    if not isinstance(payload, dict):
        raise ValueError("stage adjudication artifact must be a JSON object")
    recomputed = adjudicate_stage(stage, output_root)
    if payload != recomputed:
        raise ValueError(
            "stage adjudication no longer matches the active freeze and job artifacts"
        )
    if payload.get("verdict") == "blocked":
        raise ValueError("a published final adjudication cannot be blocked")
    return _canonical_report(payload)


__all__ = [
    "ContrastResult",
    "GateDecision",
    "adjudicate_discovery_stage",
    "adjudicate_stage",
    "decide_discovery_gate",
    "load_stage_adjudication",
    "paired_contrast",
    "positive_seed_count",
    "strongest_comparator",
    "write_stage_adjudication",
]
