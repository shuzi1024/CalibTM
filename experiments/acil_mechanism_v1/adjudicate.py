"""Paired ratio-of-sums adjudication for ACIL matched-feature controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .run_job import expected_job_identities


_ROOT = Path(__file__).resolve().parent
_CARD = _ROOT / "research_card.json"
_DATASETS = ("abilene", "geant")
_METHODS = ("full", "value_only", "no_anchor")
_FAMILIES = ("random", "internal_block", "two_burst")
_STRUCTURED = ("internal_block", "two_burst")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _card() -> dict[str, Any]:
    value = json.loads(_CARD.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("protocol") != "acil-mechanism-v1"
    ):
        raise ValueError("research card identity drifted")
    return value


def _array(value: object, *, label: str, integer: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64 if integer else np.float64)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{label} must be a nonempty vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def _identity_key(job: object) -> str:
    if not isinstance(job, dict):
        raise ValueError("result job identity is invalid")
    return _canonical_json(job).decode("ascii")


def _extract_table(
    payloads: Iterable[dict[str, object]], stage: str
) -> tuple[
    dict[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    tuple[int, ...],
]:
    values = tuple(payloads)
    expected = expected_job_identities(stage)
    expected_keys = {_identity_key(job) for job in expected}
    actual_keys = [_identity_key(value.get("job")) for value in values]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise ValueError("result identity grid is missing, duplicate or extra")

    seeds = (1,) if stage == "prototype" else (1, 2, 3)
    by_job = {
        (
            str(value["job"]["dataset"]),
            int(value["job"]["seed_bundle"]),
            str(value["job"]["method"]),
        ): value
        for value in values
    }
    table: dict[
        tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]
    ] = {}
    for dataset in _DATASETS:
        for bundle in seeds:
            jobs = {
                method: by_job[(dataset, bundle, method)]
                for method in _METHODS
            }
            cells = {method: jobs[method].get("cells") for method in _METHODS}
            if any(not isinstance(value, dict) for value in cells.values()):
                raise ValueError("result cells are invalid")
            if any(set(value) != set(_FAMILIES) for value in cells.values()):
                raise ValueError("result mask-family grid drifted")
            for family in _FAMILIES:
                reference = cells["full"][family]
                if not isinstance(reference, dict):
                    raise ValueError("result cell is invalid")
                paired_fields = (
                    "linear_error_sum_per_window",
                    "mask_grid_sha256",
                    "target_count_per_window",
                    "truth_sum_per_window",
                    "window_starts",
                )
                for method in _METHODS[1:]:
                    other = cells[method][family]
                    if not isinstance(other, dict):
                        raise ValueError("result cell is invalid")
                    for field in paired_fields:
                        if reference.get(field) != other.get(field):
                            raise ValueError(
                                f"paired feature jobs changed {field}"
                            )
                truth = _array(
                    reference["truth_sum_per_window"],
                    label="truth sums",
                )
                if np.any(truth <= 0.0):
                    raise ValueError("truth sums must be positive")
                target_count = _array(
                    reference["target_count_per_window"],
                    label="target counts",
                    integer=True,
                )
                starts = _array(
                    reference["window_starts"],
                    label="window starts",
                    integer=True,
                )
                if np.any(target_count <= 0):
                    raise ValueError("target counts must be positive")
                for method in (*_METHODS, "linear"):
                    cell = (
                        reference
                        if method == "linear"
                        else cells[method][family]
                    )
                    field = (
                        "linear_error_sum_per_window"
                        if method == "linear"
                        else "model_error_sum_per_window"
                    )
                    error = _array(cell[field], label=f"{method} errors")
                    if len(error) != len(truth) or len(starts) != len(truth):
                        raise ValueError("paired evidence vector lengths drifted")
                    if np.any(error < 0.0):
                        raise ValueError("error sums must be nonnegative")
                    table[(dataset, bundle, family, method)] = (
                        error,
                        truth,
                    )
    return table, seeds


def _dataset_effect(
    table,
    *,
    comparator: str,
    dataset: str,
    families: tuple[str, ...],
    selections: tuple[tuple[int, np.ndarray], ...],
) -> float:
    scores = {}
    for method in ("full", comparator):
        errors = []
        truths = []
        for bundle, positions in selections:
            for family in families:
                error, truth = table[(dataset, bundle, family, method)]
                errors.append(float(np.sum(error[positions], dtype=np.float64)))
                truths.append(float(np.sum(truth[positions], dtype=np.float64)))
        denominator = math.fsum(truths)
        if denominator <= 0.0:
            raise ValueError("pooled truth denominator is zero")
        scores[method] = math.fsum(errors) / denominator
    if scores[comparator] <= 0.0:
        raise ValueError("comparator NMAE is zero")
    return 1.0 - scores["full"] / scores[comparator]


def _all_positions(table, dataset: str, seeds: tuple[int, ...]):
    return tuple(
        (
            bundle,
            np.arange(
                len(table[(dataset, bundle, _STRUCTURED[0], "full")][0]),
                dtype=np.int64,
            ),
        )
        for bundle in seeds
    )


def _dataset_equal_effect(
    table,
    *,
    comparator: str,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[float, dict[str, float]]:
    per_dataset = {
        dataset: _dataset_effect(
            table,
            comparator=comparator,
            dataset=dataset,
            families=families,
            selections=_all_positions(table, dataset, seeds),
        )
        for dataset in _DATASETS
    }
    return math.fsum(per_dataset.values()) / 2.0, per_dataset


def _bootstrap(
    table,
    *,
    stage: str,
    draws: int,
) -> dict[str, object]:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("bootstrap draws must be positive")
    seeds = (1,) if stage == "prototype" else (1, 2, 3)
    rng = np.random.Generator(np.random.PCG64DXSM(83001))
    effects = {
        comparator: np.empty(draws, dtype="<f8")
        for comparator in ("value_only", "no_anchor")
    }
    plan = hashlib.sha256(b"acil-mechanism-v1:bootstrap-plan:v1\x00")
    for draw_index in range(draws):
        sampled = (
            np.array([1], dtype=np.int64)
            if stage == "prototype"
            else rng.choice(np.array(seeds), size=3, replace=True)
        )
        plan.update(draw_index.to_bytes(8, "big"))
        plan.update(sampled.astype("<i8", copy=False).tobytes())
        selections_by_dataset = {}
        for dataset in _DATASETS:
            selections = []
            for occurrence, bundle in enumerate(sampled):
                count = len(
                    table[
                        (
                            dataset,
                            int(bundle),
                            _STRUCTURED[0],
                            "full",
                        )
                    ][0]
                )
                origins = rng.integers(
                    0,
                    count,
                    size=math.ceil(count / 4),
                )
                positions = np.concatenate(
                    [
                        (int(origin) + np.arange(4, dtype=np.int64))
                        % count
                        for origin in origins
                    ]
                )[:count]
                plan.update(dataset.encode("ascii"))
                plan.update(occurrence.to_bytes(8, "big"))
                plan.update(positions.astype("<i8", copy=False).tobytes())
                selections.append((int(bundle), positions))
            selections_by_dataset[dataset] = tuple(selections)
        for comparator in effects:
            dataset_effects = [
                _dataset_effect(
                    table,
                    comparator=comparator,
                    dataset=dataset,
                    families=_STRUCTURED,
                    selections=selections_by_dataset[dataset],
                )
                for dataset in _DATASETS
            ]
            effects[comparator][draw_index] = (
                math.fsum(dataset_effects) / 2.0
            )
    contrasts = {}
    for comparator, values in effects.items():
        lower, upper = np.quantile(
            values,
            (0.025, 0.975),
            method="linear",
        )
        contrasts[f"full_over_{comparator}"] = {
            "draws_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "lower": float(lower),
            "upper": float(upper),
        }
    return {
        "contrasts": contrasts,
        "draw_plan_sha256": plan.hexdigest(),
        "draws": draws,
        "rng": "PCG64DXSM",
        "seed": 83001,
        "window_block_length": 4,
    }


def _gate(
    actual: float | int,
    operator: str,
    threshold: float | int,
) -> dict[str, object]:
    if operator == ">=":
        passed = actual >= threshold
    elif operator == ">":
        passed = actual > threshold
    elif operator == "==":
        passed = actual == threshold
    else:
        raise ValueError("unknown gate operator")
    return {
        "actual": actual,
        "operator": operator,
        "pass": bool(passed),
        "threshold": threshold,
    }


def adjudicate_payloads(
    payloads: Iterable[dict[str, object]],
    *,
    stage: str,
    bootstrap_draws: int = 10000,
) -> dict[str, object]:
    if stage not in {"prototype", "extension"}:
        raise ValueError("stage must be prototype or extension")
    table, seeds = _extract_table(payloads, stage)
    effects = {}
    per_dataset = {}
    for comparator in ("value_only", "no_anchor", "linear"):
        effect, datasets = _dataset_equal_effect(
            table,
            comparator=comparator,
            families=_STRUCTURED,
            seeds=seeds,
        )
        effects[comparator] = effect
        per_dataset[comparator] = datasets

    per_cell = {
        f"full_over_{comparator}/{dataset}/{family}": _dataset_effect(
            table,
            comparator=comparator,
            dataset=dataset,
            families=(family,),
            selections=_all_positions(table, dataset, seeds),
        )
        for comparator in ("value_only", "no_anchor")
        for dataset in _DATASETS
        for family in _STRUCTURED
    }
    random_per_dataset = {
        f"full_over_{comparator}/{dataset}": _dataset_effect(
            table,
            comparator=comparator,
            dataset=dataset,
            families=("random",),
            selections=_all_positions(table, dataset, seeds),
        )
        for comparator in ("value_only", "no_anchor")
        for dataset in _DATASETS
    }
    per_dataset_seed = {}
    for dataset in _DATASETS:
        for bundle in seeds:
            count = len(
                table[(dataset, bundle, _STRUCTURED[0], "full")][0]
            )
            selection = ((bundle, np.arange(count, dtype=np.int64)),)
            per_dataset_seed[f"{dataset}/seed{bundle}"] = {
                comparator: _dataset_effect(
                    table,
                    comparator=comparator,
                    dataset=dataset,
                    families=_STRUCTURED,
                    selections=selection,
                )
                for comparator in ("value_only", "no_anchor")
            }
    positive_both = sum(
        value["value_only"] > 0.0 and value["no_anchor"] > 0.0
        for value in per_dataset_seed.values()
    )
    dataset_both = sum(
        per_dataset["value_only"][dataset] > 0.0
        and per_dataset["no_anchor"][dataset] > 0.0
        for dataset in _DATASETS
    )
    bootstrap = _bootstrap(
        table,
        stage=stage,
        draws=bootstrap_draws,
    )

    specification = _card()[f"{stage}_gates"]
    expected_gate_keys = {
        "full_over_value_structured_min",
        "full_over_value_bootstrap_lower_strictly_positive",
        "full_over_no_anchor_structured_min",
        "full_over_no_anchor_bootstrap_lower_strictly_positive",
        "full_over_linear_structured_min",
        "full_over_linear_each_dataset_positive",
        "both_matched_controls_each_dataset_strictly_positive",
        "worst_matched_structured_cell_min",
        "worst_matched_random_dataset_min",
        "positive_dataset_seed_for_both_controls_required",
        "positive_dataset_seed_total",
    }
    if set(specification) != expected_gate_keys:
        raise ValueError("research-card gate registry drifted")
    gates = {
        "full_over_value_structured_min": _gate(
            effects["value_only"],
            ">=",
            specification["full_over_value_structured_min"],
        ),
        "full_over_value_bootstrap_lower_strictly_positive": _gate(
            bootstrap["contrasts"]["full_over_value_only"]["lower"],
            ">",
            specification[
                "full_over_value_bootstrap_lower_strictly_positive"
            ],
        ),
        "full_over_no_anchor_structured_min": _gate(
            effects["no_anchor"],
            ">=",
            specification["full_over_no_anchor_structured_min"],
        ),
        "full_over_no_anchor_bootstrap_lower_strictly_positive": _gate(
            bootstrap["contrasts"]["full_over_no_anchor"]["lower"],
            ">",
            specification[
                "full_over_no_anchor_bootstrap_lower_strictly_positive"
            ],
        ),
        "full_over_linear_structured_min": _gate(
            effects["linear"],
            ">=",
            specification["full_over_linear_structured_min"],
        ),
        "full_over_linear_each_dataset_positive": _gate(
            sum(value > 0.0 for value in per_dataset["linear"].values()),
            "==",
            specification["full_over_linear_each_dataset_positive"],
        ),
        "both_matched_controls_each_dataset_strictly_positive": _gate(
            dataset_both,
            "==",
            specification[
                "both_matched_controls_each_dataset_strictly_positive"
            ],
        ),
        "worst_matched_structured_cell_min": _gate(
            min(per_cell.values()),
            ">=",
            specification["worst_matched_structured_cell_min"],
        ),
        "worst_matched_random_dataset_min": _gate(
            min(random_per_dataset.values()),
            ">=",
            specification["worst_matched_random_dataset_min"],
        ),
        "positive_dataset_seed_for_both_controls_required": _gate(
            positive_both,
            ">=",
            specification[
                "positive_dataset_seed_for_both_controls_required"
            ],
        ),
        "positive_dataset_seed_total": _gate(
            len(per_dataset_seed),
            "==",
            specification["positive_dataset_seed_total"],
        ),
    }
    passed = all(value["pass"] for value in gates.values())
    return {
        "bootstrap": bootstrap,
        "gates": gates,
        "headline": {
            "full_over_linear": {
                "actual": effects["linear"],
                "per_dataset": per_dataset["linear"],
            },
            "full_over_no_anchor": {
                "actual": effects["no_anchor"],
                "per_dataset": per_dataset["no_anchor"],
            },
            "full_over_value_only": {
                "actual": effects["value_only"],
                "per_dataset": per_dataset["value_only"],
            },
        },
        "paper_evidence": bool(stage == "extension" and passed),
        "per_dataset_seed": per_dataset_seed,
        "random_per_dataset": random_per_dataset,
        "reason": (
            f"{stage}_gate_passed"
            if passed
            else "acil_feature_attribution_gate_failure"
        ),
        "schema": "acil-mechanism-v1:adjudication:v1",
        "stage": stage,
        "structured_per_cell": per_cell,
        "verdict": "proceed" if passed else "kill",
    }


def _load_external(
    path: Path, expected_freeze: dict[str, str]
) -> dict[str, object]:
    from experiments.acil_innovation_v1.registries import seed_bundle
    from experiments.anchorcv_v1.data_access import canonical_data_identity
    from experiments.sc_acil_v1.checkpoint import load_checkpoint

    from .model import new_acil

    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol") != "acil-mechanism-v1"
        or payload.get("status") != "succeeded"
        or payload.get("test_access") is not False
        or payload.get("probe_freeze") != expected_freeze
        or manifest.get("job") != payload.get("job")
        or manifest.get("probe_freeze") != expected_freeze
        or manifest.get("result_sha256") != _file_sha256(path)
    ):
        raise ValueError("result or manifest identity is invalid")
    training = payload.get("training")
    if (
        not isinstance(training, dict)
        or training.get("epochs_completed") != 20
        or training.get("optimizer_updates") != 320
        or training.get("parameter_count") != 5475
    ):
        raise ValueError("training budget or parameter count drifted")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint descriptor is invalid")
    job = payload.get("job")
    if not isinstance(job, dict):
        raise ValueError("job identity is invalid")
    feature_sets = {
        "full": "full",
        "value_only": "value_only",
        "no_anchor": "no_anchor_values",
    }
    method = str(job.get("method"))
    try:
        expected_feature_set = feature_sets[method]
    except KeyError:
        raise ValueError("method has no registered feature set") from None
    if payload.get("feature_set") != expected_feature_set:
        raise ValueError("method and feature_set drifted")
    dataset = str(job.get("dataset"))
    if payload.get("data_identity") != canonical_data_identity(dataset):
        raise ValueError("data identity drifted")
    if (
        payload.get("selection_cohort") != "source_dev"
        or payload.get("evaluation_cohort") != "tune"
    ):
        raise ValueError("data cohort identity drifted")
    identity = checkpoint.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("job") != payload.get("job")
        or identity.get("probe_freeze") != expected_freeze
        or identity.get("best_epoch") != training.get("best_epoch")
        or identity.get("source_dev_nmae")
        != training.get("best_source_dev_nmae")
        or identity.get("feature_set") != expected_feature_set
    ):
        raise ValueError("checkpoint identity is not cross-bound to the result")
    try:
        metadata_relative = Path(str(checkpoint["metadata"]))
        weights_relative = Path(str(checkpoint["weights"]))
        if (
            metadata_relative.is_absolute()
            or weights_relative.is_absolute()
            or ".." in metadata_relative.parts
            or ".." in weights_relative.parts
        ):
            raise ValueError("checkpoint paths must be confined relative paths")
        metadata_path = path.parent / metadata_relative
        bundle = int(job["seed_bundle"])
        model = new_acil(
            method,
            model_seed=seed_bundle(bundle).model,
        )
        record = load_checkpoint(
            model,
            metadata_path=metadata_path,
            expected_identity=identity,
        )
        if (
            record.file_sha256 != checkpoint.get("file_sha256")
            or record.identity_sha256 != checkpoint.get("identity_sha256")
            or record.tensor_sha256 != checkpoint.get("tensor_sha256")
            or record.weights_path != path.parent / weights_relative
        ):
            raise ValueError("checkpoint descriptor hash or path drifted")
    except Exception as exc:
        raise ValueError(f"checkpoint verification failed: {exc}") from exc
    return payload


def _adjudicate_external(
    paths: Iterable[Path],
    *,
    stage: str,
    expected_freeze: dict[str, str],
    bootstrap_draws: int = 10000,
) -> dict[str, object]:
    payloads = []
    issues = []
    for path in paths:
        try:
            payloads.append(_load_external(Path(path), expected_freeze))
        except Exception as exc:
            issues.append(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "path": str(path),
                }
            )
    expected = len(expected_job_identities(stage))
    if issues or len(payloads) != expected:
        return {
            "expected_primary_jobs": expected,
            "gates": {},
            "issues": issues
            or [{"error": "result count differs from frozen grid"}],
            "paper_evidence": False,
            "reason": "incomplete_or_identity_invalid",
            "schema": "acil-mechanism-v1:adjudication:v1",
            "stage": stage,
            "valid_primary_jobs": len(payloads),
            "verdict": "revise",
        }
    try:
        return adjudicate_payloads(
            payloads,
            stage=stage,
            bootstrap_draws=bootstrap_draws,
        )
    except Exception as exc:
        return {
            "expected_primary_jobs": expected,
            "gates": {},
            "issues": [
                {"error": f"{type(exc).__name__}: {exc}", "path": "paired_grid"}
            ],
            "paper_evidence": False,
            "reason": "incomplete_or_identity_invalid",
            "schema": "acil-mechanism-v1:adjudication:v1",
            "stage": stage,
            "valid_primary_jobs": len(payloads),
            "verdict": "revise",
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prototype", "extension"), required=True
    )
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", default=10000, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from .freeze import probe_identity
    from .run_job import _write_json_exclusive

    paths = sorted(
        path
        for path in args.result_root.glob("*.json")
        if not path.name.endswith(".manifest.json")
        and path.name != args.output.name
    )
    report = _adjudicate_external(
        paths,
        stage=args.stage,
        expected_freeze=probe_identity(),
        bootstrap_draws=args.bootstrap_draws,
    )
    _write_json_exclusive(args.output, report)
    _write_json_exclusive(
        args.output.with_name(args.output.name + ".manifest.json"),
        {
            "adjudication_file": args.output.name,
            "adjudication_sha256": _file_sha256(args.output),
            "freeze": probe_identity(),
            "protocol": "acil-mechanism-v1",
            "schema": "acil-mechanism-v1:adjudication-manifest:v1",
        },
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["verdict"] in {"proceed", "kill"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_adjudicate_external",
    "adjudicate_payloads",
    "build_parser",
]
