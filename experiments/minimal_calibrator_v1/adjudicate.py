"""Paired ratio-of-sums adjudication for minimal calibration controls."""

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
_METHODS = ("static_zero", "blinear_only", "value_only")
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
        or value.get("protocol") != "minimal-calibrator-v1"
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
    payloads: Iterable[dict[str, object]],
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    values = tuple(payloads)
    expected_keys = {
        _identity_key(job) for job in expected_job_identities()
    }
    actual_keys = [_identity_key(value.get("job")) for value in values]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise ValueError("result identity grid is missing, duplicate or extra")

    by_job = {
        (str(value["job"]["dataset"]), str(value["job"]["method"])): value
        for value in values
    }
    table: dict[
        tuple[str, str, str], tuple[np.ndarray, np.ndarray]
    ] = {}
    for dataset in _DATASETS:
        jobs = {
            method: by_job[(dataset, method)]
            for method in _METHODS
        }
        cells = {method: jobs[method].get("cells") for method in _METHODS}
        if any(not isinstance(value, dict) for value in cells.values()):
            raise ValueError("result cells are invalid")
        if any(set(value) != set(_FAMILIES) for value in cells.values()):
            raise ValueError("result mask-family grid drifted")
        reference_starts: list[int] | None = None
        for family in _FAMILIES:
            reference = cells["static_zero"][family]
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
                            f"paired calibrator jobs changed {field}"
                        )
            truth = _array(
                reference["truth_sum_per_window"],
                label="truth sums",
            )
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
            if np.any(truth <= 0.0) or np.any(target_count <= 0):
                raise ValueError("truth sums and target counts must be positive")
            if reference_starts is None:
                reference_starts = starts.tolist()
            elif starts.tolist() != reference_starts:
                raise ValueError("mask families changed evaluation windows")
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
                table[(dataset, family, method)] = (error, truth)
    return table


def _dataset_effect(
    table,
    *,
    candidate: str,
    comparator: str,
    dataset: str,
    families: tuple[str, ...],
    positions: np.ndarray,
) -> float:
    scores = {}
    for method in (candidate, comparator):
        errors = []
        truths = []
        for family in families:
            error, truth = table[(dataset, family, method)]
            errors.append(float(np.sum(error[positions], dtype=np.float64)))
            truths.append(float(np.sum(truth[positions], dtype=np.float64)))
        denominator = math.fsum(truths)
        if denominator <= 0.0:
            raise ValueError("pooled truth denominator is zero")
        scores[method] = math.fsum(errors) / denominator
    if scores[comparator] <= 0.0:
        raise ValueError("comparator NMAE is zero")
    return 1.0 - scores[candidate] / scores[comparator]


def _all_positions(table, dataset: str) -> np.ndarray:
    return np.arange(
        len(table[(dataset, _STRUCTURED[0], "static_zero")][0]),
        dtype=np.int64,
    )


def _dataset_equal_effect(
    table,
    *,
    candidate: str,
    comparator: str,
    families: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    per_dataset = {
        dataset: _dataset_effect(
            table,
            candidate=candidate,
            comparator=comparator,
            dataset=dataset,
            families=families,
            positions=_all_positions(table, dataset),
        )
        for dataset in _DATASETS
    }
    return math.fsum(per_dataset.values()) / 2.0, per_dataset


def _bootstrap(
    table,
    *,
    draws: int,
) -> dict[str, object]:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("bootstrap draws must be positive")
    contrasts = {
        "value_over_static": ("value_only", "static_zero"),
        "blinear_over_static": ("blinear_only", "static_zero"),
        "blinear_over_value": ("blinear_only", "value_only"),
    }
    samples = {
        name: np.empty(draws, dtype="<f8")
        for name in contrasts
    }
    rng = np.random.Generator(np.random.PCG64DXSM(83002))
    plan = hashlib.sha256(b"minimal-calibrator-v1:bootstrap-plan:v1\x00")
    for draw_index in range(draws):
        positions_by_dataset = {}
        plan.update(draw_index.to_bytes(8, "big"))
        for dataset in _DATASETS:
            count = len(
                table[(dataset, _STRUCTURED[0], "static_zero")][0]
            )
            origins = rng.integers(0, count, size=math.ceil(count / 4))
            positions = np.concatenate(
                [
                    (int(origin) + np.arange(4, dtype=np.int64)) % count
                    for origin in origins
                ]
            )[:count]
            plan.update(dataset.encode("ascii"))
            plan.update(positions.astype("<i8", copy=False).tobytes())
            positions_by_dataset[dataset] = positions
        for name, (candidate, comparator) in contrasts.items():
            effects = [
                _dataset_effect(
                    table,
                    candidate=candidate,
                    comparator=comparator,
                    dataset=dataset,
                    families=_STRUCTURED,
                    positions=positions_by_dataset[dataset],
                )
                for dataset in _DATASETS
            ]
            samples[name][draw_index] = math.fsum(effects) / 2.0
    intervals = {}
    for name, values in samples.items():
        lower, upper = np.quantile(
            values,
            (0.025, 0.975),
            method="linear",
        )
        intervals[name] = {
            "draws_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "lower": float(lower),
            "upper": float(upper),
        }
    return {
        "contrasts": intervals,
        "draw_plan_sha256": plan.hexdigest(),
        "draws": draws,
        "rng": "PCG64DXSM",
        "seed": 83002,
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
    bootstrap_draws: int = 10000,
) -> dict[str, object]:
    table = _extract_table(payloads)
    pairs = {
        "value_over_static": ("value_only", "static_zero"),
        "blinear_over_static": ("blinear_only", "static_zero"),
        "blinear_over_value": ("blinear_only", "value_only"),
        "value_over_linear": ("value_only", "linear"),
    }
    effects = {}
    per_dataset = {}
    for name, (candidate, comparator) in pairs.items():
        effect, datasets = _dataset_equal_effect(
            table,
            candidate=candidate,
            comparator=comparator,
            families=_STRUCTURED,
        )
        effects[name] = effect
        per_dataset[name] = datasets

    per_cell = {
        f"{name}/{dataset}/{family}": _dataset_effect(
            table,
            candidate=candidate,
            comparator=comparator,
            dataset=dataset,
            families=(family,),
            positions=_all_positions(table, dataset),
        )
        for name, (candidate, comparator) in (
            ("value_over_static", pairs["value_over_static"]),
            ("blinear_over_static", pairs["blinear_over_static"]),
        )
        for dataset in _DATASETS
        for family in _STRUCTURED
    }
    random_per_dataset = {
        f"{name}/{dataset}": _dataset_effect(
            table,
            candidate=candidate,
            comparator=comparator,
            dataset=dataset,
            families=("random",),
            positions=_all_positions(table, dataset),
        )
        for name, (candidate, comparator) in (
            ("value_over_static", pairs["value_over_static"]),
            ("blinear_over_static", pairs["blinear_over_static"]),
        )
        for dataset in _DATASETS
    }
    bootstrap = _bootstrap(table, draws=bootstrap_draws)
    specification = _card()["gates"]
    kill_spec = specification["kill_point_effects"]
    primary_spec = specification["primary_robustness"]
    blinear_spec = specification["blinear_simplification"]

    point_gates = {
        "value_over_static_structured_min": _gate(
            effects["value_over_static"],
            ">=",
            kill_spec["value_over_static_structured_min"],
        ),
        "value_over_linear_structured_min": _gate(
            effects["value_over_linear"],
            ">=",
            kill_spec["value_over_linear_structured_min"],
        ),
    }
    robustness_gates = {
        "value_over_static_bootstrap_lower_strictly_positive": _gate(
            bootstrap["contrasts"]["value_over_static"]["lower"],
            ">",
            primary_spec[
                "value_over_static_bootstrap_lower_strictly_positive"
            ],
        ),
        "value_over_static_each_dataset_positive": _gate(
            sum(
                value > 0.0
                for value in per_dataset["value_over_static"].values()
            ),
            "==",
            primary_spec["value_over_static_each_dataset_positive"],
        ),
        "value_over_linear_each_dataset_positive": _gate(
            sum(
                value > 0.0
                for value in per_dataset["value_over_linear"].values()
            ),
            "==",
            primary_spec["value_over_linear_each_dataset_positive"],
        ),
        "worst_value_over_static_structured_cell_min": _gate(
            min(
                value
                for key, value in per_cell.items()
                if key.startswith("value_over_static/")
            ),
            ">=",
            primary_spec["worst_value_over_static_structured_cell_min"],
        ),
        "worst_value_over_static_random_dataset_min": _gate(
            min(
                value
                for key, value in random_per_dataset.items()
                if key.startswith("value_over_static/")
            ),
            ">=",
            primary_spec["worst_value_over_static_random_dataset_min"],
        ),
    }
    primary_gates = {**point_gates, **robustness_gates}
    blinear_gates = {
        "blinear_over_static_structured_min": _gate(
            effects["blinear_over_static"],
            ">=",
            blinear_spec["blinear_over_static_structured_min"],
        ),
        "blinear_over_static_bootstrap_lower_strictly_positive": _gate(
            bootstrap["contrasts"]["blinear_over_static"]["lower"],
            ">",
            blinear_spec[
                "blinear_over_static_bootstrap_lower_strictly_positive"
            ],
        ),
        "blinear_over_static_each_dataset_positive": _gate(
            sum(
                value > 0.0
                for value in per_dataset["blinear_over_static"].values()
            ),
            "==",
            blinear_spec["blinear_over_static_each_dataset_positive"],
        ),
        "blinear_noninferior_to_value": _gate(
            bootstrap["contrasts"]["blinear_over_value"]["lower"],
            ">=",
            blinear_spec[
                "blinear_noninferior_to_value_bootstrap_lower_min"
            ],
        ),
        "worst_blinear_over_static_structured_cell_min": _gate(
            min(
                value
                for key, value in per_cell.items()
                if key.startswith("blinear_over_static/")
            ),
            ">=",
            blinear_spec["worst_blinear_over_static_structured_cell_min"],
        ),
        "worst_blinear_over_static_random_dataset_min": _gate(
            min(
                value
                for key, value in random_per_dataset.items()
                if key.startswith("blinear_over_static/")
            ),
            ">=",
            blinear_spec["worst_blinear_over_static_random_dataset_min"],
        ),
    }
    if not all(item["pass"] for item in point_gates.values()):
        verdict = "kill"
        reason = "input_conditioning_effect_too_small"
    elif not all(item["pass"] for item in primary_gates.values()):
        verdict = "revise"
        reason = "input_conditioning_robustness_failure"
    elif all(item["pass"] for item in blinear_gates.values()):
        verdict = "proceed"
        reason = "blinear_conditioning_supported"
    else:
        verdict = "proceed"
        reason = "value_scale_conditioning_supported"
    return {
        "blinear_gates": blinear_gates,
        "bootstrap": bootstrap,
        "gates": {**primary_gates, **blinear_gates},
        "headline": {
            name: {
                "actual": effects[name],
                "per_dataset": per_dataset[name],
            }
            for name in pairs
        },
        "paper_evidence": False,
        "primary_gates": primary_gates,
        "random_per_dataset": random_per_dataset,
        "reason": reason,
        "schema": "minimal-calibrator-v1:adjudication:v1",
        "stage": "prototype",
        "structured_per_cell": per_cell,
        "verdict": verdict,
    }


def _validate_training(training: object) -> None:
    if (
        not isinstance(training, dict)
        or training.get("epochs_completed") != 20
        or training.get("optimizer_updates") != 320
        or training.get("parameter_count") != 5475
    ):
        raise ValueError("training budget or parameter count drifted")
    epochs = training.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != 20:
        raise ValueError("training epoch ledger drifted")
    selected = []
    source_dev_scores = []
    for index, row in enumerate(epochs):
        if (
            not isinstance(row, dict)
            or row.get("epoch") != index
            or row.get("optimizer_updates") != 16
            or row.get("physical_batches") != 64
        ):
            raise ValueError("training epoch ledger drifted")
        error = row.get("source_dev_absolute_error_sum")
        truth = row.get("source_dev_absolute_truth_sum")
        nmae = row.get("source_dev_nmae")
        if (
            isinstance(error, bool)
            or isinstance(truth, bool)
            or isinstance(nmae, bool)
            or not isinstance(error, (int, float))
            or not isinstance(truth, (int, float))
            or not isinstance(nmae, (int, float))
            or not math.isfinite(float(error))
            or not math.isfinite(float(truth))
            or not math.isfinite(float(nmae))
            or float(truth) <= 0.0
            or not math.isclose(
                float(nmae),
                float(error) / float(truth),
                rel_tol=1e-14,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("source-dev ratio-of-sums drifted")
        if row.get("selected_as_best") is True:
            selected.append(index)
        source_dev_scores.append(float(nmae))
    earliest_minimum = min(
        range(len(source_dev_scores)),
        key=source_dev_scores.__getitem__,
    )
    if (
        selected != [earliest_minimum]
        or training.get("best_epoch") != earliest_minimum
        or not math.isclose(
            float(training["best_source_dev_nmae"]),
            source_dev_scores[earliest_minimum],
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "selected checkpoint is not the earliest source-dev minimum"
        )


def _load_external(
    path: Path, expected_freeze: dict[str, str]
) -> dict[str, object]:
    from experiments.acil_innovation_v1.registries import seed_bundle
    from experiments.anchorcv_v1.data_access import canonical_data_identity
    from experiments.sc_acil_v1.checkpoint import load_checkpoint

    from .model import new_calibrator
    from .run_job import registered_job

    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = payload.get("job")
    if not isinstance(job, dict):
        raise ValueError("job identity is invalid")
    try:
        expected_job = registered_job(
            str(job["dataset"]),
            str(job["method"]),
        )
    except Exception as exc:
        raise ValueError(f"job identity is invalid: {exc}") from exc
    if (
        job != expected_job
        or payload.get("schema") != "minimal-calibrator-v1:result:v1"
        or payload.get("protocol") != "minimal-calibrator-v1"
        or payload.get("status") != "succeeded"
        or payload.get("test_access") is not False
        or payload.get("probe_freeze") != expected_freeze
        or manifest.get("schema")
        != "minimal-calibrator-v1:result-manifest:v1"
        or manifest.get("protocol") != "minimal-calibrator-v1"
        or manifest.get("result_file") != path.name
        or manifest.get("job") != job
        or manifest.get("probe_freeze") != expected_freeze
        or manifest.get("result_sha256") != _file_sha256(path)
    ):
        raise ValueError("result or manifest identity is invalid")
    training = payload.get("training")
    _validate_training(training)
    method = str(job["method"])
    if payload.get("feature_set") != method:
        raise ValueError("method and feature set drifted")
    dataset = str(job["dataset"])
    if payload.get("data_identity") != canonical_data_identity(dataset):
        raise ValueError("data identity drifted")
    if (
        payload.get("selection_cohort") != "source_dev"
        or payload.get("evaluation_cohort") != "tune"
    ):
        raise ValueError("data cohort identity drifted")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint descriptor is invalid")
    identity = checkpoint.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("job") != job
        or identity.get("probe_freeze") != expected_freeze
        or identity.get("best_epoch") != training.get("best_epoch")
        or identity.get("source_dev_nmae")
        != training.get("best_source_dev_nmae")
        or identity.get("feature_set") != method
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
        model = new_calibrator(
            method,
            model_seed=seed_bundle(1).model,
        )
        record = load_checkpoint(
            model,
            metadata_path=path.parent / metadata_relative,
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


def _discover_results(result_root: Path, output_name: str) -> list[Path]:
    paths = []
    for path in sorted(result_root.glob("*.json")):
        if path.name == output_name or path.name.endswith(".manifest.json"):
            continue
        companion = path.with_name(path.name + ".manifest.json")
        if not companion.is_file():
            continue
        try:
            manifest = json.loads(companion.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(manifest, dict)
            and manifest.get("schema")
            == "minimal-calibrator-v1:result-manifest:v1"
            and manifest.get("protocol") == "minimal-calibrator-v1"
            and manifest.get("result_file") == path.name
        ):
            paths.append(path)
    return paths


def _adjudicate_external(
    paths: Iterable[Path],
    *,
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
    expected = len(expected_job_identities())
    if issues or len(payloads) != expected:
        return {
            "expected_primary_jobs": expected,
            "issues": issues
            or [{"error": "result count differs from frozen grid"}],
            "paper_evidence": False,
            "reason": "incomplete_or_identity_invalid",
            "schema": "minimal-calibrator-v1:adjudication:v1",
            "stage": "prototype",
            "valid_primary_jobs": len(payloads),
            "verdict": "revise",
        }
    try:
        return adjudicate_payloads(
            payloads,
            bootstrap_draws=bootstrap_draws,
        )
    except Exception as exc:
        return {
            "expected_primary_jobs": expected,
            "issues": [
                {"error": f"{type(exc).__name__}: {exc}", "path": "paired_grid"}
            ],
            "paper_evidence": False,
            "reason": "incomplete_or_identity_invalid",
            "schema": "minimal-calibrator-v1:adjudication:v1",
            "stage": "prototype",
            "valid_primary_jobs": len(payloads),
            "verdict": "revise",
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", default=10000, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from .freeze import probe_identity
    from .run_job import _write_json_exclusive

    paths = _discover_results(args.result_root, args.output.name)
    report = _adjudicate_external(
        paths,
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
            "protocol": "minimal-calibrator-v1",
            "schema": "minimal-calibrator-v1:adjudication-manifest:v1",
        },
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["verdict"] in {"proceed", "kill"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_adjudicate_external",
    "_discover_results",
    "_load_external",
    "adjudicate_payloads",
    "build_parser",
]
