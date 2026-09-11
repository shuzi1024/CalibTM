"""Adjudicate the closed one-shot tiny-KAN development comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from experiments.acil_innovation_v1.registries import (
    dataset_spec,
    ordered_window_starts,
)
from experiments.anchorcv_v1.data_access import canonical_data_identity

from .jobs import (
    DATASETS,
    MASK_FAMILIES,
    METHODS,
    PROTOCOL_ID,
    SEED_BUNDLES,
    STRUCTURED_FAMILIES,
    Job,
    canonical_json,
    expected_jobs,
)
from .result_io import verified_success, write_exclusive
from .source_identity import source_tree_sha256


CONTRASTS = {
    "kan_over_mlp": ("kan_value", "mlp_value"),
    "kan_value_over_static": ("kan_value", "kan_static"),
}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026081107
BLOCK_LENGTH = 4
_CHECKPOINT_IDENTITY_DOMAIN = b"sc-acil-v1:checkpoint-identity:v1\x00"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _float_vector(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    return array


def _int_vector(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must be a nonempty integer list")
    return np.asarray(value, dtype=np.int64)


def _locate_payload(root: Path, job: Job) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for attempt in sorted((root / "jobs" / job.job_id).glob("attempt-*")):
        payload = verified_success(attempt / "result.json")
        if payload is not None and payload.get("job") == job.to_json():
            candidates.append(payload)
    if len(candidates) != 1:
        raise ValueError(f"job {job.job_id} must have exactly one verified success")
    return candidates[0]


def load_payloads(root: str | Path) -> tuple[dict[str, object], ...]:
    location = Path(root)
    return tuple(_locate_payload(location, job) for job in expected_jobs())


def _extract_table(
    payloads: Iterable[dict[str, object]],
) -> tuple[
    dict[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    dict[str, int],
]:
    values = tuple(payloads)
    expected = {job.job_id: job for job in expected_jobs()}
    if len(values) != len(expected):
        raise ValueError("result grid is incomplete")
    seen: set[str] = set()
    table: dict[
        tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]
    ] = {}
    identities: dict[tuple[str, int, str], tuple[object, ...]] = {}
    parameter_counts: dict[str, set[int]] = {method: set() for method in METHODS}
    current_source = source_tree_sha256()
    for payload in values:
        if not isinstance(payload, dict):
            raise ValueError("result is not an object")
        job_value = payload.get("job")
        if not isinstance(job_value, dict):
            raise ValueError("result job is invalid")
        job_id = job_value.get("job_id")
        if not isinstance(job_id, str) or job_id in seen or job_id not in expected:
            raise ValueError("result job grid is duplicate or unexpected")
        job = expected[job_id]
        if job_value != job.to_json():
            raise ValueError("result job differs from the closed registry")
        seen.add(job_id)
        if (
            payload.get("schema") != f"{PROTOCOL_ID}:result:v1"
            or payload.get("protocol") != PROTOCOL_ID
            or payload.get("status") != "succeeded"
            or payload.get("source_tree_sha256") != current_source
            or payload.get("test_access") is not False
            or payload.get("fit_access") is not True
            or payload.get("selection_cohort") != "source_dev"
            or payload.get("evidence_boundary", {}).get("cohort") != "tune"
            or payload.get("evidence_boundary", {}).get("candidate_confirmation") is not False
            or payload.get("evidence_boundary", {}).get("project_wide_pristine") is not False
            or payload.get("evidence_boundary", {}).get("role")
            != "post-project one-shot development collision"
            or payload.get("data_identity") != canonical_data_identity(job.dataset)
        ):
            raise ValueError("source or evidence boundary is invalid")
        training = payload.get("training")
        architecture = payload.get("architecture")
        checkpoint = payload.get("checkpoint")
        if not isinstance(training, dict) or not isinstance(architecture, dict):
            raise ValueError("training or architecture record is missing")
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint record is missing")
        if training.get("epochs_completed") != 20 or training.get("optimizer_updates") != 320:
            raise ValueError("training budget drifted")
        best_epoch = training.get("best_epoch")
        if type(best_epoch) is not int or not 0 <= best_epoch < 20:
            raise ValueError("checkpoint-selection epoch is invalid")
        count = architecture.get("parameter_count")
        if type(count) is not int:
            raise ValueError("parameter count is invalid")
        if training.get("parameter_count") != count:
            raise ValueError("training and architecture parameter counts differ")
        expected_class = "ACILBase" if job.method == "mlp_value" else "TinyKANCalibrator"
        if architecture.get("class") != expected_class:
            raise ValueError("method label and architecture class differ")
        details = architecture.get("details")
        if job.method == "mlp_value":
            if details != {}:
                raise ValueError("exact MLP control architecture details drifted")
        else:
            if not isinstance(details, dict):
                raise ValueError("KAN architecture details are missing")
            expected_feature_set = (
                "value_only" if job.method == "kan_value" else "static_zero"
            )
            expected_kan = {
                "arithmetic": "fp32-for-fp16-or-bf16",
                "base_branch": "silu",
                "beta_e": 0.10,
                "beta_o": 0.10,
                "beta_r": 0.25,
                "feature_dim": 16,
                "feature_set": expected_feature_set,
                "grid_range": [-1.0, 1.0],
                "grid_size": 5,
                "hidden_dim": 32,
                "knot_vector": "uniform-extended-by-spline-order",
                "layer_widths": [16, 32, 3],
                "logit_dim": 3,
                "output_layer_initialization": (
                    "base-and-spline-normal-1e-4-bias-zero"
                ),
                "spline_basis_count_per_edge": 8,
                "spline_input_map": "tanh(x/2)",
                "spline_order": 3,
                "trunk": "layernorm-kan-kan",
            }
            if details != expected_kan:
                raise ValueError("KAN architecture identity drifted")
        expected_checkpoint_identity = {
            "architecture": architecture,
            "best_epoch": best_epoch,
            "job": job.to_json(),
            "source_tree_sha256": current_source,
        }
        if checkpoint.get("identity") != expected_checkpoint_identity:
            raise ValueError("checkpoint identity differs from the result")
        expected_identity_sha256 = hashlib.sha256(
            _CHECKPOINT_IDENTITY_DOMAIN + canonical_json(expected_checkpoint_identity)
        ).hexdigest()
        if checkpoint.get("identity_sha256") != expected_identity_sha256:
            raise ValueError("checkpoint identity hash differs from its payload")
        for field in ("identity_sha256", "file_sha256", "tensor_sha256"):
            if not _is_sha256(checkpoint.get(field)):
                raise ValueError("checkpoint hash is invalid")
        parameter_counts[job.method].add(count)
        cells = payload.get("cells")
        if not isinstance(cells, dict) or set(cells) != set(MASK_FAMILIES):
            raise ValueError("mask-family grid is invalid")
        for family in MASK_FAMILIES:
            cell = cells[family]
            if not isinstance(cell, dict):
                raise ValueError("cell is invalid")
            errors = _float_vector(
                cell.get("model_error_sum_per_window"), name="model errors"
            )
            linear = _float_vector(
                cell.get("linear_error_sum_per_window"), name="linear errors"
            )
            truth = _float_vector(
                cell.get("truth_sum_per_window"), name="truth sums"
            )
            targets = _int_vector(
                cell.get("target_count_per_window"), name="target counts"
            )
            starts = _int_vector(cell.get("window_starts"), name="window starts")
            if not (
                len(errors) == len(linear) == len(truth) == len(targets) == len(starts)
            ):
                raise ValueError("cell vectors differ in length")
            if np.any(errors < 0) or np.any(linear < 0) or np.any(truth <= 0) or np.any(targets <= 0):
                raise ValueError("cell sufficient statistics are invalid")
            registered_starts = np.asarray(
                ordered_window_starts(job.dataset, "tune"), dtype=np.int64
            )
            if not np.array_equal(starts, registered_starts):
                raise ValueError("cell does not use the registered tune schedule")
            expected_targets = dataset_spec(job.dataset).flows * 47
            if not np.all(targets == expected_targets):
                raise ValueError("cell target count differs from the K=3 complement")
            if not _is_sha256(cell.get("mask_grid_sha256")):
                raise ValueError("mask grid identity is invalid")
            identity = (
                cell.get("mask_grid_sha256"),
                tuple(starts.tolist()),
                tuple(targets.tolist()),
                tuple(truth.tolist()),
                tuple(linear.tolist()),
            )
            key = (job.dataset, job.seed_bundle, family)
            previous = identities.setdefault(key, identity)
            if identity != previous:
                raise ValueError("paired methods changed mask/truth/linear identity")
            table[(job.dataset, job.seed_bundle, family, job.method)] = (
                errors,
                truth,
            )
    if seen != set(expected):
        raise ValueError("result grid is incomplete")
    collapsed = {}
    for method, counts in parameter_counts.items():
        if len(counts) != 1:
            raise ValueError("parameter count varies within a method")
        collapsed[method] = next(iter(counts))
    if collapsed["mlp_value"] != 5475:
        raise ValueError("MLP control parameter count drifted")
    if collapsed["kan_value"] != collapsed["kan_static"]:
        raise ValueError("KAN arms are not parameter matched")
    if not 0.95 * 5475 <= collapsed["kan_value"] <= 1.05 * 5475:
        raise ValueError("KAN parameter count is outside +/-5%")
    return table, collapsed


def _positions(table, dataset: str, seed: int) -> np.ndarray:
    count = len(table[(dataset, seed, "random", "mlp_value")][0])
    return np.arange(count, dtype=np.int64)


def _effect(
    table: Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    *,
    dataset: str,
    seed: int,
    candidate: str,
    comparator: str,
    families: tuple[str, ...],
    positions: np.ndarray,
) -> float:
    candidate_error = 0.0
    comparator_error = 0.0
    truth_sum = 0.0
    for family in families:
        candidate_values, truth = table[(dataset, seed, family, candidate)]
        comparator_values, paired_truth = table[(dataset, seed, family, comparator)]
        if not np.array_equal(truth, paired_truth):
            raise ValueError("paired methods changed truth denominator")
        candidate_error += float(np.sum(candidate_values[positions], dtype=np.float64))
        comparator_error += float(np.sum(comparator_values[positions], dtype=np.float64))
        truth_sum += float(np.sum(truth[positions], dtype=np.float64))
    candidate_nmae = candidate_error / truth_sum
    comparator_nmae = comparator_error / truth_sum
    if comparator_nmae <= 0:
        raise ValueError("comparator NMAE is zero")
    return 1.0 - candidate_nmae / comparator_nmae


def _point_estimates(table) -> dict[str, object]:
    structured = {
        contrast: {
            dataset: {
                f"bundle{seed}": _effect(
                    table,
                    dataset=dataset,
                    seed=seed,
                    candidate=methods[0],
                    comparator=methods[1],
                    families=STRUCTURED_FAMILIES,
                    positions=_positions(table, dataset, seed),
                )
                for seed in SEED_BUNDLES
            }
            for dataset in DATASETS
        }
        for contrast, methods in CONTRASTS.items()
    }
    per_dataset = {
        contrast: {
            dataset: float(np.mean(tuple(seed_values.values())))
            for dataset, seed_values in dataset_values.items()
        }
        for contrast, dataset_values in structured.items()
    }
    headline = {
        contrast: {
            "actual": float(np.mean(tuple(dataset_values.values()))),
            "per_dataset": dataset_values,
        }
        for contrast, dataset_values in per_dataset.items()
    }
    seed_effects = {
        contrast: {
            f"bundle{seed}": float(
                np.mean(
                    [structured[contrast][dataset][f"bundle{seed}"] for dataset in DATASETS]
                )
            )
            for seed in SEED_BUNDLES
        }
        for contrast in CONTRASTS
    }
    structured_cells = {
        f"{dataset}/{family}": float(
            np.mean(
                [
                    _effect(
                        table,
                        dataset=dataset,
                        seed=seed,
                        candidate="kan_value",
                        comparator="mlp_value",
                        families=(family,),
                        positions=_positions(table, dataset, seed),
                    )
                    for seed in SEED_BUNDLES
                ]
            )
        )
        for dataset in DATASETS
        for family in STRUCTURED_FAMILIES
    }
    random_datasets = {
        dataset: float(
            np.mean(
                [
                    _effect(
                        table,
                        dataset=dataset,
                        seed=seed,
                        candidate="kan_value",
                        comparator="mlp_value",
                        families=("random",),
                        positions=_positions(table, dataset, seed),
                    )
                    for seed in SEED_BUNDLES
                ]
            )
        )
        for dataset in DATASETS
    }
    return {
        "headline": headline,
        "random_per_dataset": random_datasets,
        "seed_effects": seed_effects,
        "structured_effects": structured,
        "structured_per_dataset_mask": structured_cells,
    }


def _bootstrap(table, *, draws: int = BOOTSTRAP_DRAWS) -> dict[str, object]:
    rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED))
    samples = {contrast: np.empty(draws, dtype="<f8") for contrast in CONTRASTS}
    for draw in range(draws):
        sampled_seeds = [
            SEED_BUNDLES[int(index)]
            for index in rng.integers(0, len(SEED_BUNDLES), size=len(SEED_BUNDLES))
        ]
        effects = {
            contrast: {dataset: [] for dataset in DATASETS}
            for contrast in CONTRASTS
        }
        for seed in sampled_seeds:
            for dataset in DATASETS:
                count = len(table[(dataset, seed, "random", "mlp_value")][0])
                origins = rng.integers(0, count, size=math.ceil(count / BLOCK_LENGTH))
                positions = np.concatenate(
                    [
                        (int(origin) + np.arange(BLOCK_LENGTH, dtype=np.int64)) % count
                        for origin in origins
                    ]
                )[:count]
                for contrast, methods in CONTRASTS.items():
                    effects[contrast][dataset].append(
                        _effect(
                            table,
                            dataset=dataset,
                            seed=seed,
                            candidate=methods[0],
                            comparator=methods[1],
                            families=STRUCTURED_FAMILIES,
                            positions=positions,
                        )
                    )
        for contrast in CONTRASTS:
            samples[contrast][draw] = float(
                np.mean(
                    [float(np.mean(effects[contrast][dataset])) for dataset in DATASETS]
                )
            )
    intervals = {}
    for contrast, values in samples.items():
        lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
        intervals[contrast] = {"lower": float(lower), "upper": float(upper)}
    return {
        "block_length": BLOCK_LENGTH,
        "contrasts": intervals,
        "draws": draws,
        "rng": "PCG64DXSM",
        "seed": BOOTSTRAP_SEED,
    }


def _gate(actual: float | int, operator: str, threshold: float | int) -> dict[str, object]:
    if operator == ">=":
        passed = actual >= threshold
    elif operator == ">":
        passed = actual > threshold
    elif operator == "==":
        passed = actual == threshold
    else:
        raise ValueError("unknown operator")
    return {"actual": actual, "operator": operator, "pass": bool(passed), "threshold": threshold}


def adjudicate_payloads(payloads: Iterable[dict[str, object]]) -> dict[str, object]:
    table, parameter_counts = _extract_table(payloads)
    points = _point_estimates(table)
    bootstrap = _bootstrap(table)
    headline = points["headline"]
    primary = headline["kan_over_mlp"]
    conditioning = headline["kan_value_over_static"]
    gates = {
        "kan_over_mlp_structured_at_least_1pct": _gate(primary["actual"], ">=", 0.01),
        "kan_over_mlp_bootstrap_lower_positive": _gate(
            bootstrap["contrasts"]["kan_over_mlp"]["lower"], ">", 0.0
        ),
        "kan_over_mlp_both_datasets_positive": _gate(
            sum(value > 0 for value in primary["per_dataset"].values()), "==", 2
        ),
        "kan_over_mlp_at_least_two_seed_bundles_positive": _gate(
            sum(value > 0 for value in points["seed_effects"]["kan_over_mlp"].values()),
            ">=",
            2,
        ),
        "worst_structured_dataset_mask_no_more_than_half_pct_harm": _gate(
            min(points["structured_per_dataset_mask"].values()), ">=", -0.005
        ),
        "worst_random_dataset_no_more_than_half_pct_harm": _gate(
            min(points["random_per_dataset"].values()), ">=", -0.005
        ),
        "kan_value_over_static_at_least_half_pct": _gate(
            conditioning["actual"], ">=", 0.005
        ),
        "kan_value_over_static_bootstrap_lower_positive": _gate(
            bootstrap["contrasts"]["kan_value_over_static"]["lower"], ">", 0.0
        ),
        "kan_value_over_static_both_datasets_positive": _gate(
            sum(value > 0 for value in conditioning["per_dataset"].values()), "==", 2
        ),
    }
    passed = all(item["pass"] for item in gates.values())
    return {
        "aggregation": "ratio-of-sums within dataset x seed; seed-equal then dataset-equal",
        "bootstrap": bootstrap,
        "evidence_role": "development collision only; not independent confirmation",
        "gates": gates,
        "headline": headline,
        "latency_gate_pending_only_if_accuracy_proceeds": passed,
        "paper_evidence": False,
        "parameter_counts": parameter_counts,
        "protocol": PROTOCOL_ID,
        "random_per_dataset": points["random_per_dataset"],
        "schema": f"{PROTOCOL_ID}:adjudication:v1",
        "seed_effects": points["seed_effects"],
        "structured_effects": points["structured_effects"],
        "structured_per_dataset_mask": points["structured_per_dataset_mask"],
        "test_access": False,
        "verdict": (
            "PROCEED_TO_LATENCY_AND_FRESH_CONFIRMATION_REQUIRED"
            if passed
            else "KILL_TINY_KAN_KEEP_VALUE_ONLY"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = adjudicate_payloads(load_payloads(args.results_root))
    write_exclusive(args.output, report)
    print(json.dumps({"output": str(args.output), "verdict": report["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["adjudicate_payloads", "build_parser", "load_payloads"]
