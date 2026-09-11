"""Pure frozen gate decision plus result-grid adjudication entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .data import load_windows
from .evidence import load_rows
from .freeze import load_and_verify_freeze
from .jobs import planned_jobs
from .protocol import canonical_json_bytes, load_protocol
from .statistics import dataset_equal_bootstrap, dataset_equal_effect


@dataclass(frozen=True, slots=True)
class GateDecision:
    verdict: str
    clauses: Mapping[str, bool]


def _nmae(
    table,
    method: str,
    *,
    seeds: tuple[int, ...],
    dataset: str,
    masks: tuple[str, ...],
) -> float:
    errors = []
    truths = []
    for seed in seeds:
        for mask in masks:
            error, truth = table[(method, seed, dataset, mask)]
            errors.append(float(np.sum(error, dtype=np.float64)))
            truths.append(float(np.sum(truth, dtype=np.float64)))
    denominator = math.fsum(truths)
    if denominator <= 0:
        raise ValueError("NMAE denominator is zero")
    return math.fsum(errors) / denominator


def _effect(
    table,
    candidate: str,
    comparator: str,
    *,
    seeds: tuple[int, ...],
    dataset: str,
    masks: tuple[str, ...],
) -> float:
    candidate_score = _nmae(
        table, candidate, seeds=seeds, dataset=dataset, masks=masks
    )
    comparator_score = _nmae(
        table, comparator, seeds=seeds, dataset=dataset, masks=masks
    )
    if comparator_score <= 0:
        raise ValueError("effect comparator NMAE is zero")
    return 1.0 - candidate_score / comparator_score


def summarize_table(table) -> dict[str, object]:
    structured = ("internal_block", "two_burst")
    per_dataset = {
        dataset: _effect(
            table,
            "sc_acil",
            "acil_only",
            seeds=(4, 5, 6),
            dataset=dataset,
            masks=structured,
        )
        for dataset in ("abilene", "geant")
    }
    cells = {
        f"{dataset}/{mask}": _effect(
            table,
            "sc_acil",
            "acil_only",
            seeds=(4, 5, 6),
            dataset=dataset,
            masks=(mask,),
        )
        for dataset in ("abilene", "geant")
        for mask in structured
    }
    per_seed = {
        str(seed): math.fsum(
            _effect(
                table,
                "sc_acil",
                "acil_only",
                seeds=(seed,),
                dataset=dataset,
                masks=structured,
            )
            for dataset in ("abilene", "geant")
        )
        / 2.0
        for seed in (4, 5, 6)
    }
    return {
        "main_full_over_acil": dataset_equal_effect(
            table,
            candidate="sc_acil",
            comparator="acil_only",
            masks=structured,
        ),
        "innovation_full_over_u0": dataset_equal_effect(
            table,
            candidate="sc_acil",
            comparator="sc_acil_u0",
            masks=structured,
        ),
        "random_full_over_acil": dataset_equal_effect(
            table,
            candidate="sc_acil",
            comparator="acil_only",
            masks=("random",),
        ),
        "full_over_linear": dataset_equal_effect(
            table,
            candidate="sc_acil",
            comparator="linear_interpolation",
            masks=structured,
        ),
        "acil_over_linear": dataset_equal_effect(
            table,
            candidate="acil_only",
            comparator="linear_interpolation",
            masks=structured,
        ),
        "u0_over_acil": dataset_equal_effect(
            table,
            candidate="sc_acil_u0",
            comparator="acil_only",
            masks=structured,
        ),
        "per_dataset_main": per_dataset,
        "dataset_mask_main": cells,
        "per_seed_main": per_seed,
        "nmae": {
            method: {
                f"{dataset}/{mask}": _nmae(
                    table,
                    method,
                    seeds=(4, 5, 6),
                    dataset=dataset,
                    masks=(mask,),
                )
                for dataset in ("abilene", "geant")
                for mask in ("random", *structured)
            }
            for method in (
                "linear_interpolation",
                "acil_only",
                "sc_acil_u0",
                "sc_acil",
            )
        },
    }


def decide(
    *,
    main_point: float,
    main_ci_lower: float,
    innovation_point: float,
    innovation_ci_lower: float,
    random_effect: float,
    per_dataset_main: Mapping[str, float],
    dataset_mask_main: Mapping[str, float],
    per_seed_main: Mapping[str, float],
) -> GateDecision:
    numeric = [main_point, main_ci_lower, innovation_point, innovation_ci_lower, random_effect]
    numeric.extend(per_dataset_main.values())
    numeric.extend(dataset_mask_main.values())
    numeric.extend(per_seed_main.values())
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("gate operands must be finite")
    if set(per_dataset_main) != {"abilene", "geant"}:
        raise ValueError("per-dataset gate operands drifted")
    expected_cells = {
        "abilene/internal_block",
        "abilene/two_burst",
        "geant/internal_block",
        "geant/two_burst",
    }
    if set(dataset_mask_main) != expected_cells or set(per_seed_main) != {"4", "5", "6"}:
        raise ValueError("cell or seed gate operands drifted")
    gate = load_protocol()["gate"]
    main = gate["main_full_over_acil"]
    attribution = gate["innovation_attribution"]
    random = gate["random_no_harm"]
    clauses = {
        "main_effect": main_point >= main["dataset_equal_structured_improvement_minimum"],
        "main_ci": main_ci_lower > main["bootstrap_ci_lower_strictly_above"],
        "innovation_effect": innovation_point >= attribution["dataset_equal_structured_improvement_minimum"],
        "innovation_ci": innovation_ci_lower > attribution["bootstrap_ci_lower_strictly_above"],
        "random_no_harm": random_effect >= random["dataset_equal_improvement_minimum"],
        "dataset_no_harm": min(per_dataset_main.values()) >= main["per_dataset_minimum"],
        "cell_wins": sum(value > 0.0 for value in dataset_mask_main.values())
        >= main["positive_dataset_mask_cells_required"],
        "worst_cell": min(dataset_mask_main.values()) >= main["worst_dataset_mask_cell_minimum"],
        "positive_seeds": sum(value > 0.0 for value in per_seed_main.values())
        >= main["positive_seed_bundles_required"],
    }
    return GateDecision(
        verdict="proceed" if all(clauses.values()) else "kill",
        clauses=clauses,
    )


def _read_result(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError("job result is not canonical")
    return value


def _formal_table(output_root: Path, manifest_sha256: str):
    rows = []
    failures = []
    for job in planned_jobs("formal_gate"):
        directory = output_root / "formal_gate" / job.job_id
        result = _read_result(directory / "result.json")
        if result.get("job") != job.to_json() or result.get("manifest_sha256") != manifest_sha256:
            raise ValueError("formal result job/freeze identity drifted")
        if result.get("status") != "succeeded":
            failures.append({"job": job.to_json(), "status": result.get("status")})
            continue
        descriptor = result.get("evidence")
        if not isinstance(descriptor, dict):
            raise ValueError("formal result evidence descriptor is absent")
        values = load_rows(
            directory / str(descriptor["file"]),
            expected_sha256=str(descriptor["sha256"]),
        )
        if len(values) != int(descriptor["row_count"]):
            raise ValueError("formal evidence row count differs from descriptor")
        rows.extend(values)
    if failures:
        raise RuntimeError("algorithmic formal jobs failed; survivor-only adjudication is forbidden")
    expected_methods = (
        "linear_interpolation",
        "acil_only",
        "sc_acil_u0",
        "sc_acil",
    )
    grouped = {}
    for row in rows:
        identity = row["identity"]
        key = (
            str(identity["method"]),
            int(identity["seed_bundle"]),
            str(identity["dataset"]),
            str(identity["mask_family"]),
        )
        grouped.setdefault(key, []).append(row)
    table = {}
    for method in expected_methods:
        for seed in (4, 5, 6):
            for dataset in ("abilene", "geant"):
                starts = load_windows(dataset, "gate").absolute_starts
                for mask in ("random", "internal_block", "two_burst"):
                    key = (method, seed, dataset, mask)
                    group = grouped.get(key, [])
                    ordered = sorted(group, key=lambda row: int(row["identity"]["window_start"]))
                    if tuple(int(row["identity"]["window_start"]) for row in ordered) != starts:
                        raise ValueError(f"formal evidence grid is missing or duplicated: {key!r}")
                    table[key] = (
                        np.asarray([row["absolute_error_sum"] for row in ordered], dtype="<f8"),
                        np.asarray([row["absolute_truth_sum"] for row in ordered], dtype="<f8"),
                    )
    if len(grouped) != len(table):
        raise ValueError("formal evidence contains an unregistered method/cell")
    return table


def adjudicate(output_root: Path) -> Path:
    output_root = Path(output_root)
    freeze = load_and_verify_freeze()
    table = _formal_table(output_root, str(freeze["manifest_sha256"]))
    config = load_protocol()
    bootstrap = config["bootstrap"]
    main = dataset_equal_bootstrap(
        table,
        candidate="sc_acil",
        comparator="acil_only",
        masks=("internal_block", "two_burst"),
        draws=int(bootstrap["draws"]),
        bootstrap_seed=int(bootstrap["seed"]),
        block_length=int(bootstrap["block_length"]),
    )
    innovation = dataset_equal_bootstrap(
        table,
        candidate="sc_acil",
        comparator="sc_acil_u0",
        masks=("internal_block", "two_burst"),
        draws=int(bootstrap["draws"]),
        bootstrap_seed=int(bootstrap["seed"]),
        block_length=int(bootstrap["block_length"]),
    )
    summary = summarize_table(table)
    decision = decide(
        main_point=float(summary["main_full_over_acil"]),
        main_ci_lower=main.ci_lower,
        innovation_point=float(summary["innovation_full_over_u0"]),
        innovation_ci_lower=innovation.ci_lower,
        random_effect=float(summary["random_full_over_acil"]),
        per_dataset_main=summary["per_dataset_main"],
        dataset_mask_main=summary["dataset_mask_main"],
        per_seed_main=summary["per_seed_main"],
    )
    payload = {
        "schema": "sc-acil-v1:adjudication:v1",
        "protocol": "sc-acil-v1",
        "manifest_sha256": freeze["manifest_sha256"],
        "verdict": decision.verdict,
        "clauses": dict(decision.clauses),
        "summary": summary,
        "bootstrap": {
            "main": asdict_bootstrap(main),
            "innovation_attribution": asdict_bootstrap(innovation),
        },
        "gate": config["gate"],
        "failures": [],
        "missing_jobs": [],
    }
    directory = output_root / "_adjudication"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stage_a.json"
    content = canonical_json_bytes(payload) + b"\n"
    with path.open("xb") as handle:
        handle.write(content)
    return path


def asdict_bootstrap(value) -> dict[str, object]:
    return {
        "candidate": value.candidate,
        "comparator": value.comparator,
        "point_estimate": value.point_estimate,
        "ci_lower": value.ci_lower,
        "ci_upper": value.ci_upper,
        "draws": value.draws,
        "draws_sha256": value.draws_sha256,
        "draw_plan_sha256": value.draw_plan_sha256,
    }


__all__ = ["GateDecision", "adjudicate", "decide", "summarize_table"]
