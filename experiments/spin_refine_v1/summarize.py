"""Audit four completed GEANT refinement trajectories without loading weights.

Writes an honest development-screen summary. No GPU calls or training launch.
The original SPIN+Direct is a historical reference, not an iteration ablation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "spin-refine-development-20260917-v1"
SEEDS = (41001, 41002)
VARIANTS = ("one_step", "two_step")
CONDITIONS = ("uniform", "unequal", "unequal_gap")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def trajectory(job, *, seed, variant, cap=160, historical=False):
    config = read(job / "config.json")
    suffix = "120" if cap == 120 else ""
    result_path = job / f"result{suffix}.json"
    history_path = job / f"history{suffix}.json"
    result, history = read(result_path), read(history_path)
    identity = {"dataset": "geant", "variant": variant, "seed": seed,
                "config_sha256": sha(job / "config.json")}
    require(all(result.get(k) == v for k, v in identity.items()), f"Result identity differs: {job}")
    require(all(config.get(k) == v for k, v in identity.items() if k != "config_sha256"),
            f"Config identity differs: {job}")
    require(result.get("state") == "complete" and result.get("epochs_completed") == cap
            and result.get("role") == "development_only" and result.get("independent_test") is False
            and result.get("optimizer_updates") == 64 * cap,
            f"Complete development trajectory required: {job}")
    require(len(history) == cap and all(row["epoch"] == i + 1 and row["updates"] == 64 * (i + 1)
                                       for i, row in enumerate(history)), f"History differs: {job}")
    require(all(math.isfinite(row["selection_score"]) for row in history), f"Nonfinite history: {job}")
    selected = min(history, key=lambda row: row["selection_score"])
    require(result["best_epoch"] == selected["epoch"]
            and result["primary_eval"]["selection_score"] == selected["selection_score"],
            f"Earliest strict-best selection differs: {job}")
    require(result["primary_eval"]["mask_seed"] == 71001, f"Development masks differ: {job}")
    for kind in ("best", "last"):
        filename = result[f"{kind}_checkpoint"] if f"{kind}_checkpoint" in result else f"{kind}{suffix}.pt"
        require(sha(job / filename) == result[f"{kind}_checkpoint_sha256"], f"Checkpoint hash differs: {job}/{kind}")
    if not historical:
        require(config["protocol"] == result["protocol"] == PROTOCOL, f"Protocol differs: {job}")
        require(config.get("shared_named_parameters_exact") is True
                and config.get("reference_checkpoint_weights_loaded") is False
                and config.get("continuation_checkpoint_loaded") is False
                and config.get("imported_epoch_count") == 0 and config.get("matched_compute") is False,
                f"Refinement training semantics differ: {job}")
        expected_passes = 1 if variant == "one_step" else 2
        require(all(row["context_passes_per_prediction"] == expected_passes
                    and row["training_context_forward_passes_cumulative"] == row["updates"] * expected_passes
                    and row["training_context_window_passes_cumulative"] == row["epoch"] * 512 * expected_passes
                    and row["matched_compute"] is False for row in history), f"Pass budget differs: {job}")
    scores = [row["selection_score"] for row in history[-20:]]
    summary = {"variant": variant, "seed": seed, "cap": cap,
        "comparison_role": "historical_reference_not_iteration_ablation" if historical else "matched_skeleton_iteration_ablation",
        "best_epoch": selected["epoch"], "nmae": result["primary_eval"]["selection_score"],
        "nrmse": statistics.mean(result["primary_eval"]["metrics"][c]["nrmse"] for c in CONDITIONS),
        "last20_median_nmae": statistics.median(scores), "last20_min_nmae": min(scores),
        "last20_max_nmae": max(scores), "last_epoch_nmae": history[-1]["selection_score"],
        "training_seconds": result["training_seconds"], "parameter_count": result["parameter_count"],
        "condition_nmae": {c: result["primary_eval"]["metrics"][c]["nmae"] for c in CONDITIONS},
        "condition_nrmse": {c: result["primary_eval"]["metrics"][c]["nrmse"] for c in CONDITIONS},
        "context_passes_per_prediction": 1 if historical else config["context_passes_per_prediction"],
        "source_artifacts": {str(p): sha(p) for p in (job / "config.json", result_path, history_path)}}
    return summary, config, history, result


def summarize(output, historical_output):
    records, configs, histories, results = {}, {}, {}, {}
    missing = [str(output / "geant" / v / f"seed{s}" / "result.json")
               for v in VARIANTS for s in SEEDS
               if not (output / "geant" / v / f"seed{s}" / "result.json").is_file()]
    if missing:
        return {"protocol": PROTOCOL, "state": "incomplete", "missing": missing,
                "component_screen_passed": None, "independent_test": False,
                "no_decision_before_all_four_complete": True}
    for variant in VARIANTS:
        for seed in SEEDS:
            job = output / "geant" / variant / f"seed{seed}"
            for cap in (120, 160):
                record, config, history, result = trajectory(job, seed=seed, variant=variant, cap=cap)
                records[variant, seed, cap] = record
                configs[variant, seed] = config
                histories[variant, seed, cap] = history
                results[variant, seed, cap] = result
    exemplar = configs["one_step", SEEDS[0]]
    for key, config in configs.items():
        require(all(config[k] == exemplar[k] for k in ("data", "training", "runtime", "source_files", "model_config")),
                f"Paired recipe/data/runtime/sources differ: {key}")
    for seed in SEEDS:
        one, two = configs["one_step", seed], configs["two_step", seed]
        require(one["initial_state_sha256"] == two["initial_state_sha256"]
                and one["parameter_count"] == two["parameter_count"], f"Paired initialization differs: {seed}")
        for left, right in zip(histories["one_step", seed, 160], histories["two_step", seed, 160]):
            require(left["schedule"] == right["schedule"], f"Paired epoch schedule differs: {seed}/{left['epoch']}")
    comparisons = []
    for seed in SEEDS:
        one, two = records["one_step", seed, 160], records["two_step", seed, 160]
        comparisons.append({"seed": seed, "two_step_better_best_nmae": two["nmae"] < one["nmae"],
            "two_step_better_last20_median": two["last20_median_nmae"] < one["last20_median_nmae"],
            "relative_best_nmae_gain": (one["nmae"] - two["nmae"]) / one["nmae"],
            "per_condition_better_nmae": {c: two["condition_nmae"][c] < one["condition_nmae"][c] for c in CONDITIONS}})
    component_pass = all(c["two_step_better_best_nmae"] and c["two_step_better_last20_median"] for c in comparisons)
    historical, historical_comparisons, missing_historical = [], [], []
    for seed in SEEDS:
        job = historical_output / "geant/spin_direct" / f"seed{seed}"
        if not (job / "result.json").is_file():
            missing_historical.append(str(job / "result.json"))
            continue
        record, config, history, result = trajectory(job, seed=seed, variant="spin_direct", historical=True)
        require(config["data"] == exemplar["data"], f"Historical reference data differ: {job}")
        historical.append(record)
        two = records["two_step", seed, 160]
        historical_comparisons.append({"seed": seed,
            "two_step_better_best_nmae": two["nmae"] < record["nmae"],
            "two_step_better_last20_median": two["last20_median_nmae"] < record["last20_median_nmae"],
            "relative_best_nmae_gain": (record["nmae"] - two["nmae"]) / record["nmae"]})
    original_both = (len(historical_comparisons) == len(SEEDS)
        and all(c["two_step_better_best_nmae"] and c["two_step_better_last20_median"] for c in historical_comparisons))
    # All reported selected scores remain development-selected; no acceptance claim.
    return {"protocol": PROTOCOL, "state": "complete", "role": "development_only", "independent_test": False,
        "component_screen_rule": "two_step beats one_step in BOTH seeds on best1..160 and last20 median NMAE",
        "component_screen_passed": component_pass, "per_seed_comparisons": comparisons,
        "two_step_also_better_than_original_on_both_criteria_and_seeds": original_both,
        "eligible_to_consider_further_validation": component_pass and original_both,
        "automatic_expansion_or_main_model_replacement": False,
        "historical_reference_missing": missing_historical, "historical_comparisons": historical_comparisons,
        "records": list(records.values()), "historical_references": historical,
        "matched_compute": False,
        "limitations": ["Both arms match optimizer updates and data, not compute; two_step makes two context passes.",
            "Context pass counts exclude gradient-checkpoint backward recomputation and do not measure FLOPs.",
            "The last 20 checkpoints are correlated trajectory diagnostics, not independent repeats.",
            "Original SPIN+Direct is a historical reference; iteration attribution uses the matched one_step arm.",
            "Development-screen success does not establish independent-test generalization or publication readiness."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/spin-refine-20260917-v1")
    parser.add_argument("--historical-output", type=Path, default=ROOT / "outputs/spin-path-controls-20260917-v1")
    parser.add_argument("--report", type=Path, default=ROOT / "analysis/spin_refine_20260917/summary/SUMMARY.json")
    args = parser.parse_args()
    report = summarize(args.output, args.historical_output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if report["state"] == "complete":
        rows = []
        for record in report["records"] + report["historical_references"]:
            for condition in CONDITIONS:
                rows.append({**{k: record[k] for k in ("variant", "seed", "cap", "comparison_role", "best_epoch")},
                    "condition": condition, "nmae": record["condition_nmae"][condition],
                    "nrmse": record["condition_nrmse"][condition]})
        with args.report.with_name("RESULTS.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"state": report["state"], "report": str(args.report),
                      "component_screen_passed": report["component_screen_passed"]}))


if __name__ == "__main__":
    main()
