#!/usr/bin/env python3
"""Plot saved unified-model development histories; no model/data/GPU imports.

From the CalibTM root, in an existing matplotlib environment:
  python plot_training.py --outputs analysis/spin_sync_unified_20260916/raw_outputs \
    --references analysis/research_7h_20260916/references \
    --outdir analysis/spin_sync_unified_20260916/figures --with-time

The same command with --validate-only reads JSON using the standard library and
prints a snapshot summary without plotting or creating an output directory.
Missing jobs/baselines are disclosed. A status file alone cannot mark a run
complete; completion requires a matching completed result and saved history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

DATASETS = ("abilene", "geant")
VARIANTS = ("spin_sync_direct", "spin_direct_no_memory")
CONDITIONS = ("uniform", "unequal", "unequal_gap")
LABELS = {"spin_sync_direct": "SPIN/Direct/Sync", "spin_direct_no_memory": "SPIN/Direct (no memory)"}
COLORS = {"spin_sync_direct": "#0072B2", "spin_direct_no_memory": "#D55E00"}
BASELINES = {"spin": ("Old SPIN", "#666666", "--"),
             "direct": ("Old Direct+Sync", "#009E73", "-."),
             "linear": ("Linear", "#AA4499", ":")}


def read(path):
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def finite(value, label, nonnegative=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"Invalid {label}: {value!r}")
    return float(value)


def dev_score(evaluation):
    if evaluation.get("role") != "development_only" or evaluation.get("mask_seed") != 71001:
        raise ValueError("Baseline/result must be development_only with mask_seed 71001")
    score = finite(evaluation["selection_score"], "development score")
    mean = sum(finite(evaluation["metrics"][c]["nmae"], c) for c in CONDITIONS) / 3
    if not math.isclose(score, mean, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("Development selection score differs from the three-condition mean")
    return score


def load_job(directory, dataset, variant, seed):
    identity = {"dataset": dataset, "variant": variant, "seed": seed}
    job = {**identity, "path": str(directory), "state": "missing", "recorded_state": None,
           "epochs": [], "nmae": [], "cumulative_seconds": None, "notes": [], "input_sha256": {}}
    history, result, status = None, None, None
    for filename in ("history.json", "result.json", "status.json"):
        value, digest = read(directory / filename)
        if digest:
            job["input_sha256"][filename] = digest
        if filename == "history.json": history = value
        elif filename == "result.json": result = value
        else: status = value
    job["recorded_state"] = (status or {}).get("state")
    if history is None:
        job["notes"].append("No copied history; process liveness is unknown")
        return job
    if not isinstance(history, list):
        raise ValueError(f"History is not a list: {directory}")
    job["state"] = "incomplete"
    cumulative, timed = 0.0, []
    timing_available = True
    for index, row in enumerate(history, 1):
        if type(row.get("epoch")) is not int or row["epoch"] != index:
            raise ValueError(f"History epochs must be consecutive from 1: {directory}")
        score = finite(row["selection_score"], "history selection score")
        mean = sum(finite(row["condition_nmae"][c], c) for c in CONDITIONS) / 3
        if not math.isclose(score, mean, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError(f"History score differs from condition mean at {directory}/epoch{index}")
        job["epochs"].append(index)
        job["nmae"].append(score)
        if "training_seconds" not in row or "evaluation_seconds" not in row:
            timing_available = False
        else:
            cumulative += finite(row["training_seconds"], "training seconds") + finite(row["evaluation_seconds"], "evaluation seconds")
            timed.append(cumulative)
    if timing_available:
        job["cumulative_seconds"] = timed
    else:
        job["notes"].append("Time curve omitted: at least one epoch lacks recorded timing")
    if history:
        best = min(range(len(history)), key=lambda index: job["nmae"][index])
        job.update(best_saved_epoch=best + 1, best_saved_nmae=job["nmae"][best])
    if result and result.get("state") == "complete":
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Completed result identity differs: {directory}")
        score = dev_score(result["primary_eval"])
        if result.get("role") != "development_only" or result.get("independent_test") is not False:
            raise ValueError(f"Unexpected completed result role: {directory}")
        if result.get("epochs_completed") == len(history) and history:
            if result.get("best_epoch") != job["best_saved_epoch"] or not math.isclose(score, job["best_saved_nmae"], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"Completed result and history best checkpoint differ: {directory}")
            job["state"] = "complete"
            job["stop_reason"] = result.get("stop_reason")
        else:
            job["notes"].append("Completed result and copied history lengths differ; snapshot is incomplete")
    if job["state"] != "complete":
        job["notes"].append("No matching completed result; saved epochs are a partial snapshot")
    return job


def collect(args):
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "role": "development_only",
              "independent_test": False, "datasets": {},
              "state_note": "File snapshots only; missing files are not evidence of a running process",
              "time_note": "Sum of recorded epoch training_seconds + evaluation_seconds; excludes initialization, mask generation, checkpoints, final evaluation and unsaved work; not end-to-end wall time",
              "baseline_note": "Historical selected development scores; horizontal references have no matched training budget"}
    for dataset in DATASETS:
        item = {"jobs": [], "baselines": {}, "notes": []}
        seeds = set(args.expect_seeds)
        for variant in VARIANTS:
            seeds.update(int(path.name[4:]) for path in (args.outputs / dataset / variant).glob("seed*")
                         if path.is_dir() and path.name[4:].isdigit())
        for variant in VARIANTS:
            for seed in sorted(seeds):
                item["jobs"].append(load_job(args.outputs / dataset / variant / f"seed{seed}", dataset, variant, seed))
        for name in BASELINES:
            path = args.references / dataset / f"{name}.json"
            saved, digest = read(path)
            if saved is None:
                item["notes"].append(f"Missing baseline: {name}")
                continue
            if name != "linear":
                expected_variant = "spin_adapted" if name == "spin" else "direct_sync"
                if saved.get("dataset") != dataset or saved.get("variant") != expected_variant:
                    raise ValueError(f"Wrong historical baseline identity: {path}")
            evaluation = saved if name == "linear" else saved["primary_eval"]
            item["baselines"][name] = {"nmae": dev_score(evaluation), "path": str(path), "sha256": digest}
        report["datasets"][dataset] = item
    return report


def plot(report, outdir, axis):
    # Deliberately late import: --help and --validate-only need only Python.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    styles = ("-", "--", "-.", ":")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        item = report["datasets"][dataset]
        seeds = sorted({job["seed"] for job in item["jobs"]})
        extent = 1.0
        for job in item["jobs"]:
            x = job["epochs"] if axis == "epoch" else job["cumulative_seconds"]
            state = job["state"]
            suffix = state if x else ("no copied history" if not job["epochs"] else "timing unavailable")
            label = f"{LABELS[job['variant']]} s{job['seed']} [{suffix}]"
            style = styles[seeds.index(job["seed"]) % len(styles)]
            if x:
                extent = max(extent, x[-1])
                ax.plot(x, job["nmae"], color=COLORS[job["variant"]], linestyle=style,
                        linewidth=1.6, label=label)
                ax.plot(x[-1], job["nmae"][-1], marker="o", markersize=4,
                        markerfacecolor=COLORS[job["variant"]] if state == "complete" else "white",
                        markeredgecolor=COLORS[job["variant"]], linestyle="none")
            else:
                # Legend entry discloses the absent curve; no data are invented.
                ax.plot([], [], color=COLORS[job["variant"]], linestyle=style, linewidth=1.6, label=label)
        for name, (label, color, style) in BASELINES.items():
            baseline = item["baselines"].get(name)
            if baseline:
                ax.axhline(baseline["nmae"], color=color, linestyle=style, linewidth=1.1,
                           label=f"{label}: {baseline['nmae']:.6f}")
            else:
                ax.plot([], [], color=color, linestyle=style, label=f"{label} [missing]")
        completed = sum(job["state"] == "complete" for job in item["jobs"])
        ax.set_title(f"{'Abilene' if dataset == 'abilene' else 'GEANT'}: development only\n"
                     f"{completed}/{len(item['jobs'])} copied runs complete", fontsize=11)
        ax.set_xlabel("Epoch" if axis == "epoch" else "Cumulative recorded train + evaluation time (s)")
        ax.set_ylabel("Development NMAE (three-condition mean)")
        ax.set_xlim(0, extent * 1.025)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=axis == "epoch"))
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.93)
    title = "Unified-model training: saved development histories (no smoothing)"
    note = "Historical baselines are selected development scores; training budgets are not matched."
    if axis == "time":
        note += "\nRecorded train/evaluation seconds exclude setup, checkpoints and unsaved work; not wall-clock speedup."
    else:
        note += "\nIncomplete/missing labels describe saved files; they do not establish process liveness."
    fig.suptitle(title + "\n" + note, fontsize=10)
    stem = outdir / ("development_nmae_by_epoch" if axis == "epoch" else "development_nmae_by_recorded_seconds")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=Path("analysis/spin_sync_unified_20260916/raw_outputs"))
    parser.add_argument("--references", type=Path, default=Path("analysis/research_7h_20260916/references"))
    parser.add_argument("--outdir", type=Path, default=Path("analysis/spin_sync_unified_20260916/figures"))
    parser.add_argument("--expect-seeds", nargs="+", type=int, default=[41001], help="Expected seeds; all other discovered seeds are also included")
    parser.add_argument("--with-time", action="store_true", help="Also plot cumulative recorded training + evaluation seconds")
    parser.add_argument("--validate-only", action="store_true", help="Read/validate saved JSON and print a summary without matplotlib or output files")
    args = parser.parse_args()
    report = collect(args)
    summary = {dataset: {"jobs": [{key: job.get(key) for key in ("variant", "seed", "state", "recorded_state", "best_saved_epoch")}
                                  | {"saved_epochs": len(job["epochs"])} for job in item["jobs"]],
                         "baselines": {name: value["nmae"] for name, value in item["baselines"].items()}}
               for dataset, item in report["datasets"].items()}
    if args.validate_only:
        print(json.dumps(summary, indent=2, allow_nan=False))
        return
    try:
        import matplotlib  # availability check before creating outputs
    except ImportError:
        parser.exit(2, "matplotlib is unavailable. Use an existing plotting environment; --validate-only needs only Python's standard library.\n")
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot(report, args.outdir, "epoch")
    if args.with_time:
        plot(report, args.outdir, "time")
    report["matplotlib_version"] = matplotlib.__version__
    report["figures"] = ["development_nmae_by_epoch.pdf", "development_nmae_by_epoch.png"]
    if args.with_time:
        report["figures"] += ["development_nmae_by_recorded_seconds.pdf", "development_nmae_by_recorded_seconds.png"]
    (args.outdir / "plot_inputs.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.outdir), "figures": report["figures"], "runs": summary}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Invalid saved input: {error}", file=sys.stderr)
        raise SystemExit(2)
