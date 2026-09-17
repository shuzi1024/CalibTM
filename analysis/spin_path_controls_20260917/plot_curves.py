#!/usr/bin/env python3
"""Plot all saved development NMAE trajectories, with no smoothing or new evaluation.

The left panel shows epochs 1--160; the right shows every saved point in 100--160.
Model color and seed line style are fixed. Incomplete curves stop at the last
saved epoch and remain labelled incomplete, even when status says complete.

--validate-only requires only the standard library and creates no files.
Rendering requires an existing matplotlib environment, never a GPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = "spin-path-controls-development-20260917-v1"
VARIANTS = ("spin_direct", "context_only", "direct_only")
SEEDS = (41001, 41002)
CONDITIONS = ("uniform", "unequal", "unequal_gap")
LABELS = {"spin_direct": "SPIN + Direct", "context_only": "Context-only", "direct_only": "Direct-only"}
COLORS = {"spin_direct": "#0072B2", "context_only": "#D55E00", "direct_only": "#009E73"}
STYLES = {41001: "-", 41002: "--"}


def read(path):
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {label}: {value!r}")
    return float(value)


def near(left, right):
    return math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-13)


def load_job(path, dataset, variant, seed):
    job = {"path": str(path), "dataset": dataset, "variant": variant, "seed": seed,
           "state": "missing", "recorded_state": None, "saved_epochs": 0,
           "epochs": [], "nmae": [], "condition_nmae": {c: [] for c in CONDITIONS},
           "condition_nrmse": {c: [] for c in CONDITIONS}, "train_loss": [], "lr": [],
           "input_sha256": {}, "notes": []}
    saved = {}
    for name in ("history.json", "config.json", "result.json", "status.json"):
        record, digest = read(path / name)
        saved[name] = record
        if digest:
            job["input_sha256"][name] = digest
    job["recorded_state"] = (saved["status.json"] or {}).get("state")
    identity = {"protocol": PROTOCOL, "dataset": dataset, "variant": variant, "seed": seed}
    config = saved["config.json"]
    if config is not None and any(config.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Config identity differs: {path}")
    history = saved["history.json"]
    if history is None:
        job["notes"].append("No copied history; missing does not imply queued, running, or failed")
        return job
    if not isinstance(history, list) or len(history) > 160:
        raise ValueError(f"Expected a cumulative history with at most 160 epochs: {path}")
    job["state"] = "incomplete"
    for epoch, row in enumerate(history, 1):
        if type(row.get("epoch")) is not int or row["epoch"] != epoch:
            raise ValueError(f"History must be consecutive from epoch 1: {path}")
        score = finite(row["selection_score"], f"{path}/epoch{epoch}/NMAE")
        values = {c: finite(row["condition_nmae"][c], f"{c}/NMAE") for c in CONDITIONS}
        if not near(score, sum(values.values()) / 3):
            raise ValueError(f"Development NMAE is not the equal mean of conditions: {path}/epoch{epoch}")
        job["epochs"].append(epoch)
        job["nmae"].append(score)
        for condition in CONDITIONS:
            job["condition_nmae"][condition].append(values[condition])
            nrmse = row.get("condition_nrmse", {}).get(condition)
            job["condition_nrmse"][condition].append(finite(nrmse, "NRMSE") if nrmse is not None else None)
        job["train_loss"].append(finite(row["train_loss"], "training loss") if "train_loss" in row else None)
        lr = row.get("lr")
        if lr is not None:
            if not near(finite(lr, "learning rate"), .001 if epoch <= 120 else .0001):
                raise ValueError(f"Learning rate violates frozen schedule: {path}/epoch{epoch}")
        elif epoch > 120:
            raise ValueError(f"Learning rate missing from new history: {path}/epoch{epoch}")
        job["lr"].append(lr)
    job["saved_epochs"] = len(history)
    if history:
        best = min(range(len(history)), key=lambda index: job["nmae"][index])
        job.update(best_saved_epoch=best + 1, best_saved_nmae=job["nmae"][best])
    result = saved["result.json"]
    if result and result.get("state") == "complete":
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Completed result identity differs: {path}")
        evaluation = result["primary_eval"]
        if (result.get("role") != "development_only" or result.get("independent_test") is not False
                or evaluation.get("role") != "development_only" or evaluation.get("mask_seed") != 71001):
            raise ValueError(f"Completed result evaluation role differs: {path}")
        score = finite(evaluation["selection_score"], "completed development score")
        if not near(score, sum(finite(evaluation["metrics"][c]["nmae"], "condition NMAE") for c in CONDITIONS) / 3):
            raise ValueError(f"Completed score differs from three-condition mean: {path}")
        if result.get("epochs_completed") == 160 and len(history) == 160:
            if result.get("best_epoch") != job["best_saved_epoch"] or not near(score, job["best_saved_nmae"]):
                raise ValueError(f"Completed result and history selected checkpoint differ: {path}")
            if config is None or result.get("config_sha256") != job["input_sha256"]["config.json"]:
                raise ValueError(f"Completed result config hash unavailable or inconsistent: {path}")
            job["state"] = "complete"
        else:
            job["notes"].append("Completed marker/history length does not match 160; plotted as incomplete snapshot")
    if job["state"] != "complete":
        job["notes"].append("Incomplete saved snapshot; no extension/interpolation beyond saved epochs")
    return job


def collect(args):
    jobs = [load_job(args.outputs / args.dataset / variant / f"seed{seed}", args.dataset, variant, seed)
            for variant in VARIANTS for seed in SEEDS]
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
            "role": "development_only", "independent_test": False, "protocol": PROTOCOL,
            "metric": "equal mean of uniform, unequal and unequal_gap missing-only global NMAEs",
            "panels": {"full": [1, 160], "late": [100, 160]},
            "lr_drop_boundary": 120.5, "fixed_schedule": {"epochs1_120": .001, "epochs121_160": .0001},
            "colors": COLORS, "seed_line_styles": {str(k): v for k, v in STYLES.items()},
            "smoothing": False, "point_selection": "all saved epochs within each declared axis range",
            "uncertainty_band": None, "historical_baselines_plotted": False,
            "state_note": "saved file snapshots only; no process liveness inference; complete requires matching result/config/history at 160 epochs",
            "scientific_note": "fixed optimization diagnostic; no proof of convergence, significance, or independent test performance",
            "jobs": jobs, "state_counts": dict(Counter(job["state"] for job in jobs)),
            "all_six_jobs_complete": all(job["state"] == "complete" for job in jobs)}


def plot(report, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    for ax, (name, limits) in zip(axes, report["panels"].items()):
        visible_values = []
        for job in report["jobs"]:
            if job["epochs"]:
                ax.plot(job["epochs"], job["nmae"], color=COLORS[job["variant"]],
                        linestyle=STYLES[job["seed"]], linewidth=1.35, alpha=.95)
                if limits[0] <= job["epochs"][-1] <= limits[1]:
                    ax.plot(job["epochs"][-1], job["nmae"][-1], marker="o", markersize=4,
                            markerfacecolor=COLORS[job["variant"]] if job["state"] == "complete" else "white",
                            markeredgecolor=COLORS[job["variant"]], linestyle="none")
                visible_values += [value for epoch, value in zip(job["epochs"], job["nmae"])
                                   if limits[0] <= epoch <= limits[1]]
        ax.axvspan(120.5, 160, color="#999999", alpha=.08, zorder=0)
        ax.axvline(120.5, color="#555555", linewidth=1, linestyle=":")
        ax.text(120.5, .98, " LR drop after epoch 120", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=8, rotation=90)
        ax.set_xlim(*limits)
        if visible_values:
            low, high = min(visible_values), max(visible_values)
            pad = max((high - low) * .07, .001)
            ax.set_ylim(max(0, low - pad), high + pad)
        else:
            ax.set_ylim(0, 1)
            ax.text(.5, .5, "No saved epochs in this range", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("All saved epochs (1-160)" if name == "full" else "Late-epoch view (100-160)", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Development NMAE (three-condition mean)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(alpha=.20, linewidth=.6)
    handles = []
    for job in report["jobs"]:
        state = "complete" if job["state"] == "complete" else "incomplete"
        label = f"{LABELS[job['variant']]} / seed {job['seed']} [{state}: {job['saved_epochs']}/160]"
        handles.append(Line2D([], [], color=COLORS[job["variant"]], linestyle=STYLES[job["seed"]],
                              linewidth=1.7, label=label))
    completed = sum(job["state"] == "complete" for job in report["jobs"])
    dataset = "GEANT" if report["dataset"] == "geant" else "Abilene"
    fig.suptitle(f"{dataset}: SPIN + Direct path controls\nDevelopment histories, no smoothing; {completed}/6 trajectories complete", fontsize=12, y=.98)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075), ncol=2, fontsize=8, frameon=False)
    fig.text(.5, .025, "Incomplete lines stop at the last saved epoch. Two seeds are two trajectories; 120-to-160 continuation is not a new repeat.\n"
             "Same saved points in both panels; different vertical scales. Development-only optimization diagnostic, not a convergence or significance claim.",
             ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(top=.80, bottom=.31, left=.07, right=.98, wspace=.24)
    stem = outdir / "development_nmae_curves"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    report["matplotlib_version"] = matplotlib.__version__
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def write_csv(report, path):
    fields = ["dataset", "variant", "seed", "trajectory_state", "saved_epochs", "epoch", "mean_nmae", "train_loss", "lr"]
    fields += [f"{condition}_{metric}" for metric in ("nmae", "nrmse") for condition in CONDITIONS]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for job in report["jobs"]:
            for index, epoch in enumerate(job["epochs"]):
                row = {"dataset": report["dataset"], "variant": job["variant"], "seed": job["seed"],
                       "trajectory_state": job["state"], "saved_epochs": job["saved_epochs"], "epoch": epoch,
                       "mean_nmae": job["nmae"][index], "train_loss": job["train_loss"][index], "lr": job["lr"][index]}
                row.update({f"{condition}_{metric}": job[f"condition_{metric}"][condition][index]
                            for metric in ("nmae", "nrmse") for condition in CONDITIONS})
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path(__file__).parent / "raw_outputs")
    parser.add_argument("--dataset", choices=("geant", "abilene"), default="geant")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "figures")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report = collect(args)
    summary = {"dataset": args.dataset, "state_counts": report["state_counts"],
               "all_six_jobs_complete": report["all_six_jobs_complete"],
               "jobs": [{key: job[key] for key in ("variant", "seed", "state", "saved_epochs")} for job in report["jobs"]]}
    if not args.validate_only:
        args.outdir.mkdir(parents=True, exist_ok=True)
        artifacts = plot(report, args.outdir)
        csv_path = args.outdir / "plot_inputs.csv"
        write_csv(report, csv_path)
        artifacts.append(csv_path)
        report["artifacts"] = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
                               for path in artifacts}
        (args.outdir / "plot_inputs.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        summary["outdir"] = str(args.outdir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
