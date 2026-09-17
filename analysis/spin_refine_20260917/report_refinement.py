"""Audit fixed one-/two-step refinement for GEANT or Abilene without loading weights.

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


def trajectory(job, *, dataset, seed, variant, cap=160, historical=False):
    config = read(job / "config.json")
    suffix = "120" if cap == 120 else ""
    result_path = job / f"result{suffix}.json"
    history_path = job / f"history{suffix}.json"
    result, history = read(result_path), read(history_path)
    identity = {"dataset": dataset, "variant": variant, "seed": seed,
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
    summary = {"dataset": dataset, "variant": variant, "seed": seed, "cap": cap,
        "comparison_role": "historical_reference_not_iteration_ablation" if historical else "matched_skeleton_iteration_ablation",
        "best_epoch": selected["epoch"], "nmae": result["primary_eval"]["selection_score"],
        "nrmse": statistics.mean(result["primary_eval"]["metrics"][c]["nrmse"] for c in CONDITIONS),
        "last20_mean_nmae": statistics.mean(scores), "last20_median_nmae": statistics.median(scores), "last20_min_nmae": min(scores),
        "last20_max_nmae": max(scores), "last_epoch_nmae": history[-1]["selection_score"],
        "training_seconds": result["training_seconds"], "parameter_count": result["parameter_count"],
        "condition_nmae": {c: result["primary_eval"]["metrics"][c]["nmae"] for c in CONDITIONS},
        "condition_nrmse": {c: result["primary_eval"]["metrics"][c]["nrmse"] for c in CONDITIONS},
        "context_passes_per_prediction": 1 if historical else config["context_passes_per_prediction"],
        "source_artifacts": {str(p): sha(p) for p in (job / "config.json", result_path, history_path)}}
    return summary, config, history, result


def summarize(output, historical_output, dataset="geant"):
    require(dataset in ("geant", "abilene"), "Unsupported dataset")
    records, configs, histories, results = {}, {}, {}, {}
    missing = [str(output / dataset / v / f"seed{s}" / "result.json")
               for v in VARIANTS for s in SEEDS
               if not (output / dataset / v / f"seed{s}" / "result.json").is_file()]
    if missing:
        return {"protocol": PROTOCOL, "dataset": dataset, "state": "incomplete", "missing": missing,
                "component_screen_passed": None, "independent_test": False,
                "no_decision_before_all_four_complete": True}
    for variant in VARIANTS:
        for seed in SEEDS:
            job = output / dataset / variant / f"seed{seed}"
            for cap in (120, 160):
                record, config, history, result = trajectory(job, dataset=dataset, seed=seed, variant=variant, cap=cap)
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
        job = historical_output / dataset / "spin_direct" / f"seed{seed}"
        if not (job / "result.json").is_file():
            missing_historical.append(str(job / "result.json"))
            continue
        record, config, history, result = trajectory(job, dataset=dataset, seed=seed, variant="spin_direct", historical=True)
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
    return {"protocol": PROTOCOL, "dataset": dataset, "state": "complete", "role": "development_only", "independent_test": False,
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


def cost_reference(path, dataset):
    """Read existing fit-only benchmark output; perform no timing ourselves."""
    record = read(path)
    require(record.get("state") == "complete", f"Cost record is incomplete: {path}")
    require(dataset in record.get("datasets", {}), f"Cost report lacks {dataset}: {path}")
    data = record["datasets"][dataset]
    groups = {}
    for row in data["measurements"]:
        key = row["method"], row["batch"], row["execution_device"]
        groups.setdefault(key, []).append(row)
    rows = []
    for (method, batch, device), cells in sorted(groups.items()):
        require(len(cells) == 3 and {r["condition"] for r in cells} == set(CONDITIONS),
                f"Cost conditions incomplete or duplicate: {path}/{method}/B{batch}")
        require(all(math.isfinite(r["median_ms_per_window"]) and r["median_ms_per_window"] > 0
                    for r in cells), f"Invalid cost value: {path}/{method}/B{batch}")
        peak = [r.get("max_memory_allocated_bytes_including_resident_model") for r in cells]
        rows.append({"method": method, "batch": batch, "execution_device": device,
            "mean_condition_median_ms_per_window": statistics.mean(r["median_ms_per_window"] for r in cells),
            "max_allocated_mib_including_model": max(peak) / 2**20 if all(x is not None for x in peak) else None,
            "parameters": data.get("models", {}).get(method, {}).get("parameters")})
    return {"path": str(path), "sha256": sha(path), "dataset": dataset,
            "device_name": record.get("device_name"), "hostname": record.get("hostname"),
            "role": "provided_cost_reference_not_measured_by_this_report", "rows": rows}


def markdown(report):
    dataset = report["dataset"].upper()
    lines = [f"# {dataset} 固定两步细化结果", "", "评价角色：开发集筛选；不是独立未见测试。", ""]
    if report["state"] != "complete":
        lines += ["四条轨迹尚未全部完成，暂不做方向判断。", "", "缺少：", ""]
        lines += [f"- `{p}`" for p in report["missing"]]
        return "\n".join(lines) + "\n"
    flag = lambda value: "通过" if value else "未通过"
    lines += [f"组件筛选：**{flag(report['component_screen_passed'])}**。两个种子均须在 1—160 轮最佳 NMAE 和末 20 轮中位数上优于匹配的一步模型。", "",
              f"对原 SPIN+Direct 的两项逐种子比较：**{flag(report['two_step_also_better_than_original_on_both_criteria_and_seeds'])}**。该模型是历史参照，不承担迭代机制归因。", "",
              "## 三条件等权平均", "", "| 模型 | 种子 | 轮数上限 | 最佳轮 | NMAE | NRMSE | 末20均值 | 末20中位数 | 末20范围 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    records = report["records"] + report["historical_references"]
    for row in records:
        lines.append(f"| {row['variant']} | {row['seed']} | {row['cap']} | {row['best_epoch']} | {row['nmae']:.6f} | {row['nrmse']:.6f} | {row['last20_mean_nmae']:.6f} | {row['last20_median_nmae']:.6f} | {row['last20_min_nmae']:.6f}—{row['last20_max_nmae']:.6f} |")
    lines += ["", "## 逐条件结果", "", "| 模型 | 种子 | 轮数上限 | 条件 | NMAE | NRMSE |", "|---|---:|---:|---|---:|---:|"]
    for row in records:
        for condition in CONDITIONS:
            lines.append(f"| {row['variant']} | {row['seed']} | {row['cap']} | {condition} | {row['condition_nmae'][condition]:.6f} | {row['condition_nrmse'][condition]:.6f} |")
    lines += ["", "## 训练成本", "", "| 模型 | 种子 | 轮数上限 | 参数量 | 记录的训练秒数 | 每次预测上下文遍数 |", "|---|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['variant']} | {row['seed']} | {row['cap']} | {row['parameter_count']} | {row['training_seconds']:.1f} | {row['context_passes_per_prediction']} |")
    lines += ["", "训练秒数沿用各轨迹记录，不含每轮开发评价；跨设备和运行时段的秒数不能当作严格同场计时。两种新模型匹配更新次数和数据，未匹配计算量。", ""]
    if report.get("cost_references"):
        lines += ["## 已提供的推理成本记录", ""]
        for cost in report["cost_references"]:
            lines += [f"来源：`{cost['path']}`；设备：{cost.get('device_name') or '见原记录'}。", "",
                      "| 模型 | 设备 | Batch | 三条件平均 ms/窗 | 显存峰值 MiB（含模型） |", "|---|---|---:|---:|---:|"]
            for row in cost["rows"]:
                peak = "—" if row["max_allocated_mib_including_model"] is None else f"{row['max_allocated_mib_including_model']:.2f}"
                lines.append(f"| {row['method']} | {row['execution_device']} | {row['batch']} | {row['mean_condition_median_ms_per_window']:.5f} | {peak} |")
            lines += [""]
        lines += ["CPU Linear 与 GPU 模型分列，不计算跨设备加速比；不同成本报告的运行条件以各原始记录为准。", ""]
    lines += ["## 解释边界", "", "- 末 20 轮是相关的优化轨迹诊断，不是 20 次独立重复。",
              "- 逻辑上下文遍数不含梯度检查点在反向传播中的重算，不等于 FLOPs。",
              "- 筛选通过只支持考虑进一步验证，不自动扩充实验或替换主方法。",
              "- 120 轮结果完整保留，160 轮结果不覆盖原记录。", ""]
    return "\n".join(lines)


def curves(report, output, historical_output, destination):
    """Optional standard Matplotlib PNG/PDF artifacts; never required for scoring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    colors = {"one_step": "#2467a0", "two_step": "#c24e2a", "spin_direct": "#6a7780"}
    for column, seed in enumerate(SEEDS):
        for variant in (*VARIANTS, "spin_direct"):
            root = historical_output if variant == "spin_direct" else output
            path = root / report["dataset"] / variant / f"seed{seed}" / "history.json"
            if not path.exists():
                continue
            rows = read(path)
            epochs = [r["epoch"] for r in rows]
            scores = [r["selection_score"] for r in rows]
            for axis in axes[:, column]:
                axis.plot(epochs, scores, color=colors[variant], label=variant, linewidth=1.2)
                axis.grid(alpha=.2)
                axis.set_xlabel("Epoch")
                axis.set_ylabel("Development mean NMAE")
            axes[0, column].set_title(f"{report['dataset'].upper()} / seed {seed}")
            axes[1, column].set_xlim(141, 160)
        axes[0, column].axvline(120.5, color="black", linestyle=":", linewidth=.8)
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_title("Last 20 epochs (correlated diagnostic)")
        tail = [r for variant in (*VARIANTS, "spin_direct")
                for r in report["records"] + report["historical_references"]
                if r["variant"] == variant and r["seed"] == seed and r["cap"] == 160]
        if tail:
            lo, hi = min(r["last20_min_nmae"] for r in tail), max(r["last20_max_nmae"] for r in tail)
            pad = max((hi-lo)*.1, .0001)
            axes[1, column].set_ylim(lo-pad, hi+pad)
    paths = []
    for extension in ("png", "pdf"):
        path = destination / f"{report['dataset']}_learning_curves.{extension}"
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("geant", "abilene"), default="geant")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/spin-refine-20260917-v1")
    parser.add_argument("--historical-output", type=Path, default=ROOT / "outputs/spin-path-controls-20260917-v1")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--cost-reference", type=Path, action="append", default=[], help="Existing complete benchmark JSON; repeat to include multiple reports")
    parser.add_argument("--plot-curves", action="store_true", help="Optional Matplotlib PNG/PDF; no GPU required")
    args = parser.parse_args()
    destination = args.report_dir or ROOT / "analysis/spin_refine_20260917" / f"report_{args.dataset}"
    destination.mkdir(parents=True, exist_ok=True)
    report = summarize(args.output, args.historical_output, args.dataset)
    report["report_generator_sha256"] = sha(Path(__file__))
    report["cost_references"] = [cost_reference(path, args.dataset) for path in args.cost_reference]
    if report["state"] == "complete":
        rows = []
        for record in report["records"] + report["historical_references"]:
            for condition in CONDITIONS:
                rows.append({**{k: record[k] for k in ("dataset", "variant", "seed", "cap", "comparison_role", "best_epoch")},
                    "condition": condition, "nmae": record["condition_nmae"][condition],
                    "nrmse": record["condition_nrmse"][condition],
                    **{k:record[k] for k in ("last20_mean_nmae", "last20_median_nmae", "last20_min_nmae", "last20_max_nmae")}})
        with (destination / "RESULTS.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        if args.plot_curves:
            report["curve_artifacts"] = curves(report, args.output, args.historical_output, destination)
    (destination / "SUMMARY.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (destination / "REPORT_CN.md").write_text(markdown(report))
    print(json.dumps({"dataset": args.dataset, "state": report["state"], "report_dir": str(destination),
                      "component_screen_passed": report["component_screen_passed"]}))


if __name__ == "__main__":
    main()
