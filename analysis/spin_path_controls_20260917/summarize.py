#!/usr/bin/env python3
"""Read-only, stdlib-only summary of the frozen SPIN path-controls experiment.

No model loading, GPU calls, raw-data access, or new evaluations. Result files
are completion markers. A cumulative 1--160 history is one trajectory, not an
extra repeat. The 120-epoch cut remains visible. Historical references retain
their original budgets and are excluded from the matched route screen.
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
import statistics

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "spin-path-controls-development-20260917-v1"
VARIANTS = ("spin_direct", "context_only", "direct_only")
CONTROLS = VARIANTS[1:]
CONDITIONS = ("uniform", "unequal", "unequal_gap")
SEEDS = (41001, 41002)
TOTALS = ("count", "abs_error", "abs_truth", "sq_error", "sq_truth")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-13)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_identity(evaluation):
    key = lambda row: tuple(row[k] for k in ("condition", "cohort", "window_start"))
    rows = evaluation["rows"]
    require(len({key(row) for row in rows}) == len(rows), "duplicate evaluation row identity")
    return [(key(row), tuple(row[k] for k in ("count", "abs_truth", "sq_truth")),
             {group: tuple(values[k] for k in ("count", "abs_truth", "sq_truth"))
              for group, values in row.get("groups", {}).items()})
            for row in sorted(rows, key=key)]


def check_eval(evaluation, reference=None):
    require(evaluation["role"] == "development_only" and evaluation["mask_seed"] == 71001,
            "wrong development role or selection mask seed")
    require(set(evaluation["metrics"]) == set(CONDITIONS), "wrong evaluation conditions")
    for condition in CONDITIONS:
        metric = evaluation["metrics"][condition]
        rows = [row for row in evaluation["rows"] if row["condition"] == condition]
        require(rows and metric["abs_truth"] > 0 and metric["sq_truth"] > 0, "missing denominators")
        for name in TOTALS:
            require(near(sum(row[name] for row in rows), metric[name]), f"{condition}: {name} sum differs")
        require(near(metric["nmae"], metric["abs_error"] / metric["abs_truth"]), "NMAE formula differs")
        require(near(metric["nrmse"], math.sqrt(metric["sq_error"] / metric["sq_truth"])), "NRMSE formula differs")
        for group, aggregate in metric.get("groups", {}).items():
            for name in TOTALS:
                require(near(sum(row["groups"][group][name] for row in rows), aggregate[name]),
                        f"{condition}/{group}: {name} sum differs")
            require(near(aggregate["nmae"], aggregate["abs_error"] / aggregate["abs_truth"]), "group NMAE differs")
            require(near(aggregate["nrmse"], math.sqrt(aggregate["sq_error"] / aggregate["sq_truth"])), "group NRMSE differs")
    require(near(evaluation["selection_score"], statistics.mean(evaluation["metrics"][c]["nmae"] for c in CONDITIONS)),
            "three-condition equal mean differs")
    identity = row_identity(evaluation)
    if reference is not None:
        require(identity == row_identity(reference), "paired evaluation identities/truth denominators differ")
        if "mask_array_sha256" in evaluation and "mask_array_sha256" in reference:
            require(evaluation["mask_array_sha256"] == reference["mask_array_sha256"], "evaluation mask hashes differ")


def scores(evaluation):
    return {"mean_nmae": evaluation["selection_score"],
            "mean_nrmse": statistics.mean(evaluation["metrics"][c]["nrmse"] for c in CONDITIONS),
            "conditions": {c: {metric: evaluation["metrics"][c][metric] for metric in ("nmae", "nrmse")}
                           for c in CONDITIONS}}


def sample_summary(items):
    def apply(operation):
        return {**{metric: operation(item[metric] for item in items) for metric in ("mean_nmae", "mean_nrmse")},
                "conditions": {condition: {metric: operation(item["conditions"][condition][metric] for item in items)
                                            for metric in ("nmae", "nrmse")} for condition in CONDITIONS}}
    return {"n": len(items), "mean": apply(statistics.mean),
            "sample_sd": apply(statistics.stdev) if len(items) >= 2 else None}


def difference(candidate, control):
    def pair(a, b):
        return {"delta": a - b, "relative_improvement_percent": 100 * (b - a) / b}
    return {**{metric: pair(candidate[metric], control[metric]) for metric in ("mean_nmae", "mean_nrmse")},
            "conditions": {c: {metric: pair(candidate["conditions"][c][metric], control["conditions"][c][metric])
                               for metric in ("nmae", "nrmse")} for c in CONDITIONS}}


def distribution(values):
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "range": max(values) - min(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def history_summary(history, cap):
    prefix = [row for row in history if row["epoch"] <= cap]
    require(len(prefix) == cap, f"history does not contain complete 1--{cap} prefix")
    selected = min(prefix, key=lambda row: row["selection_score"])
    tail = prefix[-20:]
    last = prefix[-1]
    rmses = [statistics.mean(row["condition_nrmse"][c] for c in CONDITIONS)
             for row in tail if "condition_nrmse" in row]
    return {"cap": cap, "best_epoch": selected["epoch"], "best_nmae": selected["selection_score"],
            "last_epoch": last["epoch"], "last_nmae": last["selection_score"],
            "last_condition_nmae": last["condition_nmae"],
            "last_condition_nrmse": last.get("condition_nrmse"),
            "last_mean_nrmse": statistics.mean(last["condition_nrmse"].values()) if "condition_nrmse" in last else None,
            "tail_epoch_range": [tail[0]["epoch"], tail[-1]["epoch"]],
            "tail_nmae": distribution([row["selection_score"] for row in tail]),
            "tail_nrmse": distribution(rmses) if len(rmses) == len(tail) else None,
            "tail_nrmse_available_epochs": len(rmses),
            "tail_condition_nmae": {c: distribution([row["condition_nmae"][c] for row in tail]) for c in CONDITIONS},
            "tail_rows": [{key: row[key] for key in ("epoch", "selection_score", "condition_nmae", "condition_nrmse", "lr") if key in row}
                          for row in tail],
            "interpretation": "descriptive optimization diagnostic; epochs are correlated; does not replace selected-checkpoint metrics"}


def check_history(history):
    require([row["epoch"] for row in history] == list(range(1, len(history) + 1)), "history has skipped/repeated epochs")
    require(len(history) <= 160, "training exceeds frozen 160-epoch cap")
    for row in history:
        require(row["updates"] == 64 * row["epoch"], "optimizer update count differs")
        require(near(row["selection_score"], statistics.mean(row["condition_nmae"][c] for c in CONDITIONS)),
                "history selection score differs from condition mean")
        expected = .001 if row["epoch"] <= 120 else .0001
        require(("lr" not in row and row["epoch"] <= 120) or near(row["lr"], expected), "frozen learning rate differs")


def completed_slice(path, record, history, cap, identity, config, reference):
    require(record["state"] == "complete", "result is not marked complete")
    for key, value in identity.items():
        require(record[key] == value, "result identity differs: " + key)
    require(record["role"] == "development_only" and record["independent_test"] is False, "wrong result evaluation role")
    require(record["epochs_completed"] == cap and record["optimizer_updates"] == 64 * cap,
            "result cap/update count differs")
    require(record["config_sha256"] == sha(path / "config.json"), "result config hash differs")
    check_eval(record["primary_eval"], reference)
    trajectory = history_summary(history, cap)
    require(record["best_epoch"] == trajectory["best_epoch"] and near(record["primary_eval"]["selection_score"], trajectory["best_nmae"]),
            "result disagrees with strict best-development checkpoint selection")
    checkpoint = path / ("best120.pt" if cap == 120 else "best.pt")
    if checkpoint.exists():
        require(sha(checkpoint) == record["best_checkpoint_sha256"], "selected checkpoint file hash differs")
    return {"eligible": True, "metrics": scores(record["primary_eval"]), "trajectory": trajectory,
            "groups": {c: record["primary_eval"]["metrics"][c].get("groups", {}) for c in CONDITIONS},
            "best_checkpoint_sha256": record["best_checkpoint_sha256"],
            "checkpoint_file_hash_checked": checkpoint.exists(), "checkpoint_tensors_deserialized": False,
            "parameter_count": record.get("parameter_count", config.get("parameter_count")),
            "epochs_completed": cap, "best_epoch": record["best_epoch"]}


def job_summary(path, dataset, variant, seed, reference, old_outputs):
    job = {"path": str(path), "variant": variant, "seed": seed, "state": "pending", "errors": [], "warnings": [], "slices": {}}
    try:
        config, history = read(path / "config.json"), read(path / "history.json") or []
        status = read(path / "status.json") or {}
        job.update(saved_status=status, state=status.get("state", "pending"), saved_epochs=len(history), history=history)
        if history:
            check_history(history)
        if config is None:
            require(not history, "history exists without config")
            return job
        identity = {"protocol": PROTOCOL, "dataset": dataset, "variant": variant, "seed": seed}
        require(all(config[key] == value for key, value in identity.items()), "config identity differs")
        job["pairing"] = {key: config[key] for key in ("data", "training", "source_files")}
        job["config_sha256"] = sha(path / "config.json")
        job["shared_named_parameters_exact"] = config.get("shared_named_parameters_exact")
        job["anchor_parent"] = config.get("anchor_parent")
        job["continuation_checkpoint_loaded"] = config.get("continuation_checkpoint_loaded")
        require(config.get("continuation_checkpoint_loaded") is (variant == "spin_direct"), "wrong continuation versus new-control interpretation")
        require(config.get("independent_training_repeat") is (variant != "spin_direct"), "wrong independent-repeat interpretation")
        anchor_path = old_outputs / dataset / "spin_direct_no_memory" / f"seed{seed}"
        anchor = config.get("anchor_parent", {})
        checked = []
        for name, digest in anchor.get("files_sha256", {}).items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe parent artifact name")
            local = anchor_path / name
            if local.exists():
                require(sha(local) == digest, "original parent artifact hash differs: " + name)
                checked.append(name)
        job["parent_artifacts_checked"] = checked
        for cap, name in ((120, "result120.json"), (160, "result.json")):
            result = read(path / name)
            if result is None:
                continue
            job["slices"][str(cap)] = completed_slice(path, result, history, cap, identity, config, reference)
            job["slices"][str(cap)]["result_sha256"] = sha(path / name)
        if "160" in job["slices"]:
            job["state"] = "complete"
        elif job["state"] == "complete":
            job["state"] = "incomplete"
            job["warnings"].append("saved status says complete but no validated 160-epoch result")
        if "120" in job["slices"]:
            history120 = read(path / "history120.json")
            if history120 is not None:
                require(history120 == history[:120], "history120 differs from cumulative prefix")
            else:
                job["warnings"].append("history120.json not copied; cumulative prefix used")
            if variant == "spin_direct":
                parent_history = read(anchor_path / "history.json")
                parent_result = read(anchor_path / "result.json")
                if parent_history is not None and parent_result is not None:
                    require(history[:120] == parent_history, "continued model's 120-epoch history prefix changed")
                    current = read(path / "result120.json")["primary_eval"]
                    require(all(current[field] == parent_result["primary_eval"][field]
                                for field in ("selection_score", "metrics", "rows")), "continued model's original 120-epoch result changed")
                    job["original_120_history_and_evaluation_exact"] = True
                else:
                    job["warnings"].append("original 120-epoch parent JSON unavailable for independent prefix verification")
        if any(not section["checkpoint_file_hash_checked"] for section in job["slices"].values()):
            job["warnings"].append("selected checkpoint binary not copied; result hash metadata retained without tensor verification")
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        job.update(state="invalid", slices={})
        job["errors"].append(f"{type(error).__name__}: {error}")
    return job


def pair_summary(candidate, control, cap):
    key = str(cap)
    if key not in candidate["slices"] or key not in control["slices"]:
        return {"eligible": False, "reason": "one or both complete budget slices unavailable"}
    try:
        require(candidate["pairing"] == control["pairing"], "data/training/source configuration differs")
        require(candidate["shared_named_parameters_exact"] is True and control["shared_named_parameters_exact"] is True,
                "shared initialization exactness not verified by runner")
        for a, b in zip(candidate["history"][:cap], control["history"][:cap]):
            for field in ("epoch_zero_based", "mask_sha256", "order_sha256"):
                require(a["schedule"][field] == b["schedule"][field], "per-epoch mask/order schedule differs")
        a, b = candidate["slices"][key], control["slices"][key]
        tail_a, tail_b = candidate["history"][cap - 20:cap], control["history"][cap - 20:cap]
        wins = sum(x["selection_score"] < y["selection_score"] for x, y in zip(tail_a, tail_b))
        ties = sum(x["selection_score"] == y["selection_score"] for x, y in zip(tail_a, tail_b))
        return {"eligible": True, "candidate_minus_control": difference(a["metrics"], b["metrics"]),
                "best_nmae_candidate_wins": a["metrics"]["mean_nmae"] < b["metrics"]["mean_nmae"],
                "tail_median_nmae_delta": a["trajectory"]["tail_nmae"]["median"] - b["trajectory"]["tail_nmae"]["median"],
                "tail_mean_nmae_delta": a["trajectory"]["tail_nmae"]["mean"] - b["trajectory"]["tail_nmae"]["mean"],
                "tail_same_epoch_candidate_wins": wins, "tail_same_epoch_ties": ties, "tail_epochs": 20,
                "candidate_tail_epochs_better_than_control_selected_best": sum(row["selection_score"] < b["metrics"]["mean_nmae"] for row in tail_a),
                "tail_win_count_interpretation": "correlated optimization iterates, not independent repeats or significance"}
    except (ValueError, KeyError, TypeError) as error:
        return {"eligible": False, "reason": str(error)}


def historical_references(args, reference):
    refs = {}
    for variant, label in (("spin_sync_direct", "Full_fixed120"), ("spin_direct_no_memory", "SPIN_Direct_fixed120")):
        records = []
        for seed in SEEDS:
            path = args.old_outputs / args.dataset / variant / f"seed{seed}" / "result.json"
            record = read(path)
            if record is None:
                continue
            require(record["state"] == "complete" and record["dataset"] == args.dataset and record["variant"] == variant and record["seed"] == seed,
                    "historical reference identity differs")
            require(record["epochs_completed"] == 120, "historical reference is not fixed120")
            check_eval(record["primary_eval"], reference)
            records.append({"seed": seed, "metrics": scores(record["primary_eval"]), "path": str(path), "sha256": sha(path),
                            "epochs_completed": record["epochs_completed"], "best_epoch": record["best_epoch"]})
        if records:
            refs[label] = {"records": records, "summary": sample_summary([record["metrics"] for record in records]),
                           "interpretation": "historical 120-epoch recipe; not a new 160-epoch matched control; original trajectory counted once"}
    for name in ("linear", "spin", "direct"):
        path = args.references / args.dataset / f"{name}.json"
        record = read(path)
        if record is None:
            continue
        evaluation = record if name == "linear" else record["primary_eval"]
        check_eval(evaluation, reference)
        refs[name] = {"metrics": scores(evaluation), "path": str(path), "sha256": sha(path),
                      "epochs_completed": record.get("epochs_completed"), "best_epoch": record.get("best_epoch"),
                      "interpretation": "saved historical reference; no new measurement; neural recipes/depth/budgets/repeats differ; excluded from matched route screen"}
    return refs


def collect(args):
    linear_path = args.references / args.dataset / "linear.json"
    reference = read(linear_path)
    require(reference is not None, f"missing original Linear evaluation: {linear_path}")
    check_eval(reference)
    jobs = {variant: {str(seed): job_summary(args.outputs / args.dataset / variant / f"seed{seed}", args.dataset, variant, seed, reference, args.old_outputs)
                      for seed in SEEDS} for variant in VARIANTS}
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL, "dataset": args.dataset,
              "role": "development_only", "independent_test": False, "primary_metric": "equal mean of three global missing-only NMAEs",
              "expected_variants": list(VARIANTS), "expected_seeds": list(SEEDS), "jobs": jobs, "budgets": {},
              "historical_references": historical_references(args, reference),
              "state_counts": dict(Counter(job["state"] for variants in jobs.values() for job in variants.values())),
              "interpretation": "route screening on previously consumed development data; not submission confirmation, independent testing, significance, or proof of full optimization",
              "no_automatic_followup": "unclear component increment means stop/review; never add modules, seeds, or epochs automatically"}
    for cap in (120, 160):
        key = str(cap)
        block = {"seed_summaries": {}, "candidate_vs_controls": {}}
        for variant in VARIANTS:
            completed = [job["slices"][key]["metrics"] for job in jobs[variant].values() if key in job["slices"]]
            if completed:
                block["seed_summaries"][variant] = {**sample_summary(completed), "all_expected_seeds_complete": len(completed) == len(SEEDS)}
        for control in CONTROLS:
            block["candidate_vs_controls"][control] = {str(seed): pair_summary(jobs["spin_direct"][str(seed)], jobs[control][str(seed)], cap)
                                                       for seed in SEEDS}
        pairs = [pair for control in block["candidate_vs_controls"].values() for pair in control.values()]
        complete = all(pair["eligible"] for pair in pairs)
        block["matched_matrix_complete"] = complete
        block["best_nmae_both_seeds_beat_both_controls"] = all(pair["best_nmae_candidate_wins"] for pair in pairs) if complete else None
        block["tail_median_direction_consistent_all_pairs"] = all(pair["tail_median_nmae_delta"] < 0 for pair in pairs) if complete else None
        if not complete:
            block["screen_status"] = "incomplete_or_invalid_no_direction_decision"
        elif not block["best_nmae_both_seeds_beat_both_controls"]:
            block["screen_status"] = "combination_increment_not_established_stop_for_review"
        elif not block["tail_median_direction_consistent_all_pairs"]:
            block["screen_status"] = "selected_checkpoint_advantage_but_tail_direction_inconsistent_review_stability"
        else:
            block["screen_status"] = "positive_component_screen_requires_effect_size_cost_and_frozen_confirmation_review"
        block["screen_scope"] = "best-checkpoint NMAE is primary; tail diagnostics describe whether the direction persists, never redefine checkpoint selection or statistical significance"
        block["is_final_budget"] = cap == 160
        report["budgets"][key] = block
    report["all_six_jobs_complete"] = all(job["state"] == "complete" and "160" in job["slices"] for variants in jobs.values() for job in variants.values())
    return report


def markdown(report):
    lines = ["# SPIN＋Direct 路径必要性：开发结果汇总", "", f"生成：{report['generated_utc']}；数据：{report['dataset'].upper()}；状态：{report['state_counts']}。", "",
             "主指标仍为三种条件等权平均的 NMAE，检查点按开发集严格改善选择。以下是路线筛选，不是独立测试、显著性判断或充分收敛证明。",
             "每种模型两个配对初始化；120→160 为同一轨迹的固定降学习率诊断，不能计为额外重复。共有六条轨迹。全程保留 120 轮截面。",
             "Context-only 是当前骨架路径对照，不能冒称官方 SPIN；Direct-only 不接收 SPIN 上下文。路径删除也改变参数和计算量，非等容量控制。", ""]
    for cap, block in report["budgets"].items():
        stage_note = "历史预算截面的描述，不提前替代 160 轮路线判断。" if cap == "120" else "全部六条轨迹完成且配对通过后才作本轮方向判断。"
        lines += [f"## {cap} 轮预算截面", "", f"筛选状态：`{block['screen_status']}`。{stage_note}", "",
                  "| 模型 | 完成种子 | NMAE 均值 ± 样本 SD | NRMSE 均值 ± 样本 SD |", "|---|---:|---:|---:|"]
        for variant in VARIANTS:
            summary = block["seed_summaries"].get(variant)
            if not summary:
                lines.append(f"| {variant} | 0/2 | — | — |")
                continue
            def fmt(metric):
                return f"{summary['mean'][metric]:.9f} ± {summary['sample_sd'][metric]:.9f}" if summary["sample_sd"] else f"{summary['mean'][metric]:.9f}（n=1）"
            lines.append(f"| {variant} | {summary['n']}/2 | {fmt('mean_nmae')} | {fmt('mean_nrmse')} |")
        lines += ["", "### 每个种子的完整条件结果", "", "| 模型 | 种子 | 最佳轮 | 条件 | NMAE | NRMSE |", "|---|---:|---:|---|---:|---:|"]
        for variant, variants in report["jobs"].items():
            for seed, job in variants.items():
                result = job["slices"].get(cap)
                if result:
                    for condition in CONDITIONS:
                        metric = result["metrics"]["conditions"][condition]
                        lines.append(f"| {variant} | {seed} | {result['best_epoch']} | {condition} | {metric['nmae']:.9f} | {metric['nrmse']:.9f} |")
        lines += ["", "### 组合相对两条单路径的配对差", "", "差 = 组合−对照；负值为组合更好；改善率 = 100×(对照−组合)/对照。", "",
                  "| 对照 | 种子 | NMAE 差 | NMAE 改善 % | NRMSE 差 | 末20轮中位 NMAE 差 | 同轮胜出/20 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for control, pairs in block["candidate_vs_controls"].items():
            for seed, pair in pairs.items():
                if not pair["eligible"]:
                    lines.append(f"| {control} | {seed} | 未完成或不匹配：{pair['reason']} | — | — | — | — |")
                    continue
                delta = pair["candidate_minus_control"]
                lines.append(f"| {control} | {seed} | {delta['mean_nmae']['delta']:+.9f} | {delta['mean_nmae']['relative_improvement_percent']:+.4f} | {delta['mean_nrmse']['delta']:+.9f} | {pair['tail_median_nmae_delta']:+.9f} | {pair['tail_same_epoch_candidate_wins']}/20 |")
        lines += ["", "### 最佳点、末轮及末20轮轨迹", "", "末段统计只解释波动，不替代主指标。连续训练轮不是独立样本。", "",
                  "| 模型 | 种子 | 最佳轮 / NMAE | 末轮 NMAE | 末20轮中位 NMAE | 末20轮最小—最大 |", "|---|---:|---:|---:|---:|---:|"]
        for variant, variants in report["jobs"].items():
            for seed, job in variants.items():
                result = job["slices"].get(cap)
                if result:
                    trajectory = result["trajectory"]
                    tail = trajectory["tail_nmae"]
                    lines.append(f"| {variant} | {seed} | {trajectory['best_epoch']} / {trajectory['best_nmae']:.9f} | {trajectory['last_nmae']:.9f} | {tail['median']:.9f} | {tail['min']:.9f}—{tail['max']:.9f} |")
        lines += [""]
    lines += ["## 历史参考（不加入匹配筛选）", "", "下表均来自既有结果，没有新增测量。Full 使用旧 120 轮预算；旧 SPIN/Direct＋Sync 的层数、训练配方、预算与重复次数不同。Linear 无训练种子。", "",
              "| 历史参考 | NMAE | NRMSE | 说明 |", "|---|---:|---:|---|"]
    for name, reference in report["historical_references"].items():
        metric = reference["summary"]["mean"] if "summary" in reference else reference["metrics"]
        note = "旧120轮、两个种子" if "summary" in reference else ("无训练" if name == "linear" else "旧配方、一个种子")
        lines.append(f"| {name} | {metric['mean_nmae']:.9f} | {metric['mean_nrmse']:.9f} | {note} |")
    lines += ["", "## 审计与解释边界", "",
              "- 所有计划种子均保留；部分完成均值不作最终方向判断。样本 SD 只描述两个初始化。",
              "- 两个种子的组合最佳 NMAE 都需优于两条单路径；末段方向不一致则明确保留稳定性疑问。正向筛选也仍需人工审阅改善幅度、成本与后续冻结确认。",
              "- 组合增量不明确时收束审阅，不自动增加模块、种子或训练轮数。",
              "- JSON 保留逐种子/条件/掩码分组、配对差、完整已保存轨迹及末20轮诊断。继承的旧120轮未保存逐轮 NRMSE，缺失值保持为空，不补造。",
              "- 本脚本只读现有 JSON/检查点文件哈希；不加载模型、不运行推理，不把旧参考改写成新测量。", ""]
    for variant, variants in report["jobs"].items():
        for seed, job in variants.items():
            for issue in job["errors"] + job["warnings"]:
                lines.append(f"- {variant}/{seed}：{issue}")
    return "\n".join(lines) + "\n"


def write_csv(report, path):
    fields = ["dataset", "budget", "variant", "seed", "best_epoch", "condition", "nmae", "nrmse"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant, variants in report["jobs"].items():
            for seed, job in variants.items():
                for cap, result in job["slices"].items():
                    for condition, metric in result["metrics"]["conditions"].items():
                        writer.writerow({"dataset": report["dataset"], "budget": cap, "variant": variant, "seed": seed,
                                         "best_epoch": result["best_epoch"], "condition": condition, **metric})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--dataset", choices=("geant", "abilene"), default="geant")
    parser.add_argument("--references", type=Path, default=ROOT / "analysis/research_7h_20260916/references")
    parser.add_argument("--old-outputs", type=Path, default=ROOT / "analysis/spin_sync_unified_20260916/raw_fixed120")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).parent / "summary")
    args = parser.parse_args()
    report = collect(args)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "SUMMARY.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.report_dir / "SUMMARY_CN.md").write_text(markdown(report))
    write_csv(report, args.report_dir / "CONDITIONS.csv")
    print(json.dumps({"report_dir": str(args.report_dir), "state_counts": report["state_counts"],
                      "all_six_jobs_complete": report["all_six_jobs_complete"],
                      "screen160": report["budgets"]["160"]["screen_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
