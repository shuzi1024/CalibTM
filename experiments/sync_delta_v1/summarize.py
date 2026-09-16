"""Summarize the ten-run development queue without opening traffic data.

Only result artifacts are read. Paired block bootstrap intervals describe
physical-window variation on development data, not initialization uncertainty
or performance on an independent test set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("abilene", "geant")
VARIANTS = ("sync_delta", "attention", "serial_delta", "batch_ridge", "additive")
CONDITIONS = ("uniform", "unequal", "unequal_gap")
SUM_FIELDS = ("abs_error", "abs_truth", "sq_error", "sq_truth", "count")
LABELS = {"sync_delta": "Sync-Delta", "attention": "Attention", "serial_delta": "Serial-Delta",
          "batch_ridge": "Batch-Ridge", "additive": "Additive", "linear": "Linear"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("top-level JSON must be an object")
        return payload, None
    except (OSError, ValueError) as exc:
        return None, f"{path}: {exc}"


def _evaluation(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return payload.get("primary_eval", payload)


def _evaluation_summary(evaluation: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(evaluation, dict):
        return {"status": "pending"}
    metrics = evaluation.get("metrics", {})
    scores = [metrics.get(condition, {}).get("nmae") for condition in CONDITIONS]
    if not all(_finite(value) and value >= 0 for value in scores):
        return {"status": "invalid", "error": "three finite nonnegative condition NMAEs are required"}
    score = float(sum(scores) / len(scores))
    reported = evaluation.get("selection_score")
    if not _finite(reported) or not math.isclose(score, reported, rel_tol=1e-9, abs_tol=1e-12):
        return {"status": "invalid", "error": "selection_score differs from equal condition mean"}
    for condition in CONDITIONS:
        entry = metrics[condition]
        if any(not _finite(entry.get(name)) or entry[name] < 0 for name in SUM_FIELDS):
            return {"status": "invalid", "error": f"invalid nonnegative error sums: {condition}"}
        if entry["count"] <= 0 or entry["abs_truth"] <= 0 or entry["sq_truth"] <= 0:
            return {"status": "invalid", "error": f"undefined aggregate truth denominator: {condition}"}
        expected = entry["abs_error"] / entry["abs_truth"]
        if not math.isclose(entry["nmae"], expected, rel_tol=1e-9, abs_tol=1e-12):
            return {"status": "invalid", "error": f"NMAE does not match ratio of sums: {condition}"}
        nrmse = math.sqrt(entry["sq_error"] / entry["sq_truth"])
        if not _finite(entry.get("nrmse")) or not math.isclose(entry["nrmse"], nrmse, rel_tol=1e-9, abs_tol=1e-12):
            return {"status": "invalid", "error": f"NRMSE does not match ratio of sums: {condition}"}
    return {"status": "complete", "role": "development_only", "selection_score": score,
            "mask_seed": evaluation.get("mask_seed"), "metrics": metrics,
            "physical_windows": len({row["window_start"] for row in evaluation.get("rows", [])}),
            "intervention": evaluation.get("intervention")}


def _row_tensor(evaluation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = evaluation.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("paired bootstrap requires nonempty physical-window rows")
    indexed = {}
    for row in rows:
        key = (int(row["window_start"]), row["condition"])
        if key in indexed or key[1] not in CONDITIONS:
            raise ValueError("duplicate or unregistered window/condition row")
        values = [row.get(field) for field in SUM_FIELDS]
        if not all(_finite(value) and value >= 0 for value in values):
            raise ValueError("row error sums must be finite and nonnegative")
        indexed[key] = values
    starts = np.asarray(sorted({key[0] for key in indexed}), dtype=np.int64)
    if len(indexed) != len(starts) * len(CONDITIONS):
        raise ValueError("each physical window must have all three conditions")
    try:
        tensor = np.asarray([[indexed[(int(start), condition)] for condition in CONDITIONS]
                             for start in starts], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("incomplete window/condition pairing") from exc
    for ci, condition in enumerate(CONDITIONS):
        sums = tensor[:, ci].sum(axis=0)
        metric = evaluation["metrics"][condition]
        if not np.allclose(sums, [metric[field] for field in SUM_FIELDS], rtol=1e-9, atol=1e-12):
            raise ValueError(f"row sums disagree with aggregate metrics for {condition}")
    return starts, tensor


def paired_block_bootstrap(candidate: dict[str, Any], baseline: dict[str, Any], *,
                           dataset: str, draws: int = 1000, seed: int = 82001,
                           block_length: int = 4) -> dict[str, Any]:
    """Paired circular moving blocks; all condition masks follow the same windows."""
    if draws < 1 or block_length < 1:
        raise ValueError("draws and block_length must be positive")
    if candidate.get("mask_seed") != baseline.get("mask_seed"):
        raise ValueError("paired comparison has different mask seeds")
    starts, cand = _row_tensor(candidate)
    other_starts, base = _row_tensor(baseline)
    if not np.array_equal(starts, other_starts):
        raise ValueError("paired comparison has different physical windows")
    if len(starts) > 1 and np.any(np.diff(starts) != 50):
        raise ValueError("bootstrap expects contiguous stride-50 development windows")
    if not np.array_equal(cand[..., 4], base[..., 4]):
        raise ValueError("paired comparison has different missing counts")
    if not np.allclose(cand[..., (1, 3)], base[..., (1, 3)], rtol=1e-10, atol=0):
        raise ValueError("paired comparison has different truth denominators")
    identity = json.dumps({"dataset": dataset, "seed": int(seed), "starts": starts.tolist()},
                          sort_keys=True, separators=(",", ":")).encode("ascii")
    derived = int.from_bytes(hashlib.sha256(b"sync-delta-v1.1:paired-blocks\0" + identity).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(derived))
    n = len(starts)
    length = min(block_length, n)
    block_starts = rng.integers(0, n, size=(draws, math.ceil(n / length)))
    indices = ((block_starts[..., None] + np.arange(length)) % n).reshape(draws, -1)[:, :n]
    candidate_sums = cand[indices].sum(axis=1)
    baseline_sums = base[indices].sum(axis=1)

    def ratio(sums: np.ndarray) -> np.ndarray:
        result = np.full(sums.shape[:-1], np.nan, dtype=np.float64)
        np.divide(sums[..., 0], sums[..., 1], out=result, where=sums[..., 1] > 0)
        return result

    candidate_draws, baseline_draws = ratio(candidate_sums), ratio(baseline_sums)
    point_candidate, point_baseline = ratio(cand.sum(axis=0)), ratio(base.sum(axis=0))
    entries = {}
    for ci, condition in enumerate((*CONDITIONS, "equal_condition_mean")):
        if ci < len(CONDITIONS):
            c_draw, b_draw = candidate_draws[:, ci], baseline_draws[:, ci]
            c_point, b_point = float(point_candidate[ci]), float(point_baseline[ci])
        else:
            c_draw, b_draw = candidate_draws.mean(axis=1), baseline_draws.mean(axis=1)
            c_point, b_point = float(point_candidate.mean()), float(point_baseline.mean())
        delta = b_draw - c_draw
        valid = np.isfinite(delta)
        interval = np.quantile(delta[valid], (0.025, 0.975)).tolist() if valid.any() else None
        entries[condition] = {"candidate_nmae": c_point, "baseline_nmae": b_point,
            "baseline_minus_candidate": b_point - c_point,
            "relative_reduction": 1.0 - c_point / b_point if b_point > 0 else None,
            "paired_block_ci95": interval, "valid_draws": int(valid.sum())}
    return {"status": "complete", "role": "development_window_variation_only",
            "method": "paired circular moving blocks, all masks bound to physical window",
            "dataset": dataset, "draws": draws, "seed": seed, "block_length": length,
            "physical_windows": n, "positive_difference_favors": "sync_delta",
            "initialization_variation_estimated": False,
            "checkpoint_selection_bias_corrected": False, "comparisons": entries}


def _convergence(result: dict[str, Any]) -> str:
    if result.get("stopped_early") is True:
        return "early_stop_observed"
    if result.get("best_in_last_five_at_cap") is True:
        return "budget_exhausted_best_in_last_five"
    completed, maximum, best = (result.get("epochs_completed"),
                                result.get("max_epochs", result.get("training_cap")),
                                result.get("best_epoch"))
    if not all(isinstance(value, int) for value in (completed, maximum, best)):
        return "not_certified_from_available_fields"
    if completed >= maximum and best >= completed - 4:
        return "budget_exhausted_best_in_last_five"
    return "no_late_budget_peak_flag"


def _load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result, error = _load(path / "result.json")
    if error:
        return {"status": "invalid_result", "error": error, "path": str(path)}, None
    if result is None:
        status_artifact, status_error = _load(path / "status.json")
        return {"status": "running_or_pending", "path": str(path),
                "progress": status_artifact, "progress_read_error": status_error}, None
    evaluation = _evaluation(result)
    summary = _evaluation_summary(evaluation)
    if summary["status"] != "complete":
        return {"status": "invalid_result", "path": str(path), "evaluation": summary}, None
    latency, latency_error = _load(path / "latency.json")
    postfreeze, postfreeze_error = _load(path / "postfreeze.json")
    run = {"status": "complete", "path": str(path), "evaluation": summary,
           "convergence": _convergence(result),
           **{key: result.get(key) for key in ("best_epoch", "stopped_early", "training_seconds",
                                              "parameter_count", "epochs_completed", "max_epochs",
                                              "training_cap", "best_in_last_five_at_cap")},
           "latency": latency, "latency_status": "invalid" if latency_error else ("complete" if latency else "pending"),
           "latency_read_error": latency_error,
           "postfreeze_status": "invalid" if postfreeze_error else ("complete" if postfreeze else "pending"),
           "postfreeze_read_error": postfreeze_error,
           "postfreeze_evaluations": {}, "interventions": {}, "serial_permutations": {}}
    if postfreeze:
        for seed in ("71002", "71003"):
            run["postfreeze_evaluations"][seed] = _evaluation_summary(postfreeze.get("evaluations", {}).get(seed))
        for intervention, item in postfreeze.get("interventions", {}).items():
            info = _evaluation_summary(item)
            if info["status"] == "complete":
                info["nmae_change_from_full"] = info["selection_score"] - summary["selection_score"]
            run["interventions"][intervention] = info
        permutations = postfreeze.get("serial_permutations", {})
        if isinstance(permutations, list):
            permutations = {str(index): value for index, value in enumerate(permutations)}
        run["serial_permutations"] = {str(key): _evaluation_summary(value) for key, value in permutations.items()}
        if any(value["status"] != "complete" for value in run["postfreeze_evaluations"].values()):
            run["postfreeze_status"] = "pending"
        if path.parent.name == "sync_delta" and any(
                run["interventions"].get(name, {}).get("status") != "complete"
                for name in ("zero_readout", "zero_value")):
            run["postfreeze_status"] = "pending"
        if path.parent.name == "serial_delta" and (
                len(run["serial_permutations"]) != 3 or any(
                    value["status"] != "complete" for value in run["serial_permutations"].values())):
            run["postfreeze_status"] = "pending"
    return run, evaluation


def _dataset_review(dataset: dict[str, Any]) -> dict[str, Any]:
    runs, linear = dataset["runs"], dataset["linear"]
    needed = ("sync_delta", "attention", "batch_ridge", "serial_delta")
    if any(runs[name]["status"] != "complete" for name in needed) or linear["status"] != "complete":
        return {"status": "pending", "note_cn": "关键对照尚未完成，暂不判断是否扩大实验。"}
    if any(dataset["paired_comparisons"].get(name, {}).get("status") == "invalid_pairing"
           for name in ("linear", "attention", "batch_ridge", "serial_delta")):
        return {"status": "invalid_pairing", "note_cn": "关键对照的物理窗口或误差汇总不匹配，先核查记录，暂不判断方法优势。"}
    scores = {name: runs[name]["evaluation"]["selection_score"] for name in needed}
    strongest = min(("attention", "batch_ridge"), key=lambda name: scores[name])
    checks = {"below_strongest_reader": scores["sync_delta"] < scores[strongest],
              "below_serial": scores["sync_delta"] < scores["serial_delta"],
              "below_linear": scores["sync_delta"] < linear["selection_score"]}
    convergence = {name: runs[name]["convergence"] for name in needed}
    late_peak = any(value == "budget_exhausted_best_in_last_five" for value in convergence.values())
    record = {"status": "descriptive_only", "strongest_reader_on_development": strongest,
              "strongest_reader_selection_is_descriptive": True, "point_estimate_checks": checks,
              "convergence": convergence, "late_budget_peak_requires_review": late_peak,
              "cost_ratios": {}}
    full_latency = (runs["sync_delta"].get("latency") or {}).get("mean_ms")
    for name in ("attention", "batch_ridge", "serial_delta"):
        other = (runs[name].get("latency") or {}).get("mean_ms")
        record["cost_ratios"][name] = full_latency / other if _finite(full_latency) and _finite(other) and other > 0 else None
    if late_peak:
        record["signal"] = "convergence_review_required"
        record["note_cn"] = "至少一个关键臂的最优点落预算末段，先核查收敛，不能据此宣布方法失败。"
    elif all(checks.values()):
        record["signal"] = "candidate_for_seed_confirmation"
        record["note_cn"] = "开发点估计优于最强读取对照、Serial 和 Linear；可讨论补种子确认，同时核对区间、干预与成本。"
    elif all(scores["sync_delta"] >= value for value in (scores["attention"], scores["batch_ridge"], linear["selection_score"])):
        record["signal"] = "no_reader_advantage_in_this_screen"
        record["note_cn"] = "当前开发点估计未优于 Attention、Ridge 或 Linear；训练正常时，不支持直接扩大记忆主干。"
    else:
        record["signal"] = "mixed_or_small_effect"
        record["note_cn"] = "优势不一致；先解释与 Serial、最强读取对照及成本的差异，暂不形成论文机制结论。"
    return record


def summarize(queue_root: str | Path, seed: int = 41001, draws: int = 1000,
              bootstrap_seed: int = 82001) -> dict[str, Any]:
    root = Path(queue_root).resolve()
    report = {"schema_version": 1, "role": "development_only", "queue_root": str(root),
              "created_utc": datetime.now(timezone.utc).isoformat(), "initialization_seed": seed,
              "expected_training_runs": len(DATASETS) * len(VARIANTS), "completed_training_runs": 0,
              "bootstrap": {"draws": draws, "seed": bootstrap_seed, "block_length": 4,
                            "scope": "paired development physical-window variation; not initialization uncertainty"},
              "independent_test_claim": False, "datasets": {}}
    for name in DATASETS:
        linear_payload, linear_error = _load(root / name / "linear_seed71001.json")
        linear_eval = _evaluation(linear_payload)
        linear = _evaluation_summary(linear_eval)
        if linear_error:
            linear = {"status": "invalid", "error": linear_error}
        dataset = {"runs": {}, "linear": linear, "paired_comparisons": {}}
        evaluations = {"linear": linear_eval}
        for variant in VARIANTS:
            run, evaluation = _load_run(root / name / variant / f"seed{seed}")
            dataset["runs"][variant] = run
            evaluations[variant] = evaluation
            report["completed_training_runs"] += int(run["status"] == "complete")
        full = evaluations["sync_delta"]
        if full is not None:
            for comparator in ("linear", "attention", "batch_ridge", "serial_delta", "additive"):
                comparison = evaluations.get(comparator)
                if comparison is None:
                    dataset["paired_comparisons"][comparator] = {"status": "pending"}
                    continue
                try:
                    dataset["paired_comparisons"][comparator] = paired_block_bootstrap(
                        full, comparison, dataset=name, draws=draws, seed=bootstrap_seed)
                except (KeyError, ValueError, TypeError) as exc:
                    dataset["paired_comparisons"][comparator] = {"status": "invalid_pairing", "error": str(exc)}
        dataset["review"] = _dataset_review(dataset)
        report["datasets"][name] = dataset
    completed = report["completed_training_runs"]
    postfreeze_complete = all(dataset["linear"]["status"] == "complete" for dataset in report["datasets"].values()) and all(
        run.get("postfreeze_status") == "complete" and run.get("latency_status") == "complete"
        for dataset in report["datasets"].values() for run in dataset["runs"].values())
    if completed < report["expected_training_runs"]:
        report["status"] = "running_or_pending"
        report["recommendation_cn"] = f"已完成 {completed}/10 次训练；其余任务未完成，保留当前状态，暂不作完整首轮判断。"
    elif not postfreeze_complete:
        report["status"] = "training_complete_followups_pending"
        report["recommendation_cn"] = "十次训练已完成，Linear、冻结后掩码复评或成本记录尚未齐全；先补齐已授权诊断。"
    else:
        report["status"] = "complete"
        signals = [dataset["review"].get("signal") for dataset in report["datasets"].values()]
        if all(signal == "candidate_for_seed_confirmation" for signal in signals):
            report["recommendation_cn"] = "两个 WAN 均有开发点估计信号，可据配对区间、干预和成本讨论补种子确认；本脚本不自动启动后续实验。"
        elif all(signal == "no_reader_advantage_in_this_screen" for signal in signals):
            report["recommendation_cn"] = "两个 WAN 均未显示对强读取对照的开发优势；确认训练正常后，不支持继续增加记忆模块。"
        else:
            report["recommendation_cn"] = "首轮结果需要结合两 WAN 的差异、收敛、观测值干预与成本判断；当前不自动给出继续投稿实验的结论。"
    report["limitations_cn"] = [
        "全部结果来自参与 checkpoint 选择的 source-dev+tune，均为开发结果。",
        "首轮只有一个初始化种子；时间块区间不代表初始化波动，也未校正开发选型偏差。",
        "额外 mask seeds 71002/71003 复用相同物理窗口，不是独立测试集。",
        "最强 Attention/Ridge 按同一开发均值描述性选出；两项固定配对比较均保留，不挑选显著区间。",
        "成本按 latency.json 的实际计时范围报告；缺失记录保持待测，不补造数值。",
    ]
    return report


def _fmt(value: Any, digits: int = 5) -> str:
    return f"{value:.{digits}f}" if _finite(value) else "—"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["**Sync-Delta 首轮开发结果汇总**", "",
             f"状态：`{report['status']}`；已完成 {report['completed_training_runs']}/10 次训练。",
             "全部为开发结果；三条件均值按每个条件的 global missing-only NMAE 等权计算。",
             "表中各条件为 NMAE / NRMSE，数值越低越好。", ""]
    for name, dataset in report["datasets"].items():
        lines += [f"**{name.capitalize()}：开发集**", "",
                  "| 方法 | 状态 | NMAE 均值 | Uniform | Unequal | Unequal-Gap | 延迟 ms |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for variant in ("linear", *VARIANTS):
            run = dataset["linear"] if variant == "linear" else dataset["runs"][variant]
            evaluation = run if variant == "linear" else run.get("evaluation", {})
            status = run["status"]
            label = "完成" if status == "complete" else ("未完成：运行中/待执行" if status in ("pending", "running_or_pending") else "记录需核查")
            values = []
            for condition in CONDITIONS:
                metric = evaluation.get("metrics", {}).get(condition, {})
                values.append(f"{_fmt(metric.get('nmae'))} / {_fmt(metric.get('nrmse'))}")
            latency = (run.get("latency") or {}).get("mean_ms")
            lines.append(f"| {LABELS[variant]} | {label} | {_fmt(evaluation.get('selection_score'))} | "
                         + " | ".join(values) + f" | {_fmt(latency, 2)} |")
        lines += ["", dataset["review"]["note_cn"], ""]
        full = dataset["runs"]["sync_delta"]
        if full["status"] == "complete":
            strongest = dataset["review"].get("strongest_reader_on_development")
            if strongest:
                lines.append(f"同开发均值下较强的读取对照：{LABELS[strongest]}；以下保留各项固定配对比较。")
                lines.append("")
            for comparator in ("linear", "attention", "batch_ridge", "serial_delta"):
                pairing = dataset["paired_comparisons"].get(comparator, {})
                if pairing.get("status") == "complete":
                    effect = pairing["comparisons"]["equal_condition_mean"]
                    ci = effect["paired_block_ci95"]
                    interval = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "未定义"
                    lines.append(f"- {LABELS[comparator]} − Sync 的综合开发 NMAE：{_fmt(effect['baseline_minus_candidate'])}，"
                                 f"配对时间块 95% 区间 {interval}；正值有利于 Sync。")
                elif pairing.get("status") == "invalid_pairing":
                    lines.append(f"- 与 {LABELS[comparator]} 的配对记录需核查：{pairing['error']}")
            lines.append("")
        completed_runs = [(variant, run) for variant, run in dataset["runs"].items() if run["status"] == "complete"]
        if completed_runs:
            cost = []
            for variant, run in completed_runs:
                seconds = run.get("training_seconds")
                peak = (run.get("latency") or {}).get("peak_allocated_bytes")
                cost.append(f"{LABELS[variant]}：训练 {_fmt(seconds / 3600 if _finite(seconds) else None, 2)} h，"
                            f"参数 {run.get('parameter_count', '—')}，"
                            f"推理峰值显存 {_fmt(peak / 2**30 if _finite(peak) else None, 2)} GiB，"
                            f"best epoch {run.get('best_epoch', '—')} ({run.get('convergence')})")
            lines += ["成本与收敛记录：", "", *[f"- {item}" for item in cost], ""]
        postfreeze_lines = []
        for variant, run in completed_runs:
            evaluations = run.get("postfreeze_evaluations", {})
            if any(item.get("status") == "complete" for item in evaluations.values()):
                pair = " / ".join(_fmt(evaluations.get(seed, {}).get("selection_score")) for seed in ("71002", "71003"))
                postfreeze_lines.append(f"{LABELS[variant]}：{pair}")
        if postfreeze_lines:
            lines += ["冻结后额外 mask 的开发 NMAE 均值（71002 / 71003）：" + "；".join(postfreeze_lines) + "。", ""]
        for intervention, item in full.get("interventions", {}).items():
            if item.get("status") == "complete":
                lines += [f"固定权重干预 `{intervention}`：开发均值 {_fmt(item['selection_score'])}，"
                          f"相对完整读出变化 {_fmt(item['nmae_change_from_full'])}（正值表示干预后更差）。", ""]
        permutations = dataset["runs"]["serial_delta"].get("serial_permutations", {})
        complete_permutations = {key: item["selection_score"] for key, item in permutations.items() if item.get("status") == "complete"}
        if complete_permutations:
            lines += ["Serial 固定排列开发均值：" + "；".join(f"{key}={_fmt(value)}" for key, value in complete_permutations.items())
                      + "。保留全部排列，不挑最佳或默认集成。", ""]
    lines += [report["recommendation_cn"], "", "解释边界：", "",
              *[f"- {item}" for item in report["limitations_cn"]], "",
              "逐条件配对区间、原始误差汇总、子组结果与计时范围保存在 summary.json。", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=82001)
    args = parser.parse_args(argv)
    if args.draws < 1:
        parser.error("--draws must be positive")
    report = summarize(args.root, args.seed, args.draws, args.bootstrap_seed)
    output = args.output_dir if args.output_dir is not None else args.root
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "SUMMARY_CN.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "completed_training_runs": report["completed_training_runs"],
                      "json": str(output / "summary.json"), "report": str(output / "SUMMARY_CN.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
