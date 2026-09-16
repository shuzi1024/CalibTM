"""Stdlib-only audit of completed Direct-only ablations and frozen references.

No model import, checkpoint deserialization, inference, training or candidate
selection occurs here. Missing/unfinished jobs expose status, never best scores.
Use --stdout to read remote outputs without writing into their directories.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.llm_candidates_runtime_v1.report import (
    CONDITIONS,
    _checkpoint,
    _compact,
    _reference,
    _result_identity,
    assert_paired,
    atomic_write,
    checked_evaluation,
    checked_latency,
    file_sha,
    read_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/direct-only-20260914-v1"
REGISTRY_SHA = "63b168051d28f9222ad52312d298f9eddacc728f6e91ea47e19063c1fbfc17ae"
FULL_SHA = "ca2096f69fc57abc218e29d51f5d5defe80ed989f83b51d049947379ecaee194"
PARENT_SHA = "f7dae0dfac2a776e75d62e57dade164d60b4fca49d329827018740dc071d3949"
DATASETS = ("abilene", "geant")
METHODS = ("direct_sync", "direct_only", "sync_delta", "ari", "linear")
LABELS = {"direct_sync": "完整 Direct＋Sync", "direct_only": "Direct-only（重新训练）",
          "sync_delta": "原 Sync", "ari": "ARI（统一任务适配）", "linear": "Linear"}


def _check_sources(record):
    for relative, digest in record["source_files"].items():
        if file_sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen source changed: {relative}")
    return {"verified_files": len(record["source_files"]),
            "source_files": record["source_files"]}


def _reference_full(registry, full_registry, dataset, evaluations):
    ref = registry["references"][dataset]["direct_sync"]
    payload, source = read_artifact(ref["path"], ref["sha256"])
    initial = full_registry["initializations"][dataset]["direct_sync"]["41001"]
    _result_identity(payload, dataset=dataset, variant="direct_sync", seed=41001,
                     registry_sha=FULL_SHA, initial_sha=initial["state_sha256"])
    if payload["parameter_count"] != initial["parameter_count"]:
        raise RuntimeError("full-model parameter count differs from frozen initialization")
    evaluation = checked_evaluation(payload, registry["data"][dataset])
    if evaluation["selection_score"] != ref["selection_score"]:
        raise RuntimeError("full reference score differs from registry")
    for other in evaluations.values():
        assert_paired(evaluation, other)
    evaluations["direct_sync"] = evaluation
    return _compact(evaluation, source, seed=41001, state="reused_complete", payload=payload,
                    checkpoint=_checkpoint(ref["path"], payload),
                    latency=checked_latency(Path(ref["path"]).parent / "latency.json",
                                            "direct_sync", required=True))


def _new_result(output, registry, dataset, evaluations):
    directory = output / dataset / "direct_only/seed41001"
    identity = {"dataset": dataset, "variant": "direct_only", "seed": 41001,
                "registry_sha256": REGISTRY_SHA}
    status_path = directory / "status.json"
    status, status_source = read_artifact(status_path) if status_path.exists() else ({"state": "not_started"}, None)
    if any(key in status and status[key] != value for key, value in identity.items()):
        raise RuntimeError("new-job status identity differs")
    # A status row's best_score is never promoted into the comparison.
    item = {"state": status.get("state", "unknown"), "seed": 41001,
            "status_source": status_source,
            "status": {key: status[key] for key in ("state", "updated", "epochs_completed", "epoch", "error")
                       if key in status}}
    result_path = directory / "result.json"
    exit_path = output / "logs" / f"{dataset}.exitcode"
    if not result_path.exists() or status.get("state") != "complete" or not exit_path.exists():
        if result_path.exists():
            item["unscored_result_source"] = {"path": str(result_path), "sha256": file_sha(result_path)}
        if item["state"] == "complete":
            item["state"] = "incomplete_artifacts"
        return item
    if exit_path.read_text().strip() != "0":
        raise RuntimeError(f"new-job worker did not exit successfully: {dataset}")
    if (directory / "failure.json").exists():
        raise RuntimeError(f"new-job failure artifact requires explicit review: {dataset}")
    payload, source = read_artifact(result_path)
    initial = registry["initializations"][dataset]["direct_only"]["41001"]
    _result_identity(payload, dataset=dataset, variant="direct_only", seed=41001,
                     registry_sha=REGISTRY_SHA, initial_sha=initial["state_sha256"])
    if payload["protocol"] != registry["protocol"] or payload["parameter_count"] != initial["parameter_count"]:
        raise RuntimeError("new-job protocol/parameter count differs")
    if (payload["training_cap"] != registry["training"]["max_epochs"]
            or not 1 <= payload["epochs_completed"] <= payload["training_cap"]
            or payload["optimizer_updates"] != 64 * payload["epochs_completed"]
            or payload["trainable_parameter_count"] != payload["parameter_count"]):
        raise RuntimeError("new-job epoch/optimizer/parameter accounting differs")
    checkpoint = _checkpoint(result_path, payload)
    evaluation = checked_evaluation(payload, registry["data"][dataset])
    if evaluation.get("mask_array_sha256") != registry["data"][dataset]["mask_array_sha256"]:
        raise RuntimeError("new-job mask digest differs")
    for other in evaluations.values():
        assert_paired(evaluation, other)
    evaluations["direct_only"] = evaluation
    best_eval, best_eval_source = read_artifact(directory / "best_eval.json")
    checked_evaluation(best_eval, registry["data"][dataset])
    if best_eval != evaluation:
        raise RuntimeError("best evaluation differs from the final result")
    smoke_path = output / "preflight" / dataset / "direct_only/REAL_DATA_SMOKE.json"
    smoke, smoke_source = read_artifact(smoke_path)
    if any(smoke.get(key) != value for key, value in identity.items()):
        raise RuntimeError("real-data smoke identity differs")
    if (smoke.get("passed") is not True or smoke.get("exact_checkpoint_continuation") is not True
            or smoke.get("original_epoch0_masks_and_order_exact") is not True
            or smoke.get("paired_dev_denominators_exact") is not True):
        raise RuntimeError("real-data smoke did not pass required checks")
    info, info_source = read_artifact(directory / "run_info.json")
    if any(info.get(key) != value for key, value in identity.items()):
        raise RuntimeError("run-info identity differs")
    if info["training"] != registry["training"] or info["preflight_sha256"] != smoke_source["sha256"]:
        raise RuntimeError("run-info training or smoke identity differs")
    history, history_source = read_artifact(directory / "history.json")
    if len(history) != payload["epochs_completed"]:
        raise RuntimeError("history length differs from completed epoch count")
    if [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise RuntimeError("history epochs are not sequential")
    best_row = min(history, key=lambda row: row["selection_score"])
    if best_row["epoch"] != payload["best_epoch"] or best_row["selection_score"] != evaluation["selection_score"]:
        raise RuntimeError("history best epoch/score differs from final result")
    item = _compact(evaluation, source, seed=41001, state="complete", payload=payload,
                    checkpoint=checkpoint,
                    latency=checked_latency(directory / "latency.json", "direct_only", required=True))
    item["training"].update({key: payload[key] for key in
                             ("elapsed_seconds", "peak_training_allocated_bytes", "best_in_last_five_at_cap")})
    item.update(status_source=status_source, best_eval_source=best_eval_source,
                smoke_source=smoke_source, run_info_source=info_source,
                history_source=history_source,
                last_checkpoint={"path": str(directory / "last.pt"), "sha256": file_sha(directory / "last.pt")},
                worker_exit={"path": str(exit_path), "sha256": file_sha(exit_path), "returncode": 0})
    return item


def _comparison(methods):
    if methods["direct_only"]["state"] != "complete":
        return {"available": False, "reason": "Direct-only is not a verified completed result"}
    full, direct = methods["direct_sync"], methods["direct_only"]
    by_condition = {}
    for condition in CONDITIONS:
        a, b = full["metrics"][condition], direct["metrics"][condition]
        by_condition[condition] = {
            "nmae_full": a["nmae"], "nmae_direct_only": b["nmae"],
            "nrmse_full": a["nrmse"], "nrmse_direct_only": b["nrmse"],
            "nmae_relative_reduction_by_adding_sync": (b["nmae"] - a["nmae"]) / b["nmae"],
            "nrmse_relative_reduction_by_adding_sync": (b["nrmse"] - a["nrmse"]) / b["nrmse"],
        }
    return {
        "available": True, "role": "single_initialization_development_ablation",
        "mean_nmae_relative_reduction_by_adding_sync":
            (direct["selection_score"] - full["selection_score"]) / direct["selection_score"],
        "mean_nrmse_relative_reduction_by_adding_sync":
            (direct["mean_nrmse"] - full["mean_nrmse"]) / direct["mean_nrmse"],
        "full_lower_nmae_conditions": sum(full["metrics"][c]["nmae"] < direct["metrics"][c]["nmae"] for c in CONDITIONS),
        "full_lower_nrmse_conditions": sum(full["metrics"][c]["nrmse"] < direct["metrics"][c]["nrmse"] for c in CONDITIONS),
        "condition_count": len(CONDITIONS), "conditions": by_condition,
        "parameters_added_by_sync": full["training"]["parameter_count"] - direct["training"]["parameter_count"],
        "full_to_direct_only_median_latency_ratio": full["latency"]["median_ms"] / direct["latency"]["median_ms"],
        "direct_only_nmae_relative_reductions_vs_references": {
            method: (methods[method]["selection_score"] - direct["selection_score"]) / methods[method]["selection_score"]
            for method in ("sync_delta", "ari", "linear")},
        "full_nmae_relative_reductions_vs_references": {
            method: (methods[method]["selection_score"] - full["selection_score"]) / methods[method]["selection_score"]
            for method in ("sync_delta", "ari", "linear")},
    }


def summarize(output):
    output = Path(output).resolve()
    registry, registry_source = read_artifact(output / "registry.json", REGISTRY_SHA)
    if (output / "registry.sha256").read_text().strip() != REGISTRY_SHA:
        raise RuntimeError("registry sidecar digest differs")
    if (registry.get("role") != "development_only" or registry.get("maximum_formal_new_jobs") != 2
            or registry.get("protocol") != "direct-only-development-20260914-v1"
            or registry.get("deadline_unix") is not None):
        raise RuntimeError("unexpected Direct-only scope or role")
    if registry["full_registry_sha256"] != FULL_SHA or registry["parent_registry_sha256"] != PARENT_SHA:
        raise RuntimeError("frozen parent digest differs")
    full, full_source = read_artifact(Path(registry["full_parent"]) / "registry.json", FULL_SHA)
    original, parent_source = read_artifact(Path(registry["parent"]) / "registry.json", PARENT_SHA)
    ari, ari_source = read_artifact(Path(full["ari_parent"]) / "registry.json", full["ari_registry_sha256"])
    checked_sources = {"direct_only": _check_sources(registry), "full": _check_sources(full),
                       "original": _check_sources(original), "ari": _check_sources(ari)}
    cpu_record, cpu_source = read_artifact(registry["cpu_correctness"]["path"], registry["cpu_correctness"]["sha256"])
    if cpu_record.get("status") != "passed" or cpu_record.get("device") != "cpu":
        raise RuntimeError("CPU correctness report did not pass")
    for dataset in DATASETS:
        if registry["data"][dataset] != full["data"][dataset]:
            raise RuntimeError("registered full/ablation data identity differs")
        if set(registry["initializations"][dataset]) != {"direct_only"}:
            raise RuntimeError("unexpected newly registered variant")
        for record, base in ((registry, output), (full, Path(registry["full_parent"]))):
            variant = "direct_only" if record is registry else "direct_sync"
            initial = record["initializations"][dataset][variant]["41001"]
            if file_sha(base / initial["file"]) != initial["sha256"]:
                raise RuntimeError("initialization checkpoint digest differs")
        if set(registry["initializations"][dataset]["direct_only"]) != {"41001"}:
            raise RuntimeError("unexpected newly registered initialization")
    summary = {
        "updated": datetime.now(timezone.utc).isoformat(), "state": "partial", "role": "development_only",
        "registry_sha256": REGISTRY_SHA, "registry_source": registry_source,
        "parent_sources": {"full": full_source, "original": parent_source, "ari": ari_source},
        "source_verification": checked_sources,
        "report_source": {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)},
        "cpu_correctness_source": cpu_source, "output": str(output), "datasets": {},
        "expected_new_runs": 2, "completed_new_runs": 0, "initialization_seed": 41001,
        "mask_seed": 71001, "scope": registry["scope"],
        "metric_definition": "global missing-only NMAE and NRMSE within each condition; equal arithmetic means over the three conditions",
        "limitations": [
            "One initialization and reused development physical windows; no independent test or statistical significance claim.",
            "Removing Sync also removes 23,040 parameters, cross-flow observations and its longer-range own-flow state; not a parameter-matched or formula-only comparison.",
            "Latency values reuse each model's completed end-to-end measurements; complete timing details are retained, not remeasured here.",
            "Legacy no-mask-digest results require registered shared masks and exact physical-window/truth-denominator pairing, explicitly marked per method.",
        ],
    }
    for dataset in DATASETS:
        methods, evaluations = {}, {}
        for method in ("linear", "sync_delta", "ari"):
            if registry["references"][dataset][method] != full["references"][dataset][method]:
                raise RuntimeError("reference changed relative to frozen full comparison")
            methods[method] = _reference(full, dataset, method, evaluations)
        methods["direct_sync"] = _reference_full(registry, full, dataset, evaluations)
        methods["direct_only"] = _new_result(output, registry, dataset, evaluations)
        summary["completed_new_runs"] += methods["direct_only"]["state"] == "complete"
        for method, item in methods.items():
            if "metrics" in item:
                item["role"] = "development_only"
                item["missing_target_counts"] = {c: item["metrics"][c]["count"] for c in CONDITIONS}
        summary["datasets"][dataset] = {
            "data": registry["data"][dataset], "initialization_seed": 41001,
            "methods": methods, "comparison": _comparison(methods),
            "paired_evaluations_verified": sorted(evaluations),
        }
    summary["state"] = "complete" if summary["completed_new_runs"] == 2 else "partial"
    if summary["state"] == "complete":
        summary["summed_new_training_seconds"] = sum(summary["datasets"][d]["methods"]["direct_only"]["training"]["training_seconds"] for d in DATASETS)
        summary["last_new_finished"] = max(summary["datasets"][d]["methods"]["direct_only"]["training"]["finished"] for d in DATASETS)
    return summary


def markdown(summary):
    lines = ["# Direct-only 核心消融：完成结果与来源核验", "",
             "本轮只新增两个 Direct-only 训练：Abilene / GEANT，各初始化 41001；完整模型、原 Sync、ARI 和 Linear 复用历史结果。",
             "每个精度单元为缺失位置全局 NMAE / NRMSE；平均值分别对三种既有条件等权计算。Linear 无初始化 seed，所有方法使用 mask71001。",
             "全部结果仍为 development-only；未完成任务不展示中途最佳值。", ""]
    for dataset in DATASETS:
        data = summary["datasets"][dataset]
        lines += [f"## {dataset.title()}", "",
                  "| 方法 | 平均 NMAE / NRMSE | Uniform | Unequal | Unequal-Gap | 延迟中位数 ms | 参数量 | 完成 / 最佳 epoch |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            item = data["methods"][method]
            if "metrics" not in item:
                lines.append(f"| {LABELS[method]} | {item['state']} | — | — | — | — | — | — |")
                continue
            metrics = " | ".join(f"{item['metrics'][c]['nmae']:.6f} / {item['metrics'][c]['nrmse']:.6f}" for c in CONDITIONS)
            latency = "—" if item["latency"] is None else f"{item['latency']['median_ms']:.4f}"
            trained = item["training"]
            epochs = "—" if trained.get("epochs_completed") is None else f"{trained['epochs_completed']} / {trained['best_epoch']}"
            lines.append(f"| {LABELS[method]} | {item['selection_score']:.6f} / {item['mean_nrmse']:.6f} | {metrics} | {latency} | {trained.get('parameter_count', '—'):,} | {epochs} |")
        lines.append("")
        comparison = data["comparison"]
        if comparison["available"]:
            lines += [f"在 Direct-only 上增加 Sync 后，平均 NMAE 相对降低 **{100 * comparison['mean_nmae_relative_reduction_by_adding_sync']:.2f}%**，平均 NRMSE 相对降低 **{100 * comparison['mean_nrmse_relative_reduction_by_adding_sync']:.2f}%**（负值表示变差）。",
                      f"完整模型在 {comparison['full_lower_nmae_conditions']}/3 条件的 NMAE、{comparison['full_lower_nrmse_conditions']}/3 条件的 NRMSE 更低；增加 {comparison['parameters_added_by_sync']:,} 参数，中位延迟为 Direct-only 的 **{comparison['full_to_direct_only_median_latency_ratio']:.2f} 倍**。", ""]
        counts = data["methods"]["linear"]["missing_target_counts"]
        lines += [f"评分包含 {data['data']['dev_window_count']} 个开发窗口，每条件缺失值数：{json.dumps(counts, ensure_ascii=False)}；五种方法均要求真实分母与窗口身份完全配对。", ""]
    lines += ["## 解释边界", "",
              "这个消融回答当前 Direct 读取和目标 query 已存在时，整个 Sync 路径还能贡献多少。需要把精度增益和延迟代价一起解释；单初始化的开发收益不证明结构必需或统计显著。",
              "移除 Sync 同时删除了记忆参数、跨流输入及远距离本流信息，因此不能单独证明同步更新公式、跨流机制或压缩损失的因果作用。",
              "推理延迟复用各自完成时的端到端测量；没有进行新的推理或训练。参数更少与推理更快的证据只按这里的实际测量报告。", "",
              f"完成核验：{summary['completed_new_runs']}/{summary['expected_new_runs']}，状态 `{summary['state']}`。新任务成功退出、最佳 checkpoint 实际 SHA256、初始 checkpoint、CPU/真实数据检查、历史最佳 epoch 与最终评分均已核验。",
              "注册表及全部冻结源文件、每种方法的结果/检查点/计时来源 SHA256、三条件的完整分子分母与计数保存在 comparison.json。旧结果缺少独立 mask 摘要时，在对应 JSON 字段单独注明。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--stdout", action="store_true", help="Write complete JSON only to stdout; no output artifacts are changed")
    parser.add_argument("--render-json", type=Path, help="Render an already audited JSON summary locally; no remote paths are reread")
    args = parser.parse_args()
    if args.render_json:
        summary, _ = read_artifact(args.render_json)
        if summary.get("registry_sha256") != REGISTRY_SHA or summary.get("role") != "development_only":
            raise RuntimeError("unexpected render input identity")
    else:
        summary = summarize(args.output)
    if args.stdout:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return
    report = args.report or args.output / "report"
    atomic_write(report / "comparison.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write(report / "COMPARISON_CN.md", markdown(summary))
    print(json.dumps({key: summary[key] for key in ("state", "completed_new_runs", "expected_new_runs", "updated")}))


if __name__ == "__main__":
    main()
