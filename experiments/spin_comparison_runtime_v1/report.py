"""Stdlib-only audit of completed SPIN task adaptations and frozen references.

No model import, checkpoint deserialization, inference, training or candidate
selection occurs here. Missing/unfinished jobs expose status, never best scores.
Use --stdout to read remote outputs without writing into their directories.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
DEFAULT_OUTPUT = ROOT / "outputs/spin-comparison-20260914-v1"
REGISTRY_SHA = "70b8bb21285e9e72758f418c53eb8f782a8dfa56b665f4aa96b2c8d2910cd8f2"
FULL_SHA = "ca2096f69fc57abc218e29d51f5d5defe80ed989f83b51d049947379ecaee194"
PARENT_SHA = "f7dae0dfac2a776e75d62e57dade164d60b4fca49d329827018740dc071d3949"
DATASETS = ("abilene", "geant")
METHODS = ("direct_sync", "spin_adapted", "sync_delta", "ari", "linear")
LABELS = {"direct_sync": "完整 Direct＋Sync", "spin_adapted": "SPIN（官方结构任务适配）",
          "sync_delta": "原 Sync", "ari": "ARI（统一任务适配）", "linear": "Linear"}


def _check_sources(record):
    for relative, digest in record["source_files"].items():
        if file_sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen source changed: {relative}")
    return {"verified_files": len(record["source_files"]),
            "source_files": record["source_files"]}


def _upstream_provenance():
    base = ROOT / "experiments/spin_comparison_v1"
    manifest, source = read_artifact(base / "source_manifest.json")
    if (manifest["official_commit"] != "7349ba31da7306e7e96c13668a3f1f0a4df90902"
            or manifest["tsl_source_tag"] != "v0.1.1"):
        raise RuntimeError("unexpected SPIN or TSL source version")
    for relative, digest in manifest["files"].items():
        if file_sha(base / relative) != digest:
            raise RuntimeError(f"archived SPIN/TSL provenance changed: {relative}")
    return {"source": source, "verified_files": len(manifest["files"]),
            "official_repository": manifest["official_repository"],
            "official_commit": manifest["official_commit"], "tsl_source_tag": manifest["tsl_source_tag"]}


def _scheduled_lr(step):
    if step < 12:
        factor = max(.1, step / 12.)
    else:
        progress = (step - 12.) / 288.
        if progress >= 1.:
            return 0.
        cosine = .5 * (1 + math.cos(math.pi * ((3. * progress) % 1.)))
        factor = max(.1, cosine * (1. - progress * .67))
    return .0008 * factor


def _checked_history(history, payload):
    for field in ("training_seconds", "elapsed_seconds", "peak_training_allocated_bytes"):
        if not math.isfinite(payload[field]) or payload[field] <= 0:
            raise RuntimeError(f"invalid completed SPIN cost: {field}")
    if payload["elapsed_seconds"] < payload["training_seconds"]:
        raise RuntimeError("SPIN elapsed time is less than its training time")
    if not isinstance(payload["stopped_early"], bool) or not isinstance(payload["best_in_last_five_at_cap"], bool):
        raise RuntimeError("invalid SPIN stopping/cap flags")
    for row in history:
        epoch = row["epoch"]
        if row["updates"] != epoch * 64:
            raise RuntimeError("SPIN history optimizer count differs")
        for field in ("train_loss", "mean_gradient_norm", "selection_score", "training_seconds", "evaluation_seconds"):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise RuntimeError(f"nonfinite/negative SPIN history field: {field}")
        if (not math.isclose(row["learning_rate"], _scheduled_lr(epoch - 1), rel_tol=1e-12, abs_tol=1e-15)
                or not math.isclose(row["next_learning_rate"], _scheduled_lr(epoch), rel_tol=1e-12, abs_tol=1e-15)):
            raise RuntimeError("SPIN learning-rate schedule differs from original epoch schedule")
    if not math.isclose(sum(row["training_seconds"] for row in history),
                        payload["training_seconds"], rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError("SPIN training-time accounting differs")
    capped = not payload["stopped_early"]
    if capped and payload["epochs_completed"] != 300:
        raise RuntimeError("unfinished SPIN result cannot be counted as complete")
    if payload["stopped_early"] and history[-1]["bad_epochs"] < 40:
        raise RuntimeError("SPIN stopped before its declared patience")
    expected_cap_flag = capped and payload["best_epoch"] >= 296
    if payload["best_in_last_five_at_cap"] != expected_cap_flag:
        raise RuntimeError("SPIN convergence-cap flag differs")


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
    directory = output / dataset / "spin_adapted/seed41001"
    identity = {"dataset": dataset, "variant": "spin_adapted", "seed": 41001,
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
    initial = registry["initializations"][dataset]["spin_adapted"]["41001"]
    _result_identity(payload, dataset=dataset, variant="spin_adapted", seed=41001,
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
    evaluations["spin_adapted"] = evaluation
    best_eval, best_eval_source = read_artifact(directory / "best_eval.json")
    checked_evaluation(best_eval, registry["data"][dataset])
    if best_eval != evaluation:
        raise RuntimeError("best evaluation differs from the final result")
    smoke_path = output / "preflight" / dataset / "spin_adapted/REAL_DATA_SMOKE.json"
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
    _checked_history(history, payload)
    item = _compact(evaluation, source, seed=41001, state="complete", payload=payload,
                    checkpoint=checkpoint,
                    latency=checked_latency(directory / "latency.json", "spin_adapted", required=True))
    item["training"].update({key: payload[key] for key in
                             ("elapsed_seconds", "peak_training_allocated_bytes", "best_in_last_five_at_cap")})
    item.update(status_source=status_source, best_eval_source=best_eval_source,
                smoke_source=smoke_source, run_info_source=info_source,
                history_source=history_source,
                last_checkpoint={"path": str(directory / "last.pt"), "sha256": file_sha(directory / "last.pt")},
                worker_exit={"path": str(exit_path), "sha256": file_sha(exit_path), "returncode": 0})
    return item


def _comparison(methods):
    if methods["spin_adapted"]["state"] != "complete":
        return {"available": False, "reason": "SPIN is not a verified completed result"}
    full, spin = methods["direct_sync"], methods["spin_adapted"]
    return {
        "available": True, "role": "single_initialization_development_comparison",
        "direct_sync_nmae_relative_reduction_vs_spin":
            (spin["selection_score"] - full["selection_score"]) / spin["selection_score"],
        "direct_sync_nrmse_relative_reduction_vs_spin":
            (spin["mean_nrmse"] - full["mean_nrmse"]) / spin["mean_nrmse"],
        "direct_sync_lower_nmae_conditions": sum(full["metrics"][c]["nmae"] < spin["metrics"][c]["nmae"] for c in CONDITIONS),
        "direct_sync_lower_nrmse_conditions": sum(full["metrics"][c]["nrmse"] < spin["metrics"][c]["nrmse"] for c in CONDITIONS),
        "conditions": {c: {
            "direct_sync_nmae": full["metrics"][c]["nmae"], "spin_nmae": spin["metrics"][c]["nmae"],
            "direct_sync_nrmse": full["metrics"][c]["nrmse"], "spin_nrmse": spin["metrics"][c]["nrmse"],
            "direct_sync_nmae_relative_reduction_vs_spin":
                (spin["metrics"][c]["nmae"] - full["metrics"][c]["nmae"]) / spin["metrics"][c]["nmae"],
        } for c in CONDITIONS},
        "direct_sync_to_spin_parameter_ratio": full["training"]["parameter_count"] / spin["training"]["parameter_count"],
        "direct_sync_to_spin_median_latency_ratio": full["latency"]["median_ms"] / spin["latency"]["median_ms"],
        "spin_nmae_relative_reductions_vs_references": {
            name: (methods[name]["selection_score"] - spin["selection_score"]) / methods[name]["selection_score"]
            for name in ("sync_delta", "ari", "linear")},
    }


def summarize(output):
    output = Path(output).resolve()
    registry, registry_source = read_artifact(output / "registry.json", REGISTRY_SHA)
    if (output / "registry.sha256").read_text().strip() != REGISTRY_SHA:
        raise RuntimeError("registry sidecar digest differs")
    if (registry.get("role") != "development_only" or registry.get("maximum_formal_new_jobs") != 2
            or registry.get("protocol") != "spin-adapted-development-20260914-v1"
            or registry.get("deadline_unix") is not None):
        raise RuntimeError("unexpected SPIN scope or role")
    if registry["full_registry_sha256"] != FULL_SHA or registry["parent_registry_sha256"] != PARENT_SHA:
        raise RuntimeError("frozen parent digest differs")
    full, full_source = read_artifact(Path(registry["full_parent"]) / "registry.json", FULL_SHA)
    original, parent_source = read_artifact(Path(registry["parent"]) / "registry.json", PARENT_SHA)
    ari, ari_source = read_artifact(Path(full["ari_parent"]) / "registry.json", full["ari_registry_sha256"])
    source_verification = {"spin_adapted": _check_sources(registry), "full": _check_sources(full),
                           "original": _check_sources(original), "ari": _check_sources(ari),
                           "upstream_provenance": _upstream_provenance()}
    cpu, cpu_source = read_artifact(registry["cpu_correctness"]["path"], registry["cpu_correctness"]["sha256"])
    required_checks = {"upstream_outputs", "all_parameter_gradients", "checkpoint_chunk_equivalence",
                       "hidden_nan_invariance", "physical_microbatch_equivalence", "four_readout_train_step", "state_reload"}
    if (cpu.get("passed") is not True or not required_checks <= set(cpu["checks"])
            or cpu["official_source_commit"] != "7349ba31da7306e7e96c13668a3f1f0a4df90902"
            or cpu["layer_readouts_compared"] != 4
            or not 0 <= cpu["max_output_abs_diff"] <= 3e-6
            or not 0 <= cpu["max_parameter_gradient_abs_diff"] <= 3e-6):
        raise RuntimeError("CPU official-source equivalence report did not pass")
    for dataset in DATASETS:
        if registry["data"][dataset] != full["data"][dataset]:
            raise RuntimeError("registered SPIN/full data identity differs")
        if (set(registry["initializations"][dataset]) != {"spin_adapted"}
                or set(registry["initializations"][dataset]["spin_adapted"]) != {"41001"}):
            raise RuntimeError("unexpected SPIN variant or initialization")
        for record, base, variant in ((registry, output, "spin_adapted"),
                (full, Path(registry["full_parent"]), "direct_sync")):
            initial = record["initializations"][dataset][variant]["41001"]
            if file_sha(base / initial["file"]) != initial["sha256"]:
                raise RuntimeError("initialization checkpoint digest differs")
        initial = registry["initializations"][dataset]["spin_adapted"]["41001"]
        if initial["parameter_count"] != cpu["parameter_counts"][str(registry["data"][dataset]["flows"])]:
            raise RuntimeError("SPIN parameter count differs from checked architecture")
        if not initial["normalization"].get("fit_only") or initial["normalization"]["scale"] <= 0:
            raise RuntimeError("SPIN normalization is not fit-only and positive")
    summary = {
        "updated": datetime.now(timezone.utc).isoformat(), "state": "partial", "role": "development_only",
        "registry_sha256": REGISTRY_SHA, "registry_source": registry_source,
        "parent_sources": {"full": full_source, "original": parent_source, "ari": ari_source},
        "source_verification": source_verification,
        "report_source": {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)},
        "cpu_correctness_source": cpu_source, "output": str(output), "datasets": {},
        "expected_new_runs": 2, "completed_new_runs": 0, "initialization_seed": 41001,
        "mask_seed": 71001, "scope": registry["scope"], "training": registry["training"],
        "adaptations": registry["adaptations"],
        "metric_definition": "global missing-only NMAE and NRMSE within each condition; equal arithmetic means over the three conditions",
        "limitations": [
            "One initialization on reused development periods, not independent testing or statistical significance.",
            "Official SPIN architecture adapted to T50 OD-flow graphs, existing directed fit neighbors, relative native-slot covariate and common 20% masks; not a reproduction of the paper's original benchmarks.",
            "Official global fit scaling, four-readout supervision, optimizer and 300-epoch schedule retained; epoch selection adapted to common mean-condition missing NMAE.",
            "Internal optimization losses and training caps differ by method; common input permissions and raw-unit scoring do not imply identical training procedures.",
            "Small parameter count does not establish lower training/inference cost; actual measured costs are retained.",
            "Legacy no-mask-digest results require registered shared masks and exact physical-window/truth-denominator pairing, explicitly marked per method.",
            "An unfinished model's intermediate best score is never promoted into a final comparison.",
        ],
    }
    for dataset in DATASETS:
        methods, evaluations = {}, {}
        for method in ("linear", "sync_delta", "ari"):
            if registry["references"][dataset][method] != full["references"][dataset][method]:
                raise RuntimeError("reference changed relative to frozen full comparison")
            methods[method] = _reference(full, dataset, method, evaluations)
        methods["direct_sync"] = _reference_full(registry, full, dataset, evaluations)
        methods["spin_adapted"] = _new_result(output, registry, dataset, evaluations)
        summary["completed_new_runs"] += methods["spin_adapted"]["state"] == "complete"
        for method, item in methods.items():
            if "metrics" in item:
                item["role"] = "development_only"
                item["missing_target_counts"] = {c: item["metrics"][c]["count"] for c in CONDITIONS}
        summary["datasets"][dataset] = {
            "data": registry["data"][dataset], "initialization_seed": 41001,
            "normalization": registry["initializations"][dataset]["spin_adapted"]["41001"]["normalization"],
            "methods": methods, "comparison": _comparison(methods),
            "paired_evaluations_verified": sorted(evaluations),
        }
    summary["state"] = "complete" if summary["completed_new_runs"] == 2 else "partial"
    if summary["state"] == "complete":
        training = [summary["datasets"][d]["methods"]["spin_adapted"]["training"] for d in DATASETS]
        summary["summed_new_training_seconds"] = sum(item["training_seconds"] for item in training)
        summary["summed_new_elapsed_seconds"] = sum(item["elapsed_seconds"] for item in training)
        summary["last_new_finished"] = max(item["finished"] for item in training)
        summary["any_convergence_cap_warning"] = any(item["best_in_last_five_at_cap"] for item in training)
    return summary


def markdown(summary):
    lines = ["# SPIN 专用补全基线：结果与来源核验", "",
        "只新增 Abilene / GEANT 各 seed41001 一次 SPIN 训练；Direct＋Sync、原Sync、ARI和Linear均复用登记结果。",
        "SPIN使用作者官方四层结构与原优化设置，适配到本项目T50、fit相关图、相对时间及统一20%观测任务；不是原论文主表复现。",
        "表格单元为缺失位置全局 NMAE / NRMSE，平均列对三种条件分别等权计算；所有结果仍为development-only。未完成任务不展示中途最佳值。", ""]
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
            lines += [f"Direct＋Sync 相对 SPIN 的平均 NMAE 降低 **{100*comparison['direct_sync_nmae_relative_reduction_vs_spin']:.2f}%**，平均 NRMSE 降低 **{100*comparison['direct_sync_nrmse_relative_reduction_vs_spin']:.2f}%**（负值表示Direct＋Sync更差）。",
                f"Direct＋Sync在 {comparison['direct_sync_lower_nmae_conditions']}/3 条件的NMAE、{comparison['direct_sync_lower_nrmse_conditions']}/3 条件的NRMSE更低；参数量为SPIN的 {comparison['direct_sync_to_spin_parameter_ratio']:.2f} 倍，中位延迟为 {comparison['direct_sync_to_spin_median_latency_ratio']:.2f} 倍。", ""]
            training = data["methods"]["spin_adapted"]["training"]
            lines += [f"SPIN实测训练累计 {training['training_seconds']/3600:.3f} GPU小时，含评估与收尾的任务耗时 {training['elapsed_seconds']/3600:.3f} 小时；训练峰值已分配显存 {training['peak_training_allocated_bytes']/2**30:.2f} GiB。"]
            if training["best_in_last_five_at_cap"]:
                lines += ["**收敛限制：训练达到300轮上限，最佳值仍出现在最后5轮；本结果不能解释为充分收敛后的能力上限。**"]
            lines.append("")
        counts = data["methods"]["linear"]["missing_target_counts"]
        lines += [f"共 {data['data']['dev_window_count']} 个开发窗口；每条件缺失计数 {json.dumps(counts, ensure_ascii=False)}。每个已完成方法都必须通过相同物理窗口和真实分母配对。", ""]
    lines += ["## 解释与核验边界", "",
        "这是同一输入信息权限和评分任务下的模型比较，训练内部损失与原始优化日程并不完全相同。SPIN保留全局fit标准化、四层L1深监督、300轮上限/patience40及官方epoch调度；checkpoint使用共同开发NMAE选型。",
        "两个WAN各只有一次SPIN初始化，开发物理时间段参与选型。因此，无论结果有利或不利，都不声称统计显著、独立泛化结论或全面超越原论文。参数规模与计算成本分别按实际值报告。",
        "本报告只读取JSON及文件散列，不加载模型、不运行推理或训练。完整模型、ARI、Linear和原Sync计时沿用各自完成时的测量。", "",
        f"已核验完成 {summary['completed_new_runs']}/{summary['expected_new_runs']}，状态 `{summary['state']}`。对于已完成SPIN任务，核验成功退出、实际best/last checkpoint文件、初始状态文件、历史最佳轮次、完整epoch调度和原始评分分母；未完成任务保留状态。",
        "注册表、冻结实现、归档SPIN/TSL来源文件、CPU等价检查、实际数据预检、四层输出监督来源及每个结果/计时/检查点的散列保存在comparison.json。旧结果缺少独立mask摘要时另行标记。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--render-json", type=Path)
    args = parser.parse_args()
    if args.render_json:
        summary, _ = read_artifact(args.render_json)
    else:
        summary = summarize(args.output)
    if args.stdout:
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
        return
    destination = args.report or args.output / "report"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write(destination / "comparison.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    atomic_write(destination / "COMPARISON_CN.md", markdown(summary))
    print(json.dumps({"state": summary["state"], "completed_new_runs": summary["completed_new_runs"],
                      "report": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
