"""Stdlib-only, checked development report for the bounded overnight candidates.

Reads existing artifacts only. It never imports a model, loads a Torch checkpoint,
launches training, chooses a new winner, or turns partial history into final scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/llm-candidates-20260914-v1"
DATASETS = ("abilene", "geant")
CONDITIONS = ("uniform", "unequal", "unequal_gap")
VARIANTS = ("routing_sync", "direct_sync", "feature_sync")
LABELS = {"routing_sync": "动态邻流 Sync", "direct_sync": "直接读取＋Sync",
          "feature_sync": "特征映射 Sync", "sync_delta": "原 Sync-Delta",
          "ari": "ARI（统一任务适配）", "linear": "Linear"}
NO_FINAL_STATES = {"failed", "paused", "paused_budget", "interrupted", "incomplete",
                   "smoke_failed", "deadline_reached", "budget_exhausted"}
SUM_FIELDS = ("abs_error", "abs_truth", "sq_error", "sq_truth", "count")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bad_constant(value):
    raise RuntimeError(f"nonfinite JSON constant: {value}")


def read_artifact(path, expected_sha=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise RuntimeError(f"artifact checksum mismatch: {path}")
    return json.loads(raw, parse_constant=_bad_constant), {"path": str(path), "sha256": digest}


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value <= 0)):
        raise RuntimeError(f"invalid {name}: {value!r}")
    return value


def _close(actual, expected, name, *, signed=False):
    if signed:
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise RuntimeError(f"invalid {name}: {actual!r}")
    else:
        _number(actual, name)
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError(f"inconsistent {name}: {actual!r} != {expected!r}")


def checked_evaluation(payload, data=None):
    """Verify role, masks, unique windows, aggregate numerators and denominators."""
    evaluation = payload.get("primary_eval", payload)
    if evaluation.get("role") != "development_only" or evaluation.get("mask_seed") != 71001:
        raise RuntimeError("unexpected evaluation role or mask seed")
    if evaluation.get("intervention") is not None:
        raise RuntimeError("intervention evaluation cannot be a primary result")
    if set(evaluation["metrics"]) != set(CONDITIONS):
        raise RuntimeError("unexpected evaluation conditions")
    rows = evaluation["rows"]
    if not rows:
        raise RuntimeError("evaluation has no physical windows")
    seen, per_condition = set(), {c: [] for c in CONDITIONS}
    for row in rows:
        key = (row["condition"], row["window_start"], row["cohort"])
        if key in seen or row["condition"] not in per_condition:
            raise RuntimeError("duplicate or unknown physical evaluation window")
        seen.add(key)
        if not isinstance(row["window_start"], int) or isinstance(row["window_start"], bool):
            raise RuntimeError("invalid physical window start")
        if row["cohort"] not in ("source_dev", "tune"):
            raise RuntimeError("unexpected development cohort")
        for name in SUM_FIELDS:
            _number(row[name], f"row {key}/{name}")
        if not isinstance(row["count"], int) or row["count"] <= 0:
            raise RuntimeError("invalid missing-target count")
        if data is not None and row["count"] != 40 * data["flows"]:
            raise RuntimeError("row does not have the registered T=50, 80% missing count")
        per_condition[row["condition"]].append(row)
    windows = None
    for condition in CONDITIONS:
        selected = per_condition[condition]
        physical = {(r["window_start"], r["cohort"]) for r in selected}
        if windows is not None and physical != windows:
            raise RuntimeError("conditions do not score the same physical windows")
        windows = physical
        if data is not None and len(selected) != data["dev_window_count"]:
            raise RuntimeError("development window count differs from registry")
        metric = evaluation["metrics"][condition]
        totals = {name: sum(r[name] for r in selected) for name in SUM_FIELDS}
        for name in SUM_FIELDS:
            _close(metric[name], totals[name], f"{condition}/{name}")
        _number(totals["abs_truth"], "absolute truth denominator", positive=True)
        _number(totals["sq_truth"], "squared truth denominator", positive=True)
        _close(metric["nmae"], totals["abs_error"] / totals["abs_truth"], f"{condition}/NMAE")
        _close(metric["nrmse"], math.sqrt(totals["sq_error"] / totals["sq_truth"]), f"{condition}/NRMSE")
    _close(evaluation["selection_score"], sum(evaluation["metrics"][c]["nmae"] for c in CONDITIONS) / 3,
           "equal-condition mean NMAE")
    mask_sha = evaluation.get("mask_array_sha256")
    if mask_sha is not None:
        if set(mask_sha) != set(CONDITIONS):
            raise RuntimeError("unexpected mask digest conditions")
        if data is not None and mask_sha != data["mask_array_sha256"]:
            raise RuntimeError("evaluation mask digest differs from registry")
    return evaluation


def assert_paired(left, right):
    def denominator_map(evaluation):
        return {(r["condition"], r["window_start"], r["cohort"]):
                (r["abs_truth"], r["sq_truth"], r["count"]) for r in evaluation["rows"]}
    if denominator_map(left) != denominator_map(right):
        raise RuntimeError("unpaired physical windows, masks, or truth denominators")
    a, b = left.get("mask_array_sha256"), right.get("mask_array_sha256")
    if a is not None and b is not None and a != b:
        raise RuntimeError("paired evaluations have different mask digests")


def checked_latency(path, method, *, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise RuntimeError(f"completed result has no latency artifact: {path}")
        return None
    payload, source = read_artifact(path)
    latency = payload.get(method, payload)
    _number(latency.get("median_ms"), "median latency", positive=True)
    if latency.get("batch_windows") != 1:
        raise RuntimeError("latency is not for one complete window")
    if "mask_seed" in latency and latency["mask_seed"] != 71001:
        raise RuntimeError("latency mask seed differs")
    if "condition" in latency and latency["condition"] != "unequal_gap":
        raise RuntimeError("latency missingness condition differs")
    return {"median_ms": latency["median_ms"], "details": latency, "source": source}


def _checkpoint(path, payload):
    checkpoint = Path(path).parent / "best.pt"
    expected = payload.get("best_checkpoint_sha256")
    if not expected or file_sha(checkpoint) != expected:
        raise RuntimeError(f"completed best checkpoint mismatch: {checkpoint}")
    return {"path": str(checkpoint.resolve()), "sha256": expected}


def _result_identity(payload, *, dataset, variant, seed, registry_sha, initial_sha=None):
    expected = {"dataset": dataset, "variant": variant, "seed": seed,
                "registry_sha256": registry_sha, "role": "development_only"}
    if initial_sha is not None:
        expected["initial_state_sha256"] = initial_sha
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"completed result identity mismatch for {dataset}/{variant}/seed{seed}")
    if not isinstance(payload.get("finished"), str) or not payload["finished"]:
        raise RuntimeError("result does not establish completed training")
    epochs, best = payload.get("epochs_completed"), payload.get("best_epoch")
    if not isinstance(epochs, int) or not isinstance(best, int) or not 1 <= best <= epochs:
        raise RuntimeError("invalid completed/best epoch")
    _number(payload.get("parameter_count"), "parameter count", positive=True)


def _compact(evaluation, source, *, seed, state, payload=None, checkpoint=None, latency=None):
    item = {"state": state, "seed": seed, "selection_score": evaluation["selection_score"],
            "mean_nrmse": sum(evaluation["metrics"][c]["nrmse"] for c in CONDITIONS) / 3,
            "metrics": {c: {name: evaluation["metrics"][c][name] for name in
                        ("nmae", "nrmse", *SUM_FIELDS)} for c in CONDITIONS},
            "source": source, "checkpoint": checkpoint, "latency": latency,
            "mask_verification": ("evaluation_digest_matches_registry_and_exact_paired_denominators"
                if evaluation.get("mask_array_sha256") is not None else
                "registered_shared_masks_and_exact_paired_denominators; result_has_no_standalone_mask_digest")}
    item["training"] = ({key: payload[key] for key in (
        "parameter_count", "trainable_parameter_count", "epochs_completed", "best_epoch",
        "training_seconds", "optimizer_updates", "training_cap", "stopped_early", "finished") if key in payload}
        if payload is not None else {"parameter_count": 0, "epochs_completed": None, "best_epoch": None})
    return item


def _reference(registry, dataset, method, evaluations):
    ref = registry["references"][dataset][method]
    payload, source = read_artifact(ref["path"], ref["sha256"])
    evaluation = checked_evaluation(payload, registry["data"][dataset])
    if evaluation["selection_score"] != ref["selection_score"]:
        raise RuntimeError("reference score differs from registry")
    if method == "linear":
        if payload.get("method") != "linear":
            raise RuntimeError("incorrect Linear reference method")
        # Linear has no model initialization seed; 71001 is its observation mask seed.
        checkpoint, seed = None, None
    else:
        _result_identity(payload, dataset=dataset, variant=("ari_upstream_task_adapted" if method == "ari" else method),
                         seed=41001, registry_sha=registry["ari_registry_sha256" if method == "ari" else "parent_registry_sha256"])
        checkpoint, seed = _checkpoint(ref["path"], payload), 41001
    for existing in evaluations.values():
        assert_paired(evaluation, existing)
    evaluations[method] = evaluation
    latency_path = (Path(registry["references"][dataset]["ari"]["path"]).parent / "latency.json"
                    if method == "linear" else Path(ref["path"]).parent / "latency.json")
    latency = checked_latency(latency_path, method)
    return _compact(evaluation, source, seed=seed, state="reused_complete",
                    payload=None if method == "linear" else payload, checkpoint=checkpoint, latency=latency)


def _new_job(output, registry, digest, dataset, variant, seed, paired_evaluation, queue_jobs=None):
    directory = output / dataset / variant / f"seed{seed}"
    result_path, status_path = directory / "result.json", directory / "status.json"
    recorded, status_source = read_artifact(status_path) if status_path.exists() else ({"state": "not_started"}, None)
    identity = {"registry_sha256": digest, "dataset": dataset, "variant": variant, "seed": seed}
    if any(key in recorded and recorded[key] != value for key, value in identity.items()):
        raise RuntimeError(f"status identity differs: {status_path}")
    item = {"seed": seed, "state": recorded.get("state", "unknown"), "status_source": status_source,
            "status": {key: recorded[key] for key in ("state", "updated", "epochs_completed", "epoch",
                        "best_epoch", "optimizer_updates", "error", "reason") if key in recorded}}
    queued = (queue_jobs or {}).get(f"{dataset}/{variant}/seed{seed}")
    if queued is not None:
        item["scheduler_state"] = queued.get("state", "unknown")
        if status_source is None or queued.get("state") in NO_FINAL_STATES:
            item["state"] = queued.get("state", "unknown")
            item["status"].update({key: queued[key] for key in ("state", "reason", "error", "finished") if key in queued})
    if not result_path.exists() or item["state"] in NO_FINAL_STATES:
        if result_path.exists():
            item["unscored_result_source"] = {"path": str(result_path), "sha256": file_sha(result_path)}
        if item["state"] in ("complete", "training_complete"):
            item["state"] = "incomplete_artifacts"
        return item
    payload, source = read_artifact(result_path)
    if not payload.get("finished"):
        item.update(state="incomplete_artifacts", unscored_result_source=source)
        return item
    initial = registry["initializations"][dataset][variant][str(seed)]
    _result_identity(payload, dataset=dataset, variant=variant, seed=seed,
                     registry_sha=digest, initial_sha=initial["state_sha256"])
    if payload["parameter_count"] != initial["parameter_count"]:
        raise RuntimeError("completed parameter count differs from initialization registry")
    checkpoint = _checkpoint(result_path, payload)
    if not (directory / "latency.json").exists():
        item.update(state="incomplete_artifacts", unscored_result_source=source,
                    reason="completed result missing latency", checkpoint=checkpoint)
        return item
    evaluation = checked_evaluation(payload, registry["data"][dataset])
    if evaluation.get("mask_array_sha256") != registry["data"][dataset]["mask_array_sha256"]:
        raise RuntimeError("new result must carry the registered mask-array digests")
    assert_paired(evaluation, paired_evaluation)
    compact = _compact(evaluation, source, seed=seed, state="complete", payload=payload,
                       checkpoint=checkpoint, latency=checked_latency(directory / "latency.json", variant, required=True))
    compact["status_source"] = status_source
    return compact


def _decision(output, digest, datasets):
    path = output / "decision.json"
    if not path.exists():
        return None
    record, source = read_artifact(path)
    if (record.get("registry_sha256") != digest or record.get("role") != "development_only"
            or record.get("winner") not in (*VARIANTS, None)):
        raise RuntimeError("decision identity or winner differs")
    represented = set()
    for candidate in record.get("candidates", []):
        variant = candidate["variant"]
        if variant not in VARIANTS or variant in represented:
            raise RuntimeError("unknown or duplicated decision candidate")
        represented.add(variant)
        for dataset in DATASETS:
            item = datasets[dataset]["methods"][variant]
            evidence = candidate["results"][dataset]
            if item["state"] != "complete" or evidence["result_sha256"] != item["source"]["sha256"]:
                raise RuntimeError("decision uses an incomplete or changed result")
            if evidence["latency_sha256"] != item["latency"]["source"]["sha256"]:
                raise RuntimeError("decision latency artifact changed")
            _close(evidence["score"], item["selection_score"], "decision candidate score")
            baseline = datasets[dataset]["methods"]["sync_delta"]
            gain = (baseline["selection_score"] - item["selection_score"]) / baseline["selection_score"]
            _close(candidate["relative_gains"][dataset], gain, "decision relative gain", signed=True)
    if record.get("winner") is not None and record["winner"] not in represented:
        raise RuntimeError("winner has no verified completed candidate evidence")
    for dataset, baseline in record.get("baselines", {}).items():
        item = datasets[dataset]["methods"]["sync_delta"]
        if baseline["result_sha256"] != item["source"]["sha256"]:
            raise RuntimeError("decision baseline result changed")
        if item["latency"] is None or baseline["latency_sha256"] != item["latency"]["source"]["sha256"]:
            raise RuntimeError("decision baseline latency changed")
    return {"record": record, "source": source, "winner": record.get("winner"),
            "note": "scheduler development selection; not an independent significance or acceptance claim"}


def _routing_context(output, digest, dataset):
    path = output / "preflight" / dataset / "routing_sync" / "REAL_DATA_SMOKE.json"
    if not path.exists():
        return {"available": False, "reason": "no_real_data_smoke_artifact"}
    record, source = read_artifact(path)
    if record.get("registry_sha256") != digest or record.get("dataset") != dataset or record.get("variant") != "routing_sync":
        raise RuntimeError("routing smoke identity differs")
    if record.get("role") != "real_data_runner_smoke_not_formal_result":
        raise RuntimeError("routing context is not a runner smoke artifact")
    probe = record.get("routing_context_probe")
    if probe is not None and probe.get("role") != "fit_only_diagnostic":
        raise RuntimeError("routing context probe must be a fit-only diagnostic")
    fields = {key: value for key, value in record.items()
              if key.startswith(("routing_", "selected_valid_", "nonstatic_valid_", "context_"))}
    return {"available": bool(fields) and record.get("passed") is True, "source": source,
            "fields": fields if record.get("passed") is True else {},
            "scope": "single fit-window/limited-target smoke diagnostic, not population context averages or formal development scores",
            "reason": None if fields and record.get("passed") is True else "smoke_has_no_verified_context_counts"}


def _table(methods, ordered):
    lines = ["| 方法 | 状态 | 平均 NMAE / NRMSE | Uniform | Unequal | Unequal-Gap | 中位延迟 ms | 参数量 | 训练 / 最佳 epoch |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ordered:
        item = methods[method]
        if "selection_score" not in item:
            lines.append(f'| {LABELS[method]} | {item["state"]} | — | — | — | — | — | — | — |')
            continue
        cells = [f'{item["metrics"][c]["nmae"]:.6f} / {item["metrics"][c]["nrmse"]:.6f}' for c in CONDITIONS]
        latency = "未记录" if item["latency"] is None else f'{item["latency"]["median_ms"]:.3f}'
        training = item["training"]
        epoch_text = "—" if training.get("epochs_completed") is None else f'{training["epochs_completed"]} / {training["best_epoch"]}'
        lines.append(f'| {LABELS[method]} | {item["state"]} | {item["selection_score"]:.6f} / {item["mean_nrmse"]:.6f} | '
                     + " | ".join(cells) + f' | {latency} | {training.get("parameter_count", "未记录")} | {epoch_text} |')
    return lines


def summarize(output, report=None):
    output = Path(output).resolve()
    report = output / "report" if report is None else Path(report).resolve()
    registry, registry_source = read_artifact(output / "registry.json")
    digest = registry_source["sha256"]
    if digest != (output / "registry.sha256").read_text().strip().split()[0]:
        raise RuntimeError("overnight registry hash mismatch")
    if registry.get("role") != "development_only" or tuple(registry["screen_variants"]) != VARIANTS:
        raise RuntimeError("unexpected overnight scope or role")
    parents = {}
    for prefix in ("parent", "ari_parent"):
        expected = registry["parent_registry_sha256" if prefix == "parent" else "ari_registry_sha256"]
        _, parents[prefix] = read_artifact(Path(registry[prefix]) / "registry.json", expected)
    summary = {"updated": datetime.now(timezone.utc).isoformat(), "role": "development_only",
               "registry_sha256": digest, "registry_source": registry_source, "parent_sources": parents,
               "output": str(output), "datasets": {}, "completed_screen_runs": 0, "completed_repeat_runs": 0,
               "scope": "three candidates x two WANs seed41001; only selected candidate and same-seed Sync compared at seed41002",
               "mask_evidence_note": "ARI carries mask-array digests checked against registry; all results require exact paired physical windows and truth denominators; legacy/no-digest results are explicitly marked",
               "routing_context_note": "currently observed full-pool routing may write more real events than fixed self+8; not equal-context or equal-information-volume"}
    queue_jobs = {}
    queue_path = output / "scheduler" / "queue.json"
    if queue_path.exists():
        queue, summary["scheduler_source"] = read_artifact(queue_path)
        if queue.get("registry_sha256") != digest:
            raise RuntimeError("scheduler queue registry mismatch")
        for job in queue["jobs"]:
            key = f'{job["dataset"]}/{job["variant"]}/seed{job["seed"]}'
            if job.get("id") != key or key in queue_jobs:
                raise RuntimeError("duplicate or inconsistent scheduler job identity")
            queue_jobs[key] = job
    evaluations = {}
    for dataset in DATASETS:
        evaluations[dataset], methods = {}, {}
        for method in ("linear", "sync_delta", "ari"):
            methods[method] = _reference(registry, dataset, method, evaluations[dataset])
        for variant in VARIANTS:
            methods[variant] = _new_job(output, registry, digest, dataset, variant, 41001, evaluations[dataset]["linear"], queue_jobs)
            summary["completed_screen_runs"] += methods[variant]["state"] == "complete"
        summary["datasets"][dataset] = {"seed": 41001, "methods": methods,
                                       "routing_context": _routing_context(output, digest, dataset)}
    decision = _decision(output, digest, summary["datasets"])
    summary["decision"] = decision
    winner = None if decision is None else decision["winner"]
    repeats = {}
    if winner is not None:
        for dataset in DATASETS:
            methods = {winner: _new_job(output, registry, digest, dataset, winner, 41002, evaluations[dataset]["linear"], queue_jobs)}
            if dataset == "abilene":
                methods["sync_delta"] = _new_job(output, registry, digest, dataset, "sync_delta", 41002, evaluations[dataset]["linear"], queue_jobs)
                summary["completed_repeat_runs"] += methods["sync_delta"]["state"] == "complete"
            else:
                ref = registry["existing_geant_sync_repeat"]
                payload, source = read_artifact(ref["path"], ref["sha256"])
                _result_identity(payload, dataset="geant", variant="sync_delta", seed=41002,
                                 registry_sha=registry["parent_registry_sha256"])
                evaluation = checked_evaluation(payload, registry["data"][dataset])
                if evaluation["selection_score"] != ref["selection_score"] or ref["seed"] != 41002:
                    raise RuntimeError("existing GEANT repeat registry differs")
                assert_paired(evaluation, evaluations[dataset]["linear"])
                methods["sync_delta"] = _compact(evaluation, source, seed=41002, state="reused_complete", payload=payload,
                    checkpoint=_checkpoint(ref["path"], payload), latency=checked_latency(Path(ref["path"]).parent / "latency.json", "sync_delta"))
            summary["completed_repeat_runs"] += methods[winner]["state"] == "complete"
            repeats[dataset] = {"seed": 41002, "methods": methods, "paired_comparison_available":
                all("selection_score" in item for item in methods.values())}
            if repeats[dataset]["paired_comparison_available"]:
                base_score = methods["sync_delta"]["selection_score"]
                repeats[dataset]["relative_gain_vs_same_seed_sync"] = (base_score - methods[winner]["selection_score"]) / base_score
    summary["repeats"] = repeats
    summary["completed_runs"] = summary["completed_screen_runs"] + summary["completed_repeat_runs"]
    summary["expected_new_runs"] = 6 + (3 if winner is not None else 0)
    summary["state"] = ("complete" if summary["completed_runs"] == summary["expected_new_runs"] and decision is not None else "partial")
    lines = ["# LLM 借鉴候选：今晚开发实验汇总", "",
             "仅比较三个独立候选、Abilene 和 GEANT；旧 Sync、ARI 和 Linear 复用已有结果。",
             "每格精度为缺失位置全局 NMAE / NRMSE，越低越好；均值分别对三种条件等权计算，选型只使用平均 NMAE。",
             "所有结果均为 development-only。训练中、失败、暂停或缺少完整产物的任务仅列状态，不展示中途最佳分数。",
             "Linear 没有初始化 seed；此处固定使用 mask seed 71001。", ""]
    for dataset in DATASETS:
        lines += [f"## {dataset.title()} · 初始化 41001", ""]
        lines += _table(summary["datasets"][dataset]["methods"], (*VARIANTS, "sync_delta", "ari", "linear")) + [""]
        context = summary["datasets"][dataset]["routing_context"]
        lines += ["动态路由从当前桶已观测的全流池取邻流，实际写入事件数可能增加；不能表述为与固定邻域等上下文成本。",
                  ("已有单个 fit 窗口、有限目标的 smoke 上下文诊断，具体字段及来源见 comparison.json；不能当作总体上下文均值或正式性能结果。"
                   if context["available"] else "当前没有可核验的真实数据上下文计数，暂不报告实际写入数或非静态邻流比例。"), ""]
    if decision is None:
        lines += ["尚无调度器的正式候选选择记录；本汇总不代替调度器选择胜者。", ""]
    else:
        selected = "无候选进入第二初始化" if winner is None else LABELS[winner]
        lines += [f'已记录开发期选择：{selected}；原因：{decision["record"].get("reason", "未记录")}。', "",
                  "该选择来自初始化 41001 的开发结果，不构成泛化优势或统计显著性结论。", ""]
    if repeats:
        lines += ["## 初始化 41002 的配对复核", "",
                  "只比较已选择候选与同 seed 的 Sync；GEANT Sync 复用已完成的 41002，Abilene 使用本轮新增 41002。不同 seed 不混合计算胜负。", ""]
        for dataset in DATASETS:
            lines += [f"### {dataset.title()}", ""] + _table(repeats[dataset]["methods"], (winner, "sync_delta")) + [""]
            if repeats[dataset]["paired_comparison_available"]:
                lines += [f'相对同 seed Sync 的 NMAE 降幅：{100 * repeats[dataset]["relative_gain_vs_same_seed_sync"]:.2f}%。', ""]
    lines += [f'已核验本轮完整新任务：{summary["completed_runs"]}/{summary["expected_new_runs"]}；汇总状态：{summary["state"]}。', "",
              "状态来自落盘记录，不表示此刻进程仍存活。延迟复用各自端到端计时，完整计时范围保存在 JSON。",
              "所有正式结果的 best.pt 均按结果文件中的 SHA256 实际核验。部分旧结果未单独保存掩码摘要，其证据边界在 JSON 中明确记录。",
              "完整产物、检查点、延迟、注册表及选择记录的来源路径与 SHA256 均保存在 comparison.json；不输出独立测试或论文录用结论。", ""]
    atomic_write(report / "comparison.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write(report / "COMPARISON_CN.md", "\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    summary = summarize(args.output, args.report)
    print(json.dumps({key: summary[key] for key in ("state", "completed_runs", "expected_new_runs", "updated")}))


if __name__ == "__main__":
    main()
