#!/usr/bin/env python3
"""One-shot, stdlib-only summary of existing unified-model output records.

python3 summarize_results.py --outputs /path/to/spin-sync-unified-20260916-v1 \
  --references /path/to/research_7h_20260916/references --report-dir /path/to/summary

For fixed120 continuation, optionally pass --parent-outputs /local/raw_outputs.
Parent records are verified only; they are never added as extra seed results.

Inputs are read-only. No imports from the model/runtime, raw data, process probes,
monitoring loop, or checkpoint deserialization. Completed result.json is the
runner's completion marker; absent ancillary copies are disclosed as warnings.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

DATASETS = ("abilene", "geant")
FULL, REDUCED = "spin_sync_direct", "spin_direct_no_memory"
VARIANTS = (FULL, REDUCED)
CONDITIONS = ("uniform", "unequal", "unequal_gap")
SUMS = ("count", "abs_error", "abs_truth", "sq_error", "sq_truth")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def row_identity(evaluation):
    rows = evaluation["rows"]
    keys = lambda r: tuple(r[k] for k in ("condition", "cohort", "window_start"))
    require(len({keys(r) for r in rows}) == len(rows), "duplicate condition/cohort/window rows")
    return [(keys(r), tuple(r[k] for k in ("count", "abs_truth", "sq_truth")),
             {g: tuple(v[k] for k in ("count", "abs_truth", "sq_truth")) for g, v in r.get("groups", {}).items()})
            for r in sorted(rows, key=keys)]


def check_eval(evaluation, reference=None):
    require(evaluation["role"] == "development_only" and evaluation["mask_seed"] == 71001, "wrong evaluation role/mask seed")
    require(set(evaluation["metrics"]) == set(CONDITIONS), "condition set differs")
    for condition in CONDITIONS:
        rows = [r for r in evaluation["rows"] if r["condition"] == condition]
        metric = evaluation["metrics"][condition]
        require(rows and metric["abs_truth"] > 0 and metric["sq_truth"] > 0, "missing rows/denominators")
        for name in SUMS:
            require(close(sum(r[name] for r in rows), metric[name]), f"{condition}: {name} aggregate differs")
        require(close(metric["nmae"], metric["abs_error"] / metric["abs_truth"]), "NMAE formula differs")
        require(close(metric["nrmse"], math.sqrt(metric["sq_error"] / metric["sq_truth"])), "NRMSE formula differs")
    require(close(evaluation["selection_score"], statistics.mean(evaluation["metrics"][c]["nmae"] for c in CONDITIONS)), "selection score differs")
    identity = row_identity(evaluation)
    if reference:
        require(identity == row_identity(reference), "paired row identities/denominators differ")
        if "mask_array_sha256" in evaluation and "mask_array_sha256" in reference:
            require(evaluation["mask_array_sha256"] == reference["mask_array_sha256"], "mask hashes differ")


def metrics(evaluation):
    return {"mean_nmae": evaluation["selection_score"],
            "mean_nrmse": statistics.mean(evaluation["metrics"][c]["nrmse"] for c in CONDITIONS),
            "conditions": {c: {m: evaluation["metrics"][c][m] for m in ("nmae", "nrmse")} for c in CONDITIONS}}


def comparison(actual, reference):
    def pair(a, b):
        return {"delta": a - b, "relative_improvement_percent": 100 * (b - a) / b if b else None}
    return {**{m: pair(actual[m], reference[m]) for m in ("mean_nmae", "mean_nrmse")},
            "conditions": {c: {m: pair(actual["conditions"][c][m], reference["conditions"][c][m])
                               for m in ("nmae", "nrmse")} for c in CONDITIONS}}


def seed_stats(items):
    def summary(operation):
        return {**{m: operation(x[m] for x in items) for m in ("mean_nmae", "mean_nrmse")},
                "conditions": {c: {m: operation(x["conditions"][c][m] for x in items)
                                   for m in ("nmae", "nrmse")} for c in CONDITIONS}}
    return {"n": len(items), "mean": summary(statistics.mean),
            "sample_sd": summary(statistics.stdev) if len(items) >= 2 else None}


def check_continuation(config, result, history, status, parent_outputs):
    """Use the runner's existing continuation ledger, without loading tensors."""
    record = result if result and "continuation_parent" in result else config
    ledger, inherited = record["continuation_parent"], record["inherited_smoke"]
    require(record["posthoc_training_duration_diagnostic"] is True and record["independent_training_repeat"] is False
            and record["continuation_checkpoint_loaded"] is True, "continuation interpretation differs")
    if config and result:
        require(all(config[k] == result[k] for k in ("continuation_parent", "inherited_smoke")), "continuation config/result ledgers differ")
    identity, hashes = ledger["identity"], ledger["files_sha256"]
    require(identity["protocol"] == "spin-sync-unified-development-20260916-v1"
            and all(identity[k] == record[k] for k in ("dataset", "variant", "seed")), "continuation parent identity differs")
    require(identity["config_sha256"] == hashes["config.json"] and ledger["imported_checkpoint"] == "last.pt", "continuation must import original last checkpoint")
    require(inherited["new_smoke_performed"] is False and inherited["sha256"] == hashes["smoke/SMOKE.json"], "inherited smoke ledger differs")
    parent_epochs = ledger["epochs_completed"]
    require(1 <= parent_epochs <= 120, "invalid parent epoch count")
    parent_path = (parent_outputs / record["dataset"] / record["variant"] / f"seed{record['seed']}"
                   if parent_outputs else Path(ledger["path"]))
    checked, unavailable = [], []
    for name, expected in hashes.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe parent artifact path")
        artifact = parent_path / name
        if artifact.exists():
            require(sha(artifact) == expected, "continuation parent artifact hash differs: " + name)
            checked.append(name)
        else:
            unavailable.append(name)
    parent_config, parent_result = read(parent_path / "config.json"), read(parent_path / "result.json")
    parent_history, smoke = read(parent_path / "history.json"), read(parent_path / "smoke/SMOKE.json")
    for name, value in (("config", parent_config), ("result", parent_result), ("smoke", smoke)):
        if value is not None:
            fields = [k for k in identity if k != "config_sha256" or name != "config"]
            require(all(value[k] == identity[k] for k in fields), "continuation parent " + name + " identity differs")
    if parent_result:
        require(parent_result["state"] == "complete" and parent_result["epochs_completed"] == parent_epochs
                and parent_result["best_checkpoint_sha256"] == hashes["best.pt"], "continuation parent result differs")
    if parent_config:
        require(parent_config["initial_file_sha256"] == hashes["initial.pt"], "parent initial file hash differs")
        if config:
            require(all(config[k] == parent_config[k] for k in ("data", "model_config", "initial_state_sha256", "parameter_count")), "continuation model/data/initialization differs")
    if smoke:
        require(all(smoke[k] is True for k in ("passed", "exact_checkpoint_continuation", "fit_evaluation_path_checked")), "inherited parent smoke did not pass")
    if history is not None:
        require([h["epoch"] for h in history] == list(range(1, len(history) + 1)), "continuation history duplicates/skips epochs")
        require(all(h["updates"] == 64 * h["epoch"] for h in history), "continuation history update counts differ")
        if parent_history is not None:
            require(len(parent_history) == parent_epochs and history[:parent_epochs] == parent_history, "continuation history parent prefix differs")
    total = result["epochs_completed"] if result else (len(history) if history is not None else status.get("epochs_committed"))
    require(total is None or parent_epochs <= total <= 120, "continuation cumulative epochs differ")
    if result and result.get("state") == "complete":
        require(total == 120 and result["parent_epochs_completed"] == parent_epochs
                and result["additional_epochs_completed"] == total - parent_epochs
                and result["stop_reason"] == "fixed_epoch_cap" and result["stopped_early"] is False, "fixed120 completion metadata differs")
    return {"posthoc_training_duration_diagnostic": True, "independent_training_repeat": False,
            "same_training_trajectory": True, "parent_path": str(parent_path), "parent_epochs_completed": parent_epochs,
            "total_epochs_saved": total, "additional_epochs_saved": None if total is None else total - parent_epochs,
            "history_counting": "one cumulative history; original epochs are not appended or counted twice",
            "parent_artifacts_hash_checked": checked, "parent_artifacts_unavailable": unavailable,
            "parent_history_prefix_checked": history is not None and parent_history is not None,
            "checkpoint_tensors_deserialized": False, "inherited_smoke_verified_locally": smoke is not None}, smoke


def job_summary(path, dataset, variant, seed, reference, parent_outputs=None):
    out = {"path": str(path), "dataset": dataset, "variant": variant, "seed": seed,
           "state": "pending", "eligible_for_ranking": False, "warnings": [], "errors": []}
    try:
        status = read(path / "status.json") or {}
        history = read(path / "history.json")
        smoke = read(path / "smoke/SMOKE.json")
        config = read(path / "config.json")
        result = read(path / "result.json")
        continuation = any(value and "continuation_parent" in value for value in (config, result))
        if continuation:
            out["continuation"], smoke = check_continuation(config, result, history, status, parent_outputs)
            out["smoke_origin"] = "inherited_parent"
            missing = out["continuation"]["parent_artifacts_unavailable"]
            if missing:
                out["warnings"].append("Parent artifacts not copied (no new smoke is expected): " + ", ".join(missing))
        state = status.get("state")
        out.update(recorded_state=state, status=status, history=history, smoke=smoke,
                   smoke_status=read(path / "smoke/status.json"))
        if state in ("training", "evaluating", "epoch_complete", "starting", "running"):
            out["state"] = "running"
        elif state in ("stopped", "interrupted"):
            out["state"] = "stopped"
        elif state == "failed":
            out["state"] = "failed"
        elif state == "complete" or history:
            out["state"] = "incomplete"
        if history:
            out["progress"] = {"saved_epochs": len(history), "last_epoch": history[-1]["epoch"],
                "last_score": history[-1]["selection_score"], "best_epoch": history[-1]["best_epoch"],
                "best_score": history[-1]["best_score"], "not_a_completed_result": True}
        if not result or result.get("state") != "complete":
            if result:
                out["warnings"].append("result.json is not marked complete; excluded")
            return out
        out["state"] = "complete"
        out["result_sha256"] = sha(path / "result.json")
        for field, expected in {"dataset": dataset, "variant": variant, "seed": seed,
                                "role": "development_only", "independent_test": False}.items():
            require(result[field] == expected, "result identity differs: " + field)
        require(result["all_parameters_jointly_trained_from_scratch"] is True, "unexpected training interpretation")
        require(result["epochs_completed"] >= 1 and 1 <= result["best_epoch"] <= result["epochs_completed"], "invalid completed/best epoch")
        require(result["optimizer_updates"] == 64 * result["epochs_completed"], "update count differs")
        check_eval(result["primary_eval"], reference)
        artifacts = [("history.json", history), ("config.json", config)]
        if not continuation:
            artifacts.append(("smoke/SMOKE.json", smoke))
        for artifact, value in artifacts:
            if value is None:
                out["warnings"].append(f"{artifact} not copied; corresponding check unavailable")
        if history is not None:
            require(len(history) == result["epochs_completed"], "history length differs")
            require([h["epoch"] for h in history] == list(range(1, len(history) + 1)), "history epochs differ")
            chosen = min(history, key=lambda h: h["selection_score"])
            require(chosen["epoch"] == result["best_epoch"] and close(chosen["selection_score"], result["primary_eval"]["selection_score"]), "history checkpoint selection differs")
        if config is not None:
            require(sha(path / "config.json") == result["config_sha256"], "config hash differs")
            require(all(config[k] == result[k] for k in ("protocol", "dataset", "variant", "seed")), "config identity differs")
            require(config["parent_checkpoint_weights_loaded"] is False, "unexpected parent weight loading")
            out["config_pairing"] = {k: config[k] for k in ("data", "training", "source_files", "shared_named_parameters_exact")}
        if smoke is not None and not continuation:
            require(all(smoke[k] == result[k] for k in ("protocol", "dataset", "variant", "seed", "config_sha256")), "smoke identity differs")
            require(smoke["passed"] and smoke["exact_checkpoint_continuation"] and smoke["fit_evaluation_path_checked"], "smoke did not pass")
        checkpoint = path / "best.pt"
        out["checkpoint_file_hash_checked"] = checkpoint.exists()
        if checkpoint.exists():
            require(sha(checkpoint) == result["best_checkpoint_sha256"], "best checkpoint file hash differs")
        else:
            out["warnings"].append("best.pt not copied; completed result hash metadata retained, no tensor/file verification")
        out.update(eligible_for_ranking=True, metrics=metrics(result["primary_eval"]),
                   result_metadata={k: v for k, v in result.items() if k != "primary_eval"})
    except (ValueError, KeyError, TypeError, OSError, ZeroDivisionError) as error:
        out.update(state="invalid", eligible_for_ranking=False)
        out["errors"].append(f"{type(error).__name__}: {error}")
    return out


def collect(args):
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "datasets": {},
        "state_interpretation": "saved status only; no process liveness check; missing artifacts are not proof of a running job",
        "ranking_rule": "completed result records only; aggregate variant ranked only when all requested/discovered seeds for that WAN are complete",
        "seed_interpretation": "one variant/seed trajectory per outputs tree; from-scratch original models, fixed120 continuation is not an independent repeat; parent outputs are for verification only",
        "uncertainty": "all completed seeds retained; sample SD for n>=2, no significance or confidence interval",
        "role": "development_only", "independent_test": False}
    for dataset in DATASETS:
        references, evals = {}, {}
        for name in ("spin", "direct", "linear"):
            path = args.references / dataset / f"{name}.json"
            saved = read(path)
            require(saved is not None, "missing baseline: " + str(path))
            evaluation = saved if name == "linear" else saved["primary_eval"]
            check_eval(evaluation, evals.get("spin"))
            references[name] = {"source": str(path), "sha256": sha(path), "metrics": metrics(evaluation)}
            evals[name] = evaluation
        seeds = set(args.seeds or [])
        for variant in VARIANTS:
            seeds.update(int(p.name[4:]) for p in (args.outputs / dataset / variant).glob("seed[0-9]*") if p.is_dir() and p.name[4:].isdigit())
        seeds = sorted(seeds or {41001})
        block = {"requested_or_discovered_seeds": seeds, "references": references, "jobs": {},
                 "completed_seed_summary": {}, "full_vs_no_memory_matched_seeds": {}, "rankings_completed_only": {}}
        report["datasets"][dataset] = block
        for variant in VARIANTS:
            jobs = {str(seed): job_summary(args.outputs / dataset / variant / f"seed{seed}", dataset, variant, seed, evals["spin"], getattr(args, "parent_outputs", None)) for seed in seeds}
            block["jobs"][variant] = jobs
            completed = [j for j in jobs.values() if j["eligible_for_ranking"]]
            for job in completed:
                job["versus"] = {name: comparison(job["metrics"], ref["metrics"]) for name, ref in references.items()}
            if completed:
                summary = seed_stats([j["metrics"] for j in completed])
                summary.update(seeds=[j["seed"] for j in completed], all_seeds_complete=len(completed) == len(seeds))
                summary["versus"] = {name: comparison(summary["mean"], ref["metrics"]) for name, ref in references.items()}
                block["completed_seed_summary"][variant] = summary
        for seed in seeds:
            full, reduced = (block["jobs"][v][str(seed)] for v in VARIANTS)
            if not (full["eligible_for_ranking"] and reduced["eligible_for_ranking"]):
                continue
            pair = {"seed": seed, "eligible": True, "errors": [], "config_pairing_checked": False}
            a, b = full.get("config_pairing"), reduced.get("config_pairing")
            if a and b:
                pair["config_pairing_checked"] = True
                if a != b or a["shared_named_parameters_exact"] is not True:
                    pair.update(eligible=False, errors=["full/no_memory source/data/training/shared initialization metadata differ"])
            if full.get("history") and reduced.get("history"):
                for a, b in zip(full["history"], reduced["history"]):
                    if a.get("schedule") != b.get("schedule"):
                        pair.update(eligible=False, errors=["matched-seed epoch schedules differ"])
                        break
            if pair["eligible"]:
                pair["full_minus_no_memory"] = comparison(full["metrics"], reduced["metrics"])
            block["full_vs_no_memory_matched_seeds"][str(seed)] = pair
        pairs = [p for p in block["full_vs_no_memory_matched_seeds"].values() if p["eligible"]]
        if pairs:
            block["paired_mean_difference"] = {"seeds": [p["seed"] for p in pairs], "n": len(pairs),
                **{m: {"mean_delta": statistics.mean(p["full_minus_no_memory"][m]["delta"] for p in pairs),
                       "sample_sd_delta": statistics.stdev(p["full_minus_no_memory"][m]["delta"] for p in pairs) if len(pairs) >= 2 else None}
                   for m in ("mean_nmae", "mean_nrmse")}}
        candidates = {n: r["metrics"] for n, r in references.items()}
        candidates.update({n: s["mean"] for n, s in block["completed_seed_summary"].items() if s["all_seeds_complete"]})
        for metric in ("mean_nmae", "mean_nrmse"):
            block["rankings_completed_only"][metric] = [{"method": n, "value": m[metric]} for n, m in sorted(candidates.items(), key=lambda pair: pair[1][metric])]
    report["state_counts"] = dict(Counter(j["state"] for d in report["datasets"].values() for jobs in d["jobs"].values() for j in jobs.values()))
    report["all_discovered_jobs_complete"] = all(j["eligible_for_ranking"] for d in report["datasets"].values() for jobs in d["jobs"].values() for j in jobs.values())
    return report


def markdown(report):
    lines = ["# SPIN / Direct / Sync 联合模型：保存结果汇总", "", f"生成：{report['generated_utc']}；状态计数：{report['state_counts']}。", "",
        "只读快照，不探测进程存活。running/stopped 来自已保存 status；complete 需正式 result.json 标记完整且通过已有记录核对。smoke 仅为拟合路径/恢复检查，不是正式精度结果。",
        "原始 V1 从头联合训练；fixed120 是取消早停后的事后训练时长诊断，从同一轨迹的父 last.pt 继续，不是新的独立重复。父输出只用于核查，不并入种子统计；整条1–120轮 history（零起始0–119）只计一次。",
        "full 与 no_memory 按同一 seed 配对；移除记忆同时减少参数，因此不是等参数量控制。以下均为开发指标，不是独立测试。", ""]
    for dataset, data in report["datasets"].items():
        lines += [f"## {dataset.upper()}", "", "| 方法/种子 | 状态 | smoke | 已保存轮数 | 最佳轮 | 完成 NMAE | 完成 NRMSE |", "|---|---|---|---:|---:|---:|---:|"]
        issues = []
        for name, ref in data["references"].items():
            m = ref["metrics"]
            lines.append(f"| {name} 旧参考 | complete | — | — | — | {m['mean_nmae']:.9f} | {m['mean_nrmse']:.9f} |")
        for variant, jobs in data["jobs"].items():
            for seed, job in jobs.items():
                m = job.get("metrics") if job["eligible_for_ranking"] else None
                progress, result = job.get("progress", {}), job.get("result_metadata", {})
                scores = f"{m['mean_nmae']:.9f} | {m['mean_nrmse']:.9f}" if m else "— | —"
                smoke = "通过" if (job.get("smoke") or {}).get("passed") else "未同步/未通过"
                if job.get("smoke_origin") == "inherited_parent":
                    smoke = "继承通过" if job["continuation"]["inherited_smoke_verified_locally"] else "继承（父记录未同步）"
                lines.append(f"| {variant}/{seed} | {job['state']} | {smoke} | {result.get('epochs_completed', progress.get('saved_epochs', '—'))} | {result.get('best_epoch', progress.get('best_epoch', '—'))} | {scores} |")
                for issue in job["warnings"] + job["errors"]:
                    issues.append(f"- {variant}/{seed}：{issue}")
                if "continuation" in job:
                    c = job["continuation"]
                    issues.append(f"- {variant}/{seed}：同轨迹诊断，父轮数 {c['parent_epochs_completed']}，新增已保存轮数 {c['additional_epochs_saved']}，累计 {c['total_epochs_saved']}；不新增 smoke、不算独立重复。")
        if issues:
            lines += ["", "核查说明：", "", *issues]
        lines += ["", "### 已完成种子均值与样本标准差", "", "| 方法 | 已完成/计划种子 | NMAE 均值 ± SD | NRMSE 均值 ± SD |", "|---|---:|---:|---:|"]
        for name, s in data["completed_seed_summary"].items():
            def fmt(m):
                return f"{s['mean'][m]:.9f} ± {s['sample_sd'][m]:.9f}" if s['sample_sd'] else f"{s['mean'][m]:.9f}（n=1，无SD）"
            lines.append(f"| {name} | {s['n']}/{len(data['requested_or_discovered_seeds'])} | {fmt('mean_nmae')} | {fmt('mean_nrmse')} |")
        lines += ["", "只对该 WAN 计划/已发现种子全部完成的方法给出最终均值排名；部分均值仅描述已完成种子，不挑最好种子。", ""]
        for metric, ranked in data["rankings_completed_only"].items():
            lines.append(f"{metric}（低者较好）：" + " → ".join(f"{r['method']} {r['value']:.9f}" for r in ranked) + "。\n")
        lines += ["### 已完成模型相对旧参考", "", "正改善率 = 100×(参考−当前)/参考。", "", "| 方法/种子 | 参考 | NMAE 改善 % | NRMSE 改善 % |", "|---|---|---:|---:|"]
        for variant, jobs in data["jobs"].items():
            for seed, job in jobs.items():
                for name, difference in job.get("versus", {}).items():
                    lines.append(f"| {variant}/{seed} | {name} | {difference['mean_nmae']['relative_improvement_percent']:+.3f} | {difference['mean_nrmse']['relative_improvement_percent']:+.3f} |")
        lines += ["", "### 同 seed 的 full − no_memory", "", "绝对差为负表示 full 较好；仅纳入两边均完成且配对核查通过的种子。", "", "| seed | NMAE 差 | NRMSE 差 |", "|---:|---:|---:|"]
        for seed, pair in data["full_vs_no_memory_matched_seeds"].items():
            if pair["eligible"]:
                diff = pair["full_minus_no_memory"]
                lines.append(f"| {seed} | {diff['mean_nmae']['delta']:+.9f} | {diff['mean_nrmse']['delta']:+.9f} |")
            else:
                lines.append(f"\n配对异常 seed{seed}：{pair['errors']}\n")
    lines += ["", "JSON 保留全部 history、smoke、个体及三条件指标、均值/样本SD、改善率、配对差和核查状态。SD 只描述本批训练种子，不构造显著性或置信区间。未复制的 ancillary/checkpoint 文件会明确标记，仅读取已有完成标记，不创建新训练协议。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--parent-outputs", type=Path, help="Local original V1 raw_outputs root for inherited smoke and continuation ledger verification; never included as additional seeds")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent / "summary")
    parser.add_argument("--seeds", nargs="+", type=int, help="Also show these requested seeds if records are absent; otherwise discover, default seed41001")
    args = parser.parse_args()
    require(args.outputs.exists(), "outputs root does not exist; supply the actual copied outputs root")
    report = collect(args)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "SUMMARY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.report_dir / "SUMMARY_CN.md").write_text(markdown(report))
    print(json.dumps({"report_dir": str(args.report_dir), "state_counts": report["state_counts"], "all_discovered_jobs_complete": report["all_discovered_jobs_complete"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
