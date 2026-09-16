#!/usr/bin/env python3
"""Read-only stdlib summary of the completed, pinned inference benchmark."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

BENCHMARK_SHA256 = "d523e8bc84fcb5789a0ca712aea561c4f9fb42129d65bf8c6ab0b42a38c84679"
DATASETS = ("abilene", "geant")
METHODS = ("spin_sync_direct", "spin_direct_no_memory", "spin_adapted", "direct_sync")
CONDITIONS = ("uniform", "unequal", "unequal_gap")
LABELS = dict(zip(METHODS, ("Full", "no_memory", "旧 SPIN", "旧 Direct+Sync")))
WAN = {"abilene": "Abilene", "geant": "GEANT"}
HASH = re.compile(r"[0-9a-f]{64}\Z")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not positive or value > 0), f"{label}: invalid finite number")
    return value


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: invalid integer")
    return value


def checksum(value, label):
    require(isinstance(value, str) and HASH.fullmatch(value), f"{label}: invalid SHA256")


def text_field(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label}: missing text")


def percentile95(samples):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * .95
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[math.ceil(position)] * fraction


def validate(data):
    require(data.get("state") == "complete", "benchmark state must be complete; no final report for partial runs")
    require(data.get("role") == "fit_only_inference_benchmark", "unexpected benchmark role")
    protocol = data["protocol"]
    for key, expected in {"precision": "FP32; TF32 disabled", "batches": [1, 8],
                          "full_target_graph": True, "mask_observation_fraction": .2,
                          "warmups_per_condition_batch": 5, "repeats_per_condition_batch": 15,
                          "accuracy_scores_computed": False}.items():
        require(protocol.get(key) == expected, f"protocol.{key} differs")
    for key in ("timing", "memory", "fit_input", "per_window_latency"):
        text_field(protocol.get(key), f"protocol.{key}")
    require(protocol.get("excluded") == ["checkpoint loading", "fit statistics/graph setup", "mask generation"],
            "protocol exclusions differ")
    for key in ("device", "device_name", "hostname"):
        text_field(data.get(key), key)
    for key in ("python", "torch", "numpy", "cuda"):
        text_field(data["runtime"].get(key), f"runtime.{key}")
    for key in ("started_unix", "finished_unix", "deadline_unix"):
        number(data.get(key), key, positive=True)
    require(data["finished_unix"] >= data["started_unix"], "finish precedes start")
    sources = data["source_files"]
    require(isinstance(sources, dict), "source_files must be a mapping")
    for path, digest in sources.items():
        text_field(path, "source path")
        checksum(digest, path)
    for name in ("benchmark_inference.py", "engine.py", "data.py"):
        matches = [digest for path, digest in sources.items() if Path(path).name == name]
        require(len(matches) == 1, f"missing or ambiguous source binding: {name}")
        if name == "benchmark_inference.py":
            require(matches[0] == BENCHMARK_SHA256, "benchmark source SHA differs from reviewed version")
    require(set(data["datasets"]) == set(DATASETS), "exactly Abilene and GEANT required")
    cells = []
    for dataset in DATASETS:
        record = data["datasets"][dataset]
        require(record["shape"] == [8, 50, 144 if dataset == "abilene" else 462], f"{dataset}: input shape differs")
        require(record["window_indices"] == list(range(8)), f"{dataset}: fit indices differ")
        starts = record["window_starts"]
        require(isinstance(starts, list) and len(starts) == 8, f"{dataset}: 8 starts required")
        for start in starts:
            integer(start, f"{dataset}: window start")
        require(len(set(starts)) == 8, f"{dataset}: duplicate window starts")
        require(record["mask_seed"] == 71001, f"{dataset}: mask seed differs")
        checksum(record["values_sha256"], f"{dataset}: values")
        for key in ("stats_sha256", "neighbors_sha256", "train_starts_sha256"):
            checksum(record["fit_identity"][key], f"{dataset}: {key}")
        require(set(record["mask_sha256"]) == set(CONDITIONS), f"{dataset}: mask hash coverage differs")
        for condition, digest in record["mask_sha256"].items():
            checksum(digest, f"{dataset}/{condition}: mask")
        require(set(record["models"]) == set(METHODS), f"{dataset}: model coverage differs")
        for method, model in record["models"].items():
            label = f"{dataset}/{method}"
            integer(model["parameters"], label + ": parameters", 1)
            integer(model["best_epoch"], label + ": best epoch")
            require(model["seed"] == 41001, label + ": seed differs")
            require(model["internal_node_chunk"] == (None if method == "direct_sync" else 32), label + ": node chunk differs")
            checkpoint = model["checkpoint_path"]
            text_field(checkpoint, label + ": checkpoint path")
            require(Path(checkpoint).name == "best.pt", label + ": expected selected best.pt")
            checksum(model["checkpoint_sha256"], label + ": checkpoint")
            bound = model["bound_files"]
            require(isinstance(bound, dict), label + ": bound_files must be a mapping")
            for path, digest in bound.items():
                text_field(path, label + ": bound path")
                checksum(digest, label + ": " + path)
            require(bound.get(checkpoint) == model["checkpoint_sha256"], label + ": checkpoint hash not bound")
            require(str(Path(checkpoint).with_name("result.json")) in bound, label + ": result not bound")
            config = str(Path(checkpoint).with_name("config.json")) if method in METHODS[:2] else str(Path(checkpoint).parents[3] / "registry.json")
            require(config in bound, label + ": config/registry not bound")
        measurements = record["measurements"]
        require(isinstance(measurements, list) and len(measurements) == 24, f"{dataset}: 24 cells required")
        seen = set()
        expected = set(itertools.product(METHODS, (1, 8), CONDITIONS))
        for row in measurements:
            key = (row["method"], row["batch"], row["condition"])
            label = f"{dataset}/{key}"
            require(type(row["batch"]) is int and key in expected and key not in seen, label + ": unknown/duplicate cell")
            seen.add(key)
            require(type(row["warmups"]) is int and type(row["repeats"]) is int
                    and row["warmups"] == 5 and row["repeats"] == 15, label + ": expected 5 warmups and 15 repeats")
            samples = row["samples_ms_per_batch"]
            require(isinstance(samples, list) and len(samples) == 15, label + ": expected 15 samples")
            for sample in samples:
                number(sample, label + ": sample", positive=True)
            median, p95 = statistics.median(samples), percentile95(samples)
            for field, expected_value in (("median_ms_per_batch", median), ("p95_ms_per_batch", p95),
                                           ("median_ms_per_window", median / key[1]), ("p95_ms_per_window", p95 / key[1])):
                value = number(row[field], label + ": " + field, positive=True)
                require(math.isclose(value, expected_value, rel_tol=1e-10, abs_tol=1e-9), label + ": inconsistent " + field)
            resident = integer(row["resident_allocated_bytes"], label + ": resident memory")
            peak = integer(row["max_memory_allocated_bytes_including_resident_model"], label + ": peak memory")
            above = integer(row["peak_above_resident_bytes"], label + ": incremental memory")
            require(peak >= resident and above == peak - resident, label + ": inconsistent memory accounting")
            cells.append({"dataset": dataset, **row, "median_ms_per_batch": median, "p95_ms_per_batch": p95,
                          "median_ms_per_window": median / key[1], "p95_ms_per_window": p95 / key[1]})
        require(seen == expected, f"{dataset}: incomplete grid")
    return cells


def summarize(data):
    cells = validate(data)
    main, ratios = [], []
    for dataset in DATASETS:
        for method in METHODS:
            rows = [r for r in cells if r["dataset"] == dataset and r["method"] == method]
            b1, b8 = ([r for r in rows if r["batch"] == batch] for batch in (1, 8))
            main.append({"dataset": dataset, "method": method,
                         "parameters": data["datasets"][dataset]["models"][method]["parameters"],
                         "b1_mean_condition_median_ms_per_batch": statistics.mean(r["median_ms_per_batch"] for r in b1),
                         "b8_mean_condition_median_ms_per_batch": statistics.mean(r["median_ms_per_batch"] for r in b8),
                         "b8_mean_condition_median_ms_per_window": statistics.mean(r["median_ms_per_batch"] for r in b8) / 8,
                         "b8_max_allocated_mib_including_resident_model": max(r["max_memory_allocated_bytes_including_resident_model"] for r in b8) / 2**20})
        full = next(r for r in main if r["dataset"] == dataset and r["method"] == METHODS[0])
        for denominator in (METHODS[1], METHODS[2]):
            base = next(r for r in main if r["dataset"] == dataset and r["method"] == denominator)
            item = {"dataset": dataset, "numerator": METHODS[0], "denominator": denominator}
            for batch, field in ((1, "b1_mean_condition_median_ms_per_batch"), (8, "b8_mean_condition_median_ms_per_window")):
                ratio = full[field] / base[field]
                item.update({f"b{batch}_time_ratio": ratio, f"b{batch}_time_change_percent": 100 * (ratio - 1)})
            ratios.append(item)
    return {"state": "complete", "role": "inference_cost_summary", "validated_cell_count": len(cells),
            "validation_scope": "JSON completeness/internal consistency and declared source/checkpoint hash bindings; remote artifacts were not re-read",
            "definitions": {"b1": "mean of three condition medians, ms per batch of one window",
                            "b8": "mean of three condition medians divided by 8, amortized ms/window, not single-window turnaround",
                            "b8_memory": "maximum allocated GPU bytes over three conditions / 2^20, including resident model/statistics/neighbor table",
                            "p95": "15 measured samples; linear interpolation at index (n-1)*0.95; descriptive, no significance claim",
                            "ratios": "Full time / denominator time; percent change = 100*(ratio-1); positive means higher cost"},
            "benchmark_metadata": {key: value for key, value in data.items() if key != "datasets"},
            "dataset_metadata": {dataset: {key: value for key, value in record.items() if key != "measurements"}
                                 for dataset, record in data["datasets"].items()},
            "main": main, "relative_time": ratios, "cells": cells}


def markdown(result):
    metadata = result["benchmark_metadata"]
    lines = ["# 推理成本报告", "", f"状态：完成；已校验 48 格。设备：{metadata['device_name']}（{metadata['device']}）；FP32，关闭 TF32。", "",
             "仅使用两个 WAN 各自前 8 个 fit 窗口，B1 使用其中首窗；四模型、B1/B8、三种条件，每格 5 次暖身、15 次计时。以下数值均来自实际计时样本。", "",
             "## 汇总", "", "B1 为三种条件各自整批延迟中位数的算术均值。B8 为三种条件各自整批中位数均值除以 8，是摊销每窗延迟，不是独立测量的单窗往返延迟。显存取 B8 三条件最大 allocated 值，含当前常驻模型、fit 统计量与邻居表，不是 nvidia-smi 整卡占用。", "",
             "| WAN | 模型 | 参数总量 | B1 整批 ms | B8 摊销 ms/窗 | B8 最大 allocated MiB |",
             "|---|---|---:|---:|---:|---:|"]
    for row in result["main"]:
        lines.append(f"| {WAN[row['dataset']]} | {LABELS[row['method']]} | {row['parameters']:,} | {row['b1_mean_condition_median_ms_per_batch']:.3f} | {row['b8_mean_condition_median_ms_per_window']:.3f} | {row['b8_max_allocated_mib_including_resident_model']:.3f} |")
    lines += ["", "## Full 相对耗时", "", "比值 = Full / 对照；变化率 = (比值 − 1) × 100%，正值表示 Full 耗时更多。B8 比较采用上述摊销定义。", "",
              "| WAN | 对照 | B1 比值 | B1 变化率 | B8 比值 | B8 变化率 |", "|---|---|---:|---:|---:|---:|"]
    for row in result["relative_time"]:
        lines.append(f"| {WAN[row['dataset']]} | {LABELS[row['denominator']]} | {row['b1_time_ratio']:.4f} | {row['b1_time_change_percent']:+.2f}% | {row['b8_time_ratio']:.4f} | {row['b8_time_change_percent']:+.2f}% |")
    lines += ["", "## 48 格完整明细", "", "P95 使用 15 个计时样本排序后的线性插值（索引为 (n−1)×0.95）；暖身不计入样本。JSON 保留全部原始计时、检查点和环境元数据。", "",
              "| WAN | 模型 | B | 条件 | 整批 median ms | 整批 P95 ms | 摊销 median ms/窗 | 摊销 P95 ms/窗 | 最大 allocated MiB |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in sorted(result["cells"], key=lambda r: (DATASETS.index(r["dataset"]), METHODS.index(r["method"]), r["batch"], CONDITIONS.index(r["condition"]))):
        lines.append(f"| {WAN[row['dataset']]} | {LABELS[row['method']]} | {row['batch']} | {row['condition']} | {row['median_ms_per_batch']:.3f} | {row['p95_ms_per_batch']:.3f} | {row['median_ms_per_window']:.3f} | {row['p95_ms_per_window']:.3f} | {row['max_memory_allocated_bytes_including_resident_model'] / 2**20:.3f} |")
    lines += ["", "## 口径与边界", "",
              "- 计时覆盖 engine.predict_windows：输入传输、全 OD 流预测、输出回传、有限值检查、CPU 非负截断及观测值复制；每次计时前后同步 CUDA。",
              "- 不含检查点加载、fit 统计量/图准备及掩码生成；四模型逐一加载，其他模型卸载。显存是 PyTorch allocated 口径，非 reserved 或整卡占用。",
              "- 这是单设备、固定窗口、固定模型顺序的描述性计时；15 次重复不构成训练重复或显著性证据，P95 的尾部分辨率有限。",
              "- 本报告仅比较推理成本，不从训练耗时推断推理加速，不给准确率排名。",
              "- 汇总器校验 JSON 完整性、样本与统计量一致性、声明的检查点绑定和已审阅测速源码 SHA；不重新读取远端权重或运行模型。测速程序本身负责检查点与来源文件哈希核验。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/spin_sync_unified_20260916/INFERENCE_BENCHMARK.json"))
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = args.input.read_bytes()
        result = summarize(json.loads(payload))
        result["input"] = {"path": str(args.input.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}
        json_path = args.report_dir / "INFERENCE_SUMMARY.json"
        md_path = args.report_dir / "INFERENCE_SUMMARY_CN.md"
        require(args.input.resolve() not in (json_path.resolve(), md_path.resolve()), "output would overwrite input")
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        report = markdown(result)
        args.report_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(rendered, encoding="utf-8")
        md_path.write_text(report, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(2, f"Summary failed: {error}\n")
    print(f"Validated 48 cells: {json_path}\n{md_path}")


if __name__ == "__main__":
    main()
