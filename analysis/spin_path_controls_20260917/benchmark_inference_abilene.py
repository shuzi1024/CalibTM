"""Read-only Abilene inference costs: three path controls and CPU Linear.

Neural models: the completed seed41001 best-over-1..160 checkpoint, one common
authorized CUDA device, original engine.predict_windows end-to-end boundary.
Linear: the exact NumPy prediction expression used by evaluate_linear, measured
separately on CPU. No GPU/CPU acceleration ratios are computed. First eight fit
windows only; five warmups and fifteen timings for B1/B8 and each condition.
No accuracy scores, training, new masks for model selection, or checkpoint edits.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import platform
import signal
import socket
import sys
import time

METHODS = ("spin_direct", "context_only", "direct_only")
WARMUPS, REPEATS = 5, 15
PROTOCOL = "spin-path-controls-development-20260917-v1"


def records(args):
    artifacts = {}
    for method in METHODS:
        common.check_stop(args)
        job = args.run_root / "abilene" / method / "seed41001"
        path = job / "best.pt"
        config_path, result_path = job / "config.json", job / "result.json"
        config, result = common.read(config_path), common.read(result_path)
        identity = {"protocol": PROTOCOL, "dataset": "abilene", "variant": method,
                    "seed": 41001, "config_sha256": common.sha(config_path)}
        common.require(config, {k: v for k, v in identity.items() if k != "config_sha256"},
                       method + " config")
        common.require(result, {**identity, "state": "complete", "role": "development_only",
                                "epochs_completed": 160}, method + " result")
        common.require(result["primary_eval"], {"role": "development_only", "mask_seed": 71001},
                       method + " evaluation")
        history_path = job / "history.json"
        history = common.read(history_path)
        if (len(history) != 160 or [row["epoch"] for row in history] != list(range(1, 161))
                or any(not math.isfinite(row["selection_score"]) for row in history)):
            raise ValueError(method + ": complete finite 160-epoch history required")
        selected = min(history, key=lambda row: row["selection_score"])
        if (selected["epoch"] != result["best_epoch"]
                or selected["selection_score"] != result["primary_eval"]["selection_score"]
                or common.sha(path) != result["best_checkpoint_sha256"]
                or config["training"]["max_epochs"] != 160
                or config["training"]["lr_schedule"] != [
                    {"first_epoch": 1, "last_epoch": 120, "lr": .001},
                    {"first_epoch": 121, "last_epoch": 160, "lr": .0001}]):
            raise ValueError(method + ": frozen schedule/best checkpoint binding differs")
        for relative, checksum in config["source_files"].items():
            if relative.endswith(".py") and common.sha(args.repo_root / relative) != checksum:
                raise ValueError(f"Frozen executable source differs: {relative}")
        bound = {str(p): common.sha(p) for p in
                 (path, config_path, result_path, history_path)}
        artifacts[method] = {"path": path, "identity": identity, "result": result,
                             "config": config, "files": bound}
    if any(item["config"]["data"] != artifacts[METHODS[0]]["config"]["data"]
           for item in artifacts.values()):
        raise ValueError("The three models must use exactly the same frozen data")
    return artifacts


def load_model(method, record, flows, fit):
    import torch
    from experiments.spin_path_controls_v1.models import build_model
    from experiments.sync_delta_v1.run import state_sha
    model = build_model(flows, init_seed=41001, variant=method)
    model.configure_fit_statistics(fit)
    model.set_execution(node_chunk=32, gradient_checkpointing=False)
    scaler = {name: getattr(model, name).clone()
              for name in ("fit_mean", "fit_scale", "fit_configured")}
    saved = torch.load(record["path"], map_location="cpu", weights_only=False)
    common.require(saved, record["identity"], method + " checkpoint")
    result = record["result"]
    if (saved["epoch"] != result["best_epoch"]
            or saved["score"] != result["primary_eval"]["selection_score"]):
        raise ValueError(method + ": selected checkpoint identity differs")
    model.load_state_dict(saved["model"], strict=True)
    if (state_sha(model.state_dict()) != result["best_state_sha256"]
            or sum(p.numel() for p in model.parameters()) != result["parameter_count"]
            or any(not torch.equal(getattr(model, name), value) for name, value in scaler.items())):
        raise ValueError(method + ": model state, parameter count or fit statistics differ")
    return model.float().eval().requires_grad_(False)


def timing_row(method, batch, condition, samples, *, device):
    import numpy as np
    median, p95 = float(np.median(samples)), float(np.percentile(samples, 95))
    return {"method": method, "batch": batch, "condition": condition,
            "execution_device": device, "warmups": WARMUPS, "repeats": REPEATS,
            "median_ms_per_batch": median, "p95_ms_per_batch": p95,
            "median_ms_per_window": median / batch, "p95_ms_per_window": p95 / batch,
            "samples_ms_per_batch": samples}


def validate_prediction(final, values, mask):
    import numpy as np
    if (final.shape != values.shape or final.dtype != np.float32
            or not np.isfinite(final).all() or np.any(final < 0)
            or not np.array_equal(final[mask], values[mask])):
        raise ValueError("Prediction shape, finite/nonnegative output, or observed copy differs")


def cpu_metadata():
    cpu_name = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_name = next((line.split(":", 1)[1].strip()
                         for line in cpuinfo.read_text().splitlines()
                         if line.startswith("model name")), cpu_name)
    result = {"name": cpu_name or platform.machine(), "logical_cpu_count": os.cpu_count(),
              "process_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "linear_application_parallelism": "single calling thread; sequential NumPy interp loops",
              "thread_environment": {name: os.environ.get(name) for name in
                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    try:
        from threadpoolctl import threadpool_info
        result["loaded_thread_pools"] = threadpool_info()
    except ImportError:
        result["loaded_thread_pools"] = "threadpoolctl not installed; environment reported only"
    return result


def benchmark(args, output):
    import numpy as np
    import torch
    from experiments.sync_delta_v1 import data, engine
    from experiments.spin_path_controls_v1 import models
    common.check_stop(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Use one authorized idle CUDA device for the three neural models")
    torch.cuda.set_device(device)
    engine.seed_runtime(41001)
    output.update(device=str(device), device_name=torch.cuda.get_device_name(device),
        runtime={"python": platform.python_version(), "torch": torch.__version__,
                 "numpy": np.__version__, "cuda": torch.version.cuda,
                 "torch_num_threads": torch.get_num_threads(),
                 "torch_num_interop_threads": torch.get_num_interop_threads()},
        cpu=cpu_metadata(),
        source_files={str(Path(module.__file__).resolve()): common.sha(module.__file__)
                      for module in (sys.modules[__name__], common, engine, data, models)})
    artifacts = records(args)
    values, starts, fit, neighbors_cpu = data.load_fit_windows("abilene", indices=tuple(range(8)))
    fit_arrays = {key: fit[key] for key in ("mu", "sigma", "scale")}
    fit_arrays.update(C=np.asarray(fit["C"], dtype="<f8"),
                      global_scale=np.asarray(fit["global_scale"], dtype="<f8"))
    fit_identity = {"stats_sha256": data._arrays_sha256(fit_arrays),
                    "neighbors_sha256": data.array_sha256(neighbors_cpu),
                    "train_starts_sha256": data.array_sha256(data.window_starts("abilene", "fit"))}
    frozen_data = artifacts[METHODS[0]]["config"]["data"]
    if any(frozen_data[key] != value for key, value in fit_identity.items()):
        raise ValueError("Fit statistics, graph or fit-window identity differs")
    flows = values.shape[-1]
    if values.shape != (8, 50, 144):
        raise ValueError("Expected original Abilene first-eight fit-window dimensions")
    families = [data.make_mask_family(flows, int(start), 71001, dataset="abilene") for start in starts]
    masks = {condition: np.stack([family.masks[condition] for family in families])
             for condition in data.CONDITIONS}
    if any(int(mask.sum()) != 8 * 10 * flows for mask in masks.values()):
        raise ValueError("Original 20% observation budget differs")
    record = {"fit_identity": fit_identity, "window_indices": list(range(8)),
              "window_starts": starts.tolist(), "values_sha256": data.array_sha256(values),
              "shape": list(values.shape), "mask_seed": 71001,
              "mask_sha256": {k: data.array_sha256(v) for k, v in masks.items()},
              "models": {}, "measurements": []}
    output["datasets"]["abilene"] = record
    stats = {key: torch.as_tensor(fit[key], dtype=torch.float32, device=device)
             for key in ("mu", "scale")}
    neighbors = torch.as_tensor(neighbors_cpu, dtype=torch.long, device=device)
    for method in METHODS:
        common.check_stop(args)
        artifact = artifacts[method]
        model = load_model(method, artifact, flows, fit).to(device)
        record["models"][method] = {
            "execution_device": str(device), "seed": 41001, "selection_epoch_cap": 160,
            "checkpoint_path": str(artifact["path"]), "checkpoint_sha256": common.sha(artifact["path"]),
            "bound_files": artifact["files"], "best_epoch": artifact["result"]["best_epoch"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "internal_node_chunk": 32 if method != "direct_only" else None}
        for batch in (1, 8):
            for condition in data.CONDITIONS:
                def once():
                    return engine.predict_windows(model, values[:batch], masks[condition][:batch],
                        starts[:batch], stats, neighbors, dataset="abilene",
                        physical_batch=batch, target_block=flows)
                for _ in range(WARMUPS):
                    common.check_stop(args)
                    once()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                resident = torch.cuda.memory_allocated(device)
                samples = []
                for _ in range(REPEATS):
                    common.check_stop(args)
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    final, raw = once()
                    torch.cuda.synchronize(device)
                    samples.append(1000 * (time.perf_counter() - begin))
                peak = torch.cuda.max_memory_allocated(device)
                validate_prediction(final, values[:batch], masks[condition][:batch])
                row = timing_row(method, batch, condition, samples, device=str(device))
                row.update(resident_allocated_bytes=resident,
                           max_memory_allocated_bytes_including_resident_model=peak,
                           peak_above_resident_bytes=peak - resident)
                record["measurements"].append(row)
                common.write(args.output, output)
                print(f"abilene/{method}/B{batch}/{condition}: {row['median_ms_per_batch']:.3f} ms", flush=True)
                del final, raw
        del model
        gc.collect()
        torch.cuda.empty_cache()
    del stats, neighbors
    gc.collect()
    torch.cuda.empty_cache()

    # Exact existing Linear prediction stage; scoring and any added validation
    # are outside the timer. There is no GPU transfer or explicit clamp in it.
    record["models"]["linear_cpu"] = {"execution_device": "cpu", "parameters": 0,
        "checkpoint_path": None, "seed": None, "internal_node_chunk": None,
        "implementation": "np.stack([data.linear_fill(truth, mask, fit_mu) for each window])",
        "observation_copy_included": True,
        "clamp": "none in existing implementation; nonnegative observations imply nonnegative interpolation",
        "memory": "CPU peak allocation not measured; no GPU memory figure applies"}
    for batch in (1, 8):
        for condition in data.CONDITIONS:
            def once_linear():
                return np.stack([data.linear_fill(truth, mask, fit["mu"])
                                 for truth, mask in zip(values[:batch], masks[condition][:batch])])
            for _ in range(WARMUPS):
                common.check_stop(args)
                once_linear()
            samples = []
            for _ in range(REPEATS):
                common.check_stop(args)
                begin = time.perf_counter()
                final = once_linear()
                samples.append(1000 * (time.perf_counter() - begin))
            validate_prediction(final, values[:batch], masks[condition][:batch])
            row = timing_row("linear_cpu", batch, condition, samples, device="cpu")
            row.update(resident_allocated_bytes=None,
                       max_memory_allocated_bytes_including_resident_model=None,
                       peak_above_resident_bytes=None)
            record["measurements"].append(row)
            common.write(args.output, output)
            print(f"abilene/linear_cpu/B{batch}/{condition}: {row['median_ms_per_batch']:.3f} ms", flush=True)
            del final
    for artifact in artifacts.values():
        if any(common.sha(path) != checksum for path, checksum in artifact["files"].items()):
            raise ValueError("Checkpoint/config/result/history changed during benchmarking")
    common.check_stop(args)
    expected = {(method, batch, condition) for method in (*METHODS, "linear_cpu")
                for batch in (1, 8) for condition in data.CONDITIONS}
    actual = {(row["method"], row["batch"], row["condition"]) for row in record["measurements"]}
    if actual != expected or len(record["measurements"]) != len(expected):
        raise ValueError("Benchmark grid is incomplete or duplicated")
    output["summary"] = []
    for method in (*METHODS, "linear_cpu"):
        cells = [row for row in record["measurements"] if row["method"] == method]
        b1 = [row for row in cells if row["batch"] == 1]
        b8 = [row for row in cells if row["batch"] == 8]
        output["summary"].append({"method": method,
            "execution_device": record["models"][method]["execution_device"],
            "parameters": record["models"][method]["parameters"],
            "b1_mean_condition_median_ms_per_batch": float(np.mean([r["median_ms_per_batch"] for r in b1])),
            "b8_mean_condition_median_ms_per_window": float(np.mean([r["median_ms_per_batch"] for r in b8])) / 8,
            "b8_max_allocated_mib_including_resident_model": None if method == "linear_cpu" else
                max(r["max_memory_allocated_bytes_including_resident_model"] for r in b8) / 2**20})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Parent of abilene/{variant}/seed41001 completed run directories")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time() + 60:
        parser.error("A future authorized deadline with more than 60 seconds remaining is required")
    args.repo_root, args.output = args.repo_root.resolve(), args.output.resolve()
    args.run_root = (args.repo_root / args.run_root).resolve()
    if args.output.exists():
        parser.error("Use a new output JSON; existing files are never overwritten")
    # Set library thread configuration before importing NumPy/PyTorch. The
    # existing engine explicitly selects four PyTorch CPU threads as well.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    sys.path.insert(0, str(args.repo_root))
    global common
    from analysis.spin_sync_unified_20260916 import benchmark_inference as common
    signal.signal(signal.SIGTERM, common.request_stop)
    signal.signal(signal.SIGINT, common.request_stop)
    output = {"state": "running", "role": "fit_only_inference_benchmark", "datasets": {},
        "started_unix": time.time(), "deadline_unix": args.deadline_unix,
        "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": {"precision": "FP32; TF32 disabled", "batches": [1, 8],
            "neural_methods": list(METHODS), "separate_cpu_reference": "linear_cpu",
            "checkpoint": "seed41001 completed 160-epoch run; best NMAE over epochs 1..160",
            "full_target_graph": True, "mask_observation_fraction": .2,
            "fit_input": "Abilene first 8 fit windows; B1 uses the first of those windows",
            "warmups_per_condition_batch": WARMUPS, "repeats_per_condition_batch": REPEATS,
            "neural_timing": "engine.predict_windows including input/output transfers, finite check, CPU clamp/copy; CUDA sync before/after",
            "linear_timing": "existing evaluate_linear prediction stage on CPU: per-window linear_fill plus stack; includes observed-input validation/interpolation/copy, excludes scoring",
            "linear_output_validation": "shape/finite/nonnegative/copy checks outside timed region; original routine has no explicit clamp",
            "comparison_limit": "CPU Linear and GPU neural timings have different devices/boundaries; no cross-device speedup claim",
            "neural_memory": "PyTorch allocated bytes including resident model/shared fit statistics/neighbor table; not whole device",
            "linear_memory": "CPU peak memory not measured; GPU memory not applicable",
            "per_window_latency": "batch latency divided by batch size; B8 is amortized throughput cost",
            "excluded": ["checkpoint loading", "fit statistics/graph setup", "mask generation", "accuracy scoring"],
            "accuracy_scores_computed": False}}
    try:
        benchmark(args, output)
        output["state"] = "complete"
    except common.Stopped as error:
        output.update(state="stopped", reason=str(error))
    except BaseException as error:
        output.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output["finished_unix"] = time.time()
        common.write(args.output, output)


if __name__ == "__main__":
    main()
