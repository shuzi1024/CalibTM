"""Read-only FP32 inference benchmark; fit windows only, no accuracy scores.

Run on one idle, externally authorized GPU after training. Every method uses
engine.predict_windows, including input transfers, full-flow prediction,
GPU-to-CPU raw output, finite checks, and CPU clamp/observed-value copying.
The five warmups and fifteen timings per condition/batch exclude loading,
mask construction and fit-statistic/graph preparation. Memory includes the
resident model and shared GPU fit statistics/neighbor table.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import sys
import tempfile
import time

STOP = False
WARMUPS, REPEATS = 5, 15
METHODS = ("spin_sync_direct", "spin_direct_no_memory", "spin_adapted", "direct_sync")


class Stopped(RuntimeError):
    pass


def request_stop(signum, frame):
    global STOP
    STOP = True


def check_stop(args):
    if STOP or time.time() >= args.deadline_unix - 60:
        raise Stopped("signal" if STOP else "allocation_cutoff")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def require(record, expected, label):
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{label} identity differs")


def records(args, dataset):
    root = args.repo_root
    fixed = root / "outputs/spin-sync-unified-fixed120-20260916-v1"
    paths = {
        "spin_sync_direct": Path(args.full_checkpoint.format(dataset=dataset)) if args.full_checkpoint else fixed / dataset / "spin_sync_direct/seed41001/best.pt",
        "spin_direct_no_memory": Path(args.no_memory_checkpoint.format(dataset=dataset)) if args.no_memory_checkpoint else fixed / dataset / "spin_direct_no_memory/seed41001/best.pt",
        "spin_adapted": root / "outputs/spin-comparison-20260914-v1" / dataset / "spin_adapted/seed41001/best.pt",
        "direct_sync": root / "outputs/llm-candidates-20260914-v1" / dataset / "direct_sync/seed41001/best.pt"}
    result = {}
    for method, path in paths.items():
        check_stop(args)
        path = (root / path).resolve()
        record = read(path.with_name("result.json"))
        identity = {"dataset": dataset, "variant": method, "seed": 41001}
        require(record, identity, method + " result")
        require(record["primary_eval"], {"role": "development_only", "mask_seed": 71001}, method + " evaluation")
        if record.get("epochs_completed", 0) < 1 or sha(path) != record["best_checkpoint_sha256"]:
            raise ValueError(f"{method} selected checkpoint is missing or changed")
        if method in METHODS[:2]:
            require(record, {"state": "complete"}, method + " completion")
            config_path = path.with_name("config.json")
            config = read(config_path)
            identity.update(protocol=config["protocol"], config_sha256=sha(config_path))
            require(record, identity, method + " config binding")
            require(config, {"dataset": dataset, "variant": method, "seed": 41001}, method + " config")
            data = config["data"]
        else:
            config_path = path.parents[3] / "registry.json"
            config = read(config_path)
            identity["registry_sha256"] = sha(config_path)
            require(record, identity, method + " registry binding")
            require(read(path.with_name("status.json")), {**identity, "state": "complete"}, method + " completion")
            data = config["data"][dataset]
        for relative, checksum in config["source_files"].items():
            if relative.endswith(".py") and sha(root / relative) != checksum:
                raise ValueError(f"Frozen source differs: {relative}")
        result[method] = {"path": path, "result": record, "identity": identity, "data": data,
            "files": {str(path): sha(path), str(path.with_name("result.json")): sha(path.with_name("result.json")),
                      str(config_path): sha(config_path)}}
    if any(item["data"] != result[METHODS[0]]["data"] for item in result.values()):
        raise ValueError("Models do not use the same frozen data protocol")
    return result


def load_model(method, record, flows, fit):
    import torch
    from experiments.sync_delta_v1.run import state_sha
    if method in METHODS[:2]:
        from experiments.spin_sync_unified_v1.models import build_model
        model = build_model(flows, init_seed=41001, variant=method)
    elif method == "spin_adapted":
        from experiments.spin_comparison_runtime_v1.run import build_model
        model = build_model(flows, init_seed=41001)
    else:
        from experiments.llm_candidates_v1.models import build_model
        model = build_model(flows, "direct_sync", init_seed=41001)
    scaler = {}
    if method != "direct_sync":
        model.configure_fit_statistics(fit)
        model.set_execution(node_chunk=32, gradient_checkpointing=False)
        scaler = {name: getattr(model, name).clone() for name in ("fit_mean", "fit_scale", "fit_configured")}
    saved = torch.load(record["path"], map_location="cpu", weights_only=False)
    require(saved, record["identity"], method + " checkpoint")
    expected = record["result"]
    if saved["epoch"] != expected["best_epoch"] or saved["score"] != expected["primary_eval"]["selection_score"]:
        raise ValueError(f"{method} checkpoint is not its completed selected epoch")
    model.load_state_dict(saved["model"], strict=True)
    if (state_sha(model.state_dict()) != expected["best_state_sha256"]
            or sum(p.numel() for p in model.parameters()) != expected["parameter_count"]
            or any(not torch.equal(getattr(model, key), value) for key, value in scaler.items())):
        raise ValueError(f"{method} model state/parameter count/fit scaler differs")
    return model.float().eval().requires_grad_(False)


def benchmark(args, output):
    import numpy as np
    import torch
    from experiments.sync_delta_v1 import engine
    from experiments.sync_delta_v1.data import CONDITIONS, _arrays_sha256, array_sha256, load_fit_windows, make_mask_family, window_starts
    check_stop(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Use one authorized idle CUDA device")
    engine.seed_runtime(41001)  # FP32 deterministic execution, TF32 disabled.
    output.update(device=str(device), device_name=torch.cuda.get_device_name(device),
        runtime={"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
        source_files={str(Path(__file__).resolve()): sha(__file__), **{str(Path(module.__file__).resolve()): sha(module.__file__)
            for module in (engine, sys.modules["experiments.sync_delta_v1.data"])}})
    for dataset in args.datasets:
        artifacts = records(args, dataset)
        values, starts, fit, neighbors_cpu = load_fit_windows(dataset, indices=tuple(range(8)))
        fit_arrays = {key: fit[key] for key in ("mu", "sigma", "scale")}
        fit_arrays.update(C=np.asarray(fit["C"], dtype="<f8"), global_scale=np.asarray(fit["global_scale"], dtype="<f8"))
        fit_identity = {"stats_sha256": _arrays_sha256(fit_arrays), "neighbors_sha256": array_sha256(neighbors_cpu),
                        "train_starts_sha256": array_sha256(window_starts(dataset, "fit"))}
        if any(artifacts[METHODS[0]]["data"][key] != value for key, value in fit_identity.items()):
            raise ValueError("Fit statistics, graph, or window starts differ")
        flows = values.shape[-1]
        families = [make_mask_family(flows, int(start), 71001, dataset=dataset) for start in starts]
        masks = {condition: np.stack([family.masks[condition] for family in families]) for condition in CONDITIONS}
        if any(int(mask.sum()) != 8 * 10 * flows for mask in masks.values()):
            raise ValueError("Mask does not have the original 20% observation budget")
        stats = {key: torch.as_tensor(fit[key], dtype=torch.float32, device=device) for key in ("mu", "scale")}
        neighbors = torch.as_tensor(neighbors_cpu, dtype=torch.long, device=device)
        dataset_result = {"fit_identity": fit_identity, "window_indices": list(range(8)), "window_starts": starts.tolist(),
            "values_sha256": array_sha256(values), "shape": list(values.shape), "mask_seed": 71001,
            "mask_sha256": {key: array_sha256(value) for key, value in masks.items()}, "models": {}, "measurements": []}
        output["datasets"][dataset] = dataset_result
        for method in METHODS:
            check_stop(args)
            record = artifacts[method]
            model = load_model(method, record, flows, fit).to(device)
            dataset_result["models"][method] = {"checkpoint_path": str(record["path"]), "checkpoint_sha256": sha(record["path"]),
                "bound_files": record["files"], "best_epoch": record["result"]["best_epoch"], "seed": 41001,
                "parameters": sum(p.numel() for p in model.parameters()), "internal_node_chunk": 32 if method != "direct_sync" else None}
            for batch in (1, 8):
                for condition in CONDITIONS:
                    def once():
                        return engine.predict_windows(model, values[:batch], masks[condition][:batch], starts[:batch], stats, neighbors,
                            dataset=dataset, physical_batch=batch, target_block=flows)
                    for _ in range(WARMUPS):
                        check_stop(args)
                        once()
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    resident = torch.cuda.memory_allocated(device)
                    elapsed = []
                    for _ in range(REPEATS):
                        check_stop(args)
                        torch.cuda.synchronize(device)
                        begin = time.perf_counter()
                        final, raw = once()
                        torch.cuda.synchronize(device)
                        elapsed.append(1000 * (time.perf_counter() - begin))
                    peak = torch.cuda.max_memory_allocated(device)
                    if (not np.isfinite(final).all() or np.any(final < 0)
                            or not np.array_equal(final[masks[condition][:batch]], values[:batch][masks[condition][:batch]])):
                        raise ValueError("Inference output finiteness/clamp/observed copy differs")
                    row = {"method": method, "batch": batch, "condition": condition, "warmups": WARMUPS, "repeats": REPEATS,
                        "median_ms_per_batch": float(np.median(elapsed)), "p95_ms_per_batch": float(np.percentile(elapsed, 95)),
                        "median_ms_per_window": float(np.median(elapsed)) / batch, "p95_ms_per_window": float(np.percentile(elapsed, 95)) / batch,
                        "samples_ms_per_batch": elapsed, "resident_allocated_bytes": resident,
                        "max_memory_allocated_bytes_including_resident_model": peak, "peak_above_resident_bytes": peak - resident}
                    dataset_result["measurements"].append(row)
                    write(args.output, output)
                    print(json.dumps({"dataset": dataset, **{k: v for k, v in row.items() if k != "samples_ms_per_batch"}}), flush=True)
                    del final, raw
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for record in artifacts.values():
            if any(sha(path) != checksum for path, checksum in record["files"].items()):
                raise ValueError("Checkpoint/config/result changed during benchmarking")
        del stats, neighbors
        gc.collect()
        torch.cuda.empty_cache()
    check_stop(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--datasets", nargs="+", choices=("abilene", "geant"), default=["abilene", "geant"])
    parser.add_argument("--full-checkpoint", help="Optional best.pt path; {dataset} is expanded for each WAN")
    parser.add_argument("--no-memory-checkpoint", help="Optional best.pt path; {dataset} is expanded for each WAN")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time() + 60:
        parser.error("A future authorized-allocation deadline with more than 60 seconds remaining is required")
    args.repo_root, args.output = args.repo_root.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Use a new output JSON; existing files are never overwritten")
    sys.path.insert(0, str(args.repo_root))
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    output = {"state": "running", "role": "fit_only_inference_benchmark", "datasets": {}, "started_unix": time.time(),
        "deadline_unix": args.deadline_unix, "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": {"precision": "FP32; TF32 disabled", "batches": [1, 8], "full_target_graph": True,
            "mask_observation_fraction": .2, "fit_input": "first 8 fit windows; B1 is the first of those windows",
            "warmups_per_condition_batch": WARMUPS, "repeats_per_condition_batch": REPEATS,
            "timing": "engine.predict_windows end-to-end; perf_counter with CUDA synchronize before/after",
            "memory": "allocated bytes including current resident model, fit statistics and neighbor table; other models unloaded",
            "per_window_latency": "batch latency divided by batch size, not an independently timed single window",
            "excluded": ["checkpoint loading", "fit statistics/graph setup", "mask generation"], "accuracy_scores_computed": False}}
    try:
        benchmark(args, output)
        output["state"] = "complete"
    except Stopped as error:
        output.update(state="stopped", reason=str(error))
    except BaseException as error:
        output.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output["finished_unix"] = time.time()
        write(args.output, output)


if __name__ == "__main__":
    main()
