"""Two Direct-only ablations; reuse frozen data and training, never rerun prior models."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import time
import traceback

import numpy as np
import torch

from experiments.sync_delta_v1.data import CONDITIONS, array_sha256, training_epoch
from experiments.sync_delta_v1.engine import (
    device_inputs, evaluate, latency_probe, make_dev_masks, score_predictions,
    seed_runtime, train_step,
)
from experiments.sync_delta_v1.run import (
    capture_rng, cpu_state, emit, file_sha, load_epoch_masks, restore_rng,
    save_torch, state_sha, utc, verified_bundle, verify_registry as verify_parent,
    write_json,
)
from experiments.direct_only_v1.models import build_model as build_direct_only
from experiments.llm_candidates_runtime_v1.run import verify_registry as verify_full_registry

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("abilene", "geant")
VARIANT = "direct_only"
VARIANTS = ALL_VARIANTS = ("direct_only",)
SEED = 41001
PROTOCOL = "direct-only-development-20260914-v1"
FULL_SHA = "ca2096f69fc57abc218e29d51f5d5defe80ed989f83b51d049947379ecaee194"
DEFAULT_FULL = ROOT / "outputs/llm-candidates-20260914-v1"
PARENT_SHA = "f7dae0dfac2a776e75d62e57dade164d60b4fca49d329827018740dc071d3949"
ARI_SHA = "fc27d1ff237b4e240531ae59832d6947b0ee1ee0cf01f935cb35cc381a30a7d0"
DEFAULT_PARENT = ROOT / "outputs/sync-delta-pilot-20260913-v11"
DEFAULT_ARI = ROOT / "outputs/ari-comparison-20260913-v1"
DEFAULT_OUTPUT = ROOT / "outputs/direct-only-20260914-v1"
CPU_CHECK = ROOT / "analysis/direct_only_20260914/CPU_CHECKS.json"
DEADLINE_UNIX = None  # New authorization; bounded by two jobs, 120 epochs and existing early stop.
STOP_REQUESTED = False
TRAINING = {
    "max_epochs": 120, "effective_batch": 8, "physical_batch": 8,
    "target_block": "full", "evaluation_batch": 8, "evaluation_target_block": "full",
    "optimizer": "AdamW", "lr": 1e-3, "betas": [0.9, 0.999],
    "weight_decay": 1e-4, "gradient_clip": 1.0, "scheduler": None,
    "initialization_seed": 41001, "data_order_seed": 51001,
    "training_mask_seed": 61001, "selection_mask_seed": 71001,
    "early_stop_patience_starts_epoch": 16, "early_stop_patience": 10,
    "earliest_stop_epoch": 25, "improvement": "strictly smaller selection score",
    "loss": "unclamped raw missing absolute sum / (fit C * full effective batch missing count)",
    "selection": "equal mean of 3 condition global missing-only NMAE",
    "precision": "FP32; TF32 disabled", "latency_warmup": 20, "latency_repeats": 100,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def runtime():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda}


def sources():
    paths = sorted((ROOT / "experiments/direct_only_v1").glob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("__init__.py"),
              ROOT / "experiments/llm_candidates_v1/direct.py",
              ROOT / "experiments/llm_candidates_v1/models.py",
              ROOT / "experiments/llm_candidates_runtime_v1/run.py",
              ROOT / "plan/CALIBTM_DIRECT_ONLY_EXECUTION_2026-09-14_CN.md",
              ROOT / "setup/ksc_direct_only_v1.sh"]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def build_model(num_flows, init_seed):
    return build_direct_only(num_flows, init_seed=init_seed)


def initial_for(record, dataset):
    return record["initializations"][dataset][VARIANT][str(SEED)]


def stopping():
    return STOP_REQUESTED


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def paired(candidate, reference):
    key = lambda r: (r["condition"], r["window_start"], r["cohort"])
    left = {key(r): r for r in candidate["rows"]}
    right = {key(r): r for r in reference["rows"]}
    if len(left) != len(candidate["rows"]) or len(right) != len(reference["rows"]) or left.keys() != right.keys():
        raise RuntimeError("physical scoring windows differ")
    for identity in left:
        for field in ("abs_truth", "sq_truth", "count"):
            if left[identity][field] != right[identity][field]:
                raise RuntimeError(f"scoring denominator differs: {identity}/{field}")


def data_identity(bundle, masks):
    fields = ("data_sha256", "stats_sha256", "neighbors_sha256", "train_starts_sha256", "dev_starts_sha256")
    return {**{k: bundle.metadata[k] for k in fields}, "flows": bundle.flows,
            "train_window_count": len(bundle.train_windows), "dev_window_count": len(bundle.dev_windows),
            "fit_C": bundle.C, "mask_array_sha256": {c: array_sha256(masks[0][c]) for c in CONDITIONS}}


@contextmanager
def job_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare(args):
    out = args.output.resolve()
    if (out / "registry.json").exists():
        verify_registry(out)
        emit("already_prepared", output=str(out))
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("preparation needs an empty new output directory")
    cpu_check = read_json(CPU_CHECK)
    if cpu_check.get("status") != "passed" or cpu_check.get("variant") != "direct_only":
        raise RuntimeError("Direct-only CPU correctness checks missing or failed")
    parent, full = args.parent.resolve(), args.full_parent.resolve()
    if file_sha(parent / "registry.json") != PARENT_SHA or file_sha(full / "registry.json") != FULL_SHA:
        raise RuntimeError("parent/full registry mismatch")
    original = verify_parent(parent)
    full_record = verify_full_registry(full)
    data, references, initializations = {}, {}, {}
    seed_runtime(SEED)
    for dataset in DATASETS:
        bundle = verified_bundle(original, dataset)
        cache = make_dev_masks(bundle, 71001)
        data[dataset] = data_identity(bundle, cache)
        if data[dataset] != full_record["data"][dataset]:
            raise RuntimeError("data/masks/windows differ from full Direct+Sync")
        references[dataset] = dict(full_record["references"][dataset])
        direct_path = full / dataset / "direct_sync/seed41001/result.json"
        direct_result = read_json(direct_path)
        if (direct_result["registry_sha256"] != FULL_SHA or direct_result["dataset"] != dataset
                or direct_result["variant"] != "direct_sync" or direct_result["seed"] != SEED
                or direct_result["primary_eval"]["role"] != "development_only"
                or direct_result["primary_eval"]["mask_seed"] != 71001):
            raise RuntimeError("full reference identity differs")
        if file_sha(direct_path.parent / "best.pt") != direct_result["best_checkpoint_sha256"]:
            raise RuntimeError("full reference checkpoint changed")
        references[dataset]["direct_sync"] = {"path": str(direct_path), "sha256": file_sha(direct_path),
            "selection_score": direct_result["primary_eval"]["selection_score"]}
        linear = read_json(references[dataset]["linear"]["path"])
        rows = []
        for condition in CONDITIONS:
            rows.extend(score_predictions(bundle, condition, np.zeros_like(bundle.dev_windows),
                                           cache[0][condition], cache[1]))
        paired({"rows": rows}, linear)
        paired(direct_result["primary_eval"], linear)
        model = build_model(bundle.flows, init_seed=SEED)
        state = cpu_state(model)
        if sum(p.numel() for p in model.parameters()) != cpu_check["initialization"][str(bundle.flows)][str(SEED)]["parameters"]:
            raise RuntimeError("CPU checked architecture parameter count differs")
        full_initial = full_record["initializations"][dataset]["direct_sync"][str(SEED)]
        full_state = torch.load(full / full_initial["file"], map_location="cpu")
        if not all(name in full_state and torch.equal(value, full_state[name]) for name, value in state.items()):
            raise RuntimeError("retained parameter initialization differs from full Direct+Sync")
        removed = sorted(set(full_state) - set(state))
        if not removed or any(not (name.startswith("key.") or name.startswith("value.")) for name in removed):
            raise RuntimeError("unexpected architecture ablation")
        relative = f"initializations/{dataset}/direct_only/seed{SEED}.pt"
        save_torch(out / relative, state)
        initializations[dataset] = {"direct_only": {str(SEED): {
            "file": relative, "sha256": file_sha(out / relative), "state_sha256": state_sha(state),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "retained_initialization_exact": True, "removed_state_keys": removed}}}
    record = {"protocol": PROTOCOL, "created": utc(), "parent": str(parent),
        "parent_registry_sha256": PARENT_SHA, "full_parent": str(full), "full_registry_sha256": FULL_SHA,
        "training": TRAINING, "runtime": runtime(), "source_files": sources(), "data": data,
        "references": references, "initializations": initializations, "deadline_unix": DEADLINE_UNIX,
        "scope": "one Direct-only architecture ablation x two WANs x seed41001; no old reruns",
        "maximum_formal_new_jobs": 2, "role": "development_only",
        "cpu_correctness": {"path": str(CPU_CHECK), "sha256": file_sha(CPU_CHECK)},
        "authorization": "User authorized next-step work and GPUs on 2026-09-14; prior eight-hour window is not reused"}
    write_json(out / "registry.json", record)
    (out / "registry.sha256").write_text(file_sha(out / "registry.json") + "\n")
    emit("prepared", output=str(out), registry_sha256=file_sha(out / "registry.json"))


def verify_registry(out):
    if file_sha(out / "registry.json") != (out / "registry.sha256").read_text().strip():
        raise RuntimeError("registry changed")
    record = read_json(out / "registry.json")
    if record["protocol"] != PROTOCOL or record["training"] != TRAINING or record["source_files"] != sources():
        raise RuntimeError("training/source changed")
    if record["runtime"] != runtime() or record["deadline_unix"] is not None:
        raise RuntimeError("runtime/scope changed")
    if (file_sha(record["cpu_correctness"]["path"]) != record["cpu_correctness"]["sha256"]
            or read_json(record["cpu_correctness"]["path"]).get("status") != "passed"):
        raise RuntimeError("CPU correctness record changed")
    if file_sha(Path(record["parent"]) / "registry.json") != PARENT_SHA:
        raise RuntimeError("parent registry changed")
    verify_parent(Path(record["parent"]))
    if file_sha(Path(record["full_parent"]) / "registry.json") != FULL_SHA:
        raise RuntimeError("full registry changed")
    verify_full_registry(Path(record["full_parent"]))
    for dataset in DATASETS:
        for ref in record["references"][dataset].values():
            if file_sha(ref["path"]) != ref["sha256"]:
                raise RuntimeError("reference result changed")
        initial = initial_for(record, dataset)
        if file_sha(out / initial["file"]) != initial["sha256"]:
            raise RuntimeError("initialization changed")
    return record


def setup_run(args, record):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered real-data execution requires CUDA")
    seed_runtime(SEED)
    original = read_json(Path(record["parent"]) / "registry.json")
    bundle = verified_bundle(original, args.dataset)
    cache = make_dev_masks(bundle, 71001)
    if data_identity(bundle, cache) != record["data"][args.dataset]:
        raise RuntimeError("data/masks/windows/statistics changed")
    initial = initial_for(record, args.dataset)
    model = build_model(bundle.flows, init_seed=SEED).to(device)
    if state_sha(model.state_dict()) != initial["state_sha256"]:
        raise RuntimeError("seeded model initialization changed")
    model.load_state_dict(torch.load(args.output / initial["file"], map_location="cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["lr"],
        betas=tuple(TRAINING["betas"]), weight_decay=TRAINING["weight_decay"])
    stats, neighbors = device_inputs(bundle, device)
    return device, bundle, cache, model, optimizer, stats, neighbors


def epoch_schedule(record, bundle, epoch):
    shared = Path(record["parent"]) / bundle.dataset / "shared"
    path = shared / "masks" / f"masks_train_seed61001_epoch{epoch:03d}.npz"
    # Guard both paths: the old helper must never write into the parent output.
    if path.exists() and path.with_suffix(".json").exists():
        order, masks = load_epoch_masks(bundle, shared, epoch)
        origin = {"path": str(path), "sha256": file_sha(path)}
    else:
        schedule = training_epoch(bundle, epoch, data_order_seed=51001, mask_seed=61001)
        order, masks = schedule.order, schedule.masks
        origin = {"generated_with_frozen_data_module": True}
    return order, masks, {**origin, "epoch_zero_based": epoch,
        "order_sha256": array_sha256(order), "mask_sha256": array_sha256(masks)}


def run_step(model, optimizer, bundle, stats, neighbors, order, masks, offset, epoch):
    indices = order[offset:offset + TRAINING["effective_batch"]]
    return train_step(model, bundle.train_windows[indices], masks[offset:offset + len(indices)],
        bundle.train_starts[indices], stats, neighbors, bundle.C, optimizer,
        dataset=bundle.dataset, epoch=epoch, physical_batch=TRAINING["physical_batch"],
        target_block=bundle.flows)


def run_eval(model, bundle, stats, neighbors, cache):
    result = evaluate(model, bundle, stats, neighbors, seed=71001,
                    physical_batch=TRAINING["evaluation_batch"], target_block=bundle.flows,
                    mask_cache=cache)
    result["mask_array_sha256"] = {c: array_sha256(cache[0][c]) for c in CONDITIONS}
    return result



def load_resume(path, identity, model, optimizer):
    checkpoint = torch.load(path, map_location="cpu")
    for field, value in identity.items():
        if checkpoint[field] != value:
            raise RuntimeError(f"resume identity mismatch: {field}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng"])
    return checkpoint


def identity_for(args, record):
    return {"registry_sha256": file_sha(args.output / "registry.json"),
            "dataset": args.dataset, "variant": VARIANT, "seed": SEED,
            "initial_state_sha256": initial_for(record, args.dataset)["state_sha256"]}


def smoke(args):
    record = verify_registry(args.output)
    destination = args.output / "preflight" / args.dataset / VARIANT
    with job_lock(destination / "active.lock"):
        identity = identity_for(args, record)
        existing = destination / "REAL_DATA_SMOKE.json"
        if existing.exists():
            if read_json(existing)["registry_sha256"] != identity["registry_sha256"] or not read_json(existing)["passed"]:
                raise RuntimeError("preflight identity mismatch")
            emit("smoke_already_passed", dataset=args.dataset)
            return
        device, bundle, cache, model, optimizer, stats, neighbors = setup_run(args, record)
        order, masks, schedule_id = epoch_schedule(record, bundle, 0)
        generated = training_epoch(bundle, 0)
        if not np.array_equal(order, generated.order) or not np.array_equal(masks, generated.masks):
            raise RuntimeError("old training masks/order differ from frozen generator")
        torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        first = run_step(model, optimizer, bundle, stats, neighbors, order, masks, 0, 0)
        save_torch(destination / "resume_probe.pt", {**identity, "model": cpu_state(model),
            "optimizer": optimizer.state_dict(), "rng": capture_rng()})
        second = run_step(model, optimizer, bundle, stats, neighbors, order, masks, 8, 0)
        expected = state_sha(model.state_dict())
        expected_rng = capture_rng()
        rejected = False
        try:
            load_resume(destination / "resume_probe.pt", {**identity, "dataset": "wrong"}, model, optimizer)
        except RuntimeError:
            rejected = True
        if not rejected:
            raise RuntimeError("resume accepted wrong dataset")
        load_resume(destination / "resume_probe.pt", identity, model, optimizer)
        replay = run_step(model, optimizer, bundle, stats, neighbors, order, masks, 8, 0)
        if state_sha(model.state_dict()) != expected or second != replay:
            raise RuntimeError("checkpoint continuation differs from uninterrupted second update")
        actual_rng = capture_rng()
        if not torch.equal(expected_rng["torch"], actual_rng["torch"]) or any(
                not torch.equal(a, b) for a, b in zip(expected_rng["cuda"], actual_rng["cuda"])):
            raise RuntimeError("checkpoint RNG continuation changed")
        torch.cuda.synchronize(device)
        step_seconds = (time.perf_counter() - begin) / 3
        peak = torch.cuda.max_memory_allocated(device)
        evaluation = run_eval(model, bundle, stats, neighbors, cache)
        paired(evaluation, read_json(record["references"][args.dataset]["linear"]["path"]))
        context_probe = None
        report = {**identity, "passed": True, "finished": utc(),
            "role": "real_data_runner_smoke_not_formal_result", "formal_initialization_reused": False,
            "unique_updates": 2, "replayed_updates": 1, "first_step": first,
            "exact_checkpoint_continuation": True, "wrong_resume_identity_rejected": True,
            "original_epoch0_masks_and_order_exact": True, "paired_dev_denominators_exact": True,
            "dev_windows": len(bundle.dev_windows), "schedule": schedule_id,
            "physical_batch": TRAINING["physical_batch"], "target_block": bundle.flows,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "seconds_per_step_including_probe_io": step_seconds,
            "evaluation_seconds": evaluation["elapsed_seconds"], "peak_training_allocated_bytes": peak,
            "routing_context_probe": context_probe,
            "device_name": torch.cuda.get_device_name(device), "runtime": runtime()}
        write_json(existing, report)
        emit("smoke_passed", **report)


def train(args):
    record = verify_registry(args.output)
    identity = identity_for(args, record)
    report_path = args.output / "preflight" / args.dataset / VARIANT / "REAL_DATA_SMOKE.json"
    check_job_allowed(args)
    smoke_report = read_json(report_path)
    if not smoke_report["passed"] or smoke_report["registry_sha256"] != identity["registry_sha256"]:
        raise RuntimeError("runner smoke missing or does not match registry")
    job = args.output / args.dataset / VARIANT / f"seed{SEED}"
    with job_lock(job / "active.lock"):
        if (job / "result.json").exists():
            result = read_json(job / "result.json")
            if any(result[k] != v for k, v in identity.items()):
                raise RuntimeError("existing result identity differs")
            emit("already_complete", dataset=args.dataset, score=result["primary_eval"]["selection_score"])
            return
        if (job / "run_info.json").exists() and not args.resume:
            raise RuntimeError("existing run requires --resume")
        try:
            train_locked(args, record, identity, job, report_path)
        except BaseException as error:
            write_json(job / "failure.json", {**identity, "updated": utc(), "error": str(error),
                "traceback": traceback.format_exc(), "resume_from": "last.pt or registered initialization"})
            write_json(job / "status.json", {**identity, "state": "failed", "updated": utc(), "error": str(error)})
            raise


def train_locked(args, record, identity, job, report_path):
    device, bundle, cache, model, optimizer, stats, neighbors = setup_run(args, record)
    info = {**identity, "training": TRAINING, "device_name": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(),
            "hostname": socket.gethostname(), "deadline_unix": DEADLINE_UNIX,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "preflight_sha256": file_sha(report_path), "started_or_resumed": utc()}
    if (job / "run_info.json").exists():
        previous = read_json(job / "run_info.json")
        for key, value in identity.items():
            if previous[key] != value:
                raise RuntimeError("existing run metadata identity mismatch")
    history, best_score, best_epoch, bad_epochs = [], math.inf, 0, 0
    best_state, best_eval = None, None
    stopped_early, training_seconds, elapsed_seconds, peak_training = False, 0.0, 0.0, 0
    partial = None
    if args.resume and (job / "last.pt").exists():
        last = load_resume(job / "last.pt", identity, model, optimizer)
        history, best_score, best_epoch = last["history"], last["best_score"], last["best_epoch"]
        bad_epochs, stopped_early = last["bad_epochs"], last["stopped_early"]
        best_state, best_eval = last["best_model"], last["best_eval"]
        training_seconds, elapsed_seconds = last["training_seconds"], last["elapsed_seconds"]
        peak_training = last["peak_training_allocated_bytes"]
        partial = last.get("partial_epoch")
        del last
    elif (job / "history.json").exists() or (job / "best.pt").exists():
        raise RuntimeError("trained artifacts exist without last checkpoint")
    write_json(job / "run_info.json", info)
    emit("training_started", **info, epochs_completed=len(history))
    session_begin, elapsed_before = time.perf_counter(), elapsed_seconds
    for epoch in range(len(history), TRAINING["max_epochs"]):
        if stopped_early:
            break
        epoch_begin = time.perf_counter()
        order, masks, schedule_id = epoch_schedule(record, bundle, epoch)
        torch.cuda.reset_peak_memory_stats(device)
        if partial is not None and partial["epoch"] != epoch:
            raise RuntimeError("partial epoch identity changed")
        losses = list(partial["losses"]) if partial else []
        norms = list(partial["norms"]) if partial else []
        next_offset = partial["next_offset"] if partial else 0
        elapsed_partial = partial["training_seconds"] if partial else 0.0
        write_json(job / "status.json", {**identity, "state": "training", "updated": utc(),
            "epoch": epoch + 1, "epochs_completed": len(history), "update": 0,
            "updates_per_epoch": 64, "best_epoch": best_epoch,
            "best_score": best_score if math.isfinite(best_score) else None})
        begin = time.perf_counter()
        for offset in range(next_offset, len(order), TRAINING["effective_batch"]):
            if stopping():
                torch.cuda.synchronize(device)
                partial_training = elapsed_partial + time.perf_counter() - begin
                peak_training = max(peak_training, torch.cuda.max_memory_allocated(device))
                save_torch(job / "last.pt", {**identity, "model": cpu_state(model),
                    "optimizer": optimizer.state_dict(), "rng": capture_rng(), "history": history,
                    "best_model": best_state, "best_eval": best_eval, "best_score": best_score,
                    "best_epoch": best_epoch, "bad_epochs": bad_epochs, "stopped_early": stopped_early,
                    "training_seconds": training_seconds,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - session_begin,
                    "peak_training_allocated_bytes": peak_training,
                    "partial_epoch": {"epoch": epoch, "next_offset": offset,
                        "losses": losses, "norms": norms, "training_seconds": partial_training}})
                write_json(job / "status.json", {**identity, "state": "paused_by_signal", "updated": utc(),
                    "epochs_completed": len(history), "partial_epoch": epoch + 1,
                    "partial_updates": offset // TRAINING["effective_batch"], "best_epoch": best_epoch,
                    "best_score": best_score if math.isfinite(best_score) else None,
                    "resume_available": True, "eligible_as_completed_result": False})
                emit("paused_by_signal", dataset=args.dataset, variant=VARIANT, seed=SEED)
                return
            step = run_step(model, optimizer, bundle, stats, neighbors, order, masks, offset, epoch)
            losses.append(step["loss"])
            norms.append(step["gradient_norm"])
            update = offset // TRAINING["effective_batch"] + 1
            if update == 1 or update % 16 == 0:
                progress = {**identity, "state": "training", "updated": utc(), "epoch": epoch + 1,
                    "epochs_completed": len(history), "update": update, "updates_per_epoch": 64,
                    "mean_train_loss": float(np.mean(losses)), "best_epoch": best_epoch,
                    "best_score": best_score if math.isfinite(best_score) else None}
                write_json(job / "status.json", progress)
                emit("training_progress", **progress)
        torch.cuda.synchronize(device)
        epoch_training = elapsed_partial + time.perf_counter() - begin
        partial = None
        training_seconds += epoch_training
        peak_training = max(peak_training, torch.cuda.max_memory_allocated(device))
        evaluation = run_eval(model, bundle, stats, neighbors, cache)
        score = evaluation["selection_score"]
        improved = score < best_score
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch + 1, 0
            best_state, best_eval = cpu_state(model), evaluation
        elif epoch + 1 >= TRAINING["early_stop_patience_starts_epoch"]:
            bad_epochs += 1
        else:
            bad_epochs = 0
        stopped_early = epoch + 1 >= TRAINING["earliest_stop_epoch"] and bad_epochs >= TRAINING["early_stop_patience"]
        elapsed_seconds = elapsed_before + time.perf_counter() - session_begin
        row = {"epoch": epoch + 1, "updates": (epoch + 1) * 64,
            "train_loss": float(np.mean(losses)), "mean_gradient_norm": float(np.mean(norms)),
            "selection_score": score, "best_epoch": best_epoch, "best_score": best_score,
            "bad_epochs": bad_epochs, "training_seconds": epoch_training,
            "evaluation_seconds": evaluation["elapsed_seconds"],
            "epoch_seconds": time.perf_counter() - epoch_begin,
            "peak_training_allocated_bytes": peak_training,
            "condition_nmae": {c: evaluation["metrics"][c]["nmae"] for c in CONDITIONS},
            "schedule": schedule_id}
        history.append(row)
        save_torch(job / "last.pt", {**identity, "model": cpu_state(model),
            "optimizer": optimizer.state_dict(), "rng": capture_rng(), "history": history,
            "best_model": best_state, "best_eval": best_eval, "best_score": best_score,
            "best_epoch": best_epoch, "bad_epochs": bad_epochs, "stopped_early": stopped_early,
            "training_seconds": training_seconds, "elapsed_seconds": elapsed_seconds,
            "peak_training_allocated_bytes": peak_training})
        if improved:
            save_torch(job / "best.pt", {**identity, "model": best_state, "epoch": best_epoch, "score": best_score})
            write_json(job / "best_eval.json", best_eval)
        write_json(job / "evaluations" / f"epoch{epoch + 1:03d}.json", evaluation)
        write_json(job / "history.json", history)
        write_json(job / "status.json", {**identity, "state": "epoch_complete", "updated": utc(),
            "epochs_completed": epoch + 1, "best_epoch": best_epoch, "best_score": best_score,
            "last_score": score, "epoch_seconds": row["epoch_seconds"]})
        emit("epoch_complete", dataset=args.dataset, **{k: v for k, v in row.items() if k != "schedule"})
    if not history or best_state is None:
        raise RuntimeError("no committed trained epoch")
    if stopping():
        write_json(job / "status.json", {**identity, "state": "paused_before_final_verification",
            "updated": utc(), "epochs_completed": len(history), "best_epoch": best_epoch,
            "best_score": best_score, "eligible_as_completed_result": False})
        return
    # last.pt has the authoritative best copy if publication was interrupted.
    save_torch(job / "best.pt", {**identity, "model": best_state, "epoch": best_epoch, "score": best_score})
    model.load_state_dict(torch.load(job / "best.pt", map_location="cpu")["model"])
    del optimizer
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    final_eval = run_eval(model, bundle, stats, neighbors, cache)
    if stopping():
        write_json(job / "status.json", {**identity, "state": "paused_before_final_verification",
            "updated": utc(), "epochs_completed": len(history), "best_epoch": best_epoch,
            "best_score": best_score, "eligible_as_completed_result": False})
        return
    if final_eval["selection_score"] != best_score or final_eval["metrics"] != best_eval["metrics"]:
        raise RuntimeError("best checkpoint reload evaluation differs")
    paired(final_eval, read_json(record["references"][args.dataset]["linear"]["path"]))
    write_json(job / "best_eval.json", final_eval)
    write_json(job / "status.json", {**identity, "state": "measuring_latency", "updated": utc(),
        "epochs_completed": len(history), "best_epoch": best_epoch, "best_score": best_score})
    with torch.no_grad():
        latency = latency_probe(model, bundle, stats, neighbors, target_block=bundle.flows,
            warmup=TRAINING["latency_warmup"], repeats=TRAINING["latency_repeats"])
    write_json(job / "latency.json", latency)
    if stopping():
        write_json(job / "status.json", {**identity, "state": "paused_before_final_verification",
            "updated": utc(), "epochs_completed": len(history), "best_epoch": best_epoch,
            "best_score": best_score, "eligible_as_completed_result": False})
        return
    result = {**identity, "protocol": PROTOCOL, "role": "development_only",
        "epochs_completed": len(history), "optimizer_updates": len(history) * 64,
        "best_epoch": best_epoch, "training_cap": TRAINING["max_epochs"], "stopped_early": stopped_early,
        "best_in_last_five_at_cap": not stopped_early and best_epoch >= TRAINING["max_epochs"] - 4,
        "training_seconds": training_seconds, "elapsed_seconds": elapsed_before + time.perf_counter() - session_begin,
        "peak_training_allocated_bytes": peak_training, "parameter_count": info["parameter_count"],
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_state_sha256": state_sha(model.state_dict()), "best_checkpoint_sha256": file_sha(job / "best.pt"),
        "primary_eval": final_eval, "device_name": info["device_name"], "finished": utc()}
    write_json(job / "result.json", result)
    write_json(job / "status.json", {**identity, "state": "complete", "updated": utc(),
        "epochs_completed": len(history), "best_epoch": best_epoch, "best_score": best_score})
    emit("complete", dataset=args.dataset, score=best_score, epochs=len(history), best_epoch=best_epoch)


def check_job_allowed(args):
    if VARIANT != "direct_only" or SEED != 41001 or args.dataset not in DATASETS:
        raise RuntimeError("outside two registered Direct-only jobs")


def main():
    global VARIANT, SEED
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "smoke", "train"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        if command == "prepare":
            p.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
            p.add_argument("--full-parent", type=Path, default=DEFAULT_FULL)
        else:
            p.add_argument("--dataset", choices=DATASETS, required=True)
            p.add_argument("--variant", choices=ALL_VARIANTS, required=True)
            p.add_argument("--seed", type=int, choices=(41001,), default=41001)
            p.add_argument("--device", default="cuda:0")
        if command == "train":
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    VARIANT, SEED = getattr(args, "variant", "direct_only"), getattr(args, "seed", 41001)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    {"prepare": prepare, "smoke": smoke, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
