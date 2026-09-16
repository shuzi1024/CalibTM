"""Bounded overnight candidates; preserve the existing data and training protocol."""

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
from experiments.llm_candidates_v1.models import build_model as build_candidate, VARIANTS, ALL_VARIANTS

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("abilene", "geant")
VARIANT = None
SEED = 41001
PROTOCOL = "llm-candidates-development-20260914-v1"
PARENT_SHA = "f7dae0dfac2a776e75d62e57dade164d60b4fca49d329827018740dc071d3949"
ARI_SHA = "fc27d1ff237b4e240531ae59832d6947b0ee1ee0cf01f935cb35cc381a30a7d0"
DEFAULT_PARENT = ROOT / "outputs/sync-delta-pilot-20260913-v11"
DEFAULT_ARI = ROOT / "outputs/ari-comparison-20260913-v1"
DEFAULT_OUTPUT = ROOT / "outputs/llm-candidates-20260914-v1"
DEADLINE_UNIX = 1789347583.0  # 2026-09-14 00:59:43 UTC; eight hours from authorization check
STOP_REQUESTED = False
TRAINING = {
    "max_epochs": 120, "effective_batch": 8, "physical_batch": 8,
    "target_block": "full", "evaluation_batch": 8, "evaluation_target_block": "full",
    "optimizer": "AdamW", "lr": 1e-3, "betas": [0.9, 0.999],
    "weight_decay": 1e-4, "gradient_clip": 1.0, "scheduler": None,
    "initialization_seed": "job identity: 41001 or 41002", "data_order_seed": 51001,
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
    paths = sorted((ROOT / "experiments/llm_candidates_v1").glob("*.py"))
    paths += sorted(Path(__file__).parent.glob("*.py"))
    paths += [ROOT / "plan/CALIBTM_OVERNIGHT_2026-09-14_CN.md", ROOT / "setup/ksc_llm_candidates_v1.sh"]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def build_model(num_flows, init_seed):
    return build_candidate(num_flows, VARIANT, init_seed=init_seed)


def initial_for(record, dataset):
    return record["initializations"][dataset][VARIANT][str(SEED)]


def stopping():
    # Keep a minute for synchronization and a recoverable checkpoint.
    return STOP_REQUESTED or time.time() >= DEADLINE_UNIX - 60


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
    parent, ari_parent = args.parent.resolve(), args.ari_parent.resolve()
    if file_sha(parent / "registry.json") != PARENT_SHA or file_sha(ari_parent / "registry.json") != ARI_SHA:
        raise RuntimeError("reference registry mismatch")
    original = verify_parent(parent)
    ari_registry = read_json(ari_parent / "registry.json")
    data, references, initializations = {}, {}, {}
    seed_runtime(41001)
    for dataset in DATASETS:
        bundle = verified_bundle(original, dataset)
        cache = make_dev_masks(bundle, 71001)
        data[dataset] = data_identity(bundle, cache)
        if data[dataset]["mask_array_sha256"] != ari_registry["data"][dataset]["mask_array_sha256"]:
            raise RuntimeError("development masks differ from ARI comparison")
        paths = {"sync_delta": parent / dataset / "sync_delta/seed41001/result.json",
                 "linear": parent / dataset / "linear_seed71001.json",
                 "ari": ari_parent / dataset / "ari/seed41001/result.json"}
        evaluations, references[dataset] = {}, {}
        for name, path in paths.items():
            result = read_json(path)
            evaluation = result if name == "linear" else result["primary_eval"]
            if evaluation["mask_seed"] != 71001 or evaluation["role"] != "development_only":
                raise RuntimeError("reference mask/role mismatch")
            if name != "linear":
                expected_sha = ARI_SHA if name == "ari" else PARENT_SHA
                if result["registry_sha256"] != expected_sha or result["dataset"] != dataset or result["seed"] != 41001:
                    raise RuntimeError("reference result identity mismatch")
                if file_sha(path.parent / "best.pt") != result["best_checkpoint_sha256"]:
                    raise RuntimeError("reference best checkpoint changed")
            evaluations[name] = evaluation
            references[dataset][name] = {"path": str(path), "sha256": file_sha(path),
                                         "selection_score": evaluation["selection_score"]}
        rows = []
        for condition in CONDITIONS:
            rows.extend(score_predictions(bundle, condition, np.zeros_like(bundle.dev_windows),
                                           cache[0][condition], cache[1]))
        for evaluation in evaluations.values():
            paired({"rows": rows}, evaluation)
        initializations[dataset] = {}
        for variant in ALL_VARIANTS:
            initializations[dataset][variant] = {}
            for seed in (41001, 41002):
                model = build_candidate(bundle.flows, variant, init_seed=seed)
                state = cpu_state(model)
                relative = f"initializations/{dataset}/{variant}/seed{seed}.pt"
                save_torch(out / relative, state)
                initializations[dataset][variant][str(seed)] = {
                    "file": relative, "sha256": file_sha(out / relative),
                    "state_sha256": state_sha(state),
                    "parameter_count": sum(p.numel() for p in model.parameters())}
    old_repeat = ROOT / "outputs/sync-delta-followup-20260913-seed41002/geant/sync_delta/seed41002/result.json"
    old_repeat_result = read_json(old_repeat)
    if (old_repeat_result.get("dataset") != "geant" or old_repeat_result.get("variant") != "sync_delta"
            or old_repeat_result.get("seed") != 41002):
        raise RuntimeError("existing reference repetition identity differs")
    paired(old_repeat_result["primary_eval"], read_json(references["geant"]["linear"]["path"]))
    if file_sha(old_repeat.parent / "best.pt") != old_repeat_result["best_checkpoint_sha256"]:
        raise RuntimeError("existing GEANT repetition checkpoint changed")
    record = {"protocol": PROTOCOL, "created": utc(),
              "parent": str(parent), "parent_registry_sha256": PARENT_SHA,
              "ari_parent": str(ari_parent), "ari_registry_sha256": ARI_SHA,
              "training": TRAINING, "runtime": runtime(), "source_files": sources(),
              "data": data, "references": references, "initializations": initializations,
              "scope": "three independent candidates x two WANs; at most one winner repeated; one missing Sync reference repeat",
              "screen_variants": list(VARIANTS), "maximum_formal_new_jobs": 9,
              "existing_geant_sync_repeat": {"path": str(old_repeat), "sha256": file_sha(old_repeat),
                  "seed": 41002, "selection_score": old_repeat_result["primary_eval"]["selection_score"]},
              "deadline_unix": DEADLINE_UNIX, "role": "development_only"}
    write_json(out / "registry.json", record)
    (out / "registry.sha256").write_text(file_sha(out / "registry.json") + "\n")
    emit("prepared", output=str(out), registry_sha256=file_sha(out / "registry.json"))


def verify_registry(out):
    if file_sha(out / "registry.json") != (out / "registry.sha256").read_text().strip():
        raise RuntimeError("training registry digest changed")
    record = read_json(out / "registry.json")
    if record["protocol"] != PROTOCOL or record["training"] != TRAINING or record["source_files"] != sources():
        raise RuntimeError("training configuration/source changed")
    if record["runtime"] != runtime() or record["deadline_unix"] != DEADLINE_UNIX:
        raise RuntimeError("runtime/deadline identity changed")
    if file_sha(Path(record["parent"]) / "registry.json") != PARENT_SHA:
        raise RuntimeError("parent registry changed")
    verify_parent(Path(record["parent"]))
    if file_sha(Path(record["ari_parent"]) / "registry.json") != ARI_SHA:
        raise RuntimeError("ARI registry changed")
    repeat = record["existing_geant_sync_repeat"]
    if file_sha(repeat["path"]) != repeat["sha256"]:
        raise RuntimeError("existing GEANT reference repetition changed")
    for dataset in DATASETS:
        for ref in record["references"][dataset].values():
            if file_sha(ref["path"]) != ref["sha256"]:
                raise RuntimeError("existing reference result changed")
        for variants in record["initializations"][dataset].values():
            for initial in variants.values():
                if file_sha(out / initial["file"]) != initial["sha256"]:
                    raise RuntimeError("registered initialization changed")
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


@torch.no_grad()
def routing_context_probe(model, bundle, stats, neighbors, order, masks):
    if VARIANT != "routing_sync":
        return None
    device = next(model.parameters()).device
    ids = torch.as_tensor(np.unique(np.linspace(0, bundle.flows - 1, 8).astype(np.int64)), device=device)
    index = int(order[0])
    mask = torch.as_tensor(masks[:1], device=device, dtype=torch.bool)
    truth = torch.as_tensor(bundle.train_windows[index:index + 1], device=device)
    observed = torch.where(mask, truth, torch.zeros_like(truth))
    _, aux = model.predict_block(observed, mask, stats, neighbors, ids, return_aux=True)
    fixed = neighbors.index_select(0, ids)
    static = mask[:, :, fixed.clamp_min(0)] & (fixed >= 0)[None, None]
    result = {"role": "fit_only_diagnostic", "scope": "one epoch0 training window, eight fixed targets",
              "window_start": int(bundle.train_starts[index]), "target_ids": ids.cpu().tolist(),
              "routing_is_parameter_free": True,
              "static_mean_valid_writes_per_target_bucket": float(static.sum(-1).float().mean())}
    for direction in ("forward", "backward"):
        counts = aux[f"selected_valid_count_{direction}"]
        outside = aux[f"nonstatic_valid_count_{direction}"]
        result[direction] = {"mean_valid_writes_per_target_bucket": float(counts.float().mean()),
                             "max_valid_writes": int(counts.max()),
                             "nonstatic_fraction_of_valid_writes": float(outside.sum() / counts.sum().clamp_min(1))}
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
        context_probe = routing_context_probe(model, bundle, stats, neighbors, order, masks)
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
                write_json(job / "status.json", {**identity, "state": "paused_budget", "updated": utc(),
                    "epochs_completed": len(history), "partial_epoch": epoch + 1,
                    "partial_updates": offset // TRAINING["effective_batch"], "best_epoch": best_epoch,
                    "best_score": best_score if math.isfinite(best_score) else None,
                    "resume_available": True, "eligible_as_completed_result": False})
                emit("paused_budget", dataset=args.dataset, variant=VARIANT, seed=SEED)
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
    if VARIANT in VARIANTS and SEED == 41001:
        return
    if SEED != 41002:
        raise RuntimeError("unregistered seed / repeated existing experiment")
    decision_path = args.output / "decision.json"
    if not decision_path.exists():
        raise RuntimeError("repeat requires completed-screen selection")
    decision = read_json(decision_path)
    if decision.get("registry_sha256") != file_sha(args.output / "registry.json"):
        raise RuntimeError("repeat decision registry mismatch")
    if not decision.get("winner"):
        raise RuntimeError("no candidate selected for repeat")
    allowed = VARIANT == decision["winner"] or (VARIANT == "sync_delta" and args.dataset == "abilene")
    if not allowed:
        raise RuntimeError("repeat outside the single-winner budget")


def main():
    global VARIANT, SEED
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "smoke", "train"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        if command == "prepare":
            p.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
            p.add_argument("--ari-parent", type=Path, default=DEFAULT_ARI)
        else:
            p.add_argument("--dataset", choices=DATASETS, required=True)
            p.add_argument("--variant", choices=ALL_VARIANTS, required=True)
            p.add_argument("--seed", type=int, choices=(41001, 41002), default=41001)
            p.add_argument("--device", default="cuda:0")
        if command == "train":
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    VARIANT, SEED = getattr(args, "variant", None), getattr(args, "seed", 41001)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    {"prepare": prepare, "smoke": smoke, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
