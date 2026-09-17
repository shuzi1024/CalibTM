"""Frozen path controls: 120 epochs at .001, then 40 at .0001.

Development-only experiment. New controls train from scratch; spin_direct
continues the verified no-memory parent's LAST full optimizer/RNG state.
Parent files are read-only. Smoke work is separate from formal results.
Only complete epochs are committed; a partial epoch is discarded on resume.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import signal
import socket
import time

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "spin-path-controls-development-20260917-v1"
PARENT_OUTPUT = ROOT / "outputs/spin-sync-unified-fixed120-20260916-v1"
VARIANTS = ("spin_direct", "context_only", "direct_only")
STOP_REQUESTED = False
TRAINING = {
    "max_epochs": 160, "batch": 8, "evaluation_batch": 8, "node_chunk": 32,
    "gradient_checkpointing": True, "optimizer": "AdamW", "lr": .001,
    "lr_schedule": [{"first_epoch": 1, "last_epoch": 120, "lr": .001},
                    {"first_epoch": 121, "last_epoch": 160, "lr": .0001}],
    "weight_decay": .0001, "betas": [.9, .999], "gradient_clip": 1.,
    "scheduler": "one_predeclared_drop_at_epoch_121", "early_stopping": False,
    "stop_rule": "fixed_160_epochs_or_authorization_deadline",
    "data_order_seed": 51001, "training_mask_seed": 61001, "selection_mask_seed": 71001,
    "loss": "unclamped raw missing absolute sum / (fit C * batch missing count)",
    "selection": "equal mean of three condition global missing-only NMAE; strict improvement over epochs 1..cap",
    "initialization": "paired shared parameters; controls from scratch; spin_direct continues original trajectory",
    "precision": "FP32; TF32 disabled",
}
PROGRESS_KEYS = ("next_epoch", "history", "best_model", "best_eval", "best_epoch",
                 "best_score", "bad_epochs", "stopped_early", "training_seconds",
                 "peak_training_allocated_bytes")


class Stopped(RuntimeError):
    pass


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def check_stop(args):
    if STOP_REQUESTED or time.time() >= args.deadline_unix - 60:
        raise Stopped("signal" if STOP_REQUESTED else "allocation_cutoff")


def require_identity(record, identity, label):
    for key, value in identity.items():
        if record.get(key) != value:
            raise RuntimeError(f"{label} identity differs: {key}")


def legacy_sources():
    files = [ROOT / "experiments/spin_sync_unified_v1" / name
             for name in ("runner.py", "models.py", "__init__.py")]
    for package in ("spin_comparison_v1", "sync_delta_v1", "llm_candidates_v1"):
        files += sorted((ROOT / "experiments" / package).glob("*.py"))
    files += [ROOT / "experiments/direct_only_runtime_v1/run.py"]
    files += [ROOT / "experiments/spin_sync_fixed120_v1" / name
              for name in ("runner.py", "__init__.py")]
    return {str(path.relative_to(ROOT)): io.file_sha(path) for path in sorted(set(files))}


def sources():
    # Explicit executable sources only: growing reports never change scientific identity.
    files = [Path(__file__), Path(__file__).with_name("models.py"),
             Path(__file__).with_name("__init__.py")]
    for package in ("direct_only_v1", "llm_candidates_runtime_v1"):
        files += sorted((ROOT / "experiments" / package).glob("*.py"))
    return {**legacy_sources(), **{str(path.relative_to(ROOT)): io.file_sha(path)
                                  for path in sorted(set(files))}}


def verify_ledger(ledger):
    for name, digest in ledger["files_sha256"].items():
        if io.file_sha(Path(ledger["path"]) / name) != digest:
            raise RuntimeError(f"Frozen parent artifact changed: {name}")


def validate_progress(saved, cap, label):
    history, epochs = saved["history"], saved["next_epoch"]
    if (not 0 <= epochs <= cap or len(history) != epochs or saved["stopped_early"]
            or any(row["epoch"] != i + 1 or row["updates"] != (i + 1) * 64
                   for i, row in enumerate(history))):
        raise RuntimeError(f"{label}: committed epochs/history/update counts differ")
    if epochs:
        selected = min(history, key=lambda row: row["selection_score"])
        if (selected["epoch"] != saved["best_epoch"]
                or selected["selection_score"] != saved["best_score"]
                or saved["best_eval"]["selection_score"] != saved["best_score"]
                or saved["best_model"] is None):
            raise RuntimeError(f"{label}: strict best-checkpoint selection differs")
        if not saved["optimizer"]["state"]:
            raise RuntimeError(f"{label}: optimizer state missing")
    if set(saved["rng"]) != {"python", "numpy", "torch", "cuda"}:
        raise RuntimeError(f"{label}: RNG state incomplete")
    for group in saved["optimizer"]["param_groups"]:
        expected_lr = .001 if epochs <= 120 else .0001
        if (group["lr"] != expected_lr or group["weight_decay"] != .0001
                or tuple(group["betas"]) != (.9, .999)):
            raise RuntimeError(f"{label}: AdamW settings differ")


def completed_parent(args):
    path = PARENT_OUTPUT / args.dataset / "spin_direct_no_memory" / f"seed{args.seed}"
    config, result = base.read_json(path / "config.json"), base.read_json(path / "result.json")
    identity = {"protocol": PARENT_PROTOCOL, "config_sha256": io.file_sha(path / "config.json"),
                "dataset": args.dataset, "variant": "spin_direct_no_memory", "seed": args.seed}
    require_identity(result, identity, "Parent result")
    require_identity(config, {k: v for k, v in identity.items() if k != "config_sha256"}, "Parent config")
    if (result.get("state") != "complete" or result.get("role") != "development_only"
            or result.get("epochs_completed") != 120 or result.get("optimizer_updates") != 7680
            or result.get("stopped_early") is not False or config.get("role") != "development_only"
            or config["training"] != PARENT_TRAINING or config["model_config"] != PARENT_MODEL_CONFIG
            or config["source_files"] != legacy_sources()):
        raise RuntimeError("Completed fixed-120 parent/source/protocol required")
    # The parent's own previous continuation anchor must also remain intact.
    verify_ledger(config["continuation_parent"])
    initial_path = Path(config["continuation_parent"]["path"]) / "initial.pt"
    smoke_path = Path(config["inherited_smoke"]["path"])
    smoke = base.read_json(smoke_path)
    if (io.file_sha(smoke_path) != config["inherited_smoke"]["sha256"]
            or io.file_sha(initial_path) != config["initial_file_sha256"]
            or not all(smoke.get(k) is True for k in
                       ("passed", "exact_checkpoint_continuation", "fit_evaluation_path_checked"))):
        raise RuntimeError("Verified original initialization/smoke required")
    last = torch.load(path / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    require_identity(last, identity, "Parent last")
    require_identity(best, identity, "Parent best")
    validate_progress(last, 120, "Parent")
    if (last["next_epoch"] != 120 or last["history"] != base.read_json(path / "history.json")
            or best["epoch"] != result["best_epoch"] or best["epoch"] != last["best_epoch"]
            or best["score"] != last["best_score"]
            or io.file_sha(path / "last.pt") != result["last_checkpoint_sha256"]
            or io.file_sha(path / "best.pt") != result["best_checkpoint_sha256"]
            or io.state_sha(best["model"]) != result["best_state_sha256"]
            or io.state_sha(last["best_model"]) != result["best_state_sha256"]
            or any(last["best_eval"][key] != result["primary_eval"][key]
                   for key in ("selection_score", "metrics", "rows"))
            or last["training_seconds"] != result["training_seconds"]
            or last["peak_training_allocated_bytes"] != result["peak_training_allocated_bytes"]):
        raise RuntimeError("Parent checkpoint/result/history hashes or values differ")
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if io.state_sha(initial) != config["initial_state_sha256"]:
        raise RuntimeError("Parent original initialization state differs")
    ledger = {"path": str(path), "identity": identity, "epochs_completed": 120,
              "files_sha256": {name: io.file_sha(path / name) for name in
                               ("config.json", "result.json", "history.json", "last.pt", "best.pt")},
              "initial_path": str(initial_path), "initial_file_sha256": io.file_sha(initial_path)}
    return config, last, ledger, initial


def status(context, state, **fields):
    io.write_json(context["status_path"], {**context["identity"], "state": state,
        "updated": io.utc(), "deadline_unix": context["args"].deadline_unix, **fields})


def setup(args, job):
    check_stop(args)
    parent_config, parent_last, parent_ledger, parent_initial = completed_parent(args)
    engine.seed_runtime(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal runs and fit smoke require an authorized CUDA device")
    original = base.verify_parent(base.DEFAULT_PARENT)
    bundle = io.verified_bundle(original, args.dataset)
    masks = engine.make_dev_masks(bundle, 71001)
    data = base.data_identity(bundle, masks)
    full = base.read_json(base.DEFAULT_FULL / "registry.json")
    if (len(bundle.train_windows) != 512 or data != parent_config["data"]
            or data != full["data"][args.dataset]
            or io.file_sha(base.DEFAULT_FULL / "registry.json") != base.FULL_SHA
            or io.file_sha(base.DEFAULT_PARENT / "registry.json") != parent_config["parent_registry_sha256"]
            or parent_config["runtime"] != base.runtime()):
        raise RuntimeError("Parent data/masks/runtime/registry identity differs")
    references = parent_config["references"]
    for name, ref in references.items():
        if io.file_sha(ref["path"]) != ref["sha256"]:
            raise RuntimeError(f"Frozen reference changed: {name}")
    reference = base.read_json(references["linear"]["path"])
    if reference["role"] != "development_only" or reference["mask_seed"] != 71001:
        raise RuntimeError("Wrong development reference")
    models = {variant: build_model(bundle.flows, init_seed=args.seed, variant=variant) for variant in VARIANTS}
    for model in models.values():
        model.configure_from_bundle(bundle)
        model.set_execution(node_chunk=32, gradient_checkpointing=True)
        if not all(p.requires_grad for p in model.parameters()):
            raise RuntimeError("All model parameters must be trainable")
        initial = io.cpu_state(model)
        if (not initial.keys() <= parent_initial.keys()
                or any(not torch.equal(value, parent_initial[name]) for name, value in initial.items())):
            raise RuntimeError("Shared initialization differs from original paired trajectory")
    full_params = dict(models["spin_direct"].named_parameters())
    for variant in ("context_only", "direct_only"):
        params = dict(models[variant].named_parameters())
        if not params.keys() < full_params.keys():
            raise RuntimeError("Path control must remove a strict subset of combined parameters")
    if io.state_sha(models["spin_direct"].state_dict()) != parent_config["initial_state_sha256"]:
        raise RuntimeError("Combined model initialization is not identical to parent")
    model = models[args.variant]
    initial = io.cpu_state(model)
    initial_path = job / "initial.pt"
    if initial_path.exists():
        if io.state_sha(torch.load(initial_path, map_location="cpu", weights_only=False)) != io.state_sha(initial):
            raise RuntimeError("Frozen initialization changed")
    else:
        io.save_torch(initial_path, initial)
    continuation = args.variant == "spin_direct"
    if continuation and len(parent_last["rng"]["cuda"]) != torch.cuda.device_count():
        raise RuntimeError("Preserve parent visible-CUDA-device count for RNG continuation")
    config = {"protocol": PROTOCOL, "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
        "training": TRAINING, "model_config": MODEL_CONFIG, "data": data,
        "runtime": base.runtime(), "source_files": sources(),
        "parent": str(base.DEFAULT_PARENT), "parent_registry_sha256": io.file_sha(base.DEFAULT_PARENT / "registry.json"),
        "references": references, "initial_state_sha256": io.state_sha(initial),
        "initial_file_sha256": io.file_sha(initial_path), "parameter_count": sum(p.numel() for p in model.parameters()),
        "shared_named_parameters_exact": True, "reference_checkpoint_weights_loaded": False,
        "anchor_parent": parent_ledger, "continuation_checkpoint_loaded": continuation,
        "imported_epoch_count": 120 if continuation else 0,
        "independent_training_repeat": not continuation, "role": "development_only",
        "independent_test": False, "path_control_not_official_spin": args.variant == "context_only",
        "fixed_optimization_diagnostic": True}
    config_path = job / "config.json"
    if config_path.exists() and base.read_json(config_path) != config:
        raise RuntimeError("Immutable source/model/data/training configuration changed")
    if not config_path.exists():
        io.write_json(config_path, config)
    identity = {"protocol": PROTOCOL, "config_sha256": io.file_sha(config_path),
                "dataset": args.dataset, "variant": args.variant, "seed": args.seed}
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), weight_decay=.0001)
    stats, neighbors = engine.device_inputs(bundle, device)
    return {"args": args, "job": job, "device": device, "bundle": bundle, "masks": masks,
        "config": config, "identity": identity, "model": model, "optimizer": optimizer,
        "stats": stats, "neighbors": neighbors, "reference": reference,
        "status_path": job / ("smoke/status.json" if args.smoke else "status.json"),
        "parent_last": parent_last if continuation else None}


def step(context, order, masks, offset, epoch):
    check_stop(context["args"])
    bundle = context["bundle"]
    indices = order[offset:offset + 8]
    return engine.train_step(context["model"], bundle.train_windows[indices], masks[offset:offset + len(indices)],
        bundle.train_starts[indices], context["stats"], context["neighbors"], bundle.C, context["optimizer"],
        dataset=bundle.dataset, epoch=epoch, physical_batch=8, target_block=bundle.flows)


def evaluate(context):
    bundle, (masks, families) = context["bundle"], context["masks"]
    metrics, rows, begin = {}, [], time.perf_counter()
    for condition in CONDITIONS:
        final, raw = np.empty_like(bundle.dev_windows), np.empty_like(bundle.dev_windows)
        for first in range(0, len(raw), 8):
            check_stop(context["args"])
            sl = slice(first, first + 8)
            final[sl], raw[sl] = engine.predict_windows(context["model"], bundle.dev_windows[sl], masks[condition][sl],
                bundle.dev_starts[sl], context["stats"], context["neighbors"], dataset=bundle.dataset,
                physical_batch=8, target_block=bundle.flows)
        condition_rows = engine.score_predictions(bundle, condition, final, masks[condition], families, raw)
        rows.extend(condition_rows)
        metrics[condition] = engine.aggregate(condition_rows)
        metrics[condition]["groups"] = {name: engine.aggregate([row["groups"][name] for row in condition_rows])
                                       for name in condition_rows[0]["groups"]}
        metrics[condition]["negative_raw_fraction"] = sum(r["negative_raw_count"] for r in condition_rows) / metrics[condition]["count"]
    score = sum(metrics[c]["nmae"] for c in CONDITIONS) / 3
    if not math.isfinite(score):
        raise FloatingPointError("Undefined development score")
    result = {"role": "development_only", "mask_seed": 71001, "serial_order_seed": 81002,
        "selection_score": score, "metrics": metrics, "rows": rows, "intervention": None,
        "mask_array_sha256": {c: array_sha256(masks[c]) for c in CONDITIONS},
        "elapsed_seconds": time.perf_counter() - begin}
    base.paired(result, context["reference"])
    return result


def checkpoint(context, progress, path):
    io.save_torch(path, {**context["identity"], **progress, "model": io.cpu_state(context["model"]),
        "optimizer": context["optimizer"].state_dict(), "rng": io.capture_rng(),
        "invocation_deadline_unix": context["args"].deadline_unix, "committed": io.utc()})


def restore(context, path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require_identity(saved, context["identity"], "Checkpoint")
    validate_progress(saved, 160, "Checkpoint")
    if len(saved["rng"]["cuda"]) != torch.cuda.device_count():
        raise RuntimeError("Visible CUDA device count changed on resume")
    context["model"].load_state_dict(saved["model"], strict=True)
    context["optimizer"].load_state_dict(saved["optimizer"])
    io.restore_rng(saved["rng"])
    return {key: saved[key] for key in PROGRESS_KEYS}


def publish_best(context, progress, filename="best.pt"):
    io.save_torch(context["job"] / filename, {**context["identity"], "model": progress["best_model"],
        "epoch": progress["best_epoch"], "score": progress["best_score"]})


def result_payload(context, progress, final, cap, best_name, last_name):
    job = context["job"]
    return {**context["identity"], "state": "complete", "role": "development_only", "independent_test": False,
        "epochs_completed": cap, "optimizer_updates": cap * 64,
        "best_epoch": progress["best_epoch"], "stopped_early": False, "training_cap": cap,
        "stop_reason": "fixed_epoch_cap", "fixed_optimization_diagnostic": True,
        "continuation_checkpoint_loaded": context["config"]["continuation_checkpoint_loaded"],
        "reference_checkpoint_weights_loaded": False,
        "independent_training_repeat": context["config"]["independent_training_repeat"],
        "anchor_parent": context["config"]["anchor_parent"],
        "imported_epoch_count": context["config"]["imported_epoch_count"],
        "additional_epochs_completed": cap - context["config"]["imported_epoch_count"],
        "training_seconds": progress["training_seconds"], "parameter_count": context["config"]["parameter_count"],
        "trainable_parameter_count": sum(p.numel() for p in context["model"].parameters() if p.requires_grad),
        "best_state_sha256": io.state_sha(progress["best_model"]),
        "best_checkpoint": best_name, "best_checkpoint_sha256": io.file_sha(job / best_name),
        "last_checkpoint": last_name, "last_checkpoint_sha256": io.file_sha(job / last_name),
        "primary_eval": final, "peak_training_allocated_bytes": progress["peak_training_allocated_bytes"],
        "batch": 8, "effective_batch": 8, "physical_batch": 8, "evaluation_batch": 8,
        "target_block": context["bundle"].flows, "node_chunk": 32, "gradient_checkpointing": True,
        "all_parameters_jointly_trained_from_scratch": True,
        "device_name": torch.cuda.get_device_name(context["device"]), "finished": io.utc(),
        "invocation_deadline_unix": context["args"].deadline_unix}


def verify_frozen_inputs(context):
    if context["config"]["source_files"] != sources():
        raise RuntimeError("Frozen source changed during execution")
    for ref in context["config"]["references"].values():
        if io.file_sha(ref["path"]) != ref["sha256"]:
            raise RuntimeError("Reference result changed during execution")
    anchor = context["config"]["anchor_parent"]
    verify_ledger(anchor)
    if io.file_sha(anchor["initial_path"]) != anchor["initial_file_sha256"]:
        raise RuntimeError("Parent original initialization changed")


def evaluated_best(context, progress, filename="best.pt"):
    publish_best(context, progress, filename)
    best = torch.load(context["job"] / filename, map_location="cpu", weights_only=False)
    require_identity(best, context["identity"], "Selected checkpoint")
    context["model"].load_state_dict(best["model"], strict=True)
    evaluation = evaluate(context)
    if any(evaluation[field] != progress["best_eval"][field] for field in ("selection_score", "metrics", "rows")):
        raise RuntimeError("Best checkpoint reload evaluation differs")
    return evaluation


def snapshot120(context, progress):
    job = context["job"]
    result_path = job / "result120.json"
    if result_path.exists():
        record = base.read_json(result_path)
        require_identity(record, context["identity"], "120-epoch result")
        if (record.get("state") != "complete" or record.get("epochs_completed") != 120
                or io.file_sha(job / "best120.pt") != record["best_checkpoint_sha256"]
                or io.file_sha(job / "last120.pt") != record["last_checkpoint_sha256"]
                or len(base.read_json(job / "history120.json")) != 120):
            raise RuntimeError("Immutable 120-epoch snapshot differs")
        return
    if progress["next_epoch"] != 120:
        raise RuntimeError("120-epoch snapshot must be committed before epoch 121")
    check_stop(context["args"])
    # Copy the authoritative committed checkpoint, then restore it after scoring.
    saved = torch.load(job / "last.pt", map_location="cpu", weights_only=False)
    require_identity(saved, context["identity"], "120-epoch last")
    io.save_torch(job / "last120.pt", saved)
    evaluation = evaluated_best(context, progress, "best120.pt")
    verify_frozen_inputs(context)
    io.write_json(job / "history120.json", progress["history"])
    io.write_json(result_path, result_payload(context, progress, evaluation, 120, "best120.pt", "last120.pt"))
    restore(context, job / "last.pt")
    io.emit("snapshot120_complete", **context["identity"], best_epoch=progress["best_epoch"], score=progress["best_score"])


def train(context):
    args, job = context["args"], context["job"]
    if (job / "result.json").exists():
        result = base.read_json(job / "result.json")
        require_identity(result, context["identity"], "Existing result")
        if (result.get("state") != "complete" or result.get("epochs_completed") != 160
                or io.file_sha(job / "best.pt") != result["best_checkpoint_sha256"]
                or io.file_sha(job / "last.pt") != result["last_checkpoint_sha256"]):
            raise RuntimeError("Existing completed result or checkpoint differs")
        verify_frozen_inputs(context)
        io.emit("already_complete", **context["identity"])
        return
    smoke_report = base.read_json(job / "smoke/SMOKE.json")
    require_identity(smoke_report, context["identity"], "Smoke")
    if not all(smoke_report.get(key) is True for key in
               ("passed", "exact_checkpoint_continuation", "fit_evaluation_path_checked", "fit_scoring_checked")):
        raise RuntimeError("Matching training/resume/fit-scoring smoke must pass first")
    last = job / "last.pt"
    if last.exists() and not args.resume:
        raise RuntimeError("Committed checkpoint exists; explicitly pass --resume")
    if last.exists():
        progress = restore(context, last)
    else:
        if any((job / name).exists() for name in ("history.json", "best.pt", "result120.json", "last120.pt")):
            raise RuntimeError("Formal artifacts exist without authoritative checkpoint")
        parent = context["parent_last"]
        if parent is not None:
            context["model"].load_state_dict(parent["model"], strict=True)
            context["optimizer"].load_state_dict(parent["optimizer"])
            io.restore_rng(parent["rng"])
            progress = {key: parent[key] for key in PROGRESS_KEYS}
        else:
            progress = {"next_epoch": 0, "history": [], "best_model": None, "best_eval": None,
                        "best_epoch": 0, "best_score": float("inf"), "bad_epochs": 0,
                        "stopped_early": False, "training_seconds": 0., "peak_training_allocated_bytes": 0}
        checkpoint(context, progress, last)
    del context["parent_last"]
    if progress["next_epoch"] < context["config"]["imported_epoch_count"]:
        raise RuntimeError("Resume predates imported parent trajectory")
    # Rebuild derived files from the authoritative checkpoint after any crash.
    io.write_json(job / "history.json", progress["history"])
    if progress["best_model"] is not None:
        publish_best(context, progress)
    try:
        if progress["next_epoch"] >= 120:
            snapshot120(context, progress)
        for epoch in range(progress["next_epoch"], 160):
            check_stop(args)
            lr = .001 if epoch < 120 else .0001
            for group in context["optimizer"].param_groups:
                group["lr"] = lr
            order, masks, schedule = base.epoch_schedule(context["config"], context["bundle"], epoch)
            status(context, "training", epoch=epoch + 1, epochs_committed=progress["next_epoch"], lr=lr)
            torch.cuda.reset_peak_memory_stats(context["device"])
            begin, steps = time.perf_counter(), []
            for offset in range(0, 512, 8):
                steps.append(step(context, order, masks, offset, epoch))
            seconds = time.perf_counter() - begin
            peak = torch.cuda.max_memory_allocated(context["device"])
            status(context, "evaluating", epoch=epoch + 1, epochs_committed=progress["next_epoch"])
            evaluation = evaluate(context)
            if evaluation["selection_score"] < progress["best_score"]:
                progress.update(best_model=io.cpu_state(context["model"]), best_eval=evaluation,
                                best_epoch=epoch + 1, best_score=evaluation["selection_score"], bad_epochs=0)
            else:
                progress["bad_epochs"] += 1
            progress["next_epoch"] = epoch + 1
            progress["stopped_early"] = False
            progress["training_seconds"] += seconds
            progress["peak_training_allocated_bytes"] = max(peak, progress["peak_training_allocated_bytes"])
            row = {"epoch": epoch + 1, "updates": (epoch + 1) * 64, "lr": lr,
                "train_loss": float(np.mean([s["loss"] for s in steps])),
                "mean_gradient_norm": float(np.mean([s["gradient_norm"] for s in steps])),
                "selection_score": evaluation["selection_score"], "best_score": progress["best_score"],
                "best_epoch": progress["best_epoch"], "bad_epochs": progress["bad_epochs"],
                "condition_nmae": {c: evaluation["metrics"][c]["nmae"] for c in CONDITIONS},
                "condition_nrmse": {c: evaluation["metrics"][c]["nrmse"] for c in CONDITIONS},
                "training_seconds": seconds, "evaluation_seconds": evaluation["elapsed_seconds"], "schedule": schedule}
            progress["history"].append(row)
            checkpoint(context, progress, last)
            publish_best(context, progress)
            io.write_json(job / "history.json", progress["history"])
            status(context, "epoch_complete", epochs_completed=epoch + 1,
                   best_epoch=progress["best_epoch"], best_score=progress["best_score"])
            io.emit("epoch_complete", **context["identity"], **{k: v for k, v in row.items() if k != "schedule"})
            if epoch + 1 == 120:
                snapshot120(context, progress)
        check_stop(args)
        if progress["next_epoch"] != 160 or progress["best_model"] is None:
            raise RuntimeError("Completion requires 160 committed epochs")
        final = evaluated_best(context, progress)
        verify_frozen_inputs(context)
        check_stop(args)
        io.write_json(job / "history.json", progress["history"])
        io.write_json(job / "result.json", result_payload(context, progress, final, 160, "best.pt", "last.pt"))
        status(context, "complete", epochs_completed=160, best_epoch=progress["best_epoch"], best_score=progress["best_score"])
        io.emit("complete", **context["identity"], best_epoch=progress["best_epoch"], score=progress["best_score"])
    except Stopped as error:
        progress = restore(context, last)
        if progress["best_model"] is not None:
            publish_best(context, progress)
        io.write_json(job / "history.json", progress["history"])
        status(context, "stopped", reason=str(error), epochs_committed=progress["next_epoch"],
               eligible_as_completed_result=False, resume_required=True)
        io.emit("stopped", **context["identity"], reason=str(error), epochs_committed=progress["next_epoch"])



def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def smoke(context):
    epoch = 0
    if context["parent_last"] is not None:
        saved = context["parent_last"]
        context["model"].load_state_dict(saved["model"], strict=True)
        context["optimizer"].load_state_dict(saved["optimizer"])
        io.restore_rng(saved["rng"])
        for group in context["optimizer"].param_groups:
            group["lr"] = .0001
        epoch = 120
    order, masks, schedule = base.epoch_schedule(context["config"], context["bundle"], epoch)
    torch.cuda.reset_peak_memory_stats(context["device"])
    begin = time.perf_counter()
    first = step(context, order, masks, 0, epoch)
    path = context["job"] / "smoke/resume_probe.pt"
    io.save_torch(path, {**context["identity"], "model": io.cpu_state(context["model"]),
        "optimizer": context["optimizer"].state_dict(), "rng": io.capture_rng()})
    second = step(context, order, masks, 8, epoch)
    expected = io.state_sha(context["model"].state_dict())
    expected_optimizer = copy.deepcopy(context["optimizer"].state_dict())
    expected_rng = io.capture_rng()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require_identity(saved, context["identity"], "Smoke checkpoint")
    context["model"].load_state_dict(saved["model"], strict=True)
    context["optimizer"].load_state_dict(saved["optimizer"])
    io.restore_rng(saved["rng"])
    replay = step(context, order, masks, 8, epoch)
    if (second != replay or io.state_sha(context["model"].state_dict()) != expected
            or not tree_equal(context["optimizer"].state_dict(), expected_optimizer)
            or not tree_equal(io.capture_rng(), expected_rng)):
        raise RuntimeError("Fit smoke checkpoint continuation is not exact")
    check_stop(context["args"])
    update_seconds = (time.perf_counter() - begin) / 3
    training_peak = torch.cuda.max_memory_allocated(context["device"])
    torch.cuda.synchronize(context["device"])
    torch.cuda.reset_peak_memory_stats(context["device"])
    indices = order[:8]
    bundle = context["bundle"]
    begin = time.perf_counter()
    final, raw = engine.predict_windows(context["model"], bundle.train_windows[indices], masks[:8],
        bundle.train_starts[indices], context["stats"], context["neighbors"], dataset=bundle.dataset,
        physical_batch=8, target_block=bundle.flows)
    torch.cuda.synchronize(context["device"])
    evaluation_seconds = time.perf_counter() - begin
    if (final.shape != raw.shape or final.shape != bundle.train_windows[indices].shape
            or not np.isfinite(final).all() or not np.isfinite(raw).all() or np.any(final < 0)
            or not np.array_equal(final[masks[:8]], bundle.train_windows[indices][masks[:8]])):
        raise RuntimeError("Fit evaluation output shape, finiteness, clamp or observation copy differs")
    fit_sums = engine.error_sums(final, bundle.train_windows[indices], ~masks[:8])
    fit_metrics = engine.ratios(fit_sums)
    if (fit_sums["count"] != int((~masks[:8]).sum())
            or any(fit_metrics[name] is None or not math.isfinite(fit_metrics[name])
                   for name in ("nmae", "nrmse"))):
        raise RuntimeError("Fit-only missing-element scoring failed")
    verify_frozen_inputs(context)
    check_stop(context["args"])
    io.write_json(path.with_name("SMOKE.json"), {**context["identity"], "passed": True,
        "role": "fit_runner_smoke_not_formal_result", "unique_updates": 2, "replayed_updates": 1,
        "exact_checkpoint_continuation": True, "first_step": first, "second_step": second,
        "fit_evaluation_path_checked": True, "fit_evaluation_output_sha256": array_sha256(final),
        "development_scores_computed": False, "fit_scoring_checked": True,
        "fit_only_metrics_not_development": fit_metrics, "training_epoch_zero_based": epoch,
        "batch": 8, "node_chunk": 32, "target_block": context["bundle"].flows, "schedule": schedule,
        "seconds_per_update_including_resume_io": update_seconds,
        "seconds_per_evaluation_batch": evaluation_seconds, "peak_training_allocated_bytes": training_peak,
        "peak_evaluation_allocated_bytes_including_training_state": torch.cuda.max_memory_allocated(context["device"]),
        "rough_epoch_seconds_from_smoke": 64 * update_seconds + 3 * math.ceil(len(bundle.dev_windows) / 8) * evaluation_seconds,
        "parameter_count": context["config"]["parameter_count"], "finished": io.utc()})
    status(context, "smoke_passed", eligible_as_completed_result=False)
    io.emit("smoke_passed", **context["identity"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("abilene", "geant"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=(41001, 41002), default=41001)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/spin-path-controls-20260917-v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time() + 60:
        parser.error("An explicit future --deadline-unix with more than 60 seconds remaining is required")
    if args.smoke and args.resume:
        parser.error("--smoke and --resume are mutually exclusive")
    global torch, np, base, engine, io, CONDITIONS, array_sha256, build_model, MODEL_CONFIG
    global PARENT_TRAINING, PARENT_PROTOCOL, PARENT_MODEL_CONFIG
    import torch
    import numpy as np
    from experiments.direct_only_runtime_v1 import run as base
    from experiments.sync_delta_v1 import engine, run as io
    from experiments.sync_delta_v1.data import CONDITIONS, array_sha256
    from experiments.spin_sync_fixed120_v1.runner import TRAINING as PARENT_TRAINING, PROTOCOL as PARENT_PROTOCOL
    from experiments.spin_sync_unified_v1.models import MODEL_CONFIG as PARENT_MODEL_CONFIG
    from .models import build_model, MODEL_CONFIG
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    job = (args.output.resolve() / args.dataset / args.variant / f"seed{args.seed}").resolve()
    protected = (PARENT_OUTPUT, ROOT / "outputs/spin-sync-unified-20260916-v1", base.DEFAULT_PARENT, base.DEFAULT_FULL)
    if any(job.is_relative_to(path.resolve()) or path.resolve().is_relative_to(job) for path in protected):
        parser.error("Output must not overwrite a historical parent namespace")
    context = None
    with base.job_lock(job / "active.lock"):
        try:
            context = setup(args, job)
            invocation = {**context["identity"], "started": io.utc(), "pid": os.getpid(), "hostname": socket.gethostname(),
                "deadline_unix": args.deadline_unix, "device": args.device, "smoke": args.smoke, "resume": args.resume,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
            io.write_json(job / "invocations" / f"{time.time_ns()}-{os.getpid()}.json", invocation)
            (smoke if args.smoke else train)(context)
        except Stopped as error:
            if context is not None:
                status(context, "stopped", reason=str(error), eligible_as_completed_result=False)
        except BaseException as error:
            if context is not None:
                status(context, "failed", error=f"{type(error).__name__}: {error}", eligible_as_completed_result=False)
            raise


if __name__ == "__main__":
    main()
