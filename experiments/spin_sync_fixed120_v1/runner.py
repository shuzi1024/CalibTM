"""Post-hoc fixed-120 continuation of completed unified SPIN/Sync/Direct jobs.

Only original fit/development rows are used. Parent artifacts stay read-only.
Resume from the parent's LAST optimizer state, never its selected best weights.
The original verified fit smoke is inherited; this runner creates no new smoke.
A future deadline bounds an externally authorized invocation. Interrupting a
partial epoch rolls back to the last committed epoch, including optimizer/RNG.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import signal
import socket
import time

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "spin-sync-unified-fixed120-development-20260916-v1"
PARENT_OUTPUT = ROOT / "outputs/spin-sync-unified-20260916-v1"
VARIANTS = ("spin_sync_direct", "spin_direct_no_memory")
STOP_REQUESTED = False
TRAINING = {"max_epochs": 120, "batch": 8, "evaluation_batch": 8, "node_chunk": 32,
    "gradient_checkpointing": True, "optimizer": "AdamW", "lr": .001,
    "weight_decay": .0001, "betas": [.9, .999], "gradient_clip": 1., "scheduler": None,
    "patience": None, "patience_starts_epoch": None, "earliest_stop_epoch": None,
    "early_stopping": False, "stop_rule": "fixed_120_epochs",
    "data_order_seed": 51001, "training_mask_seed": 61001, "selection_mask_seed": 71001,
    "loss": "unclamped raw missing absolute sum / (fit C * batch missing count)",
    "selection": "equal mean of three condition global missing-only NMAE; strict improvement",
    "initialization": "all parameters jointly trained from scratch", "precision": "FP32; TF32 disabled"}


class Stopped(RuntimeError):
    pass


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def check_stop(args):
    if STOP_REQUESTED or time.time() >= args.deadline_unix - 60:
        raise Stopped("signal" if STOP_REQUESTED else "allocation_cutoff")


def parent_sources():
    package = ROOT / "experiments/spin_sync_unified_v1"
    files = [package / name for name in ("runner.py", "models.py", "__init__.py")]
    for name in ("spin_comparison_v1", "sync_delta_v1", "llm_candidates_v1"):
        files += sorted((ROOT / "experiments" / name).glob("*.py"))
    files.append(ROOT / "experiments/direct_only_runtime_v1/run.py")
    return {str(path.relative_to(ROOT)): io.file_sha(path) for path in sorted(set(files))}


def sources():
    return {**parent_sources(), **{str(path.relative_to(ROOT)): io.file_sha(path)
        for path in (Path(__file__), Path(__file__).with_name("__init__.py"))}}


def require_identity(record, identity, label):
    if any(record.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f"{label} identity differs")


def verify_parent_unchanged(ledger):
    for name, digest in ledger["files_sha256"].items():
        if io.file_sha(Path(ledger["path"]) / name) != digest:
            raise RuntimeError(f"Frozen continuation parent changed: {name}")


def completed_parent(args):
    path = PARENT_OUTPUT / args.dataset / args.variant / f"seed{args.seed}"
    result = base.read_json(path / "result.json")
    if result.get("state") != "complete" or result.get("role") != "development_only":
        raise RuntimeError("Parent must already have a complete development result")
    config = base.read_json(path / "config.json")
    identity = {"protocol": PARENT_PROTOCOL, "config_sha256": io.file_sha(path / "config.json"),
                "dataset": args.dataset, "variant": args.variant, "seed": args.seed}
    require_identity(result, identity, "Parent result")
    require_identity(config, {k: v for k, v in identity.items() if k != "config_sha256"}, "Parent config")
    if (config["training"] != PARENT_TRAINING or config["model_config"] != MODEL_CONFIG
            or config["source_files"] != parent_sources() or config["role"] != "development_only"):
        raise RuntimeError("Parent scientific configuration or frozen source differs")
    smoke = base.read_json(path / "smoke/SMOKE.json")
    require_identity(smoke, identity, "Parent smoke")
    if not all(smoke.get(k) is True for k in ("passed", "exact_checkpoint_continuation", "fit_evaluation_path_checked")):
        raise RuntimeError("A verified parent training/resume/evaluation smoke is required")
    if io.file_sha(path / "best.pt") != result["best_checkpoint_sha256"]:
        raise RuntimeError("Parent selected checkpoint hash differs")
    last = torch.load(path / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    require_identity(last, identity, "Parent last checkpoint")
    require_identity(best, identity, "Parent best checkpoint")
    history = base.read_json(path / "history.json")
    epochs = last["next_epoch"]
    if (not 1 <= epochs <= 120 or epochs != result["epochs_completed"] or len(history) != epochs
            or last["history"] != history or result["optimizer_updates"] != epochs * 64
            or any(row["epoch"] != i + 1 or row["updates"] != (i + 1) * 64 for i, row in enumerate(history))):
        raise RuntimeError("Parent committed epochs/history/update counts differ")
    selected = min(history, key=lambda row: row["selection_score"])
    if (last["best_epoch"] != best["epoch"] or best["epoch"] != result["best_epoch"]
            or best["epoch"] != selected["epoch"] or last["best_score"] != best["score"]
            or best["score"] != selected["selection_score"]
            or any(last["best_eval"][key] != result["primary_eval"][key] for key in ("selection_score", "metrics", "rows"))
            or last["best_score"] != last["best_eval"]["selection_score"]
            or io.state_sha(best["model"]) != result["best_state_sha256"]
            or io.state_sha(last["best_model"]) != result["best_state_sha256"]):
        raise RuntimeError("Parent best selection/checkpoint/evaluation differs")
    if (last["training_seconds"] != result["training_seconds"]
            or last["peak_training_allocated_bytes"] != result["peak_training_allocated_bytes"]
            or last["stopped_early"] != result["stopped_early"]
            or (epochs < 120 and not last["stopped_early"])):
        raise RuntimeError("Parent stopping or cumulative runtime differs")
    if (not last["optimizer"]["state"] or set(last["rng"]) != {"python", "numpy", "torch", "cuda"}
            or io.file_sha(path / "initial.pt") != config["initial_file_sha256"]):
        raise RuntimeError("Parent optimizer/RNG/initialization is incomplete")
    for group in last["optimizer"]["param_groups"]:
        if (group["lr"] != .001 or group["weight_decay"] != .0001 or tuple(group["betas"]) != (.9, .999)):
            raise RuntimeError("Parent AdamW hyperparameters differ")
    ledger = {"path": str(path), "identity": identity, "epochs_completed": epochs,
        "imported_checkpoint": "last.pt", "last_model_state_sha256": io.state_sha(last["model"]),
        "files_sha256": {name: io.file_sha(path / name) for name in
            ("config.json", "result.json", "last.pt", "best.pt", "history.json", "initial.pt", "smoke/SMOKE.json")}}
    return config, last, ledger


def status(context, state, **fields):
    io.write_json(context["status_path"], {**context["identity"], "state": state,
        "updated": io.utc(), "deadline_unix": context["args"].deadline_unix, **fields})


def setup(args, job):
    check_stop(args)
    parent_config, parent_last, parent_ledger = completed_parent(args)
    engine.seed_runtime(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal continuation requires an authorized CUDA device")
    original = base.verify_parent(base.DEFAULT_PARENT)
    bundle = io.verified_bundle(original, args.dataset)
    masks = engine.make_dev_masks(bundle, 71001)
    data = base.data_identity(bundle, masks)
    if len(bundle.train_windows) != 512:
        raise RuntimeError("Original 512 fit windows are required")
    full = base.read_json(base.DEFAULT_FULL / "registry.json")
    if data != full["data"][args.dataset] or io.file_sha(base.DEFAULT_FULL / "registry.json") != base.FULL_SHA:
        raise RuntimeError("Original Direct data or reference registry differs")
    reference_paths = {
        "linear": Path(full["references"][args.dataset]["linear"]["path"]),
        "direct_sync": base.DEFAULT_FULL / args.dataset / "direct_sync/seed41001/result.json",
        "spin_adapted": ROOT / "outputs/spin-comparison-20260914-v1" / args.dataset / "spin_adapted/seed41001/result.json"}
    references, evaluations = {}, {}
    for name, path in reference_paths.items():
        record = base.read_json(path)
        evaluation = record if name == "linear" else record["primary_eval"]
        if evaluation["role"] != "development_only" or evaluation["mask_seed"] != 71001:
            raise RuntimeError(f"Wrong original development reference: {name}")
        if name == "linear":
            if io.file_sha(path) != full["references"][args.dataset]["linear"]["sha256"]:
                raise RuntimeError("Original Linear reference changed")
        elif (record["dataset"] != args.dataset or record["variant"] != name or record["seed"] != 41001
              or io.file_sha(path.with_name("best.pt")) != record["best_checkpoint_sha256"]
              or record["registry_sha256"] != io.file_sha(path.parents[3] / "registry.json")
              or base.read_json(path.parents[3] / "registry.json")["data"][args.dataset] != data):
            raise RuntimeError(f"Original checkpoint identity differs: {name}")
        references[name] = {"path": str(path), "sha256": io.file_sha(path),
                            "selection_score": evaluation["selection_score"]}
        evaluations[name] = evaluation
    rows = []
    for condition in CONDITIONS:
        rows += engine.score_predictions(bundle, condition, np.zeros_like(bundle.dev_windows), masks[0][condition], masks[1])
    for evaluation in evaluations.values():
        base.paired({"rows": rows}, evaluation)
    models = {variant: build_model(bundle.flows, init_seed=args.seed, variant=variant) for variant in VARIANTS}
    for model in models.values():
        model.configure_from_bundle(bundle)
        model.set_execution(node_chunk=32, gradient_checkpointing=True)
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Every unified-model parameter must be trainable from scratch")
    full_params, reduced_params = [dict(models[variant].named_parameters()) for variant in VARIANTS]
    if not reduced_params.keys() < full_params.keys():
        raise RuntimeError("No-memory parameters must be a strict subset of the full model")
    if any(not torch.equal(parameter, full_params[name]) for name, parameter in reduced_params.items()):
        raise RuntimeError("Shared named parameters do not initialize exactly equally")
    model = models[args.variant]
    initial = io.cpu_state(model)
    initial_path = Path(parent_ledger["path"]) / "initial.pt"
    if (io.state_sha(torch.load(initial_path, map_location="cpu", weights_only=False)) != io.state_sha(initial)
            or io.state_sha(initial) != parent_config["initial_state_sha256"]):
        raise RuntimeError("Original shared initialization differs")
    config = {"protocol": PROTOCOL, "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
        "training": TRAINING, "model_config": MODEL_CONFIG, "data": data,
        "runtime": base.runtime(), "source_files": sources(),
        "parent": str(base.DEFAULT_PARENT), "parent_registry_sha256": io.file_sha(base.DEFAULT_PARENT / "registry.json"),
        "references": references, "initial_state_sha256": io.state_sha(initial),
        "initial_file_sha256": io.file_sha(initial_path), "parameter_count": sum(p.numel() for p in model.parameters()),
        "shared_named_parameters_exact": True, "shared_parameter_tensors": len(reduced_params),
        "no_memory_removed_parameters": sorted(full_params.keys() - reduced_params.keys()),
        "role": "development_only", "parent_checkpoint_weights_loaded": False,
        "reference_checkpoint_weights_loaded": False, "continuation_checkpoint_loaded": True,
        "parent_checkpoint_weights_loaded_scope": "pretrained SPIN/Direct reference weights only",
        "independent_training_repeat": False,
        "posthoc_training_duration_diagnostic": True, "continuation_parent": parent_ledger,
        "inherited_smoke": {"path": str(Path(parent_ledger["path"]) / "smoke/SMOKE.json"),
            "sha256": parent_ledger["files_sha256"]["smoke/SMOKE.json"], "new_smoke_performed": False}}
    parent_expected = {key: value for key, value in config.items()
        if key not in ("posthoc_training_duration_diagnostic", "continuation_parent", "inherited_smoke",
            "reference_checkpoint_weights_loaded", "continuation_checkpoint_loaded",
            "parent_checkpoint_weights_loaded_scope", "independent_training_repeat")}
    parent_expected.update(protocol=PARENT_PROTOCOL, training=PARENT_TRAINING,
                           source_files=parent_sources(), parent_checkpoint_weights_loaded=False)
    if parent_config != parent_expected:
        raise RuntimeError("Original parent model/data/runtime/reference configuration differs")
    if len(parent_last["rng"]["cuda"]) != torch.cuda.device_count():
        raise RuntimeError("Preserve parent visible-CUDA-device count for exact RNG continuation")
    verify_parent_unchanged(parent_ledger)
    config_path = job / "config.json"
    if config_path.exists() and base.read_json(config_path) != config:
        raise RuntimeError("Immutable model/source/data/training configuration changed")
    if not config_path.exists():
        io.write_json(config_path, config)
    identity = {"protocol": PROTOCOL, "config_sha256": io.file_sha(config_path),
                "dataset": args.dataset, "variant": args.variant, "seed": args.seed}
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), weight_decay=.0001)
    stats, neighbors = engine.device_inputs(bundle, device)
    return {"args": args, "job": job, "device": device, "bundle": bundle, "masks": masks,
        "config": config, "identity": identity, "model": model, "optimizer": optimizer,
        "stats": stats, "neighbors": neighbors, "reference": evaluations["linear"],
        "status_path": job / "status.json", "parent_last": parent_last}


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
    for key, expected in context["identity"].items():
        if saved[key] != expected:
            raise RuntimeError(f"Checkpoint identity differs: {key}")
    context["model"].load_state_dict(saved["model"], strict=True)
    context["optimizer"].load_state_dict(saved["optimizer"])
    io.restore_rng(saved["rng"])
    return {key: saved[key] for key in ("next_epoch", "history", "best_model", "best_eval", "best_epoch",
                                      "best_score", "bad_epochs", "stopped_early", "training_seconds", "peak_training_allocated_bytes")}


def publish_best(context, progress):
    io.save_torch(context["job"] / "best.pt", {**context["identity"], "model": progress["best_model"],
        "epoch": progress["best_epoch"], "score": progress["best_score"]})


def train(context):
    args, job = context["args"], context["job"]
    if (job / "result.json").exists():
        result = base.read_json(job / "result.json")
        if any(result.get(k) != v for k, v in context["identity"].items()) or result.get("state") != "complete":
            raise RuntimeError("Existing result identity/completion differs")
        if io.file_sha(job / "best.pt") != result["best_checkpoint_sha256"]:
            raise RuntimeError("Completed best checkpoint changed")
        io.emit("already_complete", **context["identity"])
        return
    last = job / "last.pt"
    if last.exists() and not args.resume:
        raise RuntimeError("Committed checkpoint exists; explicitly pass --resume")
    if last.exists():
        progress = restore(context, last)
        if progress["stopped_early"] or not context["config"]["continuation_parent"]["epochs_completed"] <= progress["next_epoch"] <= 120:
            raise RuntimeError("Continuation checkpoint stopping/epoch identity differs")
    else:
        if (job / "history.json").exists() or (job / "best.pt").exists():
            raise RuntimeError("Formal artifacts exist without an authoritative checkpoint")
        saved = context["parent_last"]
        context["model"].load_state_dict(saved["model"], strict=True)
        context["optimizer"].load_state_dict(saved["optimizer"])
        io.restore_rng(saved["rng"])
        progress = {key: saved[key] for key in ("next_epoch", "history", "best_model", "best_eval", "best_epoch",
            "best_score", "bad_epochs", "training_seconds", "peak_training_allocated_bytes")}
        progress["stopped_early"] = False
        checkpoint(context, progress, last)
        publish_best(context, progress)
        io.write_json(job / "history.json", progress["history"])
        io.emit("parent_last_imported", **context["identity"], epochs_completed=progress["next_epoch"])
    del context["parent_last"]
    try:
        for epoch in range(progress["next_epoch"], 120):
            check_stop(args)
            order, masks, schedule = base.epoch_schedule(context["config"], context["bundle"], epoch)
            status(context, "training", epoch=epoch + 1, epochs_committed=progress["next_epoch"])
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
            elif epoch + 1 >= 16:
                progress["bad_epochs"] += 1
            else:
                progress["bad_epochs"] = 0
            progress["next_epoch"] = epoch + 1
            progress["stopped_early"] = False  # Only stopping rule differs; bad_epochs remains descriptive.
            progress["training_seconds"] += seconds
            progress["peak_training_allocated_bytes"] = max(peak, progress["peak_training_allocated_bytes"])
            row = {"epoch": epoch + 1, "updates": (epoch + 1) * 64,
                "train_loss": float(np.mean([s["loss"] for s in steps])),
                "mean_gradient_norm": float(np.mean([s["gradient_norm"] for s in steps])),
                "selection_score": evaluation["selection_score"], "best_score": progress["best_score"],
                "best_epoch": progress["best_epoch"], "bad_epochs": progress["bad_epochs"],
                "condition_nmae": {c: evaluation["metrics"][c]["nmae"] for c in CONDITIONS},
                "training_seconds": seconds, "evaluation_seconds": evaluation["elapsed_seconds"], "schedule": schedule}
            progress["history"].append(row)
            checkpoint(context, progress, last)
            publish_best(context, progress)
            io.write_json(job / "history.json", progress["history"])
            status(context, "epoch_complete", epochs_completed=epoch + 1, best_epoch=progress["best_epoch"], best_score=progress["best_score"])
            io.emit("epoch_complete", **context["identity"], **{k: v for k, v in row.items() if k != "schedule"})
        check_stop(args)
        if progress["next_epoch"] != 120 or progress["best_model"] is None:
            raise RuntimeError("Completion requires all 120 committed epochs")
        publish_best(context, progress)
        best = torch.load(job / "best.pt", map_location="cpu", weights_only=False)
        context["model"].load_state_dict(best["model"], strict=True)
        final = evaluate(context)
        if any(final[field] != progress["best_eval"][field] for field in ("selection_score", "metrics", "rows")):
            raise RuntimeError("Best checkpoint reload evaluation differs")
        if context["config"]["source_files"] != sources():
            raise RuntimeError("Frozen source changed during training")
        if any(io.file_sha(ref["path"]) != ref["sha256"] for ref in context["config"]["references"].values()):
            raise RuntimeError("Reference result changed during training")
        verify_parent_unchanged(context["config"]["continuation_parent"])
        check_stop(args)
        result = {**context["identity"], "state": "complete", "role": "development_only", "independent_test": False,
            "epochs_completed": progress["next_epoch"], "optimizer_updates": progress["next_epoch"] * 64,
            "best_epoch": progress["best_epoch"], "stopped_early": progress["stopped_early"], "training_cap": 120,
            "stop_reason": "fixed_epoch_cap", "posthoc_training_duration_diagnostic": True,
            "continuation_checkpoint_loaded": True, "reference_checkpoint_weights_loaded": False,
            "independent_training_repeat": False,
            "continuation_parent": context["config"]["continuation_parent"],
            "inherited_smoke": context["config"]["inherited_smoke"],
            "parent_epochs_completed": context["config"]["continuation_parent"]["epochs_completed"],
            "additional_epochs_completed": 120 - context["config"]["continuation_parent"]["epochs_completed"],
            "last_checkpoint_sha256": io.file_sha(last),
            "training_seconds": progress["training_seconds"], "parameter_count": context["config"]["parameter_count"],
            "trainable_parameter_count": sum(p.numel() for p in context["model"].parameters() if p.requires_grad),
            "best_state_sha256": io.state_sha(context["model"].state_dict()),
            "best_checkpoint_sha256": io.file_sha(job / "best.pt"), "primary_eval": final,
            "peak_training_allocated_bytes": progress["peak_training_allocated_bytes"],
            "batch": 8, "effective_batch": 8, "physical_batch": 8, "evaluation_batch": 8,
            "target_block": context["bundle"].flows, "node_chunk": 32,
            "gradient_checkpointing": True, "all_parameters_jointly_trained_from_scratch": True,
            "device_name": torch.cuda.get_device_name(context["device"]), "finished": io.utc(),
            "invocation_deadline_unix": args.deadline_unix}
        io.write_json(job / "history.json", progress["history"])
        io.write_json(job / "result.json", result)
        status(context, "complete", epochs_completed=progress["next_epoch"], best_epoch=progress["best_epoch"], best_score=progress["best_score"])
        io.emit("complete", **context["identity"], best_epoch=progress["best_epoch"], score=progress["best_score"])
    except Stopped as error:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        status(context, "stopped", reason=str(error), epochs_committed=saved["next_epoch"],
               eligible_as_completed_result=False, resume_required=True)
        io.emit("stopped", **context["identity"], reason=str(error), epochs_committed=saved["next_epoch"])



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("abilene", "geant"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=(41001, 41002, 41003), default=41001)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/spin-sync-unified-fixed120-20260916-v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time() + 60:
        parser.error("An explicit future --deadline-unix with more than 60 seconds remaining is required")
    global torch, np, base, engine, io, CONDITIONS, array_sha256, build_model, MODEL_CONFIG, PARENT_TRAINING, PARENT_PROTOCOL
    import torch
    import numpy as np
    from experiments.direct_only_runtime_v1 import run as base
    from experiments.sync_delta_v1 import engine, run as io
    from experiments.sync_delta_v1.data import CONDITIONS, array_sha256
    from experiments.spin_sync_unified_v1.models import build_model, MODEL_CONFIG
    from experiments.spin_sync_unified_v1.runner import TRAINING as PARENT_TRAINING, PROTOCOL as PARENT_PROTOCOL
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    job = (args.output.resolve() / args.dataset / args.variant / f"seed{args.seed}").resolve()
    if job.is_relative_to(PARENT_OUTPUT.resolve()) or PARENT_OUTPUT.resolve().is_relative_to(job):
        parser.error("Continuation output must not overwrite the parent namespace")
    context = None
    with base.job_lock(job / "active.lock"):
        try:
            context = setup(args, job)
            invocation = {**context["identity"], "started": io.utc(), "pid": os.getpid(), "hostname": socket.gethostname(),
                "deadline_unix": args.deadline_unix, "device": args.device, "inherited_parent_smoke": True, "resume": args.resume,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
            io.write_json(job / "invocations" / f"{time.time_ns()}-{os.getpid()}.json", invocation)
            train(context)
        except Stopped as error:
            if context is not None:
                status(context, "stopped", reason=str(error), eligible_as_completed_result=False)
        except BaseException as error:
            if context is not None:
                status(context, "failed", error=f"{type(error).__name__}: {error}", eligible_as_completed_result=False)
            raise


if __name__ == "__main__":
    main()
