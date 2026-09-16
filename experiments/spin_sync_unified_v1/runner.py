"""Joint from-scratch training of unified SPIN/Sync/Direct and no-memory ablation.

Only original fit/development rows are accessible here. An explicitly supplied
future deadline bounds an externally authorized invocation; renewal does not
change scientific configuration or initialization. Smoke outputs are separate
from formal results. Interrupting a partial epoch rolls every update in that
epoch back to the last committed epoch on resume, including optimizer and RNG.
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
PROTOCOL = "spin-sync-unified-development-20260916-v1"
VARIANTS = ("spin_sync_direct", "spin_direct_no_memory")
STOP_REQUESTED = False
TRAINING = {"max_epochs": 120, "batch": 8, "evaluation_batch": 8, "node_chunk": 32,
    "gradient_checkpointing": True, "optimizer": "AdamW", "lr": .001,
    "weight_decay": .0001, "betas": [.9, .999], "gradient_clip": 1., "scheduler": None,
    "patience": 10, "patience_starts_epoch": 16, "earliest_stop_epoch": 25,
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


def sources():
    files = [Path(__file__), Path(__file__).with_name("models.py")]
    if Path(__file__).with_name("__init__.py").exists():
        files.append(Path(__file__).with_name("__init__.py"))
    for package in ("spin_comparison_v1", "sync_delta_v1", "llm_candidates_v1"):
        files += sorted((ROOT / "experiments" / package).glob("*.py"))
    files += [ROOT / "experiments/direct_only_runtime_v1/run.py"]
    return {str(path.relative_to(ROOT)): io.file_sha(path) for path in sorted(set(files))}


def status(context, state, **fields):
    io.write_json(context["status_path"], {**context["identity"], "state": state,
        "updated": io.utc(), "deadline_unix": context["args"].deadline_unix, **fields})


def setup(args, job):
    check_stop(args)
    engine.seed_runtime(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal execution and fit smoke require an authorized CUDA device")
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
    initial_path = job / "initial.pt"
    if initial_path.exists():
        if io.state_sha(torch.load(initial_path, map_location="cpu", weights_only=False)) != io.state_sha(initial):
            raise RuntimeError("Frozen initialization changed")
    else:
        io.save_torch(initial_path, initial)
    config = {"protocol": PROTOCOL, "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
        "training": TRAINING, "model_config": MODEL_CONFIG, "data": data,
        "runtime": base.runtime(), "source_files": sources(),
        "parent": str(base.DEFAULT_PARENT), "parent_registry_sha256": io.file_sha(base.DEFAULT_PARENT / "registry.json"),
        "references": references, "initial_state_sha256": io.state_sha(initial),
        "initial_file_sha256": io.file_sha(initial_path), "parameter_count": sum(p.numel() for p in model.parameters()),
        "shared_named_parameters_exact": True, "shared_parameter_tensors": len(reduced_params),
        "no_memory_removed_parameters": sorted(full_params.keys() - reduced_params.keys()),
        "role": "development_only", "parent_checkpoint_weights_loaded": False}
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
        "status_path": job / ("smoke/status.json" if args.smoke else "status.json")}


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
    smoke_report = base.read_json(job / "smoke/SMOKE.json")
    if (any(smoke_report.get(k) != v for k, v in context["identity"].items())
            or not smoke_report.get("passed") or not smoke_report.get("exact_checkpoint_continuation")
            or not smoke_report.get("fit_evaluation_path_checked")):
        raise RuntimeError("A matching fit training/resume/evaluation smoke must pass before formal training")
    if last.exists() and not args.resume:
        raise RuntimeError("Committed checkpoint exists; explicitly pass --resume")
    progress = {"next_epoch": 0, "history": [], "best_model": None, "best_eval": None, "best_epoch": 0,
        "best_score": float("inf"), "bad_epochs": 0, "stopped_early": False, "training_seconds": 0.,
        "peak_training_allocated_bytes": 0}
    if last.exists():
        progress = restore(context, last)
    else:
        if (job / "history.json").exists() or (job / "best.pt").exists():
            raise RuntimeError("Formal artifacts exist without an authoritative checkpoint")
        checkpoint(context, progress, last)
    try:
        for epoch in range(progress["next_epoch"], 120):
            if progress["stopped_early"]:
                break
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
            progress["stopped_early"] = epoch + 1 >= 25 and progress["bad_epochs"] >= 10
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
        if not progress["history"] or progress["best_model"] is None:
            raise RuntimeError("No committed training epoch")
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
        check_stop(args)
        result = {**context["identity"], "state": "complete", "role": "development_only", "independent_test": False,
            "epochs_completed": progress["next_epoch"], "optimizer_updates": progress["next_epoch"] * 64,
            "best_epoch": progress["best_epoch"], "stopped_early": progress["stopped_early"], "training_cap": 120,
            "stop_reason": "early_stop" if progress["stopped_early"] else "epoch_cap",
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


def smoke(context):
    order, masks, schedule = base.epoch_schedule(context["config"], context["bundle"], 0)
    torch.cuda.reset_peak_memory_stats(context["device"])
    begin = time.perf_counter()
    first = step(context, order, masks, 0, 0)
    path = context["job"] / "smoke/resume_probe.pt"
    io.save_torch(path, {**context["identity"], "model": io.cpu_state(context["model"]),
        "optimizer": context["optimizer"].state_dict(), "rng": io.capture_rng()})
    second = step(context, order, masks, 8, 0)
    expected = io.state_sha(context["model"].state_dict())
    saved = torch.load(path, map_location="cpu", weights_only=False)
    context["model"].load_state_dict(saved["model"], strict=True)
    context["optimizer"].load_state_dict(saved["optimizer"])
    io.restore_rng(saved["rng"])
    replay = step(context, order, masks, 8, 0)
    if second != replay or io.state_sha(context["model"].state_dict()) != expected:
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
            or not np.isfinite(final).all() or np.any(final < 0)
            or not np.array_equal(final[masks[:8]], bundle.train_windows[indices][masks[:8]])):
        raise RuntimeError("Fit evaluation output shape, finiteness, clamp or observation copy differs")
    check_stop(context["args"])
    io.write_json(path.with_name("SMOKE.json"), {**context["identity"], "passed": True,
        "role": "fit_runner_smoke_not_formal_result", "unique_updates": 2, "replayed_updates": 1,
        "exact_checkpoint_continuation": True, "first_step": first, "second_step": second,
        "fit_evaluation_path_checked": True, "fit_evaluation_output_sha256": array_sha256(final),
        "development_scores_computed": False,
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
    parser.add_argument("--seed", type=int, choices=(41001, 41002, 41003), default=41001)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/spin-sync-unified-20260916-v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time() + 60:
        parser.error("An explicit future --deadline-unix with more than 60 seconds remaining is required")
    if args.smoke and args.resume:
        parser.error("--smoke and --resume are mutually exclusive")
    global torch, np, base, engine, io, CONDITIONS, array_sha256, build_model, MODEL_CONFIG
    import torch
    import numpy as np
    from experiments.direct_only_runtime_v1 import run as base
    from experiments.sync_delta_v1 import engine, run as io
    from experiments.sync_delta_v1.data import CONDITIONS, array_sha256
    from .models import build_model, MODEL_CONFIG
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    job = args.output.resolve() / args.dataset / args.variant / f"seed{args.seed}"
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
