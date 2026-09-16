"""Prepare, train, resume, and finalize the ten-run development experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from .data import (CONDITIONS, PROTOCOL, load_dataset, save_data_manifest,
                   save_mask_registry, training_epoch)
from .engine import (device_inputs, evaluate, evaluate_linear, fit_diagnostics,
                     latency_probe, make_dev_masks, seed_runtime, train_step)
from .models import VARIANTS, build_model


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("abilene", "geant")
INIT_SEED = 41001


def utc():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps({"time": utc(), "event": event, **fields}, allow_nan=False), flush=True)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                     encoding="utf-8", delete=False) as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
        temporary = output.name
    os.replace(temporary, path)


def save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as output:
        torch.save(payload, output)
        output.flush()
        os.fsync(output.fileno())
        temporary = output.name
    os.replace(temporary, path)


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_files():
    files = list((ROOT / "experiments/sync_delta_v1").glob("*.py"))
    files += [ROOT / "setup/ksc_sync_delta.sh", ROOT / "plan/CALIBTM_DELTA_EXPERIMENT_V1_1_CN.md"]
    return {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(files)}


def prepare(args):
    out = args.output.resolve()
    if (out / "registry.json").exists():
        verify_registry(out)
        emit("already_prepared", output=str(out))
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory; partial/old outputs are not overwritten")
    out.mkdir(parents=True, exist_ok=True)
    manifests, initializations = {}, {}
    seed_runtime(INIT_SEED)
    for dataset in DATASETS:
        bundle = load_dataset(dataset, args.canonical_dir)
        shared = out / dataset / "shared"
        manifests[dataset] = save_data_manifest(shared, bundle)
        save_mask_registry(shared / "masks", bundle, seed=71001)
        write_json(out / dataset / "linear_seed71001.json", evaluate_linear(bundle))
        initializations[dataset] = {}
        for variant in VARIANTS:
            model = build_model(bundle.flows, variant, INIT_SEED)
            state = cpu_state(model)
            path = shared / f"initial_{variant}_seed{INIT_SEED}.pt"
            save_torch(path, state)
            initializations[dataset][variant] = {"file": str(path.relative_to(out)),
                "file_sha256": file_sha(path), "state_sha256": state_sha(state),
                "parameter_count": sum(p.numel() for p in model.parameters())}
        emit("data_prepared", dataset=dataset, fit_windows=len(bundle.train_starts),
             dev_windows=len(bundle.dev_starts), C=bundle.C)
    sources = source_files()
    for relative in sources:
        destination = out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    record = {"protocol": PROTOCOL, "created": utc(), "output": str(out),
        "datasets": list(DATASETS), "variants": list(VARIANTS), "initialization_seed": INIT_SEED,
        "data_order_seed": 51001, "training_mask_seed": 61001, "selection_mask_seed": 71001,
        "serial_training_order_seed": 81001, "serial_primary_eval_seed": 81002,
        "serial_extra_eval_seeds": [81003, 81004, 81005], "postfreeze_mask_seeds": [71002, 71003],
        "canonical_dir": str(Path(args.canonical_dir).resolve()) if args.canonical_dir else
                         str(ROOT / "experiments/acil_innovation_v1/canonical"),
        "source_files": sources, "data": manifests, "initializations": initializations,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "cuda": torch.version.cuda},
        "training": {"effective_batch": 8, "physical_batch": args.physical_batch,
            "target_block": args.target_block, "lr": 1e-3, "weight_decay": 1e-4,
            "betas": [0.9, 0.999], "gradient_clip": 1.0, "max_epochs": 60,
            "extension_cap": 120, "early_stop_patience_starts_epoch": 16,
            "early_stop_patience": 10, "precision": "FP32; TF32 disabled",
            "loss": "unclamped raw-unit missing absolute sum / (fit C * full effective batch missing count)"},
        "access_ledger": {"historical_fit": "complete supervised truth for training/statistics/graph",
            "online_budget": "20% of each reconstruction window; not whole-lifecycle measurement budget",
            "development": "previously accessed source_dev+tune; all present results development-only",
            "gate_or_test_access": False, "independent_final_test_certified": False,
            "postfreeze_masks": "only after all ten checkpoints are frozen; still development robustness"}}
    write_json(out / "registry.json", record)
    (out / "registry.sha256").write_text(file_sha(out / "registry.json") + "\n")
    write_json(out / "DATA_ACCESS_LEDGER.json", record["access_ledger"])
    emit("registry_frozen", sha256=file_sha(out / "registry.json"), output=str(out))


def verify_registry(out):
    path = out / "registry.json"
    if file_sha(path) != (out / "registry.sha256").read_text().strip():
        raise RuntimeError("registry digest changed")
    record = json.loads(path.read_text())
    if record["protocol"] != PROTOCOL:
        raise RuntimeError("wrong protocol")
    for relative, expected in record["source_files"].items():
        if file_sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    actual_runtime = {"python": platform.python_version(), "torch": torch.__version__,
                      "numpy": np.__version__, "cuda": torch.version.cuda}
    if record["runtime"] != actual_runtime:
        raise RuntimeError("runtime changed since preparation")
    return record


def verified_bundle(record, dataset):
    bundle = load_dataset(dataset, record["canonical_dir"])
    expected = record["data"][dataset]
    for key in ("data_sha256", "stats_sha256", "neighbors_sha256"):
        if bundle.metadata[key] != expected[key]:
            raise RuntimeError(f"{dataset} {key} drift")
    return bundle


def verify_preflight(out):
    gate = json.loads((out / "PREFLIGHT_PASSED.json").read_text())
    if not gate["passed"] or gate["registry_sha256"] != file_sha(out / "registry.json"):
        raise RuntimeError("preflight is not bound to this registry")
    for relative, expected in gate["reports"].items():
        if file_sha(out / relative) != expected:
            raise RuntimeError("preflight report changed")
    return gate


def load_epoch_masks(bundle, shared, epoch):
    """Write each global mask table before any target chunk; reuse it for all arms."""
    path = shared / "masks" / f"masks_train_seed61001_epoch{epoch:03d}.npz"
    if not path.exists() or not path.with_suffix(".json").exists():
        schedule = training_epoch(bundle, epoch)
        save_mask_registry(shared / "masks", bundle, seed=61001, split="train", epoch=epoch,
                           epoch_masks=schedule)
        return schedule.order, schedule.masks
    info = json.loads(path.with_suffix(".json").read_text())
    if (file_sha(path) != info["npz_file_sha256"] or info["data_sha256"] != bundle.metadata["data_sha256"]
            or info["seed"] != 61001 or info["epoch"] != epoch
            or info["dataset"] != bundle.dataset or info["split"] != "train"):
        raise RuntimeError("saved training mask identity mismatch")
    with np.load(path, allow_pickle=False) as data:
        order = data["window_order"].copy()
        if not np.array_equal(np.sort(order), np.arange(len(bundle.train_starts))):
            raise RuntimeError("saved training order is not a permutation")
        if not np.array_equal(data["window_starts"], bundle.train_starts[order]):
            raise RuntimeError("saved mask window starts do not match training order")
        conditions = data["selected_conditions"]
        selected = np.asarray([CONDITIONS.index(str(name)) for name in conditions])
        packed = data["packed_masks"][np.arange(len(order)), selected]
        masks = np.unpackbits(packed, axis=-1, count=50 * bundle.flows, bitorder="little")
        masks = masks.reshape(len(order), 50, bundle.flows).astype(bool)
    if not np.all(masks.sum(axis=(1, 2)) == 10 * bundle.flows):
        raise RuntimeError("saved mask budget mismatch")
    return order, masks


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def train(args):
    out = args.output.resolve()
    record = verify_registry(out)
    bundle = verified_bundle(record, args.dataset)
    device = torch.device(args.device)
    seed_runtime(args.seed)
    shared = out / args.dataset / "shared"
    job = out / args.dataset / args.variant / f"seed{args.seed}"
    if job.exists() and not args.resume:
        raise FileExistsError(f"existing run requires explicit --resume: {job}")
    job.mkdir(parents=True, exist_ok=True)
    lock = job / "active.lock"
    if lock.exists() and args.resume:
        pid = int(lock.read_text().strip())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError(f"run still has a live process: {pid}")
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        model = build_model(bundle.flows, args.variant, args.seed).to(device)
        initialization = cpu_state(model)
        if args.seed == INIT_SEED:
            expected = record["initializations"][args.dataset][args.variant]
            if state_sha(initialization) != expected["state_sha256"]:
                raise RuntimeError("explicit common initialization does not match registry")
            if file_sha(out / expected["file"]) != expected["file_sha256"]:
                raise RuntimeError("saved initialization file changed")
            state = torch.load(out / expected["file"], map_location="cpu")
            if state_sha(state) != expected["state_sha256"]:
                raise RuntimeError("saved initialization tensor identity changed")
            model.load_state_dict(state, strict=True)
        initial_sha = state_sha(initialization)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
        stats, neighbors = device_inputs(bundle, device)
        primary_masks = make_dev_masks(bundle, 71001)
        registry_sha = file_sha(out / "registry.json")
        start_epoch, best_epoch, best_score, bad_epochs = 0, 0, float("inf"), 0
        best_state, best_eval = cpu_state(model), None
        training_seconds, elapsed_seconds, history = 0.0, 0.0, []
        stopped_early = False
        skip_updates = False
        if args.resume and (job / "last.pt").exists():
            last = torch.load(job / "last.pt", map_location="cpu")
            if last["registry_sha256"] != registry_sha or last["initial_state_sha256"] != initial_sha:
                raise RuntimeError("resume identity mismatch")
            model.load_state_dict(last["model"], strict=True)
            optimizer.load_state_dict(last["optimizer"])
            start_epoch, best_epoch, best_score = last["epoch"], last["best_epoch"], last["best_score"]
            bad_epochs, best_state, best_eval = last["bad_epochs"], last["best_model"], last["best_eval"]
            training_seconds, elapsed_seconds, history = last["training_seconds"], last["elapsed_seconds"], last["history"]
            restore_rng(last["rng"])
            stopped_early = last["stopped_early"]
            if last["stopped_early"] or start_epoch >= args.max_epochs:
                skip_updates = True
                emit("resume_finalizing_terminal_checkpoint", dataset=args.dataset,
                     variant=args.variant, epoch=start_epoch)
        else:
            if args.resume:
                # An interrupted first epoch has no committed update state. Restart
                # from the registered initialization and deterministic epoch schedule.
                if (job / "history.json").exists() or (job / "best.pt").exists():
                    raise RuntimeError("trained artifacts exist but last checkpoint is missing")
                if (job / "initial.pt").exists():
                    saved_initial = torch.load(job / "initial.pt", map_location="cpu")
                    if state_sha(saved_initial) != initial_sha:
                        raise RuntimeError("pre-epoch restart initialization mismatch")
                if (job / "run_config.json").exists():
                    previous = json.loads((job / "run_config.json").read_text())
                    if previous["registry_sha256"] != registry_sha or previous["initial_state_sha256"] != initial_sha:
                        raise RuntimeError("pre-epoch restart configuration mismatch")
                emit("restart_before_first_checkpoint", dataset=args.dataset, variant=args.variant)
            save_torch(job / "initial.pt", initialization)
            write_json(job / "run_config.json", {"registry_sha256": registry_sha,
                "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
                "initial_state_sha256": initial_sha, "physical_batch": args.physical_batch,
                "target_block": args.target_block, "effective_batch": 8})
        emit("training_started", dataset=args.dataset, variant=args.variant, seed=args.seed,
             next_epoch=start_epoch + 1, max_epochs=args.max_epochs)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(start_epoch, start_epoch if skip_updates else args.max_epochs):
            epoch_start = time.perf_counter()
            order, masks = load_epoch_masks(bundle, shared, epoch)
            losses, gradient_norms = [], []
            train_start = time.perf_counter()
            for offset in range(0, len(order), 8):
                indices = order[offset:offset + 8]
                try:
                    step = train_step(model, bundle.train_windows[indices], masks[offset:offset + 8],
                        bundle.train_starts[indices], stats, neighbors, bundle.C, optimizer,
                        dataset=args.dataset, epoch=epoch, physical_batch=args.physical_batch,
                        target_block=args.target_block, order_seed=81001)
                except (RuntimeError, FloatingPointError) as error:
                    failed_masks = masks[offset:offset + 8]
                    save_torch(job / f"failure_epoch{epoch + 1:03d}_update{offset // 8 + 1:03d}.pt",
                        {"error": str(error), "epoch": epoch, "window_starts": bundle.train_starts[indices],
                         "observed": np.where(failed_masks, bundle.train_windows[indices], 0),
                         "mask": failed_masks, "model": cpu_state(model), "rng": capture_rng(),
                         "registry_sha256": registry_sha})
                    raise
                losses.append(step["loss"])
                gradient_norms.append(step["gradient_norm"])
                if (offset // 8 + 1) % 16 == 0:
                    emit("training_progress", dataset=args.dataset, variant=args.variant, epoch=epoch + 1,
                         update=offset // 8 + 1, updates_per_epoch=64, loss=float(np.mean(losses)))
            training_seconds += time.perf_counter() - train_start
            evaluation = evaluate(model, bundle, stats, neighbors, physical_batch=args.physical_batch,
                                  target_block=args.target_block, mask_cache=primary_masks)
            improved = evaluation["selection_score"] < best_score
            if improved:
                best_score, best_epoch = evaluation["selection_score"], epoch + 1
                best_state, best_eval, bad_epochs = cpu_state(model), evaluation, 0
            elif epoch + 1 >= 16:
                bad_epochs += 1
            else:
                bad_epochs = 0
            stopped_early = epoch + 1 >= 25 and bad_epochs >= 10
            elapsed_seconds += time.perf_counter() - epoch_start
            row = {"epoch": epoch + 1, "updates": (epoch + 1) * 64,
                "train_loss": float(np.mean(losses)), "mean_gradient_norm": float(np.mean(gradient_norms)),
                "selection_score": evaluation["selection_score"], "best_epoch": best_epoch,
                "best_score": best_score, "bad_epochs": bad_epochs,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "metrics": evaluation["metrics"]}
            history.append(row)
            payload = {"model": cpu_state(model), "optimizer": optimizer.state_dict(),
                "best_model": best_state, "best_eval": best_eval, "epoch": epoch + 1,
                "best_epoch": best_epoch, "best_score": best_score, "bad_epochs": bad_epochs,
                "stopped_early": stopped_early, "history": history,
                "training_seconds": training_seconds, "elapsed_seconds": elapsed_seconds,
                "rng": capture_rng(), "registry_sha256": registry_sha, "initial_state_sha256": initial_sha}
            save_torch(job / "last.pt", payload)
            if improved:
                save_torch(job / "best.pt", {"model": best_state, "epoch": best_epoch,
                    "score": best_score, "registry_sha256": registry_sha, "initial_state_sha256": initial_sha})
            write_json(job / "evaluations" / f"epoch{epoch + 1:03d}.json", evaluation)
            write_json(job / "history.json", history)
            write_json(job / "status.json", {"state": "training", "updated": utc(), **row})
            emit("epoch_complete", dataset=args.dataset, variant=args.variant, **{k: v for k, v in row.items() if k != "metrics"})
            if stopped_early:
                break
        model.load_state_dict(best_state, strict=True)
        # A last checkpoint contains a complete best-state copy, including after a crash
        # between last.pt, best.pt, and result.json publication.
        save_torch(job / "best.pt", {"model": best_state, "epoch": best_epoch,
            "score": best_score, "registry_sha256": registry_sha, "initial_state_sha256": initial_sha})
        final_eval = evaluate(model, bundle, stats, neighbors, physical_batch=args.physical_batch,
                              target_block=args.target_block, mask_cache=primary_masks)
        if abs(final_eval["selection_score"] - best_score) > 1e-6:
            raise RuntimeError("best checkpoint reload evaluation drift")
        result = {"protocol": PROTOCOL, "role": "development_only", "dataset": args.dataset,
            "variant": args.variant, "seed": args.seed, "best_epoch": best_epoch,
            "epochs_completed": history[-1]["epoch"], "optimizer_updates": history[-1]["updates"],
            "training_cap": args.max_epochs, "stopped_early": stopped_early,
            "best_in_last_five_at_cap": not stopped_early and best_epoch >= args.max_epochs - 4,
            "training_seconds": training_seconds, "elapsed_seconds": elapsed_seconds,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "peak_training_allocated_bytes": max((r.get("peak_allocated_bytes") or 0) for r in history),
            "initial_state_sha256": initial_sha, "best_state_sha256": state_sha(best_state),
            "registry_sha256": registry_sha, "best_checkpoint_sha256": file_sha(job / "best.pt"),
            "primary_eval": final_eval, "finished": utc()}
        write_json(job / f"result_cap{args.max_epochs}.json", result)
        write_json(job / "result.json", result)
        write_json(job / "status.json", {"state": "training_complete", "updated": utc(),
                                          "best_epoch": best_epoch, "score": best_score})
        emit("training_complete", dataset=args.dataset, variant=args.variant,
             best_epoch=best_epoch, score=best_score, epochs=history[-1]["epoch"])
    finally:
        lock.unlink(missing_ok=True)


def postfreeze(args):
    out = args.output.resolve()
    record = verify_registry(out)
    if not all((out / dataset / "TRAINING_FROZEN.json").exists() for dataset in DATASETS):
        raise RuntimeError("extra masks remain blocked until all ten checkpoints are frozen")
    for dataset in DATASETS:
        frozen = json.loads((out / dataset / "TRAINING_FROZEN.json").read_text())
        if frozen["registry_sha256"] != file_sha(out / "registry.json"):
            raise RuntimeError("checkpoint freeze registry mismatch")
        for variant in VARIANTS:
            checkpoint = out / dataset / variant / f"seed{args.seed}" / "best.pt"
            if file_sha(checkpoint) != frozen["checkpoints"][variant]:
                raise RuntimeError("checkpoint changed after freeze")
    bundle = verified_bundle(record, args.dataset)
    seed_runtime(args.seed)
    device = torch.device(args.device)
    stats, neighbors = device_inputs(bundle, device)
    shared = out / args.dataset / "shared"
    caches = {}
    for seed in (71002, 71003):
        save_mask_registry(shared / "masks", bundle, seed=seed)
        caches[seed] = make_dev_masks(bundle, seed)
        write_json(out / args.dataset / f"linear_seed{seed}.json", evaluate_linear(bundle, seed, caches[seed]))
    for variant in VARIANTS:
        job = out / args.dataset / variant / f"seed{args.seed}"
        if (job / "postfreeze.json").exists() and (job / "latency.json").exists():
            emit("postfreeze_already_complete", dataset=args.dataset, variant=variant)
            continue
        model = build_model(bundle.flows, variant, args.seed).to(device)
        checkpoint = torch.load(job / "best.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        output = {"role": "postfreeze_development_robustness", "checkpoint_sha256": file_sha(job / "best.pt"),
            "evaluations": {}, "interventions": {}, "serial_permutations": {}}
        for seed in (71002, 71003):
            output["evaluations"][str(seed)] = evaluate(model, bundle, stats, neighbors,
                seed=seed, physical_batch=args.physical_batch, target_block=args.target_block, mask_cache=caches[seed])
        if variant == "sync_delta":
            primary_masks = make_dev_masks(bundle, 71001)
            for intervention in ("zero_readout", "zero_value"):
                output["interventions"][intervention] = evaluate(model, bundle, stats, neighbors,
                    physical_batch=args.physical_batch, target_block=args.target_block,
                    intervention=intervention, mask_cache=primary_masks)
        if variant == "serial_delta":
            primary_masks = make_dev_masks(bundle, 71001)
            for seed in (81003, 81004, 81005):
                output["serial_permutations"][str(seed)] = evaluate(model, bundle, stats, neighbors,
                    physical_batch=args.physical_batch, target_block=args.target_block,
                    order_seed=seed, mask_cache=primary_masks)
        write_json(job / "fit_diagnostics.json", fit_diagnostics(model, bundle, stats, neighbors))
        write_json(job / "postfreeze.json", output)
        write_json(job / "latency.json", latency_probe(model, bundle, stats, neighbors, args.target_block))
        emit("postfreeze_complete", dataset=args.dataset, variant=variant)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(out / args.dataset / "COMPLETE.json", {"time": utc(), "role": "development_only"})


def queue(args):
    out = args.output.resolve()
    record = verify_registry(out)
    verify_preflight(out)
    base = [sys.executable, "-u", "-m", "experiments.sync_delta_v1.run", "train",
        "--output", str(out), "--dataset", args.dataset, "--seed", str(args.seed),
        "--device", args.device, "--physical-batch", str(args.physical_batch),
        "--target-block", str(args.target_block)]
    for variant in VARIANTS:
        job = out / args.dataset / variant / f"seed{args.seed}"
        if (job / "result.json").exists():
            continue
        command = base + ["--variant", variant, "--max-epochs", "60"]
        if job.exists():
            command += ["--resume"]
        subprocess.run(command, cwd=ROOT, check=True)
    results = {variant: json.loads((out / args.dataset / variant / f"seed{args.seed}" / "result.json").read_text())
               for variant in VARIANTS}
    extension_path = out / args.dataset / "EXTENSION_TO_120.json"
    extend = any(result["training_cap"] == 60 and result["best_in_last_five_at_cap"] for result in results.values())
    if extension_path.exists():
        extension = json.loads(extension_path.read_text())
        if extension["registry_sha256"] != file_sha(out / "registry.json"):
            raise RuntimeError("extension marker belongs to a different registry")
        extend = True
    if extend:
        if not extension_path.exists():
            write_json(extension_path, {"time": utc(), "registry_sha256": file_sha(out / "registry.json"),
                "reason": "at least one capped model has best epoch in last five; common cap raised to 120"})
        for variant, result in results.items():
            if not result["stopped_early"] and result["epochs_completed"] < 120:
                subprocess.run(base + ["--variant", variant, "--max-epochs", "120", "--resume"], cwd=ROOT, check=True)
    frozen = {variant: file_sha(out / args.dataset / variant / f"seed{args.seed}" / "best.pt")
              for variant in VARIANTS}
    write_json(out / args.dataset / "TRAINING_FROZEN.json", {"time": utc(), "checkpoints": frozen,
                                                           "registry_sha256": file_sha(out / "registry.json")})
    emit("wan_checkpoints_frozen", dataset=args.dataset)
    last_notice = 0.0
    while not all((out / dataset / "TRAINING_FROZEN.json").exists() for dataset in DATASETS):
        if time.monotonic() - last_notice > 300:
            emit("waiting_other_wan_before_extra_masks", dataset=args.dataset)
            last_notice = time.monotonic()
        time.sleep(30)
    subprocess.run([sys.executable, "-u", "-m", "experiments.sync_delta_v1.run", "postfreeze",
        "--output", str(out), "--dataset", args.dataset, "--seed", str(args.seed),
        "--device", args.device, "--physical-batch", str(args.physical_batch),
        "--target-block", str(args.target_block)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "experiments.sync_delta_v1.summarize", "--root", str(out),
                    "--output-dir", str(out / ("analysis_" + args.dataset))], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "queue", "postfreeze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=INIT_SEED)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--physical-batch", type=int, default=2)
    parser.add_argument("--target-block", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed != INIT_SEED:
        parser.error("this pilot is frozen to initialization seed 41001")
    if not 1 <= args.physical_batch <= 2 or not 1 <= args.target_block <= 64:
        parser.error("only physical batch 1..2 and target block 1..64 are allowed")
    if args.max_epochs not in (60, 120):
        parser.error("training caps are fixed to 60 or the registered extension to 120")
    if args.command != "prepare" and args.dataset is None:
        parser.error("--dataset required")
    if args.command == "train" and args.variant is None:
        parser.error("--variant required")
    {"prepare": prepare, "train": train, "queue": queue, "postfreeze": postfreeze}[args.command](args)


if __name__ == "__main__":
    main()
