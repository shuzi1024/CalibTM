"""Two-node, four-GPU bounded scheduler; standard library only.

Each worker keeps one child per leased GPU.  A shared queue lock serializes
claims and decisions; a per-job lease inherited by the child prevents a second
node from reclaiming a still-running child after its worker exits unexpectedly.
Completed results and failed attempts are never automatically retrained.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/llm-candidates-20260914-v1"
DEADLINE_UNIX = 1789347583.0
ALLOCATION_CUTOFF = DEADLINE_UNIX - 60
VARIANTS = ("routing_sync", "direct_sync", "feature_sync")
DATASETS = ("abilene", "geant")
TIE_GAIN = 0.005
SELECTION_PREFERENCE = "balanced_slight_novelty"
# Task-contribution potential, not a claim of demonstrated originality.
NOVELTY_PRIORITY = ("routing_sync", "direct_sync", "feature_sync")
TERMINAL = {"complete", "failed", "incomplete"}
STOP_REQUESTED = False


class AllocationStopped(RuntimeError):
    """The authorization boundary was reached between claim and process spawn."""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def try_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


@contextmanager
def queue_lock(out: Path):
    path = out / "scheduler/queue.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def identity(dataset: str, variant: str, seed: int, phase: str) -> dict[str, Any]:
    return {"id": f"{dataset}/{variant}/seed{seed}", "dataset": dataset,
            "variant": variant, "seed": seed, "phase": phase, "state": "pending", "attempts": []}


def screen_jobs() -> list[dict[str, Any]]:
    return [identity(dataset, variant, 41001, "screen") for dataset, variant in (
        ("geant", "routing_sync"), ("geant", "direct_sync"),
        ("abilene", "routing_sync"), ("abilene", "direct_sync"),
        ("geant", "feature_sync"), ("abilene", "feature_sync"))]


def job_dir(out: Path, job: dict[str, Any]) -> Path:
    return out / job["id"]


def lease_path(out: Path, job: dict[str, Any]) -> Path:
    return out / "scheduler/leases" / (job["id"].replace("/", "__") + ".lock")


def verify_registry(out: Path) -> tuple[dict[str, Any], str]:
    digest = sha(out / "registry.json")
    if digest != (out / "registry.sha256").read_text().strip():
        raise RuntimeError("Registry checksum mismatch")
    registry = read_json(out / "registry.json")
    if (registry["deadline_unix"] != DEADLINE_UNIX or tuple(registry["screen_variants"]) != VARIANTS
            or registry["maximum_formal_new_jobs"] != 9):
        raise RuntimeError("Scheduler scope/deadline differs from registry")
    return registry, digest


def load_queue(out: Path, digest: str) -> dict[str, Any]:
    path = out / "scheduler/queue.json"
    if path.exists():
        queue = read_json(path)
        if queue["registry_sha256"] != digest:
            raise RuntimeError("Queue registry mismatch")
        expected = {j["id"] for j in screen_jobs()}
        screen = [j for j in queue["jobs"] if j["phase"] == "screen"]
        if len(screen) != 6 or {j["id"] for j in screen} != expected or len(queue["jobs"]) > 9:
            raise RuntimeError("Queue exceeds or changes frozen experiment scope")
        if len({j["id"] for j in queue["jobs"]}) != len(queue["jobs"]):
            raise RuntimeError("Duplicate job identities in queue")
        if any(j["phase"] not in ("screen", "repeat") for j in queue["jobs"]):
            raise RuntimeError("Unknown queue phase")
        repeats = [j for j in queue["jobs"] if j["phase"] == "repeat"]
        if repeats:
            decision = read_json(out / "decision.json")
            winner = decision.get("winner")
            if decision.get("registry_sha256") != digest or winner not in VARIANTS:
                raise RuntimeError("Repeats require a matching selected candidate")
            allowed = {f"{d}/{winner}/seed41002" for d in DATASETS} | {"abilene/sync_delta/seed41002"}
            if any(j["id"] not in allowed for j in repeats):
                raise RuntimeError("Queue contains an unselected repeat")
        return queue
    return {"registry_sha256": digest, "created": utc(), "deadline_unix": DEADLINE_UNIX,
            "jobs": screen_jobs()}


def save_queue(out: Path, queue: dict[str, Any]) -> None:
    queue["updated"] = utc()
    atomic_json(out / "scheduler/queue.json", queue)


def valid_result(out: Path, job: dict[str, Any], registry: dict[str, Any], digest: str) -> dict[str, Any]:
    path = job_dir(out, job) / "result.json"
    result = read_json(path)
    expected = {"registry_sha256": digest, "dataset": job["dataset"],
                "variant": job["variant"], "seed": job["seed"],
                "initial_state_sha256": registry["initializations"][job["dataset"]][job["variant"]][str(job["seed"])]["state_sha256"]}
    if any(result.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Result identity mismatch")
    evaluation = result["primary_eval"]
    score = float(evaluation["selection_score"])
    if (result.get("role") != "development_only" or evaluation.get("role") != "development_only"
            or evaluation.get("mask_seed") != 71001 or not math.isfinite(score) or score < 0):
        raise RuntimeError("Invalid completed evaluation")
    # The trainer reloads and hashes the checkpoint before committing result.
    # Do not repeatedly read checkpoint tensors while holding the queue lock.
    if not (path.parent / "best.pt").is_file() or not result.get("best_checkpoint_sha256"):
        raise RuntimeError("Completed checkpoint record missing")
    median = float(read_json(path.parent / "latency.json")["median_ms"])
    if not math.isfinite(median) or median <= 0:
        raise RuntimeError("Invalid latency record")
    return {"score": score, "median_ms": median, "result_sha256": sha(path),
            "latency_sha256": sha(path.parent / "latency.json"),
            "best_in_last_five_at_cap": result.get("best_in_last_five_at_cap", False)}


def reconcile_job(out: Path, job: dict[str, Any], registry: dict[str, Any], digest: str,
                  now: float) -> None:
    """Caller holds this job's lease; never infer remote liveness from a PID."""
    directory = job_dir(out, job)
    exit_path = directory / "worker_exit.json"
    exit_record = read_json(exit_path) if exit_path.exists() else None
    if exit_record and exit_record.get("registry_sha256") != digest:
        job.update(state="failed", error="Exit record registry mismatch")
        return
    if (directory / "result.json").exists():
        try:
            summary = valid_result(out, job, registry, digest)
            if exit_record and exit_record.get("returncode") != 0:
                raise RuntimeError("Result exists but managed process exited unsuccessfully")
            job.update(state="complete", summary=summary, finished=utc())
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            job.update(state="failed", error=str(error))
        return
    if exit_record:
        state = "failed" if exit_record.get("returncode") != 0 and not exit_record.get("stop_requested") else "incomplete"
        job.update(state=state, reason="process_exited_without_complete_result", finished=exit_record["finished"])
        return
    if (directory / "failure.json").exists():
        job.update(state="failed", reason="runner_failure_record")
        return
    if job["state"] in TERMINAL:
        return
    if now >= ALLOCATION_CUTOFF:
        job.update(state="incomplete", reason="authorization_deadline", finished=utc())
    elif job["state"] == "running":
        # The inherited job lease is now free and no exit/result/failure was
        # committed: recover only this interrupted identity, with --resume.
        job.update(state="pending", reason="orphaned_worker_recover_same_job")


def choose_candidate(rows: list[dict[str, Any]]) -> tuple[str | None, str, list[str]]:
    eligible = [r for r in rows if r["mean_relative_gain"] > 0]
    preferred = [r for r in eligible if all(g >= 0 for g in r["relative_gains"].values())]
    pool = preferred or eligible
    if not pool:
        return None, "no_positive_mean_improvement", []
    best = max(r["mean_relative_gain"] for r in pool)
    tied = [r for r in pool if best - r["mean_relative_gain"] <= TIE_GAIN + 1e-12]
    chosen = min(tied, key=lambda r: (NOVELTY_PRIORITY.index(r["variant"]), r["mean_normalized_latency"]))
    return chosen["variant"], ("both_wans_nonregressing" if preferred else "uncertain_one_wan_regression"), [r["variant"] for r in tied]


def maybe_decide(out: Path, queue: dict[str, Any], registry: dict[str, Any], digest: str) -> None:
    screen = [j for j in queue["jobs"] if j["phase"] == "screen"]
    if not all(j["state"] in TERMINAL for j in screen):
        return
    path = out / "decision.json"
    if path.exists():
        decision = read_json(path)
        if decision["registry_sha256"] != digest or decision.get("winner") not in (*VARIANTS, None):
            raise RuntimeError("Existing decision identity/winner differs")
    else:
        baselines = {}
        for dataset in DATASETS:
            reference = registry["references"][dataset]["sync_delta"]
            reference_path = Path(reference["path"])
            if sha(reference_path) != reference["sha256"]:
                raise RuntimeError("Reference score source changed")
            latency_path = reference_path.parent / "latency.json"
            score = float(reference["selection_score"])
            median = float(read_json(latency_path)["median_ms"])
            if not math.isfinite(score) or score <= 0 or not math.isfinite(median) or median <= 0:
                raise RuntimeError("Invalid reference score/latency")
            baselines[dataset] = {"selection_score": score, "median_ms": median,
                                  "result_sha256": reference["sha256"], "latency_sha256": sha(latency_path)}
        rows, excluded = [], {}
        for variant in VARIANTS:
            jobs = {j["dataset"]: j for j in screen if j["variant"] == variant}
            if not all(jobs[d]["state"] == "complete" for d in DATASETS):
                excluded[variant] = {d: jobs[d]["state"] for d in DATASETS}
                continue
            gains = {d: (baselines[d]["selection_score"] - jobs[d]["summary"]["score"])
                      / baselines[d]["selection_score"] for d in DATASETS}
            normalized = {d: jobs[d]["summary"]["median_ms"] / baselines[d]["median_ms"] for d in DATASETS}
            rows.append({"variant": variant, "relative_gains": gains,
                         "mean_relative_gain": sum(gains.values()) / 2,
                         "normalized_latencies": normalized,
                         "mean_normalized_latency": sum(normalized.values()) / 2,
                         "results": {d: jobs[d]["summary"] for d in DATASETS}})
        winner, reason, tied = choose_candidate(rows)
        existing_repeat = registry["existing_geant_sync_repeat"]
        if existing_repeat["seed"] != 41002 or sha(Path(existing_repeat["path"])) != existing_repeat["sha256"]:
            raise RuntimeError("Frozen existing GEANT reference repeat changed")
        decision = {"registry_sha256": digest, "created": utc(), "winner": winner,
                    "reason": reason, "role": "development_only", "candidates": rows,
                    "excluded_candidates": excluded, "baselines": baselines,
                    "tie_gain_threshold": TIE_GAIN, "near_tied_variants": tied,
                    "selection_preference": SELECTION_PREFERENCE,
                    "novelty_priority": list(NOVELTY_PRIORITY),
                    "novelty_priority_interpretation": "Task-contribution potential, not demonstrated originality",
                    "tie_break": "frozen task-contribution potential order, then mean normalized median latency",
                    "screen_states": {j["id"]: j["state"] for j in screen},
                    "scope": "At most one candidate gets seed41002; no candidate combinations",
                    "geant_sync_seed41002": {"action": "reuse_only_never_schedule", **existing_repeat}}
        atomic_json(path, decision)
    winner = decision["winner"]
    allowed_repeats = [] if winner is None else [
        identity("geant", winner, 41002, "repeat"), identity("abilene", winner, 41002, "repeat"),
        identity("abilene", "sync_delta", 41002, "repeat")]
    permitted = {j["id"] for j in allowed_repeats}
    if any(j["id"] not in permitted for j in queue["jobs"] if j["phase"] == "repeat"):
        raise RuntimeError("Queue contains an unselected repeat")
    existing = {j["id"] for j in queue["jobs"]}
    queue["jobs"].extend(j for j in allowed_repeats if j["id"] not in existing)
    assert len(queue["jobs"]) <= 9


def claim(out: Path, registry: dict[str, Any], digest: str, owner: dict[str, Any],
          now: float | None = None):
    """Return (job, lease); mutation and selection happen under one CPFS lock."""
    now = time.time() if now is None else now
    with queue_lock(out):
        queue = load_queue(out, digest)
        leases = {}
        try:
            for job in queue["jobs"]:
                lease = try_lock(lease_path(out, job))
                if lease is not None:
                    leases[job["id"]] = lease
                    reconcile_job(out, job, registry, digest, now)
            maybe_decide(out, queue, registry, digest)
            # Repeats can have been appended by the decision in this iteration.
            for job in queue["jobs"]:
                if job["id"] not in leases and job["state"] == "pending":
                    lease = try_lock(lease_path(out, job))
                    if lease is not None:
                        leases[job["id"]] = lease
                        reconcile_job(out, job, registry, digest, now)
            selected = None
            if now < ALLOCATION_CUTOFF and not STOP_REQUESTED:
                for job in queue["jobs"]:
                    if job["state"] == "pending" and job["id"] in leases:
                        attempt = {**owner, "claimed": utc(), "attempt": len(job["attempts"]) + 1}
                        job["attempts"].append(attempt)
                        job.update(state="running", owner=attempt)
                        atomic_json(job_dir(out, job) / "worker_assignment.json", {"registry_sha256": digest, **job})
                        selected = (dict(job), leases.pop(job["id"]))
                        break
            save_queue(out, queue)
            return selected, all(j["state"] in TERMINAL for j in queue["jobs"])
        finally:
            for lease in leases.values():
                lease.close()


def command(out: Path, job: dict[str, Any], stage: str) -> list[str]:
    args = [sys.executable, "-m", "experiments.llm_candidates_runtime_v1.run", stage,
            "--dataset", job["dataset"], "--variant", job["variant"], "--seed", str(job["seed"]),
            "--output", str(out)]
    directory = job_dir(out, job)
    if stage == "train" and any((directory / p).exists() for p in ("run_info.json", "last.pt")):
        args.append("--resume")
    return args


def launch(out: Path, job: dict[str, Any], lease, gpu: int, stage: str, gpu_lease=None) -> dict[str, Any]:
    if STOP_REQUESTED or time.time() >= ALLOCATION_CUTOFF:
        raise AllocationStopped("No subprocess starts during shutdown or after the allocation cutoff")
    directory = job_dir(out, job)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"worker_{stage}.log"
    log = log_path.open("a", buffering=1)
    args = command(out, job, stage)
    log.write(json.dumps({"started": utc(), "command": args, "gpu": gpu}) + "\n")
    log.flush()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    try:
        descriptors = (lease.fileno(),) if gpu_lease is None else (lease.fileno(), gpu_lease.fileno())
        process = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=descriptors)
    except BaseException:
        log.close()
        raise
    return {"job": job, "lease": lease, "gpu": gpu, "stage": stage, "process": process,
            "log": log, "log_path": str(log_path), "started": utc(), "term_time": None}


def finish(out: Path, active: dict[str, Any], digest: str, code: int, *, stop: bool = False) -> None:
    record = {"registry_sha256": digest, "job_id": active["job"]["id"], "stage": active["stage"],
              "returncode": code, "stop_requested": stop, "started": active["started"], "finished": utc(),
              "pid": active["process"].pid, "hostname": socket.gethostname(), "gpu": active["gpu"],
              "log": active["log_path"]}
    atomic_json(job_dir(out, active["job"]) / f"worker_exit_{active['stage']}.json", record)
    atomic_json(job_dir(out, active["job"]) / "worker_exit.json", record)
    active["log"].close()
    active["lease"].close()


def spawn_failed(out: Path, job: dict[str, Any], lease, digest: str, error: Exception) -> None:
    atomic_json(job_dir(out, job) / "worker_exit.json", {
        "registry_sha256": digest, "job_id": job["id"], "stage": "spawn",
        "returncode": 1, "finished": utc(), "error": repr(error),
        "stop_requested": isinstance(error, AllocationStopped)})
    lease.close()


def request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def gpu_description() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                                       text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def refresh_report(out: Path, node_id: str) -> bool:
    """Reporting never runs under the queue lock; failures remain visible."""
    try:
        from .report import summarize
        summarize(out)
        record = {"updated": utc(), "passed": True}
    except Exception as error:
        record = {"updated": utc(), "passed": False, "error": repr(error)}
        with (out / "scheduler" / f"watch_{node_id}.log").open("a") as handle:
            handle.write(json.dumps({"event": "report_failed", **record}) + "\n")
    atomic_json(out / "scheduler/workers" / f"{node_id}_summary_status.json", record)
    return record["passed"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--node-id", choices=("primary", "extra"), required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus or any(g not in (0, 1) for g in args.gpus):
        raise ValueError("Use each physical GPU 0/1 at most once")
    out = args.output.resolve()
    registry, digest = verify_registry(out)
    hostname = socket.gethostname()
    held = []
    for name in [f"node_{args.node_id}", *(f"gpu_{hostname}_{g}" for g in args.gpus)]:
        lease = try_lock(out / "scheduler/workers" / (name + ".lock"))
        if lease is None:
            for acquired in held:
                acquired.close()
            raise RuntimeError(f"Worker/GPU already leased: {name}")
        held.append(lease)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    worker = {"node_id": args.node_id, "hostname": hostname, "worker_pid": os.getpid(),
              "gpus": args.gpus, "gpu_description": gpu_description(), "started": utc(),
              "registry_sha256": digest}
    gpu_leases = dict(zip(args.gpus, held[1:]))
    active: dict[int, dict[str, Any]] = {}
    report_ok = True
    status_path = out / "scheduler/workers" / f"{args.node_id}.json"
    try:
        while True:
            now = time.time()
            decision_existed = (out / "decision.json").exists()
            finished_job = False
            for gpu, task in list(active.items()):
                process = task["process"]
                if (STOP_REQUESTED or now >= DEADLINE_UNIX) and task["term_time"] is None:
                    if process.poll() is None:
                        process.terminate()
                    task["term_time"] = now
                if task["term_time"] is not None and now - task["term_time"] >= 30 and process.poll() is None:
                    # The trainer checkpoints before the deadline; bound an
                    # unresponsive process without inventing a final result.
                    process.kill()
                code = process.poll()
                if code is None:
                    continue
                if task["stage"] == "smoke" and code == 0 and now < ALLOCATION_CUTOFF and not STOP_REQUESTED:
                    smoke = out / "preflight" / task["job"]["dataset"] / task["job"]["variant"] / "REAL_DATA_SMOKE.json"
                    report = read_json(smoke)
                    if report.get("passed") and report.get("registry_sha256") == digest:
                        atomic_json(job_dir(out, task["job"]) / "worker_exit_smoke.json",
                                    {"registry_sha256": digest, "returncode": code, "finished": utc(), "log": task["log_path"]})
                        task["log"].close()
                        try:
                            active[gpu] = launch(out, task["job"], task["lease"], gpu, "train", gpu_leases[gpu])
                        except Exception as error:
                            spawn_failed(out, task["job"], task["lease"], digest, error)
                            del active[gpu]
                            finished_job = True
                        continue
                    code = 1
                finish(out, task, digest, code, stop=STOP_REQUESTED or now >= ALLOCATION_CUTOFF)
                del active[gpu]
                finished_job = True
            all_done = False
            for gpu in args.gpus:
                if STOP_REQUESTED:
                    break
                if gpu in active:
                    continue
                selected, all_done = claim(out, registry, digest, {**worker, "gpu": gpu}, now)
                if selected:
                    job, lease = selected
                    try:
                        active[gpu] = launch(out, job, lease, gpu, "smoke", gpu_leases[gpu])
                    except Exception as error:
                        spawn_failed(out, job, lease, digest, error)
                        finished_job = True
            if finished_job or (not decision_existed and (out / "decision.json").exists()):
                report_ok = refresh_report(out, args.node_id)
            atomic_json(status_path, {**worker, "updated": utc(), "state": "stopping" if STOP_REQUESTED or now >= ALLOCATION_CUTOFF else "running",
                "active": {str(g): {"job_id": t["job"]["id"], "stage": t["stage"], "pid": t["process"].pid} for g, t in active.items()}})
            if not active and (all_done or STOP_REQUESTED or now >= DEADLINE_UNIX):
                break
            time.sleep(2)
    finally:
        # A local scheduler error must not leave unsupervised GPU work behind.
        for task in active.values():
            process = task["process"]
            if process.poll() is None:
                process.terminate()
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            finish(out, task, digest, code, stop=True)
        report_ok = refresh_report(out, args.node_id)
        atomic_json(status_path, {**worker, "updated": utc(),
            "state": "stopped" if report_ok else "stopped_with_report_error", "active": {},
            "report_passed": report_ok})
        for lease in held:
            lease.close()
    if not report_ok:
        raise RuntimeError("Worker stopped, but final summary failed; see scheduler/watch log")


if __name__ == "__main__":
    main()
