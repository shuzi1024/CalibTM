#!/usr/bin/env python3
"""Summarize finished GPU process intervals after read-only idle checks.

This does not stop jobs. Capture inspect_gpu_processes.py on both hosts first.
Process wall time includes loading, CPU work and I/O; it is not GPU utilization
time, energy, or a billing measurement. Prior 120-epoch training is excluded.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import time

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    auth = read(HERE / "AUTHORIZATION.json")
    inventories = {host: read(HERE / f"GPU_FINAL_{host}.json") for host in auth["hosts"]}
    for inventory in inventories.values():
        assert not inventory["experiment_processes"]
        assert not inventory["compute_processes_csv"]
        observed = datetime.fromisoformat(inventory["utc"]).timestamp()
        assert 0 <= time.time() - observed < 600, "Idle inventory must be recent"
    intervals = []
    for path in sorted((HERE / "raw_outputs/logs").glob("*.exit.json")):
        record = read(path)
        assert record["returncode"] == 0
        assert record["finished_unix"] <= auth["deadline_unix"]
        assert record["finished_unix"] >= record["started_unix"]
        intervals.append(dict(path=str(path.relative_to(HERE)), kind="smoke" if "--smoke" in record["command"] else "train",
                              hostname=record["hostname"], gpu=record["gpu"],
                              start=record["started_unix"], finish=record["finished_unix"]))
    assert len(intervals) == 24, "Expect 12 smoke and 12 formal processes"
    assert sum(row["kind"] == "train" for row in intervals) == 12
    for suffix in ("", "_ABILENE"):
        record = read(HERE / f"INFERENCE_BENCHMARK{suffix}.json")
        launch = read(HERE / f"INFERENCE_BENCHMARK{suffix}_LAUNCH.json")
        assert record["state"] == "complete"
        assert record["finished_unix"] <= auth["deadline_unix"]
        intervals.append(dict(path=f"INFERENCE_BENCHMARK{suffix}.json", kind="inference_benchmark",
                              hostname=record["hostname"], gpu=launch["gpu"],
                              start=record["started_unix"], finish=record["finished_unix"]))
    events = []
    for row in intervals:
        events.extend([(row["start"], 1), (row["finish"], -1)])
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    assert active == 0 and peak <= auth["max_gpu_count"]
    per_device = {}
    for row in intervals:
        per_device.setdefault((row["hostname"], row["gpu"]), []).append(row)
    for key, rows in per_device.items():
        rows.sort(key=lambda row: row["start"])
        assert all(left["finish"] <= right["start"] for left, right in zip(rows, rows[1:])), key
    summaries = {}
    for dataset, dirname in (("geant", "summary"), ("abilene", "summary_abilene")):
        path = HERE / dirname / "SUMMARY.json"
        summary = read(path)
        assert summary["all_six_jobs_complete"]
        assert all(not job["errors"] and not job["warnings"]
                   for jobs in summary["jobs"].values() for job in jobs.values())
        summaries[dataset] = hashlib.sha256(path.read_bytes()).hexdigest()
    last_finish = max(row["finish"] for row in intervals)
    start = datetime.fromisoformat(auth["started_utc"]).timestamp()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "state": "complete",
              "authorization": auth, "gpu_process_inventories": inventories,
              "summary_sha256": summaries, "complete_trajectories": 12,
              "new_from_scratch_trajectories": 8, "continued_prior_trajectories": 4,
              "max_simultaneous_processes": peak, "all_finished_before_deadline": True,
              "allocation_to_last_gpu_work_wall_hours": (last_finish - start) / 3600,
              "sum_recorded_gpu_process_wall_hours": sum(row["finish"] - row["start"] for row in intervals) / 3600,
              "process_time_definition": "12 smoke + 12 formal process wall intervals + 2 inference benchmarks; includes CPU preparation, loading and I/O; excludes historical parent training, idle allocation, launch overhead and CPU-only checks; not utilization/billing",
              "last_gpu_work_beijing": datetime.fromtimestamp(last_finish, timezone(timedelta(hours=8))).isoformat(),
              "intervals": intervals, "no_queued_followups": True,
              "next_experiments_started": False, "external_git_push_attempted": False}
    target = HERE / "GPU_CLOSEOUT.json"
    if target.exists():
        raise RuntimeError("Preserve the previous closeout record")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("intervals", "authorization", "gpu_process_inventories")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
