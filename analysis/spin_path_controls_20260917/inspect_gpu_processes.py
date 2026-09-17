#!/usr/bin/env python3
"""Read-only inventory of this experiment's processes and both host GPUs.

Run on each authorized Linux host. This never kills a process. It deliberately
prints only matching experiment command lines, not unrelated users' arguments.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    filenames = {"launch_queue.py", "benchmark_inference.py", "benchmark_inference_abilene.py"}
    namespace = "/analysis/spin_path_controls_20260917/"
    processes = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            arguments = [arg.decode(errors="replace") for arg in (entry / "cmdline").read_bytes().split(b"\0") if arg]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        matches = any(arg == "experiments.spin_path_controls_v1.runner" or
                      (namespace in arg and Path(arg).name in filenames) for arg in arguments)
        if matches:
            processes.append({"pid": int(entry.name), "arguments": arguments})
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True).strip()
    print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
                      "experiment_processes": processes, "gpus_csv": gpu,
                      "compute_processes_csv": compute, "read_only": True}))


if __name__ == "__main__":
    main()
