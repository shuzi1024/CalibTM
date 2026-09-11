"""Explicit exact single-job retry runner for audited infrastructure failures."""

from __future__ import annotations

import argparse as _argparse
import os as _os
from pathlib import Path as _Path
import subprocess as _subprocess
import sys as _sys
from typing import Mapping as _Mapping

from .config import (
    CUBLAS_WORKSPACE_CONFIG as _CUBLAS_WORKSPACE_CONFIG,
    canonical_json_bytes as _canonical_json_bytes,
    protocol_config_sha256 as _protocol_config_sha256,
)
from .gpu_queue import _worker_command as _worker_command
from .recovery import (
    _active_freeze_hashes as _active_freeze_hashes,
    _canonical_gpu as _canonical_gpu,
    _EX_TEMPFAIL as _EX_TEMPFAIL,
    _fsync_directory as _fsync_directory,
    _lexists as _lexists,
    _mkdir_plain as _mkdir_plain,
    _plain_directory as _plain_directory,
    _resolve_active_job as _resolve_active_job,
    _retry_directory as _retry_directory,
    _retry_paths as _retry_paths,
    _started_payload as _started_payload,
    _validate_archive_chain as _validate_archive_chain,
)
from .result_io import (
    _read_canonical_json as _read_canonical_json,
    load_job_result as _load_job_result,
    regular_file_sha256 as _regular_file_sha256,
    write_canonical_json_exclusive as _write_canonical_json_exclusive,
)


def _build_parser() -> _argparse.ArgumentParser:
    parser = _argparse.ArgumentParser(
        description="Explicitly retry one archived ACIL-Innovation infrastructure failure"
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", required=True, type=_Path)
    parser.add_argument("--gpu", required=True)
    return parser


def _classify_worker_outcome(
    *,
    returncode: int | None,
    launch_error: str | None,
    stage: str,
    job,
    output_root: _Path,
    manifest_sha256: str,
    provenance_sha256: str,
) -> tuple[str, int | None, int | None, str | None, str | None]:
    """Classify control artifacts only; never open an unknown partial file."""

    staging = output_root / stage / f".{job.job_id}.inprogress"
    final = output_root / stage / job.job_id
    if launch_error is not None:
        if _lexists(staging) or _lexists(final):
            raise RuntimeError("failed worker launch unexpectedly mutated job artifacts")
        return "launch_failed", None, None, None, None
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise RuntimeError("retry worker did not return an integer exit code")
    signum = -returncode if returncode < 0 else None
    final_exists = _lexists(final)
    staging_exists = _lexists(staging)
    if final_exists and staging_exists:
        raise RuntimeError("retry produced both final and in-progress job directories")
    if final_exists:
        _plain_directory(final, label="retry final job")
        payload = _load_job_result(
            stage=stage, job_id=job.job_id, output_root=output_root
        )
        if not isinstance(payload, _Mapping):
            raise ValueError("retry final result is not an object")
        if (
            payload.get("manifest_sha256") != manifest_sha256
            or payload.get("provenance_sha256") != provenance_sha256
        ):
            raise ValueError("retry final result is not bound to the active freeze")
        status = payload.get("status")
        if status == "succeeded" and returncode != 0:
            raise RuntimeError("successful retry returned a nonzero exit code")
        if status == "algorithmic_failure" and returncode == 0:
            raise RuntimeError("algorithmic retry failure returned zero")
        if status not in {"succeeded", "algorithmic_failure"}:
            raise ValueError("retry final result has an unknown status")
        started_sha256 = payload.get("started_sha256")
        if not isinstance(started_sha256, str):
            raise ValueError("retry final result has no started hash")
        return str(status), returncode, signum, started_sha256, None
    if staging_exists:
        _plain_directory(staging, label="retry in-progress job")
        if _lexists(staging / "result.json"):
            _, started_sha256 = _started_payload(
                staging / "started.json",
                job_payload=job.to_json(),
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
            )
            return "unpublished_result", returncode, signum, started_sha256, None
        _, started_sha256 = _started_payload(
            staging / "started.json",
            job_payload=job.to_json(),
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
        )
        if returncode == 0:
            return "zero_exit_orphan", returncode, signum, started_sha256, None
        if returncode == _EX_TEMPFAIL:
            return "failed", returncode, None, started_sha256, "ex_tempfail"
        if signum is not None:
            return "terminated", returncode, signum, started_sha256, "signal"
        return "non_retryable_exit", returncode, None, started_sha256, None
    if returncode == _EX_TEMPFAIL:
        failure_kind = "ex_tempfail"
    elif signum is not None:
        failure_kind = "signal"
    else:
        failure_kind = None
    return "failed_without_orphan", returncode, signum, None, failure_kind


def retry_infrastructure_job(
    stage: str, job_id: str, output_root: _Path, gpu: str
) -> _Path:
    """Run the identical frozen worker once after one validated archived failure."""

    job = _resolve_active_job(stage, job_id)
    gpu = _canonical_gpu(gpu)
    root = _Path(output_root)
    manifest_sha256, provenance_sha256, freeze_sha256 = _active_freeze_hashes()
    if not _lexists(root):
        raise FileNotFoundError("an infrastructure attempt archive is required before retry")
    _plain_directory(root, label="output root")
    archive_parent = (
        root / "_infrastructure_attempts" / stage / job.job_id
    )
    if not _lexists(archive_parent):
        raise FileNotFoundError(
            "an infrastructure attempt archive is required before retry"
        )
    stage_root = root / stage
    _plain_directory(stage_root, label="stage output")
    final = stage_root / job.job_id
    staging = stage_root / f".{job.job_id}.inprogress"
    if _lexists(final):
        raise FileExistsError("a final or completed job can never be retried")
    if _lexists(staging):
        raise RuntimeError("an in-progress orphan must be archived before another retry")

    attempts = _validate_archive_chain(
        stage=stage,
        job=job,
        output_root=root,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        freeze_sha256=freeze_sha256,
    )
    if not attempts:
        raise FileNotFoundError("an infrastructure attempt archive is required before retry")
    number = len(attempts)
    source_attempt = f"attempt-{number:04d}"
    source_incident_sha256 = str(attempts[-1]["incident_sha256"])
    summary_path, log_path = _retry_paths(root, stage, job.job_id, number)
    if _lexists(summary_path) or _lexists(log_path):
        raise FileExistsError("retry summary/log is append-only and already exists")

    # Mutate only the isolated operational retry namespace after the full chain
    # has validated.  The worker command has no GPU, seed, data, or method override.
    _mkdir_plain(root / "_infrastructure_retries")
    _mkdir_plain(root / "_infrastructure_retries" / stage)
    _mkdir_plain(_retry_directory(root, stage, job.job_id))
    command = tuple(_worker_command(stage, job, root))
    environment = _os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    launch_error: str | None = None
    returncode: int | None = None
    with log_path.open("xb") as log_handle:
        try:
            process = _subprocess.Popen(
                command,
                stdin=_subprocess.DEVNULL,
                stdout=log_handle,
                stderr=_subprocess.STDOUT,
                env=environment,
                close_fds=True,
                start_new_session=True,
            )
        except Exception as exc:
            launch_error = f"{type(exc).__name__}: {exc}"
        else:
            returncode = int(process.wait())
        log_handle.flush()
        _os.fsync(log_handle.fileno())
    log_sha256 = _regular_file_sha256(log_path)
    state, exit_code, signum, started_sha256, failure_kind = _classify_worker_outcome(
        returncode=returncode,
        launch_error=launch_error,
        stage=stage,
        job=job,
        output_root=root,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )
    retry_name = f"retry-{number:04d}"
    summary = {
        "command": list(command),
        "config_sha256": _protocol_config_sha256(),
        "exit_code": exit_code,
        "failure_kind": failure_kind,
        "freeze_sha256": freeze_sha256,
        "gpu": gpu,
        "job": job.to_json(),
        "launch_error": launch_error,
        "log": log_path.relative_to(root).as_posix(),
        "log_sha256": log_sha256,
        "manifest_sha256": manifest_sha256,
        "protocol": "acil-innovation-v1",
        "provenance_sha256": provenance_sha256,
        "retry": retry_name,
        "schema": "acil-innovation-v1:infrastructure-retry:v1",
        "signal": signum,
        "source_attempt": source_attempt,
        "source_incident_sha256": source_incident_sha256,
        "started_sha256": started_sha256,
        "state": state,
    }
    _write_canonical_json_exclusive(summary_path, summary)
    _fsync_directory(summary_path.parent)
    return summary_path


def main() -> int:
    arguments = _build_parser().parse_args()
    summary_path = retry_infrastructure_job(
        arguments.stage,
        arguments.job_id,
        arguments.output_root,
        arguments.gpu,
    )
    payload, _ = _read_canonical_json(summary_path)
    print(
        _canonical_json_bytes(
            {"retry_summary": str(summary_path), "state": payload["state"]}
        ).decode("ascii"),
        flush=True,
    )
    exit_code = payload.get("exit_code")
    if payload.get("state") == "succeeded":
        return 0
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return exit_code if exit_code > 0 else 128 + abs(exit_code)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "retry_infrastructure_job"]
