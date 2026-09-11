"""Fail-closed archival for explicit, exact infrastructure retries.

Only canonical control artifacts are opened here.  Unknown files in a worker's
in-progress directory are moved as opaque bytes and are never read or parsed.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path
import re as _re
import stat as _stat
from typing import Any as _Any, Mapping as _Mapping, Sequence as _Sequence

from .config import protocol_config_sha256 as _protocol_config_sha256
from .execution import (
    _active_freeze as _active_freeze,
    _read_json_exact as _read_json_exact,
    resolve_planned_job as _resolve_planned_job,
)
from .gpu_queue import _worker_command as _worker_command
from .manifest import active_manifest_stages as _active_manifest_stages
from .result_io import (
    _read_canonical_json as _read_canonical_json,
    _rename_directory_noreplace as _rename_directory_noreplace,
    regular_file_sha256 as _regular_file_sha256,
    write_canonical_json_exclusive as _write_canonical_json_exclusive,
)


_ARCHIVABLE_STATES = frozenset({"failed", "terminated", "aborted"})
_EX_TEMPFAIL = getattr(_os, "EX_TEMPFAIL", 75)
_ATTEMPT_NAME = _re.compile(r"attempt-([0-9]{4})\Z")
_RETRY_NAME = _re.compile(r"retry-([0-9]{4})\.(json|log)\Z")
_RETRY_SCHEMA = "acil-innovation-v1:infrastructure-retry:v1"
_WORKER_MODULE = "experiments.acil_innovation_v1.worker"


def _lexists(path: _Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _plain_directory(path: _Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} is missing") from None
    except OSError as exc:
        raise ValueError(f"{label} is inaccessible") from exc
    if _stat.S_ISLNK(metadata.st_mode) or not _stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a plain directory")


def _plain_regular_file(path: _Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is inaccessible") from exc
    if _stat.S_ISLNK(metadata.st_mode) or not _stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a plain regular file")


def _require_sha256(value: object, *, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_gpu(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("GPU must be one canonical nonnegative integer string")
    if not value.isdecimal() or str(int(value)) != value:
        raise ValueError("GPU must be one canonical nonnegative integer string")
    return value


def _active_freeze_hashes() -> tuple[str, str, str]:
    """Return the three currently rebuilt and fixed freeze identities."""

    manifest_sha256, provenance_sha256 = _active_freeze()
    from .scripts.build_freeze import _FREEZE_DIRECTORY

    root = _Path(__file__).resolve().parent / "generated" / _FREEZE_DIRECTORY
    anchor = _read_json_exact(root / "anchor.json")
    if not isinstance(anchor, _Mapping):
        raise ValueError("active freeze anchor is not an object")
    freeze_sha256 = _require_sha256(
        anchor.get("freeze_sha256"), label="freeze hash"
    )
    if (
        anchor.get("manifest_sha256") != manifest_sha256
        or anchor.get("provenance_sha256") != provenance_sha256
    ):
        raise ValueError("active freeze anchor hash binding drifted")
    return manifest_sha256, provenance_sha256, freeze_sha256


def _resolve_active_job(stage: str, job_id: str):
    if not isinstance(stage, str) or stage not in _active_manifest_stages():
        raise ValueError(f"stage {stage!r} is outside the active manifest")
    try:
        return _resolve_planned_job(stage, job_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("job ID is not a registered job in the active manifest") from exc


def _started_payload(
    path: _Path,
    *,
    job_payload: _Mapping[str, object],
    manifest_sha256: str,
    provenance_sha256: str,
) -> tuple[dict[str, _Any], str]:
    value, artifact = _read_canonical_json(path)
    expected = {
        "schema": "acil-innovation-v1:job-started:v1",
        "status": "started",
        "job": dict(job_payload),
        "config_sha256": _protocol_config_sha256(),
        "manifest_sha256": manifest_sha256,
        "provenance_sha256": provenance_sha256,
    }
    if value != expected:
        raise ValueError("in-progress started identity differs from the active freeze")
    return dict(value), artifact.sha256


def _failure_evidence(
    state: object, exit_code: object, signum: object
) -> tuple[int | None, str]:
    if state not in _ARCHIVABLE_STATES:
        raise RuntimeError(
            "attempt is live, successful, algorithmic, or not infrastructure-archivable"
        )
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise ValueError("infrastructure exit code is invalid")
    if signum is not None and (
        isinstance(signum, bool) or not isinstance(signum, int) or signum <= 0
    ):
        raise ValueError("infrastructure signal is invalid")
    if exit_code == _EX_TEMPFAIL:
        derived_signal = None
        failure_kind = "ex_tempfail"
    elif isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code < 0:
        derived_signal = -exit_code
        failure_kind = "signal"
    else:
        raise RuntimeError(
            "only EX_TEMPFAIL=75 or a negative worker signal is infrastructure-archivable"
        )
    if signum is not None and signum != derived_signal:
        raise ValueError("recorded worker signal differs from the negative return code")
    if state == "terminated" and derived_signal is None:
        raise RuntimeError("terminated infrastructure attempt has no signal")
    return derived_signal, failure_kind


def _queue_attempt(
    path: _Path,
    *,
    stage: str,
    job_id: str,
    output_root: _Path,
) -> tuple[dict[str, _Any], str, int | None, str]:
    value, artifact = _read_canonical_json(path)
    if not isinstance(value, _Mapping):
        raise ValueError("queue summary is not an object")
    required = {
        "completed_job_count",
        "gpus",
        "jobs",
        "launched_job_count",
        "planned_job_count",
        "queue_directory",
        "schema_version",
        "signal",
        "stage",
        "status",
        "worker_module",
    }
    if set(value) != required:
        raise ValueError("queue summary schema drifted")
    if (
        value.get("schema_version") != 1
        or value.get("stage") != stage
        or value.get("worker_module") != _WORKER_MODULE
        or value.get("queue_directory")
        != str(output_root / "_gpu_queue" / stage)
    ):
        raise ValueError("queue summary identity drifted")
    rows = value.get("jobs")
    if not isinstance(rows, list):
        raise ValueError("queue summary job grid is absent")
    from .jobs import planned_jobs as _planned_jobs

    jobs = _planned_jobs(stage)
    if value.get("planned_job_count") != len(jobs) or len(rows) != len(jobs):
        raise ValueError("queue summary planned grid size drifted")
    row_keys = {
        "attempt_count",
        "command",
        "dataset",
        "exit_code",
        "gpu",
        "job_id",
        "launch_error",
        "log_path",
        "method",
        "seed_bundle",
        "stage",
        "state",
    }
    matches: list[dict[str, _Any]] = []
    launched = 0
    completed = 0
    completed_states = {"succeeded", "failed", "terminated", "aborted"}
    for planned, supplied in zip(jobs, rows):
        if not isinstance(supplied, _Mapping) or set(supplied) != row_keys:
            raise ValueError("queue summary job row schema drifted")
        row = dict(supplied)
        if any(
            row.get(name) != expected
            for name, expected in {
                "dataset": planned.dataset,
                "job_id": planned.job_id,
                "method": planned.method,
                "seed_bundle": planned.seed_bundle,
                "stage": planned.stage,
            }.items()
        ):
            raise ValueError("queue summary planned job identity drifted")
        attempt_count = row.get("attempt_count")
        if isinstance(attempt_count, bool) or attempt_count not in {0, 1}:
            raise ValueError("queue summary attempt count is invalid")
        launched += int(attempt_count == 1)
        completed += int(row.get("state") in completed_states)
        if planned.job_id == job_id:
            matches.append(row)
    if (
        value.get("launched_job_count") != launched
        or value.get("completed_job_count") != completed
    ):
        raise ValueError("queue summary aggregate counts drifted")
    if len(matches) != 1:
        raise ValueError("queue summary does not bind exactly one requested job")
    row = matches[0]
    planned = _resolve_active_job(stage, job_id)
    expected_command = list(_worker_command(stage, planned, output_root))
    expected_log = str(
        output_root / "_gpu_queue" / stage / "logs" / f"{job_id}.log"
    )
    if (
        row.get("command") != expected_command
        or row.get("attempt_count") != 1
        or row.get("launch_error") is not None
        or row.get("log_path") != expected_log
        or _canonical_gpu(row.get("gpu")) not in value.get("gpus", [])
    ):
        raise ValueError("queue attempt differs from the frozen worker launch")
    queue_signal = value.get("signal")
    if queue_signal is not None and (
        isinstance(queue_signal, bool)
        or not isinstance(queue_signal, int)
        or queue_signal <= 0
    ):
        raise ValueError("queue-level signal is invalid")
    worker_signal, failure_kind = _failure_evidence(
        row.get("state"), row.get("exit_code"), None
    )
    return row, artifact.sha256, worker_signal, failure_kind


def _attempt_directories(parent: _Path) -> tuple[_Path, ...]:
    if not _lexists(parent):
        return ()
    _plain_directory(parent, label="infrastructure job archive")
    attempts: list[_Path] = []
    for expected_number, child in enumerate(
        sorted(parent.iterdir(), key=lambda item: item.name), start=1
    ):
        metadata = child.lstat()
        match = _ATTEMPT_NAME.fullmatch(child.name)
        if (
            match is None
            or int(match.group(1)) != expected_number
            or _stat.S_ISLNK(metadata.st_mode)
            or not _stat.S_ISDIR(metadata.st_mode)
        ):
            raise ValueError("infrastructure attempts must be contiguous plain directories")
        attempts.append(child)
    return tuple(attempts)


def _validate_opaque_tree(root: _Path) -> None:
    """Reject links/special entries without opening any opaque worker payload."""

    pending = [root]
    while pending:
        directory = pending.pop()
        _plain_directory(directory, label="infrastructure attempt directory")
        for child in directory.iterdir():
            metadata = child.lstat()
            if _stat.S_ISLNK(metadata.st_mode):
                raise ValueError("infrastructure attempt contains a symlink")
            if _stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
            elif not _stat.S_ISREG(metadata.st_mode):
                raise ValueError("infrastructure attempt contains a non-regular entry")


def _retry_directory(output_root: _Path, stage: str, job_id: str) -> _Path:
    return output_root / "_infrastructure_retries" / stage / job_id


def _retry_paths(
    output_root: _Path, stage: str, job_id: str, number: int
) -> tuple[_Path, _Path]:
    parent = _retry_directory(output_root, stage, job_id)
    stem = f"retry-{number:04d}"
    return parent / f"{stem}.json", parent / f"{stem}.log"


def _validate_retry_directory(
    output_root: _Path,
    *,
    stage: str,
    job_id: str,
    expected_count: int,
) -> None:
    parent = _retry_directory(output_root, stage, job_id)
    if expected_count == 0 and not _lexists(parent):
        return
    _plain_directory(parent, label="infrastructure retry archive")
    expected = {
        path.name
        for number in range(1, expected_count + 1)
        for path in _retry_paths(output_root, stage, job_id, number)
    }
    actual = {path.name for path in parent.iterdir()}
    if actual != expected:
        raise ValueError("retry summaries/logs are missing, extra, or non-contiguous")
    for child in parent.iterdir():
        _plain_regular_file(child, label="infrastructure retry artifact")


def _retry_artifact_count(output_root: _Path, *, stage: str, job_id: str) -> int:
    parent = _retry_directory(output_root, stage, job_id)
    if not _lexists(parent):
        return 0
    _plain_directory(parent, label="infrastructure retry archive")
    by_number: dict[int, set[str]] = {}
    for child in parent.iterdir():
        _plain_regular_file(child, label="infrastructure retry artifact")
        match = _RETRY_NAME.fullmatch(child.name)
        if match is None:
            raise ValueError("retry archive contains a malformed artifact name")
        number = int(match.group(1))
        by_number.setdefault(number, set()).add(match.group(2))
    if any(kinds != {"json", "log"} for kinds in by_number.values()):
        raise ValueError("retry summary/log pair is incomplete")
    numbers = sorted(by_number)
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("retry summary/log pairs are not contiguous")
    return len(numbers)


def _retry_summary(
    path: _Path,
    *,
    number: int,
    stage: str,
    job,
    output_root: _Path,
    source_incident_sha256: str,
    manifest_sha256: str,
    provenance_sha256: str,
    freeze_sha256: str,
    archivable_only: bool,
) -> tuple[dict[str, _Any], str]:
    value, artifact = _read_canonical_json(path)
    if not isinstance(value, _Mapping):
        raise ValueError("retry summary is not an object")
    expected_keys = {
        "command",
        "config_sha256",
        "exit_code",
        "failure_kind",
        "freeze_sha256",
        "gpu",
        "job",
        "launch_error",
        "log",
        "log_sha256",
        "manifest_sha256",
        "protocol",
        "provenance_sha256",
        "retry",
        "schema",
        "signal",
        "source_attempt",
        "source_incident_sha256",
        "started_sha256",
        "state",
    }
    if set(value) != expected_keys:
        raise ValueError("retry summary schema drifted")
    retry_name = f"retry-{number:04d}"
    expected_command = list(_worker_command(stage, job, output_root))
    expected_log_path = _retry_paths(output_root, stage, job.job_id, number)[1]
    expected_identity = {
        "schema": _RETRY_SCHEMA,
        "protocol": "acil-innovation-v1",
        "retry": retry_name,
        "source_attempt": f"attempt-{number:04d}",
        "source_incident_sha256": source_incident_sha256,
        "job": job.to_json(),
        "command": expected_command,
        "config_sha256": _protocol_config_sha256(),
        "manifest_sha256": manifest_sha256,
        "provenance_sha256": provenance_sha256,
        "freeze_sha256": freeze_sha256,
        "log": expected_log_path.relative_to(output_root).as_posix(),
    }
    if any(value.get(name) != expected for name, expected in expected_identity.items()):
        raise ValueError("retry summary identity, command, source, or freeze drifted")
    _canonical_gpu(value.get("gpu"))
    expected_log_hash = _require_sha256(
        value.get("log_sha256"), label="retry log hash"
    )
    if _regular_file_sha256(expected_log_path) != expected_log_hash:
        raise ValueError("retry operational log hash drifted")
    state = value.get("state")
    exit_code = value.get("exit_code")
    signum = value.get("signal")
    failure_kind = value.get("failure_kind")
    launch_error = value.get("launch_error")
    started_sha256 = value.get("started_sha256")
    if state in _ARCHIVABLE_STATES:
        if launch_error is not None:
            raise ValueError("archivable retry unexpectedly records a launch error")
        derived_signal, derived_kind = _failure_evidence(state, exit_code, signum)
        if signum != derived_signal or failure_kind != derived_kind:
            raise ValueError("retry infrastructure failure taxonomy drifted")
        _require_sha256(started_sha256, label="retry started hash")
    elif state == "succeeded":
        if (
            exit_code != 0
            or signum is not None
            or failure_kind is not None
            or launch_error is not None
        ):
            raise ValueError("successful retry control outcome drifted")
        _require_sha256(started_sha256, label="retry started hash")
    elif state == "algorithmic_failure":
        if (
            exit_code != 20
            or signum is not None
            or failure_kind is not None
            or launch_error is not None
        ):
            raise ValueError("algorithmic retry control outcome drifted")
        _require_sha256(started_sha256, label="retry started hash")
    elif state == "launch_failed":
        if (
            not isinstance(launch_error, str)
            or not launch_error
            or exit_code is not None
            or signum is not None
            or failure_kind is not None
            or started_sha256 is not None
        ):
            raise ValueError("launch-failed retry control outcome drifted")
    elif state == "failed_without_orphan":
        if launch_error is not None or started_sha256 is not None:
            raise ValueError("orphan-free retry control outcome drifted")
        if exit_code == _EX_TEMPFAIL:
            expected_signal, expected_kind = None, "ex_tempfail"
        elif isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code < 0:
            expected_signal, expected_kind = -exit_code, "signal"
        else:
            expected_signal, expected_kind = None, None
        if signum != expected_signal or failure_kind != expected_kind:
            raise ValueError("orphan-free retry failure taxonomy drifted")
    elif state in {"non_retryable_exit", "zero_exit_orphan", "unpublished_result"}:
        if (
            launch_error is not None
            or signum is not None
            or failure_kind is not None
            or isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
        ):
            raise ValueError("non-retryable worker outcome drifted")
        if state == "non_retryable_exit" and (
            exit_code <= 0 or exit_code == _EX_TEMPFAIL
        ):
            raise ValueError("non-retryable exit taxonomy drifted")
        if state == "zero_exit_orphan" and exit_code != 0:
            raise ValueError("zero-exit orphan taxonomy drifted")
        if started_sha256 is not None:
            _require_sha256(started_sha256, label="retry started hash")
    else:
        raise ValueError("retry summary state is unknown")
    if archivable_only and state not in _ARCHIVABLE_STATES:
        raise RuntimeError("retry summary is not an archivable infrastructure attempt")
    return dict(value), artifact.sha256


def _retry_attempt(
    path: _Path,
    *,
    number: int,
    stage: str,
    job,
    output_root: _Path,
    source_incident_sha256: str,
    manifest_sha256: str,
    provenance_sha256: str,
    freeze_sha256: str,
) -> tuple[dict[str, _Any], str]:
    return _retry_summary(
        path,
        number=number,
        stage=stage,
        job=job,
        output_root=output_root,
        source_incident_sha256=source_incident_sha256,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        freeze_sha256=freeze_sha256,
        archivable_only=True,
    )


def _incident_payload(
    *,
    attempt_name: str,
    job,
    command: _Sequence[str],
    exit_code: int | None,
    failure_kind: str,
    signum: int | None,
    state: str,
    started_sha256: str,
    source_path: _Path,
    source_sha256: str,
    output_root: _Path,
    manifest_sha256: str,
    provenance_sha256: str,
    freeze_sha256: str,
) -> dict[str, object]:
    return {
        "attempt": attempt_name,
        "command": list(command),
        "config_sha256": _protocol_config_sha256(),
        "exit_code": exit_code,
        "failure_kind": failure_kind,
        "freeze_sha256": freeze_sha256,
        "job": job.to_json(),
        "manifest_sha256": manifest_sha256,
        "protocol": "acil-innovation-v1",
        "provenance_sha256": provenance_sha256,
        "queue_summary": source_path.relative_to(output_root).as_posix(),
        "queue_summary_sha256": source_sha256,
        "schema": "acil-innovation-v1:infrastructure-attempt:v1",
        "signal": signum,
        "started_sha256": started_sha256,
        "state": state,
    }


def _validate_archive_chain(
    *,
    stage: str,
    job,
    output_root: _Path,
    manifest_sha256: str,
    provenance_sha256: str,
    freeze_sha256: str,
    expected_retry_count: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Semantically validate every archived incident without opening partials."""

    parent = output_root / "_infrastructure_attempts" / stage / job.job_id
    attempts = _attempt_directories(parent)
    retry_count = max(0, len(attempts) - 1)
    if expected_retry_count is not None:
        retry_count = expected_retry_count
    _validate_retry_directory(
        output_root,
        stage=stage,
        job_id=job.job_id,
        expected_count=retry_count,
    )
    records: list[dict[str, object]] = []
    previous_incident_sha256: str | None = None
    queue_path = output_root / "_gpu_queue" / stage / "queue_summary.json"
    for number, attempt in enumerate(attempts, start=1):
        _validate_opaque_tree(attempt)
        _, started_sha256 = _started_payload(
            attempt / "started.json",
            job_payload=job.to_json(),
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
        )
        if number == 1:
            row, source_sha256, signum, failure_kind = _queue_attempt(
                queue_path,
                stage=stage,
                job_id=job.job_id,
                output_root=output_root,
            )
            source_path = queue_path
            command = row["command"]
            exit_code = row["exit_code"]
            state = row["state"]
        else:
            if previous_incident_sha256 is None:
                raise RuntimeError("infrastructure chain lost its previous incident")
            source_path, _ = _retry_paths(
                output_root, stage, job.job_id, number - 1
            )
            retry, source_sha256 = _retry_attempt(
                source_path,
                number=number - 1,
                stage=stage,
                job=job,
                output_root=output_root,
                source_incident_sha256=previous_incident_sha256,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
                freeze_sha256=freeze_sha256,
            )
            if retry["started_sha256"] != started_sha256:
                raise ValueError("retry summary started hash differs from archived attempt")
            command = retry["command"]
            exit_code = retry["exit_code"]
            failure_kind = retry["failure_kind"]
            signum = retry["signal"]
            state = retry["state"]
        expected = _incident_payload(
            attempt_name=attempt.name,
            job=job,
            command=command,
            exit_code=exit_code,
            failure_kind=str(failure_kind),
            signum=signum,
            state=state,
            started_sha256=started_sha256,
            source_path=source_path,
            source_sha256=source_sha256,
            output_root=output_root,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            freeze_sha256=freeze_sha256,
        )
        incident, incident_artifact = _read_canonical_json(attempt / "incident.json")
        if incident != expected:
            raise ValueError("infrastructure incident semantic binding drifted")
        previous_incident_sha256 = incident_artifact.sha256
        records.append(
            {
                "attempt": attempt.name,
                "incident": dict(expected),
                "incident_sha256": incident_artifact.sha256,
                "started_sha256": started_sha256,
            }
        )
    return tuple(records)


def _validate_audited_retry_chain(
    *,
    stage: str,
    job,
    output_root: _Path,
    manifest_sha256: str,
    provenance_sha256: str,
    freeze_sha256: str,
) -> tuple[tuple[dict[str, object], ...], dict[str, _Any] | None]:
    """Validate archived attempts plus one optional append-only terminal tail."""

    archive_parent = (
        output_root / "_infrastructure_attempts" / stage / job.job_id
    )
    attempt_count = len(_attempt_directories(archive_parent))
    retry_count = _retry_artifact_count(
        output_root, stage=stage, job_id=job.job_id
    )
    ready_count = max(0, attempt_count - 1)
    if retry_count not in {ready_count, attempt_count}:
        raise ValueError("retry and infrastructure-attempt sequence lengths drifted")
    records = _validate_archive_chain(
        stage=stage,
        job=job,
        output_root=output_root,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        freeze_sha256=freeze_sha256,
        expected_retry_count=retry_count,
    )
    final = output_root / stage / job.job_id
    staging = output_root / stage / f".{job.job_id}.inprogress"
    final_exists = _lexists(final)
    staging_exists = _lexists(staging)
    if retry_count != attempt_count or attempt_count == 0:
        if (final_exists or staging_exists) and attempt_count:
            raise ValueError("unrecorded worker execution follows the latest archive")
        return records, None
    source_incident_sha256 = str(records[-1]["incident_sha256"])
    summary_path, _ = _retry_paths(
        output_root, stage, job.job_id, retry_count
    )
    tail, _ = _retry_summary(
        summary_path,
        number=retry_count,
        stage=stage,
        job=job,
        output_root=output_root,
        source_incident_sha256=source_incident_sha256,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        freeze_sha256=freeze_sha256,
        archivable_only=False,
    )
    state = tail["state"]
    if state in _ARCHIVABLE_STATES:
        if final_exists or not staging_exists:
            raise ValueError("archivable retry tail has no sole in-progress orphan")
        _plain_directory(staging, label="retry-tail in-progress job")
        if _lexists(staging / "result.json"):
            raise ValueError("archivable retry tail contains a result.json")
        _, started_sha256 = _started_payload(
            staging / "started.json",
            job_payload=job.to_json(),
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
        )
        if tail["started_sha256"] != started_sha256:
            raise ValueError("retry tail started hash differs from its orphan")
    elif state in {"succeeded", "algorithmic_failure"}:
        if staging_exists or not final_exists:
            raise ValueError("terminal retry tail has no sole final result")
        _plain_directory(final, label="retry-tail final job")
        result, _ = _read_canonical_json(final / "result.json")
        if not isinstance(result, _Mapping) or any(
            result.get(name) != expected
            for name, expected in {
                "status": state,
                "job": job.to_json(),
                "config_sha256": _protocol_config_sha256(),
                "manifest_sha256": manifest_sha256,
                "provenance_sha256": provenance_sha256,
                "started_sha256": tail["started_sha256"],
            }.items()
        ):
            raise ValueError("terminal retry summary differs from the final result")
    elif state in {"launch_failed", "failed_without_orphan"}:
        if final_exists or staging_exists:
            raise ValueError("orphan-free retry tail unexpectedly has job artifacts")
    elif state in {"non_retryable_exit", "zero_exit_orphan", "unpublished_result"}:
        if final_exists or not staging_exists:
            raise ValueError("non-retryable tail has no sole in-progress evidence")
        _plain_directory(staging, label="non-retryable in-progress job")
        _, started_sha256 = _started_payload(
            staging / "started.json",
            job_payload=job.to_json(),
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
        )
        if tail["started_sha256"] != started_sha256:
            raise ValueError("non-retryable tail started hash differs from its orphan")
        if state == "unpublished_result" and not _lexists(staging / "result.json"):
            raise ValueError("unpublished-result tail has no result.json")
        if state != "unpublished_result" and _lexists(staging / "result.json"):
            raise ValueError("non-retryable orphan unexpectedly contains result.json")
    else:  # pragma: no cover - _retry_summary rejects unknown states
        raise ValueError("retry tail state is unknown")
    return records, tail


def _mkdir_plain(path: _Path) -> None:
    if _lexists(path):
        _plain_directory(path, label="infrastructure archive parent")
        return
    path.mkdir(mode=0o700)
    _plain_directory(path, label="infrastructure archive parent")


def _fsync_directory(path: _Path) -> None:
    descriptor = _os.open(path, _os.O_RDONLY | getattr(_os, "O_DIRECTORY", 0))
    try:
        _os.fsync(descriptor)
    finally:
        _os.close(descriptor)


def archive_infrastructure_attempt(stage: str, job_id: str, output_root: _Path) -> _Path:
    """Archive one documented failed orphan; never delete or retry a job."""

    job = _resolve_active_job(stage, job_id)
    root = _Path(output_root)
    manifest_sha256, provenance_sha256, freeze_sha256 = _active_freeze_hashes()
    _plain_directory(root, label="output root")
    stage_root = root / stage
    _plain_directory(stage_root, label="stage output")
    staging = stage_root / f".{job.job_id}.inprogress"
    final = stage_root / job.job_id
    if _lexists(final):
        raise FileExistsError("a final planned-job directory already exists")
    _plain_directory(staging, label="in-progress job directory")
    if _lexists(staging / "result.json"):
        raise RuntimeError("a finalized algorithmic result cannot be infrastructure-recovered")
    _validate_opaque_tree(staging)
    _, started_sha256 = _started_payload(
        staging / "started.json",
        job_payload=job.to_json(),
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )

    archive_parent = root / "_infrastructure_attempts" / stage / job.job_id
    existing = _attempt_directories(archive_parent)
    number = len(existing) + 1
    if existing:
        records = _validate_archive_chain(
            stage=stage,
            job=job,
            output_root=root,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            freeze_sha256=freeze_sha256,
            expected_retry_count=len(existing),
        )
        retry_path, _ = _retry_paths(root, stage, job.job_id, len(existing))
        retry, source_sha256 = _retry_attempt(
            retry_path,
            number=len(existing),
            stage=stage,
            job=job,
            output_root=root,
            source_incident_sha256=str(records[-1]["incident_sha256"]),
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            freeze_sha256=freeze_sha256,
        )
        if retry["started_sha256"] != started_sha256:
            raise ValueError("latest retry does not bind the current orphan started hash")
        source_path = retry_path
        command = retry["command"]
        exit_code = retry["exit_code"]
        failure_kind = retry["failure_kind"]
        signum = retry["signal"]
        state = retry["state"]
    else:
        queue_path = root / "_gpu_queue" / stage / "queue_summary.json"
        row, source_sha256, signum, failure_kind = _queue_attempt(
            queue_path,
            stage=stage,
            job_id=job.job_id,
            output_root=root,
        )
        source_path = queue_path
        command = row["command"]
        exit_code = row["exit_code"]
        state = row["state"]
    destination = archive_parent / f"attempt-{number:04d}"
    if _lexists(destination):
        raise FileExistsError("infrastructure attempt archive already exists")
    incident = _incident_payload(
        attempt_name=destination.name,
        job=job,
        command=command,
        exit_code=exit_code,
        failure_kind=str(failure_kind),
        signum=signum,
        state=state,
        started_sha256=started_sha256,
        source_path=source_path,
        source_sha256=source_sha256,
        output_root=root,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        freeze_sha256=freeze_sha256,
    )

    # No mutation occurs until all identity/freeze/command/source checks pass.
    _mkdir_plain(root / "_infrastructure_attempts")
    _mkdir_plain(root / "_infrastructure_attempts" / stage)
    _mkdir_plain(archive_parent)
    incident_path = staging / "incident.json"
    if _lexists(incident_path):
        prepared, _ = _read_canonical_json(incident_path)
        if prepared != incident:
            raise ValueError("pre-existing prepared incident semantic binding drifted")
    else:
        _write_canonical_json_exclusive(incident_path, incident)
    _fsync_directory(staging)
    _rename_directory_noreplace(staging, destination)
    _fsync_directory(archive_parent)
    _fsync_directory(stage_root)
    return destination


__all__ = ["archive_infrastructure_attempt"]
