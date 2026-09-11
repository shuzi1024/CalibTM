from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import signal

import pytest


_MANIFEST_SHA256 = "b" * 64
_PROVENANCE_SHA256 = "c" * 64
_FREEZE_SHA256 = "d" * 64


def _patch_freeze(monkeypatch, *modules) -> None:
    for module in modules:
        monkeypatch.setattr(
            module,
            "_active_freeze_hashes",
            lambda: (_MANIFEST_SHA256, _PROVENANCE_SHA256, _FREEZE_SHA256),
        )


def _queue_payload(output_root: Path, job) -> dict[str, object]:
    from experiments.acil_innovation_v1.gpu_queue import _worker_command
    from experiments.acil_innovation_v1.jobs import planned_jobs

    rows = []
    for candidate in planned_jobs(job.stage):
        selected = candidate.job_id == job.job_id
        rows.append(
            {
                "attempt_count": 1 if selected else 0,
                "command": (
                    list(_worker_command(job.stage, candidate, output_root))
                    if selected
                    else None
                ),
                "dataset": candidate.dataset,
                "exit_code": 75 if selected else None,
                "gpu": "0" if selected else None,
                "job_id": candidate.job_id,
                "launch_error": None,
                "log_path": (
                    str(
                        output_root
                        / "_gpu_queue"
                        / job.stage
                        / "logs"
                        / f"{candidate.job_id}.log"
                    )
                    if selected
                    else None
                ),
                "method": candidate.method,
                "seed_bundle": candidate.seed_bundle,
                "stage": candidate.stage,
                "state": "failed" if selected else "not_launched",
            }
        )
    return {
        "completed_job_count": 1,
        "gpus": ["0"],
        "jobs": rows,
        "launched_job_count": 1,
        "planned_job_count": len(rows),
        "queue_directory": str(output_root / "_gpu_queue" / job.stage),
        "schema_version": 1,
        "signal": None,
        "stage": job.stage,
        "status": "complete_with_failures",
        "worker_module": "experiments.acil_innovation_v1.worker",
    }


def _initial_archive(tmp_path: Path, monkeypatch):
    import experiments.acil_innovation_v1.recovery as recovery
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import (
        begin_job,
        write_canonical_json_exclusive,
    )

    _patch_freeze(monkeypatch, recovery)
    job = planned_jobs("stage0_acil_tune")[0]
    writer = begin_job(
        stage=job.stage,
        job_id=job.job_id,
        output_root=tmp_path,
        manifest_sha256=_MANIFEST_SHA256,
        provenance_sha256=_PROVENANCE_SHA256,
    )
    (writer.staging_directory / "partial.bin").write_bytes(b"first failure")
    queue_root = tmp_path / "_gpu_queue" / job.stage
    (queue_root / "logs").mkdir(parents=True)
    write_canonical_json_exclusive(
        queue_root / "queue_summary.json", _queue_payload(tmp_path, job)
    )
    archived = recovery.archive_infrastructure_attempt(
        job.stage, job.job_id, tmp_path
    )
    return job, archived


def _failing_process_factory(*, exit_code: int = 9):
    class FailingProcess:
        calls: list["FailingProcess"] = []

        def __init__(self, command, **kwargs):
            from experiments.acil_innovation_v1.result_io import begin_job

            self.command = tuple(command)
            self.kwargs = kwargs
            arguments = list(command)
            stage = arguments[arguments.index("--stage") + 1]
            job_id = arguments[arguments.index("--job-id") + 1]
            output_root = Path(arguments[arguments.index("--output-root") + 1])
            writer = begin_job(
                stage=stage,
                job_id=job_id,
                output_root=output_root,
                manifest_sha256=_MANIFEST_SHA256,
                provenance_sha256=_PROVENANCE_SHA256,
            )
            (writer.staging_directory / "partial-loss.bin").write_bytes(
                b"never inspect this loss"
            )
            kwargs["stdout"].write(b"worker exited before finalization\n")
            kwargs["stdout"].flush()
            self.returncode = exit_code
            self.__class__.calls.append(self)

        def wait(self):
            return self.returncode

    return FailingProcess


def test_retry_api_and_cli_expose_only_fixed_job_output_and_operational_gpu() -> None:
    from experiments.acil_innovation_v1 import retry

    assert tuple(inspect.signature(retry.retry_infrastructure_job).parameters) == (
        "stage",
        "job_id",
        "output_root",
        "gpu",
    )
    destinations = {
        action.dest for action in retry._build_parser()._actions if action.dest != "help"
    }
    assert destinations == {"stage", "job_id", "output_root", "gpu"}
    assert {"data", "split", "method", "seed", "gpus"}.isdisjoint(destinations)


@pytest.mark.parametrize("gpu", ("", "00", "-1", "0,1", "gpu0", 0))
def test_retry_rejects_noncanonical_single_gpu_without_mutation(
    tmp_path: Path, monkeypatch, gpu
) -> None:
    import experiments.acil_innovation_v1.retry as retry

    job, _ = _initial_archive(tmp_path, monkeypatch)
    _patch_freeze(monkeypatch, retry)
    with pytest.raises((TypeError, ValueError)):
        retry.retry_infrastructure_job(job.stage, job.job_id, tmp_path, gpu)
    assert not (tmp_path / "_infrastructure_retries").exists()


def test_retry_requires_a_latest_valid_archive_and_never_runs_a_final_job(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.retry as retry
    from experiments.acil_innovation_v1.jobs import planned_jobs

    _patch_freeze(monkeypatch, retry)
    job = planned_jobs("stage0_acil_tune")[0]
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError), match="archive|attempt"):
        retry.retry_infrastructure_job(job.stage, job.job_id, tmp_path, "0")
    assert not tmp_path.exists() or not (tmp_path / "_infrastructure_retries").exists()

    job, _ = _initial_archive(tmp_path, monkeypatch)
    (tmp_path / job.stage / job.job_id).mkdir(parents=True)
    with pytest.raises((FileExistsError, RuntimeError), match="final|completed"):
        retry.retry_infrastructure_job(job.stage, job.job_id, tmp_path, "0")
    assert not (tmp_path / "_infrastructure_retries").exists()


def test_failed_retry_is_append_only_exact_and_recovery_extends_chain(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.recovery as recovery
    import experiments.acil_innovation_v1.retry as retry
    from experiments.acil_innovation_v1.config import (
        CUBLAS_WORKSPACE_CONFIG,
        protocol_config_sha256,
    )
    from experiments.acil_innovation_v1.gpu_queue import _worker_command

    job, first = _initial_archive(tmp_path, monkeypatch)
    _patch_freeze(monkeypatch, recovery, retry)
    process = _failing_process_factory(exit_code=75)
    monkeypatch.setattr(retry._subprocess, "Popen", process)

    summary_path = retry.retry_infrastructure_job(
        job.stage, job.job_id, tmp_path, "2"
    )

    assert summary_path == (
        tmp_path
        / "_infrastructure_retries"
        / job.stage
        / job.job_id
        / "retry-0001.json"
    )
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    incident_path = first / "incident.json"
    command = list(_worker_command(job.stage, job, tmp_path))
    assert summary == {
        "command": command,
        "config_sha256": protocol_config_sha256(),
        "exit_code": 75,
        "failure_kind": "ex_tempfail",
        "freeze_sha256": _FREEZE_SHA256,
        "gpu": "2",
        "job": job.to_json(),
        "launch_error": None,
        "log": str(summary_path.with_suffix(".log").relative_to(tmp_path)),
        "log_sha256": hashlib.sha256(
            summary_path.with_suffix(".log").read_bytes()
        ).hexdigest(),
        "manifest_sha256": _MANIFEST_SHA256,
        "protocol": "acil-innovation-v1",
        "provenance_sha256": _PROVENANCE_SHA256,
        "retry": "retry-0001",
        "schema": "acil-innovation-v1:infrastructure-retry:v1",
        "signal": None,
        "source_attempt": "attempt-0001",
        "source_incident_sha256": hashlib.sha256(
            incident_path.read_bytes()
        ).hexdigest(),
        "started_sha256": hashlib.sha256(
            (
                tmp_path
                / job.stage
                / f".{job.job_id}.inprogress"
                / "started.json"
            ).read_bytes()
        ).hexdigest(),
        "state": "failed",
    }
    assert process.calls[0].command == tuple(command)
    assert process.calls[0].kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert (
        process.calls[0].kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"]
        == CUBLAS_WORKSPACE_CONFIG
    )
    with pytest.raises((FileExistsError, RuntimeError), match="in-progress|archive"):
        retry.retry_infrastructure_job(job.stage, job.job_id, tmp_path, "2")

    second = recovery.archive_infrastructure_attempt(
        job.stage, job.job_id, tmp_path
    )
    assert second.name == "attempt-0002"
    second_incident = json.loads(
        (second / "incident.json").read_text(encoding="ascii")
    )
    assert second_incident["queue_summary"] == str(summary_path.relative_to(tmp_path))
    assert second_incident["queue_summary_sha256"] == hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    assert second_incident["started_sha256"] == summary["started_sha256"]
    assert second_incident["command"] == command
    assert second_incident["exit_code"] == 75
    assert second_incident["failure_kind"] == "ex_tempfail"

    process2 = _failing_process_factory(exit_code=-signal.SIGTERM)
    monkeypatch.setattr(retry._subprocess, "Popen", process2)
    summary2 = retry.retry_infrastructure_job(
        job.stage, job.job_id, tmp_path, "2"
    )
    assert summary2.name == "retry-0002.json"
    payload2 = json.loads(summary2.read_text(encoding="ascii"))
    assert payload2["source_attempt"] == "attempt-0002"
    assert payload2["exit_code"] == -signal.SIGTERM
    assert payload2["failure_kind"] == "signal"
    assert payload2["signal"] == signal.SIGTERM
    third = recovery.archive_infrastructure_attempt(
        job.stage, job.job_id, tmp_path
    )
    assert third.name == "attempt-0003"


def test_ordinary_exit_is_recorded_but_never_archivable_or_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.recovery as recovery
    import experiments.acil_innovation_v1.retry as retry

    job, _ = _initial_archive(tmp_path, monkeypatch)
    _patch_freeze(monkeypatch, recovery, retry)
    process = _failing_process_factory(exit_code=1)
    monkeypatch.setattr(retry._subprocess, "Popen", process)

    summary_path = retry.retry_infrastructure_job(
        job.stage, job.job_id, tmp_path, "0"
    )
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    assert summary["state"] == "non_retryable_exit"
    assert summary["failure_kind"] is None
    assert summary["exit_code"] == 1
    with pytest.raises((RuntimeError, ValueError), match="archiv|temporary|signal"):
        recovery.archive_infrastructure_attempt(job.stage, job.job_id, tmp_path)
    with pytest.raises((FileExistsError, RuntimeError, ValueError), match="archive|in-progress"):
        retry.retry_infrastructure_job(job.stage, job.job_id, tmp_path, "0")


def test_algorithmic_retry_final_is_terminal_and_cannot_run_again(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.recovery as recovery
    import experiments.acil_innovation_v1.retry as retry

    job, _ = _initial_archive(tmp_path, monkeypatch)
    _patch_freeze(monkeypatch, recovery, retry)

    class AlgorithmicProcess:
        def __init__(self, command, **kwargs):
            from experiments.acil_innovation_v1.result_io import begin_job

            arguments = list(command)
            writer = begin_job(
                stage=arguments[arguments.index("--stage") + 1],
                job_id=arguments[arguments.index("--job-id") + 1],
                output_root=Path(arguments[arguments.index("--output-root") + 1]),
                manifest_sha256=_MANIFEST_SHA256,
                provenance_sha256=_PROVENANCE_SHA256,
            )
            writer.fail_algorithmically(
                failure_type="SyntheticMethodFailure",
                message="algorithmic failures are never infrastructure retries",
            )
            kwargs["stdout"].write(b"algorithmic failure\n")
            kwargs["stdout"].flush()
            self.returncode = 20

        def wait(self):
            return self.returncode

    monkeypatch.setattr(retry._subprocess, "Popen", AlgorithmicProcess)
    summary_path = retry.retry_infrastructure_job(
        job.stage, job.job_id, tmp_path, "0"
    )
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    assert summary["state"] == "algorithmic_failure"
    assert summary["failure_kind"] is None
    assert summary["exit_code"] == 20
    with pytest.raises(FileExistsError, match="final|completed"):
        retry.retry_infrastructure_job(job.stage, job.job_id, tmp_path, "0")
    with pytest.raises(FileExistsError, match="final"):
        recovery.archive_infrastructure_attempt(job.stage, job.job_id, tmp_path)


def test_retry_rejects_tampered_latest_incident_before_launch(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.retry as retry

    job, archive = _initial_archive(tmp_path, monkeypatch)
    _patch_freeze(monkeypatch, retry)
    payload = json.loads((archive / "incident.json").read_text(encoding="ascii"))
    payload["command"] = ["python", "wrong.py"]
    (archive / "incident.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="incident|command"):
        retry.retry_infrastructure_job(job.stage, job.job_id, tmp_path, "0")
    assert not (tmp_path / "_infrastructure_retries").exists()


def test_retry_rejects_symlink_inserted_into_archived_opaque_partials(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.retry as retry

    job, archive = _initial_archive(tmp_path, monkeypatch)
    _patch_freeze(monkeypatch, retry)
    partial = archive / "partial.bin"
    partial.unlink()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"must not be followed")
    partial.symlink_to(target)

    def forbidden_launch(*args, **kwargs):
        raise AssertionError("worker launched from a symlink-tainted archive")

    monkeypatch.setattr(retry._subprocess, "Popen", forbidden_launch)
    with pytest.raises(ValueError, match="symlink|plain|regular"):
        retry.retry_infrastructure_job(job.stage, job.job_id, tmp_path, "0")
    assert not (tmp_path / "_infrastructure_retries").exists()
