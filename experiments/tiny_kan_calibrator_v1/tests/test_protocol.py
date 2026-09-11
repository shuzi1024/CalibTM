from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys


def test_closed_job_grid_and_queue_preview() -> None:
    from experiments.tiny_kan_calibrator_v1.jobs import expected_jobs
    from experiments.tiny_kan_calibrator_v1.queue import queue_plan

    jobs = expected_jobs()
    assert len(jobs) == 18
    assert len({job.job_id for job in jobs}) == 18
    assert {
        (job.dataset, job.seed_bundle, job.method) for job in jobs
    } == {
        (dataset, seed, method)
        for dataset in ("abilene", "geant")
        for seed in (1, 2, 3)
        for method in ("mlp_value", "kan_value", "kan_static")
    }
    plan = queue_plan(max_workers=3)
    assert plan["test_access"] is False
    assert len(plan["jobs"]) == 18
    assert len(plan["source"]["aggregate_sha256"]) == 64
    assert "max_workers" not in plan


def test_queue_detects_only_a_matching_surviving_worker(tmp_path) -> None:
    from experiments.tiny_kan_calibrator_v1.jobs import expected_jobs
    from experiments.tiny_kan_calibrator_v1.queue import _matching_live_worker

    job = expected_jobs()[0]
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    expected_output = str((attempt / "result.json").resolve())
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "experiments.tiny_kan_calibrator_v1.run_job",
            job.dataset,
            str(job.seed_bundle),
            job.method,
            expected_output,
        ]
    )
    try:
        (attempt / "pid").write_text(f"{process.pid}\n", encoding="ascii")
        assert _matching_live_worker(attempt, job)
        (attempt / "pid").write_text(f"{os.getpid()}\n", encoding="ascii")
        assert not _matching_live_worker(attempt, job)
    finally:
        process.terminate()
        process.wait(timeout=5)


def _synthetic_payloads(*, kan_error: float):
    from experiments.acil_innovation_v1.registries import (
        dataset_spec,
        ordered_window_starts,
    )
    from experiments.anchorcv_v1.data_access import canonical_data_identity
    from experiments.tiny_kan_calibrator_v1.jobs import (
        PROTOCOL_ID,
        canonical_json,
        expected_jobs,
    )
    from experiments.tiny_kan_calibrator_v1.model import (
        count_parameters,
        new_calibrator,
    )
    from experiments.tiny_kan_calibrator_v1.result_io import RESULT_SCHEMA
    from experiments.tiny_kan_calibrator_v1.source_identity import source_tree_sha256

    source = source_tree_sha256()
    architectures = {}
    for method in ("mlp_value", "kan_value", "kan_static"):
        model = new_calibrator(method, model_seed=41001)
        custom = getattr(model, "architecture_record", None)
        architectures[method] = {
            "class": type(model).__name__,
            "details": custom() if callable(custom) else {},
            "parameter_count": count_parameters(model),
        }
    results = []
    for job in expected_jobs():
        model_error = {
            "mlp_value": 10.0,
            "kan_value": kan_error,
            "kan_static": 10.5,
        }[job.method]
        starts = list(ordered_window_starts(job.dataset, "tune"))
        cells = {}
        for family_index, family in enumerate(
            ("random", "internal_block", "two_burst")
        ):
            cells[family] = {
                "linear_error_sum_per_window": [12.0] * len(starts),
                "mask_grid_sha256": f"{family_index + 1:064x}",
                "model_error_sum_per_window": [model_error] * len(starts),
                "target_count_per_window": [
                    dataset_spec(job.dataset).flows * 47
                ] * len(starts),
                "truth_sum_per_window": [100.0] * len(starts),
                "window_starts": starts,
            }
        architecture = architectures[job.method]
        checkpoint_identity = {
            "architecture": architecture,
            "best_epoch": 3,
            "job": job.to_json(),
            "source_tree_sha256": source,
        }
        checkpoint_identity_sha = hashlib.sha256(
            b"sc-acil-v1:checkpoint-identity:v1\x00"
            + canonical_json(checkpoint_identity)
        ).hexdigest()
        results.append(
            {
                "architecture": architecture,
                "cells": cells,
                "checkpoint": {
                    "file_sha256": "b" * 64,
                    "identity": checkpoint_identity,
                    "identity_sha256": checkpoint_identity_sha,
                    "metadata": "checkpoints/synthetic.json",
                    "tensor_sha256": "c" * 64,
                    "weights": "checkpoints/synthetic.pt",
                },
                "data_identity": canonical_data_identity(job.dataset),
                "evidence_boundary": {
                    "candidate_confirmation": False,
                    "cohort": "tune",
                    "project_wide_pristine": False,
                    "role": "post-project one-shot development collision",
                },
                "fit_access": True,
                "job": job.to_json(),
                "protocol": PROTOCOL_ID,
                "schema": RESULT_SCHEMA,
                "selection_cohort": "source_dev",
                "source_tree_sha256": source,
                "status": "succeeded",
                "test_access": False,
                "training": {
                    "best_epoch": 3,
                    "epochs_completed": 20,
                    "optimizer_updates": 320,
                    "parameter_count": architecture["parameter_count"],
                },
            }
        )
    return results


def test_synthetic_adjudicator_passes_and_kills() -> None:
    from experiments.tiny_kan_calibrator_v1.adjudicate import adjudicate_payloads

    passed = adjudicate_payloads(_synthetic_payloads(kan_error=9.8))
    assert passed["verdict"] == "PROCEED_TO_LATENCY_AND_FRESH_CONFIRMATION_REQUIRED"
    assert all(row["pass"] for row in passed["gates"].values())
    killed = adjudicate_payloads(_synthetic_payloads(kan_error=10.1))
    assert killed["verdict"] == "KILL_TINY_KAN_KEEP_VALUE_ONLY"
    assert not killed["gates"]["kan_over_mlp_structured_at_least_1pct"]["pass"]


def test_result_manifest_round_trip(tmp_path) -> None:
    from experiments.tiny_kan_calibrator_v1.jobs import (
        PROTOCOL_ID,
        canonical_json,
        expected_jobs,
    )
    from experiments.tiny_kan_calibrator_v1.result_io import (
        RESULT_SCHEMA,
        file_sha256,
        verified_success,
        write_exclusive,
        write_result,
    )

    output = tmp_path / "result.json"
    job = expected_jobs()[0].to_json()
    source = "a" * 64
    architecture = {
        "class": "ACILBase",
        "details": {},
        "parameter_count": 5475,
    }
    identity = {
        "architecture": architecture,
        "best_epoch": 0,
        "job": job,
        "source_tree_sha256": source,
    }
    identity_sha = hashlib.sha256(
        b"sc-acil-v1:checkpoint-identity:v1\x00" + canonical_json(identity)
    ).hexdigest()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    weights = checkpoint_dir / f"{identity_sha}.pt"
    weights.write_bytes(b"synthetic checkpoint bytes")
    tensor_sha = "b" * 64
    metadata = {
        "file_sha256": file_sha256(weights),
        "identity": identity,
        "identity_sha256": identity_sha,
        "schema": "sc-acil-v1:checkpoint:v1",
        "tensor_sha256": tensor_sha,
        "weights_file": weights.name,
    }
    metadata_path = checkpoint_dir / f"{identity_sha}.json"
    write_exclusive(metadata_path, metadata)
    payload = {
        "architecture": architecture,
        "checkpoint": {
            "file_sha256": metadata["file_sha256"],
            "identity": identity,
            "identity_sha256": identity_sha,
            "metadata": metadata_path.relative_to(tmp_path).as_posix(),
            "tensor_sha256": tensor_sha,
            "weights": weights.relative_to(tmp_path).as_posix(),
        },
        "job": job,
        "protocol": PROTOCOL_ID,
        "schema": RESULT_SCHEMA,
        "source_tree_sha256": source,
        "status": "succeeded",
        "test_access": False,
        "training": {"best_epoch": 0},
    }
    write_result(output, payload)
    assert verified_success(output) == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload
