from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


def _fake_provenance():
    return {
        "schema_version": 1,
        "protocol": "acil-innovation-v1",
        "config_sha256": "a" * 64,
        "git_available": False,
        "git_commit": None,
        "production": {"sha256": "b" * 64},
        "tests": {
            "sha256": "c" * 64,
            "files": [{"path": "tests/test_protocol.py", "sha256": "2" * 64}],
        },
        "parsed_arrays": {
            "sha256": "d" * 64,
            "payload_bytes_read": True,
            "records": [
                {
                    "dataset": dataset,
                    "split": split,
                    "content_verified": True,
                }
                for dataset in ("abilene", "geant")
                for split in ("train", "val")
            ],
        },
        "gpt_assets": {"sha256": "e" * 64},
        "runtime": {
            "implementation": "CPython",
            "packages": {
                "numpy": "1.24.4",
                "safetensors": "0.4.3",
                "torch": "2.3.0",
                "transformers": "4.30.1",
            },
            "python": "3.10.12",
        },
        "runtime_sha256": "f" * 64,
        "provenance_sha256": "1" * 64,
    }


def _all_planned_handlers():
    from experiments.acil_innovation_v1.jobs import planned_jobs

    return frozenset(
        (job.stage, job.method)
        for stage in (
            "stage0_acil_tune",
            "stage_h",
            "stage_i",
            "full_tune",
        )
        for job in planned_jobs(stage)
    )


def test_public_manifest_and_freeze_builders_have_no_latest_path_or_split_argument():
    from experiments.acil_innovation_v1 import manifest
    from experiments.acil_innovation_v1.scripts import build_freeze

    assert tuple(inspect.signature(manifest.build_manifest).parameters) == ()
    assert tuple(inspect.signature(build_freeze.build_freeze_record).parameters) == ()
    assert tuple(inspect.signature(build_freeze.write_freeze).parameters) == ()
    assert tuple(inspect.signature(build_freeze.main).parameters) == ()


def test_manifest_is_exact_complete_ordered_job_grid_and_build_is_deterministic():
    from experiments.acil_innovation_v1 import jobs, manifest

    first = manifest._build_manifest(_fake_provenance())
    second = manifest._build_manifest(_fake_provenance())

    expected = tuple(
        job.to_json()
        for stage in (
            "stage0_acil_tune",
            "stage_h",
            "stage_i",
            "full_tune",
        )
        for job in jobs.planned_jobs(stage)
    )
    assert first == second
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["version"] == "v1"
    assert "latest" not in json.dumps(first).lower()
    assert tuple(first["jobs"]) == expected
    assert len(first["jobs"]) == 42
    assert len({item["job_id"] for item in first["jobs"]}) == 42
    assert all("mask_family" not in item for item in first["jobs"])
    assert first["stage_order"] == [
        "stage0_acil_tune",
        "stage_h",
        "stage_i",
        "full_tune",
    ]
    assert tuple(first["gate_registry"]) == tuple(first["stage_order"])
    assert "formal_gate" not in first["gate_registry"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "reordered"])
def test_manifest_validation_rejects_grid_shape_or_order_drift(mutation):
    from experiments.acil_innovation_v1 import manifest

    value = manifest._build_manifest(_fake_provenance())
    jobs = list(value["jobs"])
    if mutation == "missing":
        jobs.pop()
    elif mutation == "extra":
        jobs.append(dict(jobs[-1], job_id="9" * 64))
    elif mutation == "duplicate":
        jobs[-1] = jobs[0]
    else:
        jobs[0], jobs[1] = jobs[1], jobs[0]
    changed = dict(value, jobs=jobs)
    changed.pop("manifest_sha256")
    with pytest.raises(ValueError, match=mutation):
        manifest.validate_manifest(changed, expected_provenance=_fake_provenance())


def test_freeze_record_binds_draft_config_and_all_scientific_hashes():
    from experiments.acil_innovation_v1 import manifest
    from experiments.acil_innovation_v1.scripts import build_freeze

    provenance = _fake_provenance()
    planned = manifest._build_manifest(provenance)
    bundle = build_freeze._build_freeze_record(
        provenance, planned, supported_handlers=_all_planned_handlers()
    )

    assert bundle.record == {
        "config_sha256": "a" * 64,
        "config_status_at_freeze": "draft_protocol",
        "cuda_determinism": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "torch_deterministic_algorithms": True,
        },
        "git_available": False,
        "git_commit": None,
        "gpt_assets_sha256": "e" * 64,
        "manifest_sha256": planned["manifest_sha256"],
        "parsed_arrays_sha256": "d" * 64,
        "production_sha256": "b" * 64,
        "protocol": "acil-innovation-v1",
        "provenance_sha256": "1" * 64,
        "runtime_sha256": "f" * 64,
        "schema_version": 1,
        "state": "protocol_v2_frozen",
        "preflight": {
            "deterministic_gpu_smoke_repeat_required_before_launch": True,
            "full_test_suite_required_before_launch": True,
            "handler_coverage": "verified",
            "parsed_array_payloads": "verified",
            "runtime_dependencies": "verified",
            "test_execution_claimed": False,
            "test_sources": "content_hashed_not_execution_evidence",
        },
        "tests_sha256": "c" * 64,
        "version": "v2",
    }
    assert len(bundle.freeze_sha256) == 64


def test_freeze_writes_only_fixed_namespace_and_never_overwrites(tmp_path):
    from experiments.acil_innovation_v1 import manifest
    from experiments.acil_innovation_v1.scripts import build_freeze

    provenance = _fake_provenance()
    planned = manifest._build_manifest(provenance)
    bundle = build_freeze._build_freeze_record(
        provenance, planned, supported_handlers=_all_planned_handlers()
    )
    before = dict(provenance)

    artifact = build_freeze._write_freeze_for_test(tmp_path, provenance, planned, bundle)

    assert artifact.root == tmp_path / "freeze_v2"
    assert artifact.root.is_dir()
    assert sorted(path.name for path in artifact.root.iterdir()) == [
        "anchor.json",
        f"freeze-{bundle.freeze_sha256}.json",
        f"manifest-{planned['manifest_sha256']}.json",
        "provenance-" + provenance["provenance_sha256"] + ".json",
    ]
    assert provenance == before
    with pytest.raises(FileExistsError):
        build_freeze._write_freeze_for_test(tmp_path, provenance, planned, bundle)


def test_freeze_preflight_rejects_any_planned_job_without_static_handler():
    from experiments.acil_innovation_v1 import manifest
    from experiments.acil_innovation_v1.scripts import build_freeze

    provenance = _fake_provenance()
    planned = manifest._build_manifest(provenance)
    handlers = set(_all_planned_handlers())
    handlers.remove(next(iter(handlers)))

    with pytest.raises(ValueError, match="handler coverage"):
        build_freeze._build_freeze_record(
            provenance, planned, supported_handlers=frozenset(handlers)
        )


@pytest.mark.parametrize("broken", ["runtime", "parsed_arrays", "tests"])
def test_freeze_preflight_rejects_unverified_dependencies_data_or_test_sources(broken):
    from experiments.acil_innovation_v1 import manifest
    from experiments.acil_innovation_v1.scripts import build_freeze

    provenance = _fake_provenance()
    if broken == "runtime":
        provenance["runtime"]["packages"]["transformers"] = None
    elif broken == "parsed_arrays":
        provenance["parsed_arrays"]["records"][0]["content_verified"] = False
    else:
        provenance["tests"]["files"] = []
    planned = manifest._build_manifest(provenance)

    with pytest.raises(ValueError, match="runtime|parsed|test"):
        build_freeze._build_freeze_record(
            provenance, planned, supported_handlers=_all_planned_handlers()
        )


def test_runner_public_entrypoint_has_no_data_test_or_split_override():
    from experiments.acil_innovation_v1.run_job import run_planned_job

    parameters = tuple(inspect.signature(run_planned_job).parameters)
    assert parameters == ("stage", "job_id", "output_root")
    assert not ({"data", "path", "test", "split", "split_override"} & set(parameters))
