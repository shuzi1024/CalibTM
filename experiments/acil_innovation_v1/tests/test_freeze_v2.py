from __future__ import annotations

from pathlib import Path
import json

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


def test_freeze_v2_record_changes_only_implementation_freeze_identity() -> None:
    from experiments.acil_innovation_v1 import manifest
    from experiments.acil_innovation_v1.scripts import build_freeze

    provenance = _fake_provenance()
    planned = manifest._build_manifest(provenance)
    bundle = build_freeze._build_freeze_record(
        provenance, planned, supported_handlers=_all_planned_handlers()
    )

    assert bundle.record["protocol"] == "acil-innovation-v1"
    assert bundle.record["version"] == "v2"
    assert bundle.record["state"] == "protocol_v2_frozen"
    assert planned["version"] == "v1"


def test_existing_freeze_v1_does_not_block_first_v2_but_v2_never_overwrites(
    tmp_path: Path,
) -> None:
    from experiments.acil_innovation_v1 import manifest
    from experiments.acil_innovation_v1.scripts import build_freeze

    old_root = tmp_path / "freeze_v1"
    old_root.mkdir()
    sentinel = old_root / "immutable-sentinel"
    sentinel.write_bytes(b"abandoned-v1")

    provenance = _fake_provenance()
    planned = manifest._build_manifest(provenance)
    bundle = build_freeze._build_freeze_record(
        provenance, planned, supported_handlers=_all_planned_handlers()
    )
    artifact = build_freeze._write_freeze_for_test(
        tmp_path, provenance, planned, bundle
    )

    assert artifact.root == tmp_path / "freeze_v2"
    assert sentinel.read_bytes() == b"abandoned-v1"
    anchor = json.loads(artifact.anchor_path.read_text(encoding="ascii"))
    assert anchor["version"] == "v2"
    with pytest.raises(FileExistsError, match="freeze_v2"):
        build_freeze._write_freeze_for_test(tmp_path, provenance, planned, bundle)


def test_active_execution_anchor_never_falls_back_to_existing_freeze_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.acil_innovation_v1 import execution

    fake_module = tmp_path / "experiment" / "execution.py"
    old_root = fake_module.parent / "generated" / "freeze_v1"
    old_root.mkdir(parents=True)
    (old_root / "anchor.json").write_bytes(b"{}\n")
    reads: list[Path] = []

    monkeypatch.setattr(execution, "__file__", str(fake_module))
    monkeypatch.setattr(
        execution,
        "_read_json_exact",
        lambda path: reads.append(Path(path)) or {},
    )

    with pytest.raises(ValueError, match="freeze_v2"):
        execution._active_freeze()
    assert reads == []


def test_recovery_anchor_reads_v2_path_even_when_v1_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.acil_innovation_v1 import recovery

    fake_module = tmp_path / "experiment" / "recovery.py"
    old_root = fake_module.parent / "generated" / "freeze_v1"
    old_root.mkdir(parents=True)
    (old_root / "anchor.json").write_bytes(b"{}\n")
    reads: list[Path] = []

    monkeypatch.setattr(recovery, "__file__", str(fake_module))
    monkeypatch.setattr(recovery, "_active_freeze", lambda: ("a" * 64, "b" * 64))
    monkeypatch.setattr(
        recovery,
        "_read_json_exact",
        lambda path: reads.append(Path(path)) or {},
    )

    with pytest.raises(ValueError, match="freeze hash"):
        recovery._active_freeze_hashes()
    assert reads == [fake_module.parent / "generated" / "freeze_v2" / "anchor.json"]


def test_adjudication_anchor_never_falls_back_to_existing_freeze_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.acil_innovation_v1 import adjudication

    fake_module = tmp_path / "experiment" / "adjudication.py"
    old_root = fake_module.parent / "generated" / "freeze_v1"
    old_root.mkdir(parents=True)
    (old_root / "anchor.json").write_bytes(b"{}\n")
    monkeypatch.setattr(adjudication, "__file__", str(fake_module))

    with pytest.raises(FileNotFoundError, match="freeze_v2"):
        adjudication._active_freeze_hashes()
