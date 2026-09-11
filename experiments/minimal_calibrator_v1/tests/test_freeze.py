from __future__ import annotations

import os

import pytest


def test_freeze_is_deterministic_read_only_and_detects_drift(
    tmp_path, monkeypatch
) -> None:
    from experiments.minimal_calibrator_v1 import freeze

    output = tmp_path / "freeze.json"
    record = freeze.create_freeze_record(output)
    assert record == freeze.verify_freeze_record(output)
    assert record["git_available"] is False
    assert record["git_commit"] is None
    assert len(record["source_tree_sha256"]) == 64
    assert os.stat(output).st_mode & 0o777 == 0o444

    original = freeze._SOURCE_NAMES
    monkeypatch.setattr(freeze, "_SOURCE_NAMES", original + ("missing.py",))
    with pytest.raises(ValueError, match="regular file"):
        freeze.verify_freeze_record(output)


def test_freeze_creation_is_write_once(tmp_path) -> None:
    from experiments.minimal_calibrator_v1.freeze import create_freeze_record

    output = tmp_path / "freeze.json"
    create_freeze_record(output)
    with pytest.raises(FileExistsError):
        create_freeze_record(output)


def test_runtime_import_dependencies_are_frozen() -> None:
    from experiments.minimal_calibrator_v1 import freeze

    assert "experiments/acil_innovation_v1/models.py" in freeze._DEPENDENCY_PATHS
