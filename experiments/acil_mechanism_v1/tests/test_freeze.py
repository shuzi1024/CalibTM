from __future__ import annotations

import json
import stat

import pytest


def test_freeze_is_deterministic_read_only_and_detects_drift(
    tmp_path, monkeypatch
) -> None:
    import experiments.acil_mechanism_v1.freeze as freeze

    path = tmp_path / "freeze.json"
    first = freeze.create_freeze_record(path)
    assert first == freeze.verify_freeze_record(path)
    assert first["git_available"] is False
    assert first["git_commit"] is None
    assert len(first["source_tree_sha256"]) == 64
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert json.loads(path.read_text(encoding="ascii")) == first

    monkeypatch.setattr(
        freeze,
        "_current_record",
        lambda: {**first, "source_tree_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="drifted"):
        freeze.verify_freeze_record(path)


def test_freeze_creation_is_write_once(tmp_path) -> None:
    from experiments.acil_mechanism_v1.freeze import create_freeze_record

    path = tmp_path / "freeze.json"
    create_freeze_record(path)
    with pytest.raises(FileExistsError):
        create_freeze_record(path)
