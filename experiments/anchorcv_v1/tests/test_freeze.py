from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.anchorcv_v1 import freeze, protocol


def _install_minimal_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    package_root = tmp_path / "experiments" / "anchorcv_v1"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("# package\n", encoding="utf-8")
    (package_root / "method.py").write_text("METHOD = 'anchorcv'\n", encoding="utf-8")
    (package_root / "research_card.yaml").write_text("{}\n", encoding="utf-8")
    upstream = tmp_path / "upstream" / "dependency.py"
    upstream.parent.mkdir()
    upstream.write_text("UPSTREAM = 1\n", encoding="utf-8")

    monkeypatch.setattr(freeze, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(freeze, "_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(
        freeze,
        "UPSTREAM_DEPENDENCY_PATHS",
        ("upstream/dependency.py",),
    )
    monkeypatch.setattr(
        freeze.protocol,
        "load_protocol",
        lambda: type(
            "Spec",
            (),
            {"protocol_id": "anchorcv-v1", "datasets": ("abilene", "geant")},
        )(),
    )
    monkeypatch.setattr(freeze.protocol, "fingerprint", lambda spec: "c" * 64)
    monkeypatch.setattr(
        freeze.data_access,
        "canonical_data_identity",
        lambda dataset: {
            "dataset": dataset,
            "hash_scope": "canonical_parsed_permitted_split_arrays_only",
            "parsed_array_sha256": {
                "train": ("a" if dataset == "abilene" else "b") * 64,
                "val": ("d" if dataset == "abilene" else "e") * 64,
            },
            "data_sha256": ("1" if dataset == "abilene" else "2") * 64,
        },
    )
    return package_root, upstream


def _rewrite_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_create_freeze_records_exact_protocol_data_and_no_git_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    expected = {
        dataset: {
            "dataset": dataset,
            "hash_scope": "canonical_parsed_permitted_split_arrays_only",
            "parsed_array_sha256": {
                "train": ("c" if dataset == "abilene" else "d") * 64,
                "val": ("e" if dataset == "abilene" else "f") * 64,
            },
            "data_sha256": ("a" if dataset == "abilene" else "b") * 64,
        }
        for dataset in ("abilene", "geant")
    }
    monkeypatch.setattr(
        freeze.data_access,
        "canonical_data_identity",
        lambda dataset: calls.append(dataset) or expected[dataset],
    )
    output = tmp_path / "anchorcv-freeze.json"

    record = freeze.create_freeze_record(output)

    assert calls == ["abilene", "geant", "abilene", "geant"]
    assert record["protocol"] == "anchorcv-v1"
    assert record["git_available"] is False
    assert record["git_commit"] is None
    assert record["config_sha256"] == protocol.fingerprint()
    assert record["data_identities"] == expected
    assert output.exists() and not output.is_symlink()
    assert output.stat().st_mode & 0o222 == 0
    assert freeze.verify_freeze_record(output) == record


def test_freeze_bytes_and_source_tree_hash_are_deterministic(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = freeze.create_freeze_record(first_path)
    second = freeze.create_freeze_record(second_path)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert len(first["source_tree_sha256"]) == 64
    assert all(character in "0123456789abcdef" for character in first["source_tree_sha256"])


def test_create_refuses_to_overwrite_any_existing_path(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    output.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="overwrite"):
        freeze.create_freeze_record(output)

    assert output.read_bytes() == b"sentinel"


def test_inventory_includes_research_sources_card_and_explicit_upstream_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, _ = _install_minimal_source_tree(tmp_path, monkeypatch)
    (package_root / "scripts").mkdir()
    (package_root / "scripts" / "launch.py").write_text("RUN = 1\n", encoding="utf-8")
    (package_root / "occupancy.py").write_text(
        "OPERATIONAL_ONLY = True\n", encoding="utf-8"
    )
    for excluded in ("results", "freezes", "handoffs", "__pycache__", "tests"):
        directory = package_root / excluded
        directory.mkdir()
        (directory / "should_not_be_read.py").write_text(
            "raise RuntimeError('forbidden read')\n", encoding="utf-8"
        )
    (package_root / ".pytest_cache").mkdir()
    (package_root / ".pytest_cache" / "temporary.py").write_text(
        "raise RuntimeError('forbidden read')\n", encoding="utf-8"
    )

    record = freeze.create_freeze_record(tmp_path / "freeze.json")
    inventory = {
        item["path"]: item["role"] for item in record["source_files"]
    }

    assert inventory == {
        "experiments/anchorcv_v1/__init__.py": "anchorcv_source",
        "experiments/anchorcv_v1/method.py": "anchorcv_source",
        "experiments/anchorcv_v1/research_card.yaml": "research_card",
        "experiments/anchorcv_v1/scripts/launch.py": "anchorcv_source",
        "upstream/dependency.py": "upstream_dependency",
    }
    assert record["upstream_dependency_paths"] == ["upstream/dependency.py"]
    assert record["source_inventory_policy"]["operational_excluded_files"] == [
        "experiments/anchorcv_v1/occupancy.py"
    ]
    assert freeze.verify_freeze_record(tmp_path / "freeze.json") == record


def test_verify_recomputes_every_file_and_rejects_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, _ = _install_minimal_source_tree(tmp_path, monkeypatch)
    output = tmp_path / "freeze.json"
    freeze.create_freeze_record(output)
    (package_root / "method.py").write_text("METHOD = 'drifted'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source .*drift"):
        freeze.verify_freeze_record(output)


def test_verify_rejects_source_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, _ = _install_minimal_source_tree(tmp_path, monkeypatch)
    output = tmp_path / "freeze.json"
    freeze.create_freeze_record(output)
    source = package_root / "method.py"
    replacement = package_root / "replacement.txt"
    replacement.write_text("METHOD = 'same bytes are irrelevant'\n", encoding="utf-8")
    source.unlink()
    source.symlink_to(replacement)

    with pytest.raises(ValueError, match="symlink"):
        freeze.verify_freeze_record(output)


def test_verify_rejects_freeze_record_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_minimal_source_tree(tmp_path, monkeypatch)
    real = tmp_path / "real.json"
    freeze.create_freeze_record(real)
    link = tmp_path / "link.json"
    link.symlink_to(real)

    with pytest.raises(ValueError, match="symlink"):
        freeze.verify_freeze_record(link)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "nested_unknown"])
def test_verify_rejects_missing_and_unknown_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _install_minimal_source_tree(tmp_path, monkeypatch)
    original = tmp_path / "original.json"
    record = freeze.create_freeze_record(original)
    payload = json.loads(json.dumps(record))
    if mutation == "missing":
        del payload["git_commit"]
    elif mutation == "unknown":
        payload["invented_commit"] = "deadbeef"
    else:
        payload["source_files"][0]["unknown"] = True
    tampered = tmp_path / f"{mutation}.json"
    _rewrite_json(tampered, payload)

    with pytest.raises(ValueError, match="fields"):
        freeze.verify_freeze_record(tampered)


def test_verify_rejects_config_and_canonical_data_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_minimal_source_tree(tmp_path, monkeypatch)
    output = tmp_path / "freeze.json"
    freeze.create_freeze_record(output)

    monkeypatch.setattr(freeze.protocol, "fingerprint", lambda spec: "f" * 64)
    with pytest.raises(ValueError, match="config_sha256"):
        freeze.verify_freeze_record(output)

    monkeypatch.setattr(freeze.protocol, "fingerprint", lambda spec: "c" * 64)
    monkeypatch.setattr(
        freeze.data_access,
        "canonical_data_identity",
        lambda dataset: {
            "dataset": dataset,
            "hash_scope": "canonical_parsed_permitted_split_arrays_only",
            "parsed_array_sha256": {"train": "a" * 64, "val": "b" * 64},
            "data_sha256": "f" * 64,
        },
    )
    with pytest.raises(ValueError, match="data identities"):
        freeze.verify_freeze_record(output)


def test_verify_rejects_duplicate_json_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_minimal_source_tree(tmp_path, monkeypatch)
    output = tmp_path / "duplicate.json"
    output.write_text(
        '{"schema_version":1,"schema_version":1}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate"):
        freeze.verify_freeze_record(output)


def test_upstream_dependency_inventory_is_explicit_and_repo_relative() -> None:
    assert freeze.UPSTREAM_DEPENDENCY_PATHS
    assert tuple(sorted(freeze.UPSTREAM_DEPENDENCY_PATHS)) == (
        freeze.UPSTREAM_DEPENDENCY_PATHS
    )
    for value in freeze.UPSTREAM_DEPENDENCY_PATHS:
        path = Path(value)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert value.startswith("experiments/")
    assert {
        "experiments/od_orbit_gpt_v1/__init__.py",
        "experiments/od_orbit_gpt_v1/config.py",
        "experiments/od_orbit_gpt_v1/configs/protocol_v1.json",
        "experiments/od_orbit_gpt_v1/data.py",
        "experiments/od_orbit_gpt_v1/protocol.py",
    }.issubset(freeze.UPSTREAM_DEPENDENCY_PATHS)
