from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


_CANONICAL_SOURCE = Path(__file__).resolve().parents[1] / "canonical"


def _synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "acil_innovation_v1"
    files = {
        "__init__.py": "PROTOCOL_ID = 'acil-innovation-v1'\n",
        "config.py": "import json\n",
        "module.py": "from .config import VALUE\n",
        "scripts/build_freeze.py": "from .. import config\n",
        "tests/test_a.py": "from experiments.acil_innovation_v1 import module\n",
        "tests/test_b.py": "def test_b(): pass\n",
        "configs/protocol_v1.json": json.dumps(
            {
                "protocol": {"id": "acil-innovation-v1", "status": "draft_protocol"},
                "freeze": {"git_available": False, "git_commit": None},
            }
        ),
        "research_card_v1.md": "# card\n",
        "assets/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json": "{}\n",
        "assets/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors": "weights",
        "results/old_result.json": "MUST NOT READ",
        "old_test_cache/test.npz": "MUST NOT READ",
        "raw.csv": "MUST NOT READ",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _build(root: Path):
    from experiments.acil_innovation_v1 import provenance

    return provenance._build_provenance(
        root,
        runtime_versions={
            "implementation": "CPython",
            "packages": {
                "numpy": "1.24.4",
                "safetensors": "0.4.3",
                "torch": "2.3.0",
                "transformers": "4.30.1",
            },
            "python": "3.10.12",
        },
        expected_gpt_hashes=None,
        canonical_root=_CANONICAL_SOURCE,
    )


def test_public_provenance_builder_has_no_caller_selected_paths_or_splits():
    from experiments.acil_innovation_v1 import provenance

    assert tuple(inspect.signature(provenance.build_provenance).parameters) == ()
    assert tuple(inspect.signature(provenance.runtime_versions).parameters) == ()


def test_provenance_separates_production_import_closure_and_all_test_sources(tmp_path):
    value = _build(_synthetic_root(tmp_path))

    production_paths = [item["path"] for item in value["production"]["files"]]
    test_paths = [item["path"] for item in value["tests"]["files"]]
    assert production_paths == [
        "__init__.py",
        "config.py",
        "module.py",
        "scripts/build_freeze.py",
    ]
    assert test_paths == ["tests/test_a.py", "tests/test_b.py"]
    assert value["production"]["import_closure"] == production_paths
    assert len(value["production"]["sha256"]) == 64
    assert len(value["tests"]["sha256"]) == 64
    assert value["git_available"] is False
    assert value["git_commit"] is None


def test_builder_hashes_only_four_permitted_canonical_arrays_not_raw_cache_or_results(
    tmp_path, monkeypatch
):
    root = _synthetic_root(tmp_path)
    forbidden = (
        root / "results" / "old_result.json",
        root / "old_test_cache" / "test.npz",
        root / "raw.csv",
    )
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text

    def guard_bytes(path, *args, **kwargs):
        if path in forbidden:
            raise AssertionError(f"forbidden evidence read: {path}")
        return real_read_bytes(path, *args, **kwargs)

    def guard_text(path, *args, **kwargs):
        if path in forbidden:
            raise AssertionError(f"forbidden evidence read: {path}")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guard_bytes)
    monkeypatch.setattr(Path, "read_text", guard_text)

    from experiments.acil_innovation_v1 import data

    real_load = data.np.load
    loaded = []

    def audited_load(path, *args, **kwargs):
        loaded.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(data.np, "load", audited_load)

    value = _build(root)

    all_hashed_paths = {
        item["path"]
        for section in ("production", "tests")
        for item in value[section]["files"]
    }
    assert not any("canonical/" in path for path in all_hashed_paths)
    assert not any("results/" in path for path in all_hashed_paths)
    assert loaded == [
        _CANONICAL_SOURCE / "abilene_train.npy",
        _CANONICAL_SOURCE / "abilene_val.npy",
        _CANONICAL_SOURCE / "geant_train.npy",
        _CANONICAL_SOURCE / "geant_val.npy",
    ]
    assert value["parsed_arrays"]["payload_bytes_read"] is True
    assert all(item["content_verified"] is True for item in value["parsed_arrays"]["records"])


def test_parsed_array_semantics_and_upstream_acil_identity_are_exact(tmp_path):
    value = _build(_synthetic_root(tmp_path))

    assert value["parsed_arrays"]["upstream"] == {
        "config_sha256": "30689bc52176a095b307ab47ec38dfd502e40704c20d92bd6a326f6db2018fad",
        "protocol": "od-orbit-gpt-v1",
        "provider": "experiments.od_orbit_gpt_v1.protocol",
    }
    records = {
        (item["dataset"], item["split"]): item
        for item in value["parsed_arrays"]["records"]
    }
    assert records[("abilene", "train")]["sha256"] == (
        "5ec5799c6a57c3a3971ab8b33fe95a8ea7c93a01955bec92440a75f38115d8d2"
    )
    assert records[("geant", "val")]["shape"] == [1600, 462]
    assert all(item["dtype"] == "<f4" for item in records.values())
    assert all(item["content_verified"] is True for item in records.values())
    assert value["acil_core_upstream"] == {
        "locator": "Imputation/models/geoanchor_v2_modules.py",
        "sha256": "5bb37678a94e9253703d2050efa679bfbf463beb9633e0cf12544f6fbf5a1450",
        "upstream_bytes_read": False,
    }


def test_gpt_assets_are_fixed_and_runtime_versions_are_recorded(tmp_path):
    value = _build(_synthetic_root(tmp_path))

    assert value["gpt_assets"]["revision"] == (
        "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    assert [item["path"] for item in value["gpt_assets"]["files"]] == [
        "assets/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json",
        "assets/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors",
    ]
    assert value["runtime"] == {
        "implementation": "CPython",
        "packages": {
            "numpy": "1.24.4",
            "safetensors": "0.4.3",
            "torch": "2.3.0",
            "transformers": "4.30.1",
        },
        "python": "3.10.12",
    }


@pytest.mark.parametrize("missing", ["numpy", "safetensors", "torch", "transformers"])
def test_provenance_rejects_missing_required_runtime_distribution(tmp_path, missing):
    from experiments.acil_innovation_v1 import provenance

    packages = {
        "numpy": "1.24.4",
        "safetensors": "0.4.3",
        "torch": "2.3.0",
        "transformers": "4.30.1",
    }
    packages[missing] = None
    with pytest.raises(ValueError, match=missing):
        provenance._build_provenance(
            _synthetic_root(tmp_path),
            runtime_versions={
                "implementation": "CPython",
                "packages": packages,
                "python": "3.10.12",
            },
            expected_gpt_hashes=None,
        )


@pytest.mark.parametrize(
    ("distribution", "version"),
    [
        ("numpy", "2.0.0"),
        ("safetensors", "0.2.9"),
        ("torch", "2.0.1"),
        ("transformers", "4.31.0"),
    ],
)
def test_provenance_rejects_unsupported_runtime_version(
    tmp_path, distribution, version
):
    from experiments.acil_innovation_v1 import provenance

    packages = {
        "numpy": "1.24.4",
        "safetensors": "0.4.3",
        "torch": "2.3.0",
        "transformers": "4.30.1",
    }
    packages[distribution] = version
    with pytest.raises(ValueError, match=distribution):
        provenance._build_provenance(
            _synthetic_root(tmp_path),
            runtime_versions={
                "implementation": "CPython",
                "packages": packages,
                "python": "3.10.12",
            },
            expected_gpt_hashes=None,
        )


def test_two_builds_are_identical_and_source_changes_are_not_hidden(tmp_path):
    root = _synthetic_root(tmp_path)
    first = _build(root)
    second = _build(root)
    assert first == second
    assert first["provenance_sha256"] == second["provenance_sha256"]

    (root / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed_production = _build(root)
    assert changed_production["production"]["sha256"] != first["production"]["sha256"]
    assert changed_production["tests"]["sha256"] == first["tests"]["sha256"]

    (root / "tests" / "test_b.py").write_text(
        "def test_b(): assert False\n", encoding="utf-8"
    )
    changed_test = _build(root)
    assert changed_test["tests"]["sha256"] != changed_production["tests"]["sha256"]


def test_source_or_asset_symlinks_fail_closed(tmp_path):
    root = _synthetic_root(tmp_path)
    real = root / "module.py"
    alias = root / "alias.py"
    alias.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        _build(root)
