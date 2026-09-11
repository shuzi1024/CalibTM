from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


def test_public_loader_has_no_path_split_or_test_override() -> None:
    from experiments.acil_innovation_v1.data import load_registered_windows

    parameters = inspect.signature(load_registered_windows).parameters
    assert tuple(parameters) == ("stage", "dataset", "cohort")
    assert not ({"path", "split", "test", "split_override"} & set(parameters))


def test_capability_rejects_sealed_and_unregistered_cohorts() -> None:
    from experiments.acil_innovation_v1.data import load_registered_windows

    with pytest.raises(ValueError, match="capability"):
        load_registered_windows("stage_h", "geant", "test")
    with pytest.raises(ValueError, match="capability"):
        load_registered_windows("stage_h", "abilene", "gate")


def test_v1_loader_rejects_future_formal_gate_before_parent_access(
    monkeypatch,
) -> None:
    from experiments.acil_innovation_v1 import data

    parent_calls = []
    monkeypatch.setattr(data, "_parent", lambda *args: parent_calls.append(args))
    with pytest.raises(ValueError, match="active manifest"):
        data.load_registered_windows("formal_gate", "geant", "gate")
    assert parent_calls == []


def test_registered_windows_are_read_only_flow_by_time() -> None:
    from experiments.acil_innovation_v1.data import load_registered_windows

    windows = load_registered_windows("stage_h", "geant", "source_dev")
    assert windows.values.shape == (10, 462, 50)
    assert windows.values.dtype == np.dtype("<f4")
    assert not windows.values.flags.writeable
    assert windows.absolute_starts[0] == 7072
    assert windows.absolute_starts[-1] == 7522


def test_fit_fallback_uses_only_fit_cohort_and_is_stable() -> None:
    from experiments.acil_innovation_v1.data import fit_fallback

    first = fit_fallback("geant")
    second = fit_fallback("geant")
    assert first == second
    assert np.isfinite(first.mean)
    assert np.isfinite(first.std)
    assert first.std >= 1e-6


def test_registered_canonical_arrays_are_bound_by_upstream_parsed_semantics() -> None:
    from experiments.acil_innovation_v1 import data

    path = Path(data.__file__).with_name("canonical") / "geant_val.npy"
    values = np.load(path, allow_pickle=False, mmap_mode="r")
    expected = data.expected_parsed_array_sha256("geant", "val")

    assert data.parsed_array_sha256(values, "geant", "val") == expected
    assert data._load_canonical_array(path, "geant", "val", expected) is not None


def test_same_shape_canonical_payload_tampering_fails_closed(tmp_path) -> None:
    from experiments.acil_innovation_v1 import data

    source = Path(data.__file__).with_name("canonical") / "geant_val.npy"
    changed = np.array(
        np.load(source, allow_pickle=False, mmap_mode="r"),
        dtype="<f4",
        order="C",
        copy=True,
    )
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(np.inf))
    path = tmp_path / "geant_val.npy"
    np.save(path, changed, allow_pickle=False)

    expected = data.expected_parsed_array_sha256("geant", "val")
    assert data.parsed_array_sha256(changed, "geant", "val") != expected
    with pytest.raises(ValueError, match="parsed-array SHA-256"):
        data._load_canonical_array(path, "geant", "val", expected)


def test_loader_cache_contains_only_semantically_verified_parents() -> None:
    from experiments.acil_innovation_v1 import data

    data._parent.cache_clear()
    first = data._parent("abilene", "val")
    second = data._parent("abilene", "val")

    assert first is second
    assert data.parsed_array_sha256(first, "abilene", "val") == (
        data.expected_parsed_array_sha256("abilene", "val")
    )
