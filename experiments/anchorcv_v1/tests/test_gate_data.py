from __future__ import annotations

import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from experiments.anchorcv_v1 import gate_identity
from experiments.anchorcv_v1.gate_identity import VerifiedGateAuthority


_EXPECTED = {
    "abilene": {
        "shape": (72, 144, 50),
        "starts": tuple(range(37534, 41134, 50)),
    },
    "geant": {
        "shape": (16, 462, 50),
        "starts": tuple(range(8372, 9172, 50)),
    },
}


def _sealed_authority() -> VerifiedGateAuthority:
    bindings = [
        {
            "dataset": dataset,
            "seed_bundle": bundle,
        }
        for bundle in (1, 2, 3)
        for dataset in ("abilene", "geant")
    ]
    payload = {
        "schema": "anchorcv-v1:gate-authority-payload:v1",
        "protocol": "anchorcv-v1",
        "gate": "anchorcv-v1:final-gate:v1",
        "source_tree_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "git_available": False,
        "git_commit": None,
        "test_access": False,
        "freeze_record_file_sha256": "c" * 64,
        "extension_report_canonical_sha256": "d" * 64,
        "extension_report_schema": "anchorcv-v1:extension-review:v1",
        "extension_report_verdict": "proceed",
        "checkpoint_bindings": bindings,
    }
    method_hash = gate_identity._method_freeze_sha256(payload)
    return VerifiedGateAuthority(
        payload,
        method_hash,
        _seal=gate_identity._AUTHORITY_SEAL,
    )


def _valid_upstream(dataset: str) -> tuple[np.ndarray, tuple[int, ...]]:
    identity = _EXPECTED[dataset]
    values = np.zeros(identity["shape"], dtype="<f4", order="C")
    values.setflags(write=False)
    return values, identity["starts"]


def test_schedule_hash_is_fixed_deterministic_and_array_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.anchorcv_v1 import gate_data

    calls = 0

    def forbidden_loader(dataset: str, cohort: str):
        nonlocal calls
        calls += 1
        raise AssertionError("schedule hashing must not load gate values")

    monkeypatch.setattr(gate_data._upstream_data, "load_windows", forbidden_loader)

    first = gate_data.gate_window_schedule_sha256("abilene")
    second = gate_data.gate_window_schedule_sha256("abilene")
    geant = gate_data.gate_window_schedule_sha256("geant")

    assert first == second
    assert first != geant
    assert len(first) == 64
    assert int(first, 16) >= 0
    assert calls == 0
    assert gate_data.__all__ == ["gate_window_schedule_sha256"]
    with pytest.raises(ValueError, match="dataset"):
        gate_data.gate_window_schedule_sha256("other")


@pytest.mark.parametrize("dataset", ["abilene", "geant"])
def test_private_loader_uses_only_fixed_gate_cohort_and_wraps_registered_windows(
    dataset: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.acil_innovation_v1.data import RegisteredWindows
    from experiments.anchorcv_v1 import gate_data

    calls: list[tuple[str, str]] = []
    values, starts = _valid_upstream(dataset)

    def fake_loader(requested_dataset: str, cohort: str):
        calls.append((requested_dataset, cohort))
        return values, starts

    monkeypatch.setattr(gate_data._upstream_data, "load_windows", fake_loader)
    registered = gate_data._load_gate_windows(_sealed_authority(), dataset)

    assert type(registered) is RegisteredWindows
    assert registered.dataset == dataset
    assert registered.cohort == "gate"
    assert registered.absolute_starts == starts
    assert registered.values is values
    assert not registered.values.flags.writeable
    assert calls == [(dataset, "gate")]


@pytest.mark.parametrize(
    "forgery",
    [
        None,
        object(),
        SimpleNamespace(_record_bytes=b"fake"),
    ],
)
def test_missing_or_forged_authority_fails_before_upstream_loader(
    forgery: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.anchorcv_v1 import gate_data

    calls = 0

    def forbidden_loader(dataset: str, cohort: str):
        nonlocal calls
        calls += 1
        raise AssertionError("unverified authority reached gate loader")

    monkeypatch.setattr(gate_data._upstream_data, "load_windows", forbidden_loader)

    with pytest.raises(TypeError, match="authority"):
        gate_data._load_gate_windows(forgery, "geant")
    assert calls == 0


def test_partially_forged_exact_authority_fails_before_upstream_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.anchorcv_v1 import gate_data

    forged = object.__new__(VerifiedGateAuthority)
    object.__setattr__(forged, "_record_bytes", b"forged")
    object.__setattr__(forged, "_method_freeze_sha256", "0" * 64)
    object.__setattr__(forged, "_payload", MappingProxyType({}))
    object.__setattr__(forged, "_locked", True)
    calls = 0

    def forbidden_loader(dataset: str, cohort: str):
        nonlocal calls
        calls += 1
        raise AssertionError("forged authority reached gate loader")

    monkeypatch.setattr(gate_data._upstream_data, "load_windows", forbidden_loader)

    with pytest.raises(TypeError, match="authority"):
        gate_data._load_gate_windows(forged, "geant")
    assert calls == 0


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("legacy_529", "shape"),
        ("wrong_start", "start"),
        ("writable", "read-only"),
        ("nan", "finite"),
        ("negative", "nonnegative"),
        ("fortran", "C-order"),
        ("float64", "float32"),
    ],
)
def test_gate_loader_rejects_drifted_arrays_and_registry(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.anchorcv_v1 import gate_data

    dataset = "geant"
    values, starts = _valid_upstream(dataset)
    if case == "legacy_529":
        values = np.zeros((16, 529, 50), dtype="<f4")
        values.setflags(write=False)
    elif case == "wrong_start":
        starts = (starts[0] + 1, *starts[1:])
    elif case == "writable":
        values = values.copy(order="C")
    elif case == "nan":
        values = values.copy(order="C")
        values[0, 0, 0] = np.nan
        values.setflags(write=False)
    elif case == "negative":
        values = values.copy(order="C")
        values[0, 0, 0] = -1.0
        values.setflags(write=False)
    elif case == "fortran":
        values = np.asfortranarray(values)
        values.setflags(write=False)
    elif case == "float64":
        values = values.astype(np.float64)
        values.setflags(write=False)
    else:  # pragma: no cover - guards the parametrized test itself
        raise AssertionError(case)

    monkeypatch.setattr(
        gate_data._upstream_data,
        "load_windows",
        lambda requested_dataset, cohort: (values, starts),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        gate_data._load_gate_windows(_sealed_authority(), dataset)


def test_gate_schedule_hash_binds_exact_starts_shape_and_domain() -> None:
    from experiments.anchorcv_v1 import gate_data

    metadata = json.dumps(
        {
            "cohort": "gate",
            "dataset": "geant",
            "dtype": "<f4",
            "shape": [16, 462, 50],
            "window_starts": list(range(8372, 9172, 50)),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected = hashlib.sha256(
        b"anchorcv-v1:gate-window-schedule:v1\x00" + metadata
    ).hexdigest()
    assert gate_data.gate_window_schedule_sha256("geant") == expected
