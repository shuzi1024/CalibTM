"""Final-only, authority-gated access to the fixed confirmation cohort.

This module deliberately has no path, split, cohort, range, or test override.
Its sole public function hashes the static schedule without loading data.  The
private loader is reserved for a final-gate worker holding a factory-sealed
``VerifiedGateAuthority``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

from experiments.acil_innovation_v1.data import RegisteredWindows
from experiments.sc2_ari_v1 import data as _upstream_data

from . import gate_identity as _gate_identity
from .gate_identity import VerifiedGateAuthority


_SCHEDULE_DOMAIN = b"anchorcv-v1:gate-window-schedule:v1\x00"
_COHORT = "gate"
_DTYPE = np.dtype("<f4")


@dataclass(frozen=True, slots=True)
class _GateSchedule:
    starts: tuple[int, ...]
    shape: tuple[int, int, int]


_GATE_REGISTRY = MappingProxyType(
    {
        "abilene": _GateSchedule(
            starts=tuple(range(37534, 41134, 50)),
            shape=(72, 144, 50),
        ),
        "geant": _GateSchedule(
            starts=tuple(range(8372, 9172, 50)),
            shape=(16, 462, 50),
        ),
    }
)


def _schedule(dataset: str) -> _GateSchedule:
    if type(dataset) is not str or dataset not in _GATE_REGISTRY:
        raise ValueError("dataset must be exactly 'abilene' or 'geant'")
    return _GATE_REGISTRY[dataset]


def _schedule_metadata(dataset: str, schedule: _GateSchedule) -> bytes:
    return json.dumps(
        {
            "cohort": _COHORT,
            "dataset": dataset,
            "dtype": "<f4",
            "shape": list(schedule.shape),
            "window_starts": list(schedule.starts),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def gate_window_schedule_sha256(dataset: str) -> str:
    """Hash the fixed gate starts and ``[W,F,50]`` identity without data I/O."""

    schedule = _schedule(dataset)
    return hashlib.sha256(
        _SCHEDULE_DOMAIN + _schedule_metadata(dataset, schedule)
    ).hexdigest()


def _plain_authority_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_authority_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_plain_authority_value(item) for item in value]
    return value


def _require_sealed_authority(authority: object) -> VerifiedGateAuthority:
    """Reject non-factory or internally inconsistent values before data access."""

    if type(authority) is not VerifiedGateAuthority:
        raise TypeError(
            "authority must be a factory-sealed VerifiedGateAuthority"
        )
    try:
        if authority._locked is not True:
            raise ValueError("authority is not locked")
        record_bytes = authority._record_bytes
        if type(record_bytes) is not bytes:
            raise TypeError("authority record is not immutable bytes")
        record = _gate_identity._decode_json(
            record_bytes,
            label="gate authority record",
        )
        if _gate_identity._canonical_json(record) != record_bytes:
            raise ValueError("authority record is not canonical")
        _gate_identity._validate_authority_record(record)
        if (
            record.get("method_freeze_sha256")
            != authority.method_freeze_sha256
        ):
            raise ValueError("authority method hash differs from its record")
        record_payload = record.get("authority_payload")
        if (
            _gate_identity._canonical_json(record_payload)
            != _gate_identity._canonical_json(
                _plain_authority_value(authority.payload)
            )
        ):
            raise ValueError("authority payload differs from its record")
    except Exception as exc:
        raise TypeError(
            "authority must be a consistent factory-sealed "
            "VerifiedGateAuthority"
        ) from exc
    return authority


def _load_gate_windows(
    authority: VerifiedGateAuthority,
    dataset: str,
) -> RegisteredWindows:
    """Load exactly one fixed gate cohort after validating sealed authority."""

    _require_sealed_authority(authority)
    schedule = _schedule(dataset)
    schedule_hash_before = gate_window_schedule_sha256(dataset)

    loaded = _upstream_data.load_windows(dataset, _COHORT)
    if type(loaded) is not tuple or len(loaded) != 2:
        raise TypeError("gate loader must return exactly (values, starts)")
    values, starts = loaded

    if type(starts) is not tuple or any(type(start) is not int for start in starts):
        raise TypeError("gate window starts must be an exact integer tuple")
    if starts != schedule.starts:
        raise ValueError("gate window starts differ from the fixed registry")
    if type(values) is not np.ndarray:
        raise TypeError("gate values must be a NumPy ndarray")
    if values.shape != schedule.shape:
        raise ValueError(
            f"gate values shape {values.shape!r} differs from "
            f"registered shape {schedule.shape!r}"
        )
    if values.dtype != _DTYPE:
        raise TypeError("gate values must have exact float32 dtype")
    if not values.flags.c_contiguous:
        raise ValueError("gate values must have C-order contiguous storage")
    if values.flags.writeable:
        raise ValueError("gate values must be read-only")
    if not np.isfinite(values).all():
        raise ValueError("gate values must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("gate values must be nonnegative")

    registered = RegisteredWindows(
        dataset=dataset,
        cohort=_COHORT,
        absolute_starts=starts,
        values=values,
    )
    schedule_hash_after = gate_window_schedule_sha256(dataset)
    if schedule_hash_after != schedule_hash_before:
        raise RuntimeError("gate schedule identity changed during loading")
    if (
        registered.absolute_starts != schedule.starts
        or registered.values.shape != schedule.shape
    ):
        raise RuntimeError("registered gate wrapper changed schedule identity")
    return registered


__all__ = ["gate_window_schedule_sha256"]
