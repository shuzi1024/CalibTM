"""Fixed branch-sealed loader for canonical permitted arrays.

The public API accepts only a frozen stage, dataset, and registered cohort.  It
has no raw-file, split, test, or override argument by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import stat

import numpy as np
from numpy.typing import NDArray

from .config import load_protocol_config
from .preprocessing import FitFallback
from .registries import cohort_spec, dataset_spec, ordered_window_starts


_CANONICAL = Path(__file__).with_name("canonical")
_PARSED_ARRAY_PROTOCOL_VERSION = "od-orbit-gpt-v1"
_PARSED_ARRAY_HASH_DOMAIN = b"od-orbit-gpt-v1:parsed-array-sha256:v1\x00"
_EXPECTED_PARSED_ARRAY_SHA256 = {
    ("abilene", "train"): "5ec5799c6a57c3a3971ab8b33fe95a8ea7c93a01955bec92440a75f38115d8d2",
    ("abilene", "val"): "561be18adfff24be226c23e5a8937b9bceed5a5804bdc8c56c530ef8afe5626e",
    ("geant", "train"): "233cb96106d01c6ef483ae01041d71188a0a37c65aa73fb1074b635f75e796f2",
    ("geant", "val"): "337d1a0b6d1583d94332b80e22d2ecaf297827e3450666e3cc4f678eb45e05dc",
}


@dataclass(frozen=True, slots=True)
class RegisteredWindows:
    dataset: str
    cohort: str
    absolute_starts: tuple[int, ...]
    values: NDArray[np.float32]


def expected_parsed_array_sha256(dataset: str, split: str) -> str:
    """Return the sole pinned upstream semantic identity for one parent."""

    if not isinstance(dataset, str) or dataset not in {"abilene", "geant"}:
        raise ValueError("dataset must be exactly 'abilene' or 'geant'")
    if not isinstance(split, str):
        raise TypeError("split must be the exact string 'train' or 'val'")
    if split not in {"train", "val"}:
        raise ValueError("split must be exactly 'train' or 'val'")
    try:
        return _EXPECTED_PARSED_ARRAY_SHA256[(dataset, split)]
    except KeyError:  # defensive against a source-registry edit
        raise RuntimeError("permitted parsed-array identity is not pinned") from None


def parsed_array_sha256(values, dataset: str, split: str) -> str:
    """Recompute the existing OD-Orbit parsed-array semantic SHA-256."""

    specification = dataset_spec(dataset)
    if split not in specification.permitted_splits:
        raise ValueError("split must be exactly 'train' or 'val'")
    bounds = specification.permitted_splits[split]
    expected_shape = (bounds[1] - bounds[0], specification.flows)
    with np.errstate(over="ignore", invalid="ignore"):
        canonical = np.asarray(values, dtype="<f4", order="C")
    if canonical.shape != expected_shape:
        raise ValueError(
            f"array shape {canonical.shape!r} does not match {expected_shape!r} "
            f"for {dataset}/{split}"
        )
    if not canonical.flags.c_contiguous:
        canonical = np.ascontiguousarray(canonical, dtype="<f4")
    if not np.isfinite(canonical).all():
        raise ValueError("canonical permitted array must contain only finite values")
    metadata = json.dumps(
        {
            "bounds": list(bounds),
            "dataset": dataset,
            "dtype": canonical.dtype.str,
            "order": "C",
            "protocol_version": _PARSED_ARRAY_PROTOCOL_VERSION,
            "shape": list(canonical.shape),
            "split": split,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(_PARSED_ARRAY_HASH_DOMAIN)
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(canonical.nbytes.to_bytes(8, "big"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _load_canonical_array(
    path: Path, dataset: str, split: str, expected_sha256: str
) -> NDArray[np.float32]:
    """Load and semantically bind exactly one fixed permitted ``.npy`` parent."""

    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot stat pinned canonical permitted array: {path.name}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("canonical permitted array must be a regular non-symlink file")
    specification = dataset_spec(dataset)
    if split not in specification.permitted_splits:
        raise ValueError("requested split is outside the permitted parent registry")
    bounds = specification.permitted_splits[split]
    try:
        mapped = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot load pinned canonical permitted array: {path.name}") from exc
    expected_shape = (bounds[1] - bounds[0], specification.flows)
    if mapped.shape != expected_shape or mapped.dtype != np.dtype("<f4"):
        raise ValueError("canonical permitted array shape/dtype drifted")
    if mapped.flags.writeable or not mapped.flags.c_contiguous:
        raise ValueError("canonical permitted array must be read-only C-order")
    # Detach the verified parent from its backing file before caching it.  A
    # later filesystem mutation therefore cannot alter an already-bound run.
    values = np.array(mapped, dtype="<f4", order="C", copy=True)
    values.setflags(write=False)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("canonical permitted traffic must be finite and nonnegative")
    actual_sha256 = parsed_array_sha256(values, dataset, split)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"canonical array does not match the pinned parsed-array SHA-256 "
            f"for {dataset}/{split}"
        )
    return values


def _capability(stage: str, dataset: str, cohort: str) -> None:
    config = load_protocol_config()
    if stage not in tuple(config["freeze"]["active_manifest_stages"]):
        raise ValueError(f"stage {stage!r} is outside the active manifest")
    capability = config.get("capabilities", {}).get(stage)
    if not isinstance(capability, dict):
        raise ValueError(f"unknown stage capability {stage!r}")
    if dataset not in tuple(capability.get("datasets", ())):
        raise ValueError(f"stage capability forbids dataset {dataset!r}")
    if cohort not in tuple(capability.get("cohorts", ())):
        raise ValueError(f"stage capability forbids cohort {cohort!r}")


@lru_cache(maxsize=4)
def _parent(dataset: str, split: str) -> NDArray[np.float32]:
    specification = dataset_spec(dataset)
    if split not in specification.permitted_splits:
        raise ValueError("requested split is outside the permitted parent registry")
    path = _CANONICAL / f"{dataset}_{split}.npy"
    return _load_canonical_array(
        path,
        dataset,
        split,
        expected_parsed_array_sha256(dataset, split),
    )


def load_registered_windows(
    stage: str, dataset: str, cohort: str
) -> RegisteredWindows:
    _capability(stage, dataset, cohort)
    specification = cohort_spec(dataset, cohort)
    if specification.is_gap:
        raise ValueError("gap cohorts cannot be loaded as registered windows")
    starts = ordered_window_starts(dataset, cohort)
    parent = _parent(dataset, specification.split)
    parent_start = dataset_spec(dataset).permitted_splits[specification.split][0]
    rows = []
    for absolute_start in starts:
        local_start = absolute_start - parent_start
        local_stop = local_start + 50
        if local_start < 0 or local_stop > parent.shape[0]:
            raise ValueError("registered window escapes its permitted parent")
        rows.append(np.asarray(parent[local_start:local_stop], dtype="<f4").T)
    values = np.ascontiguousarray(np.stack(rows, axis=0), dtype="<f4")
    values.setflags(write=False)
    return RegisteredWindows(
        dataset=dataset,
        cohort=cohort,
        absolute_starts=starts,
        values=values,
    )


@lru_cache(maxsize=2)
def fit_fallback(dataset: str) -> FitFallback:
    specification = cohort_spec(dataset, "fit")
    parent = _parent(dataset, specification.split)
    parent_start = dataset_spec(dataset).permitted_splits[specification.split][0]
    start = specification.start - parent_start
    stop = specification.stop - parent_start
    fit_values = np.asarray(parent[start:stop], dtype=np.float64)
    mean = float(np.mean(fit_values, dtype=np.float64))
    std = float(np.std(fit_values, dtype=np.float64))
    return FitFallback(mean=mean, std=max(std, 1e-6))


__all__ = [
    "RegisteredWindows",
    "expected_parsed_array_sha256",
    "fit_fallback",
    "load_registered_windows",
    "parsed_array_sha256",
]
