"""Compact paired per-window evidence with flows summed before pooling."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import torch

from .protocol import canonical_json_bytes


def window_metric_rows(
    *,
    method: str,
    seed_bundle: int,
    dataset: str,
    mask_family: str,
    absolute_starts: tuple[int, ...],
    truth: torch.Tensor,
    prediction: torch.Tensor,
    observed: torch.Tensor,
    mask_registry_sha256: str,
) -> tuple[dict[str, object], ...]:
    if truth.shape != prediction.shape or truth.shape != observed.shape or truth.ndim != 3:
        raise ValueError("truth, prediction, and observed must match [W,F,T]")
    if observed.dtype is not torch.bool or len(absolute_starts) != truth.shape[0]:
        raise ValueError("invalid observed tensor or window schedule")
    if len(mask_registry_sha256) != 64:
        raise ValueError("mask registry SHA-256 is invalid")
    if not torch.isfinite(truth).all().item() or not torch.isfinite(prediction).all().item():
        raise ValueError("metric operands must be finite")
    target = ~observed
    difference = prediction.double() - truth.double()
    rows = []
    for index, start in enumerate(absolute_starts):
        selected_error = difference[index].masked_select(target[index])
        selected_truth = truth[index].double().masked_select(target[index])
        row = {
            "identity": {
                "dataset": dataset,
                "mask_family": mask_family,
                "method": method,
                "seed_bundle": seed_bundle,
                "window_start": int(start),
            },
            "absolute_error_sum": float(selected_error.abs().sum(dtype=torch.float64).item()),
            "absolute_truth_sum": float(selected_truth.abs().sum(dtype=torch.float64).item()),
            "mask_registry_sha256": mask_registry_sha256,
            "squared_error_sum": float(selected_error.square().sum(dtype=torch.float64).item()),
            "squared_truth_sum": float(selected_truth.square().sum(dtype=torch.float64).item()),
            "target_count": int(target[index].sum().item()),
        }
        if any(
            not math.isfinite(float(row[field]))
            for field in (
                "absolute_error_sum",
                "absolute_truth_sum",
                "squared_error_sum",
                "squared_truth_sum",
            )
        ):
            raise ValueError("window metric row is nonfinite")
        rows.append(row)
    return tuple(rows)


def mask_registry_sha256(mask_hash_grid: tuple[tuple[str, ...], ...]) -> str:
    payload = [list(row) for row in mask_hash_grid]
    return hashlib.sha256(
        b"sc-acil-v1:mask-grid:v1\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    path = Path(path)
    values = tuple(dict(row) for row in rows)
    if not values:
        raise ValueError("cannot write an empty evidence table")
    content = b"".join(canonical_json_bytes(row) + b"\n" for row in values)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
    return {
        "file": path.name,
        "row_count": len(values),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def load_rows(path: Path, *, expected_sha256: str | None = None):
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("evidence file SHA-256 drifted")
    rows = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("evidence row lacks newline terminator")
        value = json.loads(line[:-1].decode("ascii"))
        if line != canonical_json_bytes(value) + b"\n":
            raise ValueError("evidence row is not canonical")
        rows.append(value)
    if not rows:
        raise ValueError("evidence table is empty")
    return tuple(rows)


__all__ = [
    "load_rows",
    "mask_registry_sha256",
    "window_metric_rows",
    "write_rows",
]

