"""Path-free access to the already audited, permitted AnchorCV cohorts."""

from __future__ import annotations

import hashlib
import json

from experiments.acil_innovation_v1.data import (
    RegisteredWindows,
    expected_parsed_array_sha256,
    fit_fallback,
    load_registered_windows,
)
from experiments.acil_innovation_v1.preprocessing import FitFallback


_DATASETS = ("abilene", "geant")
_COHORTS = ("fit", "source_dev", "tune")
# This upstream capability exposes exactly the same three permitted cohorts.
# Callers cannot supply it, so AnchorCV has no stage/split override.
_UPSTREAM_CAPABILITY = "full_tune"
_DATA_DOMAIN = b"anchorcv-v1:permitted-parsed-arrays:v1\x00"


def _dataset_name(dataset: object) -> str:
    if not isinstance(dataset, str) or dataset not in _DATASETS:
        raise ValueError("dataset must be exactly abilene or geant")
    return dataset


def load_permitted_windows(dataset: str, cohort: str) -> RegisteredWindows:
    """Load one fixed fit/source-dev/tune schedule without path authority."""

    dataset = _dataset_name(dataset)
    if not isinstance(cohort, str) or cohort not in _COHORTS:
        raise ValueError("cohort must be exactly fit, source_dev, or tune")
    result = load_registered_windows(_UPSTREAM_CAPABILITY, dataset, cohort)
    if result.dataset != dataset or result.cohort != cohort:
        raise RuntimeError("upstream registered-window identity drifted")
    return result


def permitted_fit_fallback(dataset: str) -> FitFallback:
    """Return observation-normalization fallback computed from permitted fit."""

    return fit_fallback(_dataset_name(dataset))


def canonical_data_identity(dataset: str) -> dict[str, object]:
    """Return the pinned semantic identities of permitted parsed parents.

    The singular ``data_sha256`` is a deterministic digest over the two
    already-pinned parsed-array hashes.  No raw file path or test bytes are
    exposed or read here.
    """

    dataset = _dataset_name(dataset)
    parsed = {
        split: expected_parsed_array_sha256(dataset, split)
        for split in ("train", "val")
    }
    payload = {
        "dataset": dataset,
        "hash_scope": "canonical_parsed_permitted_split_arrays_only",
        "parsed_array_sha256": parsed,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        **payload,
        "data_sha256": hashlib.sha256(_DATA_DOMAIN + canonical).hexdigest(),
    }


__all__ = [
    "canonical_data_identity",
    "load_permitted_windows",
    "permitted_fit_fallback",
]
