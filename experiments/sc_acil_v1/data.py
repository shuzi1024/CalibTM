"""Fixed loaders for fit/source-dev/gate windows from permitted arrays only."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from experiments.acil_innovation_v1.data import (
    RegisteredWindows,
    _parent,
    expected_parsed_array_sha256,
)
from experiments.acil_innovation_v1.registries import (
    cohort_spec,
    dataset_spec,
    ordered_window_starts,
)


_COHORTS = frozenset({"fit", "source_dev", "gate"})


def permitted_parent_hashes() -> dict[str, str]:
    return {
        f"{dataset}/{split}": expected_parsed_array_sha256(dataset, split)
        for dataset in ("abilene", "geant")
        for split in ("train", "val")
    }


@lru_cache(maxsize=6)
def load_windows(dataset: str, cohort: str) -> RegisteredWindows:
    if dataset not in {"abilene", "geant"} or cohort not in _COHORTS:
        raise ValueError("dataset/cohort is outside the registered SC-ACIL boundary")
    specification = cohort_spec(dataset, cohort)
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


__all__ = ["load_windows", "permitted_parent_hashes"]

