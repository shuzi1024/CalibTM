"""Build auditable per-flow metric records on immutable target registries."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .masks import MaskBundle, oracle_partition
from .metrics import MetricRecord, metric_record


def records_for_window(
    *,
    method: str,
    seed_bundle: int,
    dataset: str,
    mask_family: str,
    window_start: int,
    truth: np.ndarray,
    prediction: np.ndarray,
    bundles: Sequence[MaskBundle],
    oracle: bool,
) -> tuple[MetricRecord, ...]:
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 50:
        raise ValueError("truth and prediction must be matching [F,50] arrays")
    if len(bundles) != truth.shape[0]:
        raise ValueError("mask bundle count must equal flow count")
    rows = []
    for flow, bundle in enumerate(bundles):
        identity = bundle.identity
        if (
            identity.dataset != dataset
            or identity.window_start != window_start
            or identity.flow_index != flow
            or identity.family != mask_family
            or identity.seed_bundle != seed_bundle
        ):
            raise ValueError("mask identity does not match evaluation cell")
        target = oracle_partition(bundle).evaluation if oracle else bundle.target
        rows.append(
            metric_record(
                identity={
                    "method": method,
                    "seed_bundle": seed_bundle,
                    "dataset": dataset,
                    "mask_family": mask_family,
                    "window_start": window_start,
                    "flow": flow,
                    "oracle": oracle,
                },
                truth=truth[flow],
                prediction=prediction[flow],
                observed=bundle.observed,
                target=target,
            )
        )
    return tuple(rows)


__all__ = ["records_for_window"]
