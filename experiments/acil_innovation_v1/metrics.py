"""Immutable-target metric records for ACIL-Innovation v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


_TARGET_HASH_DOMAIN = b"acil-innovation-v1:target:v1\x00"


def _bool50(value: ArrayLike, label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype(np.bool_) or result.shape != (50,):
        raise TypeError(f"{label} must be a boolean vector of shape (50,)")
    return np.ascontiguousarray(result)


def target_sha256(target: ArrayLike) -> str:
    canonical = _bool50(target, "target")
    digest = hashlib.sha256(_TARGET_HASH_DOMAIN)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def validate_qe_partition(
    observed: ArrayLike, q: ArrayLike, e: ArrayLike
) -> None:
    """Require ``Q`` and ``E`` to partition the unobserved complement exactly."""

    observed_array = _bool50(observed, "observed")
    q_array = _bool50(q, "q")
    e_array = _bool50(e, "e")
    if np.any(q_array & e_array):
        raise ValueError("Q and E must be disjoint")
    if np.any(observed_array & (q_array | e_array)):
        raise ValueError("Q/E cannot contain an observed timestamp")
    if not np.array_equal(q_array | e_array, ~observed_array):
        raise ValueError("Q union E must equal the unobserved complement")
    if int(q_array.sum()) != 1:
        raise ValueError("oracle Q must contain exactly one timestamp")


@dataclass(frozen=True, slots=True)
class MetricRecord:
    identity: Mapping[str, object]
    absolute_error_sum: float
    absolute_truth_sum: float
    squared_error_sum: float
    squared_truth_sum: float
    target_count: int
    target_set_sha256: str

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["identity"] = dict(self.identity)
        return payload


@dataclass(frozen=True, slots=True)
class MetricSums:
    absolute_error_sum: float
    absolute_truth_sum: float
    squared_error_sum: float
    squared_truth_sum: float
    target_count: int
    record_count: int

    @property
    def nmae(self) -> float:
        if self.absolute_truth_sum == 0.0:
            raise ValueError("pooled absolute-truth denominator is zero")
        return self.absolute_error_sum / self.absolute_truth_sum

    @property
    def nrmse(self) -> float:
        if self.squared_truth_sum == 0.0:
            raise ValueError("pooled squared-truth denominator is zero")
        return math.sqrt(self.squared_error_sum / self.squared_truth_sum)


def metric_record(
    *,
    identity: Mapping[str, object],
    truth: ArrayLike,
    prediction: ArrayLike,
    observed: ArrayLike,
    target: ArrayLike,
) -> MetricRecord:
    if not isinstance(identity, Mapping) or not identity:
        raise TypeError("identity must be a nonempty mapping")
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if truth_array.shape != (50,) or prediction_array.shape != (50,):
        raise ValueError("truth and prediction must have shape (50,)")
    if not np.isfinite(truth_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("truth and prediction must be finite")
    if np.any(truth_array < 0.0) or np.any(prediction_array < 0.0):
        raise ValueError("traffic truth and prediction must be nonnegative")
    observed_array = _bool50(observed, "observed")
    target_array = _bool50(target, "target")
    if np.any(observed_array & target_array):
        raise ValueError("observed and target must be disjoint")
    if not np.array_equal(prediction_array[observed_array], truth_array[observed_array]):
        raise ValueError("observed hard projection drifted")
    difference = prediction_array[target_array] - truth_array[target_array]
    selected_truth = truth_array[target_array]
    return MetricRecord(
        identity=json.loads(json.dumps(dict(identity), sort_keys=True)),
        absolute_error_sum=math.fsum(map(float, np.abs(difference))),
        absolute_truth_sum=math.fsum(map(float, np.abs(selected_truth))),
        squared_error_sum=math.fsum(map(float, np.square(difference))),
        squared_truth_sum=math.fsum(map(float, np.square(selected_truth))),
        target_count=int(target_array.sum()),
        target_set_sha256=target_sha256(target_array),
    )


def _field(record: MetricRecord | Mapping[str, object], name: str) -> object:
    return getattr(record, name) if isinstance(record, MetricRecord) else record[name]


def aggregate_records(
    records: Sequence[MetricRecord | Mapping[str, object]] | Iterable[MetricRecord | Mapping[str, object]],
) -> MetricSums:
    values = tuple(records)
    if not values:
        raise ValueError("cannot aggregate an empty record table")
    numeric_fields = (
        "absolute_error_sum",
        "absolute_truth_sum",
        "squared_error_sum",
        "squared_truth_sum",
    )
    for record in values:
        for name in numeric_fields:
            value = float(_field(record, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"record {name} must be finite and nonnegative")
    return MetricSums(
        absolute_error_sum=math.fsum(float(_field(item, "absolute_error_sum")) for item in values),
        absolute_truth_sum=math.fsum(float(_field(item, "absolute_truth_sum")) for item in values),
        squared_error_sum=math.fsum(float(_field(item, "squared_error_sum")) for item in values),
        squared_truth_sum=math.fsum(float(_field(item, "squared_truth_sum")) for item in values),
        target_count=sum(int(_field(item, "target_count")) for item in values),
        record_count=len(values),
    )


def relative_improvement(candidate: MetricSums, comparator: MetricSums) -> float:
    denominator = comparator.nmae
    if denominator == 0.0:
        raise ValueError("comparator NMAE is zero")
    return 1.0 - candidate.nmae / denominator


__all__ = [
    "MetricRecord",
    "MetricSums",
    "aggregate_records",
    "metric_record",
    "relative_improvement",
    "target_sha256",
    "validate_qe_partition",
]
