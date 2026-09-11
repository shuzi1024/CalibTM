"""Immutable dataset, cohort, seed, and stage registries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from types import MappingProxyType
from typing import Mapping

from .config import canonical_json_bytes, load_protocol_config, protocol_config_sha256


_SCHEDULE_HASH_DOMAIN = b"acil-innovation-v1:window-schedule:v1\x00"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    flows: int
    permitted_splits: Mapping[str, tuple[int, int]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "permitted_splits",
            MappingProxyType(dict(self.permitted_splits)),
        )


@dataclass(frozen=True, slots=True)
class CohortSpec:
    dataset: str
    name: str
    split: str
    start: int
    stop: int
    is_gap: bool

    @property
    def bounds(self) -> tuple[int, int]:
        return (self.start, self.stop)


@dataclass(frozen=True, slots=True)
class SeedBundle:
    bundle: int
    model: int
    data_order: int
    training_mask: int
    evaluation_mask: int


@dataclass(frozen=True, slots=True)
class CheckpointBindingSpec:
    source_stage: str
    source_method: str
    consumer_roles: tuple[str, ...]
    match_fields: tuple[str, ...]
    selection: str
    relationship: str
    required_exact_matches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    role: str
    datasets: tuple[str, ...]
    train_cohorts: tuple[str, ...]
    eval_cohort: str | None
    methods: tuple[str, ...]
    learned_methods: tuple[str, ...]
    result_carriers: Mapping[str, tuple[str, ...]]
    reuse_methods: tuple[str, ...]
    dependency_stages: tuple[str, ...]
    checkpoint_hash_bindings: tuple[CheckpointBindingSpec, ...]
    seed_bundles: tuple[int, ...]
    training_has_mask_axis: bool
    learned_fit_count: int
    result_cell_count: int


def _require_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _require_exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def dataset_spec(name: str) -> DatasetSpec:
    """Resolve one exact permitted dataset identity."""

    name = _require_name(name, "dataset")
    return _cached_dataset_spec(name)


@lru_cache(maxsize=None)
def _cached_dataset_spec(name: str) -> DatasetSpec:
    config = load_protocol_config()
    try:
        payload = config["datasets"][name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}") from None
    splits = {
        split: (int(bounds[0]), int(bounds[1]))
        for split, bounds in payload["permitted_splits"].items()
    }
    if tuple(splits) != ("train", "val"):
        raise RuntimeError("permitted split registry drifted")
    if splits["train"][1] != splits["val"][0]:
        raise RuntimeError("permitted train and val parents must be contiguous")
    return DatasetSpec(name=name, flows=int(payload["flows"]), permitted_splits=splits)


def cohort_spec(dataset: str, cohort: str) -> CohortSpec:
    """Resolve one exact data cohort or deliberately unscheduled gap."""

    dataset = _require_name(dataset, "dataset")
    cohort = _require_name(cohort, "cohort")
    return _cached_cohort_spec(dataset, cohort)


@lru_cache(maxsize=None)
def _cached_cohort_spec(dataset: str, cohort: str) -> CohortSpec:
    config = load_protocol_config()
    try:
        payload = config["datasets"][dataset]["cohorts"][cohort]
    except KeyError:
        if dataset not in config["datasets"]:
            raise ValueError(f"unknown dataset {dataset!r}") from None
        raise ValueError(f"unknown cohort {cohort!r} for {dataset!r}") from None
    start, stop = map(int, payload["bounds"])
    if start < 0 or stop <= start:
        raise RuntimeError("cohort interval is invalid")
    split = str(payload["split"])
    parent_start, parent_stop = dataset_spec(dataset).permitted_splits[split]
    if not (parent_start <= start < stop <= parent_stop):
        raise RuntimeError("cohort escapes its permitted parsed parent")
    kind = str(payload["kind"])
    if kind not in {"data", "gap"}:
        raise RuntimeError("cohort kind must be data or gap")
    return CohortSpec(
        dataset=dataset,
        name=cohort,
        split=split,
        start=start,
        stop=stop,
        is_gap=kind == "gap",
    )


def ordered_window_starts(dataset: str, cohort: str) -> tuple[int, ...]:
    """Generate the exact audited window schedule without consulting data."""

    dataset = _require_name(dataset, "dataset")
    cohort = _require_name(cohort, "cohort")
    return _cached_ordered_window_starts(dataset, cohort)


@lru_cache(maxsize=None)
def _cached_ordered_window_starts(dataset: str, cohort: str) -> tuple[int, ...]:
    spec = cohort_spec(dataset, cohort)
    if spec.is_gap:
        raise ValueError(f"gap cohort {dataset}/{cohort} has no window schedule")
    payload = load_protocol_config()["datasets"][dataset]["cohorts"][cohort]
    schedule = payload.get("window_schedule")
    if not isinstance(schedule, dict):
        raise RuntimeError("data cohort has no window schedule")
    count = _require_exact_int(schedule.get("count"), "window count")
    if count < 1:
        raise RuntimeError("window count must be positive")
    window_length = int(load_protocol_config()["protocol"]["window_length"])
    maximum_start = spec.stop - window_length
    kind = schedule.get("kind")
    if kind == "linspace_floor":
        if count == 1:
            starts = (spec.start,)
        else:
            span = maximum_start - spec.start
            starts = tuple(
                spec.start + (index * span) // (count - 1)
                for index in range(count)
            )
    elif kind == "stride":
        stride = _require_exact_int(schedule.get("stride"), "window stride")
        if stride <= 0:
            raise RuntimeError("window stride must be positive")
        starts = tuple(spec.start + index * stride for index in range(count))
    else:
        raise RuntimeError("unknown window schedule kind")
    if len(starts) != count or len(set(starts)) != count:
        raise RuntimeError("window schedule count or uniqueness drifted")
    if any(start < spec.start or start + window_length > spec.stop for start in starts):
        raise RuntimeError("window schedule escapes its cohort")
    return starts


def window_schedule_sha256(dataset: str, cohort: str) -> str:
    """Hash one ordered schedule together with all interval semantics."""

    dataset = _require_name(dataset, "dataset")
    cohort = _require_name(cohort, "cohort")
    return _cached_window_schedule_sha256(dataset, cohort)


@lru_cache(maxsize=None)
def _cached_window_schedule_sha256(dataset: str, cohort: str) -> str:
    spec = cohort_spec(dataset, cohort)
    starts = ordered_window_starts(dataset, cohort)
    payload = {
        "bounds": list(spec.bounds),
        "cohort": cohort,
        "config_sha256": protocol_config_sha256(),
        "dataset": dataset,
        "protocol": "acil-innovation-v1",
        "window_length": 50,
        "window_starts": list(starts),
    }
    return hashlib.sha256(_SCHEDULE_HASH_DOMAIN + canonical_json_bytes(payload)).hexdigest()


def seed_bundle(bundle: int) -> SeedBundle:
    """Resolve one of the six pre-registered seed bundles."""

    bundle = _require_exact_int(bundle, "seed bundle")
    return _cached_seed_bundle(bundle)


@lru_cache(maxsize=None)
def _cached_seed_bundle(bundle: int) -> SeedBundle:
    matches = [
        item
        for item in load_protocol_config()["seeds"]["bundles"]
        if item["bundle"] == bundle
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown seed bundle {bundle!r}")
    item = matches[0]
    return SeedBundle(
        bundle=bundle,
        model=int(item["model"]),
        data_order=int(item["data_order"]),
        training_mask=int(item["training_mask"]),
        evaluation_mask=int(item["evaluation_mask"]),
    )


def stage_spec(name: str) -> StageSpec:
    """Resolve a complete stage grid; callers cannot add methods or seeds."""

    name = _require_name(name, "stage")
    return _cached_stage_spec(name)


@lru_cache(maxsize=None)
def _cached_stage_spec(name: str) -> StageSpec:
    config = load_protocol_config()
    try:
        item = config["stages"][name]
    except KeyError:
        raise ValueError(f"unknown stage {name!r}") from None
    eval_cohort_payload = item["eval_cohort"]
    if eval_cohort_payload is not None and not isinstance(eval_cohort_payload, str):
        raise RuntimeError("stage eval cohort must be a string or null")
    raw_dependencies = item.get("dependency_stages", [])
    raw_bindings = item.get("checkpoint_hash_bindings", [])
    if not isinstance(raw_dependencies, list) or not isinstance(raw_bindings, list):
        raise RuntimeError("stage dependencies and bindings must be arrays")
    dependency_stages = tuple(
        _require_name(value, "dependency stage") for value in raw_dependencies
    )
    bindings: list[CheckpointBindingSpec] = []
    for binding in raw_bindings:
        if not isinstance(binding, dict):
            raise RuntimeError("checkpoint hash binding must be an object")
        consumers = binding.get("consumer_roles")
        match_fields = binding.get("match_fields")
        exact_matches = binding.get("required_exact_matches")
        if not all(
            isinstance(value, list)
            for value in (consumers, match_fields, exact_matches)
        ):
            raise RuntimeError("checkpoint binding list fields must be arrays")
        bindings.append(
            CheckpointBindingSpec(
                source_stage=_require_name(
                    binding.get("source_stage"), "checkpoint source stage"
                ),
                source_method=_require_name(
                    binding.get("source_method"), "checkpoint source method"
                ),
                consumer_roles=tuple(
                    _require_name(value, "checkpoint consumer role")
                    for value in consumers
                ),
                match_fields=tuple(
                    _require_name(value, "checkpoint match field")
                    for value in match_fields
                ),
                selection=_require_name(
                    binding.get("selection"), "checkpoint selection"
                ),
                relationship=_require_name(
                    binding.get("relationship"), "checkpoint relationship"
                ),
                required_exact_matches=tuple(
                    _require_name(value, "checkpoint exact hash field")
                    for value in exact_matches
                ),
            )
        )
    spec = StageSpec(
        name=name,
        role=str(item.get("role", "train_and_evaluate")),
        datasets=tuple(item["datasets"]),
        train_cohorts=tuple(item["train_cohorts"]),
        eval_cohort=eval_cohort_payload,
        methods=tuple(item["methods"]),
        learned_methods=tuple(item["learned_methods"]),
        result_carriers=MappingProxyType(
            {
                method: tuple(carried)
                for method, carried in item["result_carriers"].items()
            }
        ),
        reuse_methods=tuple(item.get("reuse_methods", ())),
        dependency_stages=dependency_stages,
        checkpoint_hash_bindings=tuple(bindings),
        seed_bundles=tuple(int(value) for value in item["seed_bundles"]),
        training_has_mask_axis=bool(item["training_has_mask_axis"]),
        learned_fit_count=int(item["learned_fit_count"]),
        result_cell_count=int(item["result_cell_count"]),
    )
    expected_fits = len(spec.learned_methods) * len(spec.datasets) * len(spec.seed_bundles)
    expected_cells = (
        sum(len(carried) for carried in spec.result_carriers.values())
        * len(spec.datasets)
        * len(spec.seed_bundles)
        * len(config["masks"]["families"])
    )
    if spec.training_has_mask_axis:
        raise RuntimeError("v1 training grid must not have a mask-family axis")
    if spec.learned_fit_count != expected_fits:
        raise RuntimeError("learned fit count does not match the frozen stage product")
    if spec.result_cell_count != expected_cells:
        raise RuntimeError("result cell count does not match the frozen stage product")
    if tuple(spec.result_carriers) != spec.learned_methods:
        raise RuntimeError("result carrier jobs do not match learned-method job order")
    carried_methods = tuple(
        method
        for carried in spec.result_carriers.values()
        for method in carried
    )
    if len(carried_methods) != len(set(carried_methods)):
        raise RuntimeError("a result method is carried by multiple planned jobs")
    if spec.eval_cohort is None:
        if carried_methods:
            raise RuntimeError("training-only stage cannot carry evaluation records")
    elif set(carried_methods) != set(spec.methods):
        raise RuntimeError("result carriers do not cover the evaluation method registry")
    if set(spec.learned_methods).intersection(spec.reuse_methods):
        raise RuntimeError("learned and reused method registries must be disjoint")
    if not set(spec.learned_methods).union(spec.reuse_methods).issubset(spec.methods):
        raise RuntimeError("learned or reused method is absent from the stage methods")
    if spec.eval_cohort is None:
        if spec.role != "training_only_dependency" or spec.result_cell_count != 0:
            raise RuntimeError("null-evaluation stage must be a training-only dependency")
        if spec.dependency_stages or spec.checkpoint_hash_bindings:
            raise RuntimeError("training-only dependency cannot itself bind a dependency")
    elif spec.role != "train_and_evaluate":
        raise RuntimeError("evaluation stage role drifted")
    if len(spec.dependency_stages) != len(set(spec.dependency_stages)):
        raise RuntimeError("stage dependency registry contains duplicates")
    binding_dependency_order = tuple(
        dict.fromkeys(binding.source_stage for binding in spec.checkpoint_hash_bindings)
    )
    if binding_dependency_order != spec.dependency_stages:
        raise RuntimeError("checkpoint bindings do not exactly cover dependency stages")
    covered_top_level_consumers: set[str] = set()
    stage_order = tuple(config["stages"])
    for binding in spec.checkpoint_hash_bindings:
        if binding.source_stage not in config["stages"]:
            raise RuntimeError("stage dependency is not registered")
        if stage_order.index(binding.source_stage) >= stage_order.index(name):
            raise RuntimeError("checkpoint dependency must precede its consumer stage")
        source = config["stages"][binding.source_stage]
        if binding.source_method not in source["learned_methods"]:
            raise RuntimeError("checkpoint source method is not learned in its source stage")
        if tuple(source["datasets"]) != spec.datasets or tuple(
            int(value) for value in source["seed_bundles"]
        ) != spec.seed_bundles:
            raise RuntimeError("checkpoint dependency dataset or seed coverage drifted")
        if binding.match_fields != ("dataset", "seed_bundle"):
            raise RuntimeError("checkpoint dependency match fields drifted")
        if binding.selection != "source_stage_best_source_dev_checkpoint":
            raise RuntimeError("checkpoint selection rule drifted")
        if binding.required_exact_matches != (
            "checkpoint_identity_sha256",
            "checkpoint_file_sha256",
            "checkpoint_tensor_sha256",
        ):
            raise RuntimeError("dependency checkpoint exact hash binding drifted")
        if binding.relationship not in {
            "exact_checkpoint_reuse",
            "method_label_rename_exact_same_checkpoint",
        }:
            raise RuntimeError("checkpoint relationship drifted")
        if not binding.consumer_roles or len(binding.consumer_roles) != len(
            set(binding.consumer_roles)
        ):
            raise RuntimeError("checkpoint consumer roles are empty or duplicated")
        for consumer in binding.consumer_roles:
            method = consumer.split(".", 1)[0]
            if method not in spec.methods:
                raise RuntimeError("checkpoint consumer method is absent from stage")
            if "." not in consumer:
                covered_top_level_consumers.add(consumer)
        if binding.relationship == "method_label_rename_exact_same_checkpoint":
            if any("." in consumer for consumer in binding.consumer_roles):
                raise RuntimeError("method-label rename must bind a top-level reused method")
        elif binding.source_method != "acil":
            raise RuntimeError("embedded base reuse must source ACIL")
    if covered_top_level_consumers != set(spec.reuse_methods):
        raise RuntimeError("reused methods are not exactly covered by checkpoint bindings")
    return spec


__all__ = [
    "CheckpointBindingSpec",
    "CohortSpec",
    "DatasetSpec",
    "SeedBundle",
    "StageSpec",
    "cohort_spec",
    "dataset_spec",
    "ordered_window_starts",
    "seed_bundle",
    "stage_spec",
    "window_schedule_sha256",
]
