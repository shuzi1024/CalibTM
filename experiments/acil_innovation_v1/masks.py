"""Deterministic K=3 masks, legal LOO anchors, and oracle Q/E identities."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import canonical_json_bytes, load_protocol_config, protocol_config_sha256
from .registries import (
    CohortSpec,
    DatasetSpec,
    cohort_spec,
    dataset_spec,
    ordered_window_starts,
    seed_bundle as registered_seed_bundle,
    window_schedule_sha256,
)


_MASK_RNG_DOMAIN: Final = b"acil-innovation-v1:mask-rng:v1\x00"
_MASK_HASH_DOMAIN: Final = b"acil-innovation-v1:observed-mask:v1\x00"
_ORACLE_RNG_DOMAIN: Final = b"acil-innovation-v1:oracle-q-rng:v1\x00"
_ORACLE_SUPPORT_HASH_DOMAIN: Final = b"acil-innovation-v1:oracle-q:v1\x00"
_ORACLE_EVAL_HASH_DOMAIN: Final = b"acil-innovation-v1:oracle-e:v1\x00"
_FAMILIES: Final = ("random", "internal_block", "two_burst")
_EVALUATION_COHORTS: Final = ("source_dev", "tune", "gate")
_FACTORY_TOKEN: Final = object()


@dataclass(frozen=True, slots=True)
class MaskIdentity:
    protocol: str
    config_sha256: str
    dataset: str
    cohort: str
    split: str
    cohort_bounds: tuple[int, int]
    window_schedule_sha256: str
    window_start: int
    flow_index: int
    family: str
    observed_count: int
    seed_bundle: int
    purpose: str
    mask_seed: int
    replica: int
    rng_algorithm: str


@dataclass(frozen=True, slots=True)
class _MaskContext:
    dataset: DatasetSpec
    cohort: CohortSpec
    window_starts: frozenset[int]
    config_sha256: str
    window_schedule_sha256: str


def _owned_bool50(value: ArrayLike, label: str) -> NDArray[np.bool_]:
    source = np.asarray(value)
    if source.dtype != np.dtype(np.bool_) or source.shape != (50,):
        raise TypeError(f"{label} must be a boolean vector with shape (50,)")
    contiguous = np.ascontiguousarray(source, dtype=np.bool_)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_).reshape(50)
    return result


def _is_lower_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class MaskBundle:
    identity: MaskIdentity
    observed: NDArray[np.bool_]
    target: NDArray[np.bool_]
    controlled_gap: NDArray[np.bool_]
    _factory_token: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _factory_sha256: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.identity, MaskIdentity):
            raise TypeError("mask identity must be a MaskIdentity")
        observed = _owned_bool50(self.observed, "observed")
        target = _owned_bool50(self.target, "target")
        controlled = _owned_bool50(self.controlled_gap, "controlled gap")
        if int(observed.sum()) != self.identity.observed_count:
            raise ValueError("observed count does not match mask identity")
        if not np.array_equal(target, np.logical_not(observed)):
            raise ValueError("target must be the exact observed complement")
        if np.any(controlled & observed):
            raise ValueError("controlled gap cannot contain an observed anchor")
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "controlled_gap", controlled)


@dataclass(frozen=True, slots=True)
class OraclePartition:
    support: NDArray[np.bool_]
    evaluation: NDArray[np.bool_]
    mask_sha256: str
    support_sha256: str
    evaluation_sha256: str

    def __post_init__(self) -> None:
        support = _owned_bool50(self.support, "oracle support")
        evaluation = _owned_bool50(self.evaluation, "oracle evaluation")
        if int(support.sum()) != 1 or int(evaluation.sum()) != 46:
            raise ValueError("oracle Q/E must contain exactly 1/46 timestamps")
        if np.any(support & evaluation):
            raise ValueError("oracle Q and E must be disjoint")
        for label, value in (
            ("mask SHA-256", self.mask_sha256),
            ("support SHA-256", self.support_sha256),
            ("evaluation SHA-256", self.evaluation_sha256),
        ):
            if not _is_lower_sha256(value):
                raise ValueError(f"{label} is invalid")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "evaluation", evaluation)


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _require_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def mixed_training_family(epoch: int, epoch_order_position: int, seed_bundle: int) -> str:
    """Return the method-independent family assigned to one fit-window visit."""

    epoch = _require_int(epoch, "epoch")
    epoch_order_position = _require_int(epoch_order_position, "epoch-order position")
    seed_bundle = _require_int(seed_bundle, "seed bundle")
    if not 0 <= epoch < 20:
        raise ValueError("epoch must be a zero-based integer in range(0,20)")
    if not 0 <= epoch_order_position < 512:
        raise ValueError("epoch-order position must be in range(0,512)")
    seed_bundle_record = registered_seed_bundle(seed_bundle)
    del seed_bundle_record
    global_position = epoch * 512 + epoch_order_position
    return _FAMILIES[(global_position + seed_bundle - 1) % len(_FAMILIES)]


@lru_cache(maxsize=None)
def _cached_mask_context(dataset: str, cohort: str) -> _MaskContext:
    dataset_record = dataset_spec(dataset)
    cohort_record = cohort_spec(dataset, cohort)
    if cohort_record.is_gap:
        raise ValueError("gap cohorts cannot own masks")
    return _MaskContext(
        dataset=dataset_record,
        cohort=cohort_record,
        window_starts=frozenset(ordered_window_starts(dataset, cohort)),
        config_sha256=protocol_config_sha256(),
        window_schedule_sha256=window_schedule_sha256(dataset, cohort),
    )


def _make_identity(
    *,
    dataset: str,
    cohort: str,
    window_start: int,
    flow_index: int,
    family: str,
    seed_bundle_id: int,
    purpose: str,
    replica: int,
) -> MaskIdentity:
    dataset = _require_name(dataset, "dataset")
    cohort = _require_name(cohort, "cohort")
    family = _require_name(family, "family")
    purpose = _require_name(purpose, "purpose")
    window_start = _require_int(window_start, "window start")
    flow_index = _require_int(flow_index, "flow index")
    seed_bundle_id = _require_int(seed_bundle_id, "seed bundle")
    replica = _require_int(replica, "replica")
    if family not in _FAMILIES:
        raise ValueError(f"unknown mask family {family!r}")
    context = _cached_mask_context(dataset, cohort)
    if window_start not in context.window_starts:
        raise ValueError("window start is absent from the registered schedule")
    if not 0 <= flow_index < context.dataset.flows:
        raise ValueError("flow index is outside the registered dataset")
    seeds = registered_seed_bundle(seed_bundle_id)
    if purpose == "training":
        if cohort != "fit":
            raise ValueError("training masks require the fit cohort")
        if not 0 <= replica < 20:
            raise ValueError("training replica must be a registered epoch")
        mask_seed = seeds.training_mask
    elif purpose == "evaluation":
        if cohort not in _EVALUATION_COHORTS:
            raise ValueError("evaluation masks require source_dev, tune, or gate")
        if replica != 0:
            raise ValueError("evaluation replica must equal zero")
        mask_seed = seeds.evaluation_mask
    else:
        raise ValueError("purpose must be exactly training or evaluation")
    return MaskIdentity(
        protocol="acil-innovation-v1",
        config_sha256=context.config_sha256,
        dataset=dataset,
        cohort=cohort,
        split=context.cohort.split,
        cohort_bounds=context.cohort.bounds,
        window_schedule_sha256=context.window_schedule_sha256,
        window_start=window_start,
        flow_index=flow_index,
        family=family,
        observed_count=3,
        seed_bundle=seed_bundle_id,
        purpose=purpose,
        mask_seed=mask_seed,
        replica=replica,
        rng_algorithm="PCG64DXSM",
    )


@lru_cache(maxsize=8192)
def _identity_json_bytes(identity: MaskIdentity) -> bytes:
    return canonical_json_bytes(
        {
            "cohort": identity.cohort,
            "cohort_bounds": identity.cohort_bounds,
            "config_sha256": identity.config_sha256,
            "dataset": identity.dataset,
            "family": identity.family,
            "flow_index": identity.flow_index,
            "mask_seed": identity.mask_seed,
            "observed_count": identity.observed_count,
            "protocol": identity.protocol,
            "purpose": identity.purpose,
            "replica": identity.replica,
            "rng_algorithm": identity.rng_algorithm,
            "seed_bundle": identity.seed_bundle,
            "split": identity.split,
            "window_schedule_sha256": identity.window_schedule_sha256,
            "window_start": identity.window_start,
        }
    )


def _rng_for_identity(identity: MaskIdentity) -> np.random.Generator:
    payload = _identity_json_bytes(identity)
    digest = hashlib.sha256(
        _MASK_RNG_DOMAIN + len(payload).to_bytes(8, "big") + payload
    ).digest()
    seed = int.from_bytes(digest[:16], "big", signed=False)
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _generate_arrays(
    identity: MaskIdentity,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    rng = _rng_for_identity(identity)
    observed = np.zeros(50, dtype=np.bool_)
    controlled = np.zeros(50, dtype=np.bool_)
    if identity.family == "random":
        pool = list(range(50))
        for index in range(3):
            selected = int(rng.integers(index, 50))
            pool[index], pool[selected] = pool[selected], pool[index]
        observed[pool[:3]] = True
    elif identity.family == "internal_block":
        gap_length = 8
        gap_start = int(rng.integers(1, 50 - gap_length))
        candidates = [
            timestamp
            for timestamp in range(50)
            if timestamp < gap_start - 1 or timestamp > gap_start + gap_length
        ]
        third = candidates[int(rng.integers(0, len(candidates)))]
        observed[[gap_start - 1, gap_start + gap_length, third]] = True
        controlled[gap_start : gap_start + gap_length] = True
    elif identity.family == "two_burst":
        first_length = int(rng.integers(3, 9))
        second_length = int(rng.integers(3, 9))
        first_anchor = int(rng.integers(0, 50 - first_length - second_length - 2))
        middle_anchor = first_anchor + first_length + 1
        last_anchor = middle_anchor + second_length + 1
        observed[[first_anchor, middle_anchor, last_anchor]] = True
        controlled[first_anchor + 1 : middle_anchor] = True
        controlled[middle_anchor + 1 : last_anchor] = True
    else:  # protected by identity construction
        raise RuntimeError("unsupported mask family")
    return observed, controlled


def _bundle(identity: MaskIdentity) -> MaskBundle:
    observed, controlled = _generate_arrays(identity)
    bundle = MaskBundle(
        identity=identity,
        observed=observed,
        target=np.logical_not(observed),
        controlled_gap=controlled,
    )
    digest = _array_sha256(_MASK_HASH_DOMAIN, identity, bundle.observed)
    object.__setattr__(bundle, "_factory_sha256", digest)
    object.__setattr__(bundle, "_factory_token", _FACTORY_TOKEN)
    return bundle


def build_evaluation_mask(
    dataset: str,
    cohort: str,
    window_start: int,
    flow_index: int,
    family: str,
    seed_bundle: int,
) -> MaskBundle:
    """Build one fixed evaluation mask; no method or truth operand exists."""

    identity = _make_identity(
        dataset=dataset,
        cohort=cohort,
        window_start=window_start,
        flow_index=flow_index,
        family=family,
        seed_bundle_id=seed_bundle,
        purpose="evaluation",
        replica=0,
    )
    return _bundle(identity)


def build_mixed_training_mask(
    dataset: str,
    window_start: int,
    flow_index: int,
    seed_bundle: int,
    epoch: int,
    epoch_order_position: int,
) -> MaskBundle:
    """Build a fit mask whose family is fixed by the mixed-mask schedule."""

    family = mixed_training_family(epoch, epoch_order_position, seed_bundle)
    identity = _make_identity(
        dataset=dataset,
        cohort="fit",
        window_start=window_start,
        flow_index=flow_index,
        family=family,
        seed_bundle_id=seed_bundle,
        purpose="training",
        replica=epoch,
    )
    return _bundle(identity)


def _validated_bundle(bundle: MaskBundle) -> MaskBundle:
    if not isinstance(bundle, MaskBundle):
        raise TypeError("bundle must be a MaskBundle")
    identity = _make_identity(
        dataset=bundle.identity.dataset,
        cohort=bundle.identity.cohort,
        window_start=bundle.identity.window_start,
        flow_index=bundle.identity.flow_index,
        family=bundle.identity.family,
        seed_bundle_id=bundle.identity.seed_bundle,
        purpose=bundle.identity.purpose,
        replica=bundle.identity.replica,
    )
    if identity != bundle.identity:
        raise ValueError("mask identity does not match the fixed protocol")
    expected_observed, expected_controlled = _generate_arrays(identity)
    if not np.array_equal(bundle.observed, expected_observed):
        raise ValueError("observed mask content drifted")
    if not np.array_equal(bundle.controlled_gap, expected_controlled):
        raise ValueError("controlled-gap content drifted")
    if not np.array_equal(bundle.target, np.logical_not(expected_observed)):
        raise ValueError("target content drifted")
    return bundle


@lru_cache(maxsize=8192)
def _array_hash_metadata(
    identity: MaskIdentity, dtype_string: str, shape: tuple[int, ...]
) -> bytes:
    # This is the exact sorted-key canonical JSON layout.  Reusing the already
    # canonical identity bytes avoids rebuilding the identity mapping
    # for every generated mask while preserving the protocol hash byte-for-byte.
    return b"".join(
        (
            b'{"bit_order":"little","dtype":',
            canonical_json_bytes(dtype_string),
            b',"identity":',
            _identity_json_bytes(identity),
            b',"shape":',
            canonical_json_bytes(list(shape)),
            b"}",
        )
    )


def _array_sha256(
    domain: bytes, identity: MaskIdentity, array: NDArray[np.bool_]
) -> str:
    metadata = _array_hash_metadata(identity, array.dtype.str, tuple(array.shape))
    packed = np.packbits(array, bitorder="little").tobytes()
    digest = hashlib.sha256(domain)
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(packed).to_bytes(8, "big"))
    digest.update(packed)
    return digest.hexdigest()


def mask_sha256(bundle: MaskBundle) -> str:
    bundle = _validated_bundle(bundle)
    digest = _array_sha256(_MASK_HASH_DOMAIN, bundle.identity, bundle.observed)
    if (
        bundle._factory_token is _FACTORY_TOKEN
        and bundle._factory_sha256 != digest
    ):
        raise ValueError("factory-bound mask SHA-256 differs from validated content")
    return digest


def factory_mask_sha256(bundle: MaskBundle) -> str:
    """Return the hash bound by an official factory without regenerating a mask."""

    if not isinstance(bundle, MaskBundle):
        raise TypeError("bundle must be a MaskBundle")
    digest = bundle._factory_sha256
    if bundle._factory_token is not _FACTORY_TOKEN or not _is_lower_sha256(digest):
        raise ValueError("bundle does not carry an official factory-bound hash")
    if any(
        value.flags.writeable
        for value in (bundle.observed, bundle.target, bundle.controlled_gap)
    ):
        raise ValueError("factory-bound mask arrays must remain immutable")
    return digest


def deployable_target(bundle: MaskBundle) -> NDArray[np.bool_]:
    """Return the immutable all-missing E used by every deployable method."""

    bundle = _validated_bundle(bundle)
    return _owned_bool50(bundle.target, "deployable target")


def middle_anchor_index(bundle: MaskBundle) -> int:
    """Return the sole strictly two-sided LOO anchor available at K=3."""

    bundle = _validated_bundle(bundle)
    anchors = np.flatnonzero(bundle.observed)
    if anchors.shape != (3,) or not int(anchors[0]) < int(anchors[1]) < int(anchors[2]):
        raise ValueError("K=3 anchors do not admit one strict middle anchor")
    return int(anchors[1])


def loo_observed(bundle: MaskBundle) -> NDArray[np.bool_]:
    """Drop only the sorted middle observed anchor, leaving two bracketing anchors."""

    bundle = _validated_bundle(bundle)
    result = np.array(bundle.observed, dtype=np.bool_, order="C", copy=True)
    result[middle_anchor_index(bundle)] = False
    result.setflags(write=False)
    return result


def _oracle_partition_from_bundle(
    bundle: MaskBundle, mask_digest: str
) -> OraclePartition:
    missing = bundle.target
    if bundle.identity.family in {"internal_block", "two_burst"}:
        eligible_mask = missing & np.logical_not(bundle.controlled_gap)
        if not eligible_mask.any():
            eligible_mask = missing
    else:
        eligible_mask = missing
    eligible = np.flatnonzero(eligible_mask)
    payload = canonical_json_bytes(
        {
            "cohort": bundle.identity.cohort,
            "dataset": bundle.identity.dataset,
            "flow_index": bundle.identity.flow_index,
            "mask_family": bundle.identity.family,
            "mask_sha256": mask_digest,
            "protocol": bundle.identity.protocol,
            "seed_bundle": bundle.identity.seed_bundle,
            "window_schedule_sha256": bundle.identity.window_schedule_sha256,
            "window_start": bundle.identity.window_start,
        }
    )
    digest = hashlib.sha256(
        _ORACLE_RNG_DOMAIN + len(payload).to_bytes(8, "big") + payload
    ).digest()
    selected = int(eligible[int.from_bytes(digest[:16], "big") % len(eligible)])
    support = np.zeros(50, dtype=np.bool_)
    support[selected] = True
    evaluation = missing & np.logical_not(support)
    support_sha256 = _array_sha256(
        _ORACLE_SUPPORT_HASH_DOMAIN, bundle.identity, support
    )
    evaluation_sha256 = _array_sha256(
        _ORACLE_EVAL_HASH_DOMAIN, bundle.identity, evaluation
    )
    return OraclePartition(
        support=support,
        evaluation=evaluation,
        mask_sha256=mask_digest,
        support_sha256=support_sha256,
        evaluation_sha256=evaluation_sha256,
    )


def factory_oracle_partition(bundle: MaskBundle) -> OraclePartition:
    """Derive Q/E from one official factory bundle without regenerating its mask."""

    return _oracle_partition_from_bundle(bundle, factory_mask_sha256(bundle))


def oracle_partition(bundle: MaskBundle) -> OraclePartition:
    """Select one truth-only diagnostic Q and immutable E without reading truth."""

    bundle = _validated_bundle(bundle)
    digest = _array_sha256(_MASK_HASH_DOMAIN, bundle.identity, bundle.observed)
    if (
        bundle._factory_token is _FACTORY_TOKEN
        and bundle._factory_sha256 != digest
    ):
        raise ValueError("factory-bound mask SHA-256 differs from validated content")
    return _oracle_partition_from_bundle(bundle, digest)


__all__ = [
    "MaskBundle",
    "MaskIdentity",
    "OraclePartition",
    "build_evaluation_mask",
    "build_mixed_training_mask",
    "deployable_target",
    "factory_mask_sha256",
    "factory_oracle_partition",
    "loo_observed",
    "mask_sha256",
    "middle_anchor_index",
    "mixed_training_family",
    "oracle_partition",
]
