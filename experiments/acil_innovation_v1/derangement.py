"""Fixed Stage-I window-flow innovation permutation diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor

from .config import canonical_json_bytes, load_protocol_config
from .registries import (
    dataset_spec,
    ordered_window_starts,
    seed_bundle as registered_seed_bundle,
    window_schedule_sha256,
)


_PERMUTATION_HASH_DOMAIN = (
    b"acil-innovation-v1:stage-i-derangement-permutation-hash:v1\x00"
)
_HASH_SEMANTICS = "output_flat_equals_input_flat_at_permutation"
_IDENTITY_FIELDS = (
    "protocol",
    "dataset",
    "cohort",
    "window_schedule_sha256",
    "seed_bundle",
    "mask_family",
    "evaluation_mask_seed",
    "registered_window_count",
    "flow_count",
)


@dataclass(frozen=True, slots=True)
class DerangementIdentity:
    protocol: str
    dataset: str
    cohort: str
    window_schedule_sha256: str
    seed_bundle: int
    mask_family: str
    evaluation_mask_seed: int
    registered_window_count: int
    flow_count: int


def _identity_mapping(identity: DerangementIdentity) -> dict[str, object]:
    return {
        "protocol": identity.protocol,
        "dataset": identity.dataset,
        "cohort": identity.cohort,
        "window_schedule_sha256": identity.window_schedule_sha256,
        "seed_bundle": identity.seed_bundle,
        "mask_family": identity.mask_family,
        "evaluation_mask_seed": identity.evaluation_mask_seed,
        "registered_window_count": identity.registered_window_count,
        "flow_count": identity.flow_count,
    }


def _permutation_sha256(
    identity: DerangementIdentity, permutation: NDArray[np.int64]
) -> str:
    metadata = canonical_json_bytes(
        {
            "dtype": permutation.dtype.str,
            "identity": _identity_mapping(identity),
            "semantics": _HASH_SEMANTICS,
            "shape": list(permutation.shape),
        }
    )
    raw = permutation.tobytes(order="C")
    digest = hashlib.sha256(_PERMUTATION_HASH_DOMAIN)
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _owned_permutation(
    value: ArrayLike, expected_count: int
) -> NDArray[np.int64]:
    source = np.asarray(value)
    if source.dtype != np.dtype("<i8") or source.shape != (expected_count,):
        raise ValueError("permutation must be a little-endian int64 vector")
    contiguous = np.ascontiguousarray(source, dtype="<i8")
    owned = np.frombuffer(contiguous.tobytes(order="C"), dtype="<i8")
    expected = np.arange(expected_count, dtype="<i8")
    if not np.array_equal(np.sort(owned), expected):
        raise ValueError("permutation must contain every registered scalar once")
    return owned


def _is_lower_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class DerangementPlan:
    identity: DerangementIdentity
    permutation: NDArray[np.int64]
    permutation_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DerangementIdentity):
            raise TypeError("identity must be DerangementIdentity")
        count = (
            self.identity.registered_window_count * self.identity.flow_count
        )
        if count <= 0:
            raise ValueError("derangement identity must have a positive scalar count")
        permutation = _owned_permutation(self.permutation, count)
        if not _is_lower_sha256(self.permutation_sha256):
            raise ValueError("permutation SHA-256 is invalid")
        if _permutation_sha256(self.identity, permutation) != self.permutation_sha256:
            raise ValueError("permutation SHA-256 does not bind identity and content")
        object.__setattr__(self, "permutation", permutation)

    def apply(self, innovations: Tensor) -> Tensor:
        """Jointly permute scalar [W,F,1] innovations, leaving no side operands."""

        if not isinstance(innovations, Tensor):
            raise TypeError("innovations must be a torch tensor")
        expected_shape = (
            self.identity.registered_window_count,
            self.identity.flow_count,
            1,
        )
        if tuple(innovations.shape) != expected_shape:
            raise ValueError("innovations shape must match registered [W,F,1]")
        if innovations.device.type != "cpu":
            raise ValueError("derangement apply is a CPU-only diagnostic")
        if not innovations.is_floating_point():
            raise TypeError("innovations tensor must be floating point")
        if not torch.isfinite(innovations).all().item():
            raise ValueError("innovations tensor must be finite")
        indices = torch.from_numpy(
            np.array(self.permutation, dtype="<i8", order="C", copy=True)
        )
        return innovations.reshape(-1).index_select(0, indices).reshape(
            expected_shape
        )


def _require_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _require_exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


@lru_cache(maxsize=1)
def _contract() -> tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    config = load_protocol_config()
    stage = config["stages"]["stage_i"]
    item = stage["derangement"]
    if item["identity_fields"] != list(_IDENTITY_FIELDS):
        raise RuntimeError("Stage-I derangement identity fields drifted")
    expected = {
        "cohort": "tune",
        "domain": "acil-innovation-v1:stage-i-derangement:v1",
        "fixed_fields": ["query_geometry", "middle_indicator", "truth", "targets"],
        "joint_permutation_axis": (
            "all_registered_window_flow_scalar_loo_innovations"
        ),
        "method_outcome_in_identity": False,
        "permutation": "single_fixed_permutation_without_replacement",
        "retraining": False,
        "rng": "PCG64DXSM",
        "scope": "each_dataset_seed_bundle_mask_family",
        "seed_derivation": (
            "sha256_domain_nul_canonical_identity_first_128_bits_big_endian"
        ),
        "signal": "scalar_loo_innovation",
    }
    for key, value in expected.items():
        if item.get(key) != value:
            raise RuntimeError(f"Stage-I derangement field {key!r} drifted")
    return (
        str(item["domain"]),
        str(item["cohort"]),
        tuple(config["masks"]["families"]),
        tuple(stage["datasets"]),
        tuple(int(value) for value in stage["seed_bundles"]),
    )


@lru_cache(maxsize=18)
def _cached_plan(
    dataset: str, seed_bundle: int, mask_family: str
) -> DerangementPlan:
    domain, cohort, _, _, _ = _contract()
    starts = ordered_window_starts(dataset, cohort)
    specification = dataset_spec(dataset)
    seeds = registered_seed_bundle(seed_bundle)
    identity = DerangementIdentity(
        protocol="acil-innovation-v1",
        dataset=dataset,
        cohort=cohort,
        window_schedule_sha256=window_schedule_sha256(dataset, cohort),
        seed_bundle=seed_bundle,
        mask_family=mask_family,
        evaluation_mask_seed=seeds.evaluation_mask,
        registered_window_count=len(starts),
        flow_count=specification.flows,
    )
    canonical_identity = canonical_json_bytes(_identity_mapping(identity))
    seed_digest = hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_identity
    ).digest()
    rng_seed = int.from_bytes(seed_digest[:16], "big", signed=False)
    count = identity.registered_window_count * identity.flow_count
    permutation = np.ascontiguousarray(
        np.random.Generator(np.random.PCG64DXSM(rng_seed)).permutation(count),
        dtype="<i8",
    )
    return DerangementPlan(
        identity=identity,
        permutation=permutation,
        permutation_sha256=_permutation_sha256(identity, permutation),
    )


def build_derangement(
    dataset: str, seed_bundle: int, mask_family: str
) -> DerangementPlan:
    """Build the sole fixed Stage-I tune permutation for one registered cell."""

    dataset = _require_name(dataset, "dataset")
    seed_bundle = _require_exact_int(seed_bundle, "seed bundle")
    mask_family = _require_name(mask_family, "mask family")
    _, _, families, datasets, seed_bundles = _contract()
    if dataset not in datasets:
        raise ValueError(f"dataset {dataset!r} is not registered for Stage I")
    if seed_bundle not in seed_bundles:
        raise ValueError(f"seed bundle {seed_bundle!r} is not registered for Stage I")
    if mask_family not in families:
        raise ValueError(f"mask family {mask_family!r} is not registered")
    return _cached_plan(dataset, seed_bundle, mask_family)


__all__ = [
    "DerangementIdentity",
    "DerangementPlan",
    "build_derangement",
]
