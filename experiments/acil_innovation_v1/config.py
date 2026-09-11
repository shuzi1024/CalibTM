"""Fixed, source-pinned protocol configuration for ACIL-Innovation v1."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any


_CONFIG_PATH = Path(__file__).with_name("configs") / "protocol_v1.json"
_HASH_DOMAIN = b"acil-innovation-v1:protocol-config:v1\x00"
_EXPECTED_PROTOCOL_CONFIG_SHA256 = (
    "bedb60c85e9808678531bbb5e2a9689bcfd7f67f700c95815d8eb4898ae67f79"
)
_TOP_LEVEL_KEYS = {
    "adjudication",
    "bootstrap",
    "capabilities",
    "datasets",
    "deployable_partition",
    "evidence_boundary",
    "freeze",
    "gates",
    "innovation",
    "masks",
    "metrics",
    "model_boundary",
    "oracle_partition",
    "protocol",
    "rng",
    "schema_version",
    "seeds",
    "stages",
    "training",
}
_FORBIDDEN_KEYS = {
    "data_path",
    "raw_csv",
    "raw_path",
    "split_override",
    "test_path",
    "test_split",
}
_FREEZE_KEYS = {
    "active_manifest_stages",
    "allow_latest",
    "future_formal_protocol",
    "git_available",
    "git_commit",
    "publication_contract",
    "revision_limit",
    "source_tree_sha256_required",
}
_PUBLICATION_CONTRACT = "protocol_controlled_single_publisher_no_replace"

# CUDA determinism is an execution contract, not a tunable scientific option.
# Queue workers set it explicitly and direct workers fail closed if it differs.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite JSON with a unique byte representation."""

    def reject_nonfinite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("protocol JSON cannot contain a non-finite number")
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError("protocol JSON object keys must be strings")
                reject_nonfinite(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                reject_nonfinite(child)

    reject_nonfinite(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate protocol JSON key {key!r}")
        result[key] = value
    return result


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key.lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) != _TOP_LEVEL_KEYS:
        raise ValueError("protocol config has nonexact top-level keys")
    if config.get("schema_version") != 1:
        raise ValueError("protocol schema version must be exactly 1")
    if config.get("protocol") != {
        "branch_sealed": True,
        "id": "acil-innovation-v1",
        "status": "draft_protocol",
        "window_length": 50,
    }:
        raise ValueError("protocol identity drifted")
    freeze = config.get("freeze")
    if not isinstance(freeze, dict):
        raise ValueError("freeze registry must be an object")
    if set(freeze) != _FREEZE_KEYS:
        raise ValueError(
            "freeze registry must contain the exact publication contract fields"
        )
    if freeze.get("git_available") is not False or freeze.get("git_commit") is not None:
        raise ValueError("invalid Git metadata: this checkout has no commit authority")
    if freeze.get("allow_latest") is not False:
        raise ValueError("freeze must forbid latest artifact selection")
    if freeze.get("publication_contract") != _PUBLICATION_CONTRACT:
        raise ValueError("freeze publication contract drifted")

    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or tuple(datasets) != ("abilene", "geant"):
        raise ValueError("dataset registry must be exactly abilene then geant")
    masks = config.get("masks")
    if not isinstance(masks, dict):
        raise ValueError("mask registry must be an object")
    if masks.get("observed_count") != 3 or masks.get("k5_registered") is not False:
        raise ValueError("v1 registers exactly K=3 and must not register K=5")
    if masks.get("families") != ["random", "internal_block", "two_burst"]:
        raise ValueError("mask-family order drifted")

    mixed = config.get("training", {}).get("mixed_masks")
    if mixed != {
        "family_formula": "families[(epoch*512+epoch_order_position+bundle-1)%3]",
        "one_checkpoint_per_dataset_seed_method": True,
        "training_job_has_mask_axis": False,
        "window_level_family_assignment": True,
    }:
        raise ValueError("mixed-mask training semantics drifted")
    if config.get("training", {}).get("checkpoint") != {
        "metric": "source_dev_three_mask_raw_ratio_of_sums_nmae",
        "tie_break": "earlier_epoch",
    }:
        raise ValueError("source-dev checkpoint rule drifted")

    oracle = config.get("oracle_partition")
    deployable = config.get("deployable_partition")
    if not isinstance(oracle, dict) or oracle.get("support_count_per_flow") != 1:
        raise ValueError("oracle must register exactly one support per flow")
    if oracle.get("evaluation_target_count_per_flow") != 46:
        raise ValueError("oracle evaluation target must contain 46 timestamps")
    if deployable != {
        "oracle_support_count": 0,
        "evaluation_set": "unobserved_complement",
        "target_count_per_flow": 47,
    }:
        raise ValueError("deployable target semantics drifted")

    if _FORBIDDEN_KEYS.intersection(_walk_keys(config)):
        raise ValueError("protocol config exposes a forbidden path or split override")
    canonical_json_bytes(config)
    return config


def _read_fixed_config() -> dict[str, Any]:
    if _CONFIG_PATH.is_symlink() or not _CONFIG_PATH.is_file():
        raise ValueError("fixed protocol config must be a regular non-symlink file")
    try:
        config = json.loads(
            _CONFIG_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value!r}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("fixed protocol config cannot be decoded") from exc
    config = _validate_config(config)
    actual = hashlib.sha256(_HASH_DOMAIN + canonical_json_bytes(config)).hexdigest()
    if actual != _EXPECTED_PROTOCOL_CONFIG_SHA256:
        raise ValueError("fixed protocol config semantic SHA-256 drifted")
    return config


@lru_cache(maxsize=1)
def _cached_fixed_config() -> dict[str, Any]:
    """Cache the sole verified snapshot; callers never receive this object."""

    return _read_fixed_config()


def load_protocol_config() -> dict[str, Any]:
    """Return an owned copy of the sole source-pinned protocol config."""

    return deepcopy(_cached_fixed_config())


def protocol_config_sha256() -> str:
    """Return the verified semantic identity of the active config."""

    _cached_fixed_config()
    return _EXPECTED_PROTOCOL_CONFIG_SHA256


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "canonical_json_bytes",
    "load_protocol_config",
    "protocol_config_sha256",
]
