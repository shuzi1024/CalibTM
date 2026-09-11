"""Complete content-addressed planned-job manifest for protocol v1."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .config import canonical_json_bytes, load_protocol_config
from .jobs import planned_jobs
from .provenance import build_provenance


_STAGE_ORDER = (
    "stage0_acil_tune",
    "stage_h",
    "stage_i",
    "full_tune",
)
_MANIFEST_HASH_DOMAIN = b"acil-innovation-v1:planned-manifest:v1\x00"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_jobs() -> tuple[dict[str, object], ...]:
    return tuple(
        job.to_json()
        for stage in _STAGE_ORDER
        for job in planned_jobs(stage)
    )


def _manifest_sha256(payload: Mapping[str, Any]) -> str:
    if "manifest_sha256" in payload:
        raise ValueError("manifest payload cannot hash itself")
    encoded = canonical_json_bytes(payload)
    digest = hashlib.sha256(_MANIFEST_HASH_DOMAIN)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _require_provenance(provenance: Mapping[str, Any]) -> None:
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    if provenance.get("protocol") != "acil-innovation-v1":
        raise ValueError("provenance protocol identity drifted")
    if provenance.get("git_available") is not False or provenance.get("git_commit") is not None:
        raise ValueError("provenance Git metadata drifted")
    for name in ("config_sha256", "provenance_sha256"):
        if not _is_sha256(provenance.get(name)):
            raise ValueError(f"provenance {name} is not a SHA-256")


def _base_manifest(provenance: Mapping[str, Any]) -> dict[str, Any]:
    _require_provenance(provenance)
    config = load_protocol_config()
    if config["protocol"]["status"] != "draft_protocol":
        raise ValueError("manifest must be built before mutating draft config status")
    if tuple(config["freeze"]["active_manifest_stages"]) != _STAGE_ORDER:
        raise ValueError("active manifest stage registry drifted")
    return {
        "config_sha256": provenance["config_sha256"],
        "config_status_at_build": "draft_protocol",
        "gate_registry": {stage: config["gates"][stage] for stage in _STAGE_ORDER},
        "jobs": list(_expected_jobs()),
        "protocol": "acil-innovation-v1",
        "provenance_sha256": provenance["provenance_sha256"],
        "schema_version": 1,
        "stage_order": list(_STAGE_ORDER),
        "version": "v1",
    }


def validate_manifest(
    manifest: Mapping[str, Any], *, expected_provenance: Mapping[str, Any]
) -> None:
    """Reject any missing, extra, duplicate, reordered, or rebound planned job."""

    _require_provenance(expected_provenance)
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    expected_keys = {
        "config_sha256",
        "config_status_at_build",
        "gate_registry",
        "jobs",
        "protocol",
        "provenance_sha256",
        "schema_version",
        "stage_order",
        "version",
    }
    if set(manifest) != expected_keys:
        raise ValueError("manifest has missing or extra top-level fields")
    if manifest["protocol"] != "acil-innovation-v1" or manifest["version"] != "v1":
        raise ValueError("manifest protocol/version identity drifted")
    if manifest["config_status_at_build"] != "draft_protocol":
        raise ValueError("manifest was not built from draft_protocol")
    if manifest["config_sha256"] != expected_provenance["config_sha256"]:
        raise ValueError("manifest config provenance mismatch")
    if manifest["provenance_sha256"] != expected_provenance["provenance_sha256"]:
        raise ValueError("manifest provenance mismatch")
    if tuple(manifest["stage_order"]) != _STAGE_ORDER:
        raise ValueError("manifest stage order drifted")

    supplied = tuple(manifest["jobs"])
    expected = _expected_jobs()
    if len(supplied) < len(expected):
        raise ValueError("manifest has a missing planned job")
    if len(supplied) > len(expected):
        raise ValueError("manifest has an extra planned job")
    identifiers = [item.get("job_id") for item in supplied if isinstance(item, Mapping)]
    if len(identifiers) != len(supplied):
        raise ValueError("manifest contains an invalid planned job")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("manifest contains a duplicate planned job")
    if supplied != expected:
        expected_ids = [item["job_id"] for item in expected]
        if set(identifiers) == set(expected_ids):
            raise ValueError("manifest planned jobs are reordered")
        raise ValueError("manifest planned job identity mismatch")
    expected_gates = {
        stage: load_protocol_config()["gates"][stage] for stage in _STAGE_ORDER
    }
    if manifest["gate_registry"] != expected_gates:
        raise ValueError("manifest gate registry drifted")


def active_manifest_stages() -> tuple[str, ...]:
    """Return the sole ordered v1 stage registry included in the manifest."""

    return _STAGE_ORDER


def _build_manifest(provenance: Mapping[str, Any]) -> dict[str, Any]:
    payload = _base_manifest(provenance)
    validate_manifest(payload, expected_provenance=provenance)
    result = dict(payload)
    result["manifest_sha256"] = _manifest_sha256(payload)
    return result


def build_manifest() -> dict[str, Any]:
    """Build the sole v1 manifest from fixed provenance and job registries."""

    return _build_manifest(build_provenance())


__all__ = ["active_manifest_stages", "build_manifest", "validate_manifest"]
