"""Load and validate the sole SC-ACIL v1 protocol."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Mapping


_CONFIG = Path(__file__).with_name("configs") / "protocol_v1.json"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


@lru_cache(maxsize=1)
def load_protocol() -> Mapping[str, object]:
    try:
        value = json.loads(_CONFIG.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load the fixed SC-ACIL protocol") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("SC-ACIL protocol schema drifted")
    protocol = value.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("id") != "sc-acil-v1":
        raise ValueError("SC-ACIL protocol identity drifted")
    if protocol.get("git_available") is not False or protocol.get("git_commit") is not None:
        raise ValueError("invalid Git identity in SC-ACIL protocol")
    boundary = value.get("evidence_boundary")
    if not isinstance(boundary, dict) or any(
        boundary.get(field) is not False
        for field in (
            "sealed_test_access",
            "legacy_test_cache_access",
            "whole_raw_csv_access",
            "whole_raw_csv_hashing",
        )
    ):
        raise ValueError("SC-ACIL evidence boundary drifted")
    return value


def protocol_sha256() -> str:
    return hashlib.sha256(
        b"sc-acil-v1:protocol:v1\x00" + canonical_json_bytes(load_protocol())
    ).hexdigest()


__all__ = ["canonical_json_bytes", "load_protocol", "protocol_sha256"]

