"""Independent final-gate artifact audit and frozen 7+2 adjudication."""

from __future__ import annotations

from dataclasses import fields
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments.acil_innovation_v1.batching import (
    EvaluationBatch,
    build_evaluation_batch,
)
from experiments.acil_innovation_v1.preprocessing import FitFallback

from .data_access import permitted_fit_fallback
from .evaluation import CaseEvidence, summarize_case_evidence
from .gate_data import _load_gate_windows
from .gate_identity import GateJobSpec, verify_gate_authority
from .gate_launcher import build_gate_specs
from .gate_runtime import (
    _build_source_spec,
    _verify_source_checkpoint,
    verify_completed_gate_job,
)
from .job_runtime import (
    mask_identity_sha256,
    scientific_array_sha256,
)
from .multibundle import adjudicate_multibundle
from .replay import replay_gate_evidence
from .review import (
    file_sha256,
    load_evidence_npz,
    safe_job_artifact_path,
    verify_registered_evidence,
)
from .uncertainty import analyze_extension_uncertainty


_SCHEMA = "anchorcv-v1:final-gate-review:v1"
_FAMILIES = ("random", "internal_block", "two_burst")
_EXPECTED_NAMES = {
    "attempt_manifest.json",
    "result.json",
    "runtime.json",
    "manifest.json",
    *(f"evidence_{family}.npz" for family in _FAMILIES),
}
_RESULT_FIELDS = {
    "schema",
    "job_id",
    "status",
    "scientific_identity",
    "training_performed",
    "source_binding",
    "neural_parameter_count",
    "prior_checkpoint",
    "cells",
}
_RESULT_SCHEMA = "anchorcv-v1:final-gate-result:v1"
_MANIFEST_SCHEMA = "anchorcv-v1:final-gate-manifest:v1"
_MANIFEST_ADDITIONAL_FIELDS = {
    "schema",
    "job_id",
    "status",
    "result_file",
    "result_sha256",
    "training_performed",
    "source_binding",
}
_CELL_FIELDS = {
    "summary",
    "evidence_file",
    "evidence_file_sha256",
    "evidence_content_sha256",
    "mask_sha256",
    "oracle_q_sha256",
    "oracle_e_sha256",
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot decode {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _scientifically_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _scientifically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _scientifically_equal(a, b) for a, b in zip(left, right)
        )
    if (
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        return math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
        )
    return left == right


def _assert_replay_equal(
    persisted: CaseEvidence,
    replayed: CaseEvidence,
) -> None:
    """Require equal dtype, shape, and bits for every persisted case field."""

    if not isinstance(persisted, CaseEvidence) or not isinstance(
        replayed, CaseEvidence
    ):
        raise TypeError("replay comparison requires CaseEvidence")
    for item in fields(CaseEvidence):
        left = np.asarray(getattr(persisted, item.name))
        right = np.asarray(getattr(replayed, item.name))
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left.view(np.uint8), right.view(np.uint8))
        ):
            raise ValueError(
                f"{item.name}: replay differs bitwise from persisted evidence"
            )


def _audit_persisted_cell(
    *,
    job_directory: Path,
    family: str,
    cell: Mapping[str, object],
    registered: EvaluationBatch,
    fit_fallback: FitFallback,
) -> CaseEvidence:
    """Audit one fixed NPZ against its result row and registered gate batch."""

    if family not in _FAMILIES:
        raise ValueError("gate family is outside the frozen registry")
    if not isinstance(cell, Mapping) or set(cell) != _CELL_FIELDS:
        raise ValueError(f"{family}: cell schema drifted")
    expected_name = f"evidence_{family}.npz"
    evidence_path = safe_job_artifact_path(
        job_directory,
        cell.get("evidence_file"),
        expected_name=expected_name,
    )
    if file_sha256(evidence_path) != cell.get("evidence_file_sha256"):
        raise ValueError(f"{family}: evidence file hash mismatch")
    evidence = load_evidence_npz(evidence_path)
    if (
        scientific_array_sha256(evidence.as_npz_dict())
        != cell.get("evidence_content_sha256")
    ):
        raise ValueError(f"{family}: evidence content hash mismatch")
    if not _scientifically_equal(
        summarize_case_evidence(evidence), cell.get("summary")
    ):
        raise ValueError(f"{family}: summary differs from evidence")
    verify_registered_evidence(
        evidence,
        registered=registered,
        fit_fallback=fit_fallback,
    )
    expected_identities = {
        "mask_sha256": mask_identity_sha256(registered.mask_sha256),
        "oracle_q_sha256": mask_identity_sha256(
            registered.oracle_q_sha256
        ),
        "oracle_e_sha256": mask_identity_sha256(
            registered.oracle_e_sha256
        ),
    }
    if any(
        cell.get(name) != expected
        for name, expected in expected_identities.items()
    ):
        raise ValueError(f"{family}: registered mask identity mismatch")
    return evidence


def _grid_row(
    spec: GateJobSpec,
    *,
    verified: bool,
    result_hash: str | None = None,
    manifest_hash: str | None = None,
) -> dict[str, object]:
    return {
        "dataset": spec.dataset,
        "seed_bundle": spec.seed_bundle,
        "job_id": spec.job_id,
        "verified": verified,
        "result_file_sha256": result_hash,
        "manifest_file_sha256": manifest_hash,
    }


def _preflight_grid(
    specs: tuple[GateJobSpec, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Check the complete six-job envelope before opening any evidence NPZ."""

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    # First reject a missing, partial, symlinked, or checkpoint-bearing grid
    # without parsing a result or evidence artifact from any completed cell.
    structural: list[tuple[GateJobSpec, str | None]] = []
    for spec in specs:
        directory = spec.job_directory
        error: str | None = None
        if directory.is_symlink() or not directory.is_dir():
            error = "gate job directory is missing or is a symlink"
        else:
            names = {entry.name for entry in directory.iterdir()}
            if names != _EXPECTED_NAMES:
                error = (
                    "gate job directory must contain exactly the seven fixed "
                    "artifacts and no checkpoint"
                )
            elif any(
                (directory / name).is_symlink()
                or not (directory / name).is_file()
                for name in _EXPECTED_NAMES
            ):
                error = "gate artifact must be a regular non-symlink file"
        structural.append((spec, error))
    if any(error is not None for _, error in structural):
        for spec, error in structural:
            rows.append(_grid_row(spec, verified=False))
            if error is not None:
                failures.append(
                    {
                        "phase": "preflight",
                        "job_id": spec.job_id,
                        "message": error,
                    }
                )
        return rows, failures

    for spec, _ in structural:
        try:
            if not verify_completed_gate_job(spec):
                raise ValueError("gate job is incomplete")
            result_path = safe_job_artifact_path(
                spec.job_directory,
                "result.json",
                expected_name="result.json",
            )
            manifest_path = safe_job_artifact_path(
                spec.job_directory,
                "manifest.json",
                expected_name="manifest.json",
            )
            rows.append(
                _grid_row(
                    spec,
                    verified=True,
                    result_hash=file_sha256(result_path),
                    manifest_hash=file_sha256(manifest_path),
                )
            )
        except Exception as exc:
            rows.append(_grid_row(spec, verified=False))
            failures.append(
                {
                    "phase": "preflight",
                    "job_id": spec.job_id,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows, failures


def _audit_gate_job(
    spec: GateJobSpec,
    *,
    gate_windows,
    prototype_output_root: Path,
    extension_output_root: Path,
) -> dict[str, CaseEvidence]:
    """Independently reconstruct one result, registry, source, and replay."""

    if not verify_completed_gate_job(spec):
        raise ValueError("gate job became incomplete during review")
    directory = spec.job_directory
    if {entry.name for entry in directory.iterdir()} != _EXPECTED_NAMES:
        raise ValueError("gate job directory changed after preflight")
    result_path = safe_job_artifact_path(
        directory, "result.json", expected_name="result.json"
    )
    manifest_path = safe_job_artifact_path(
        directory, "manifest.json", expected_name="manifest.json"
    )
    result = _json_object(result_path, label="gate result")
    manifest = _json_object(manifest_path, label="gate manifest")
    if set(result) != _RESULT_FIELDS:
        raise ValueError("gate result schema drifted")
    if set(manifest) != (
        set(spec.scientific_identity) | _MANIFEST_ADDITIONAL_FIELDS
    ):
        raise ValueError("gate manifest schema drifted")
    binding = dict(
        spec.authority.checkpoint_binding(spec.dataset, spec.seed_bundle)
    )
    identity = spec.scientific_identity
    if (
        result.get("schema") != _RESULT_SCHEMA
        or manifest.get("schema") != _MANIFEST_SCHEMA
        or result.get("job_id") != spec.job_id
        or manifest.get("job_id") != spec.job_id
        or result.get("status") != "succeeded"
        or manifest.get("status") != "succeeded"
        or result.get("scientific_identity") != identity
        or {field: manifest.get(field) for field in identity} != identity
        or result.get("training_performed") is not False
        or manifest.get("training_performed") is not False
        or result.get("source_binding") != binding
        or manifest.get("source_binding") != binding
        or manifest.get("result_file") != "result.json"
        or manifest.get("result_sha256") != file_sha256(result_path)
    ):
        raise ValueError("gate result/manifest identity or hash drifted")
    prior = result.get("prior_checkpoint")
    cells = result.get("cells")
    if (
        not isinstance(prior, Mapping)
        or set(prior) != {"file_sha256"}
        or prior.get("file_sha256")
        != binding["acil_checkpoint_file_sha256"]
        or not isinstance(cells, Mapping)
        or set(cells) != set(_FAMILIES)
    ):
        raise ValueError("gate nested result schema or source binding drifted")

    registered_batches = {
        family: build_evaluation_batch(
            gate_windows,
            seed_bundle=spec.seed_bundle,
            family=family,
        )
        for family in _FAMILIES
    }
    fallback = permitted_fit_fallback(spec.dataset)
    persisted = {
        family: _audit_persisted_cell(
            job_directory=directory,
            family=family,
            cell=cells[family],  # type: ignore[arg-type]
            registered=registered_batches[family],
            fit_fallback=fallback,
        )
        for family in _FAMILIES
    }
    source_spec, reconstructed_binding = _build_source_spec(
        spec,
        prototype_output_root=prototype_output_root,
        extension_output_root=extension_output_root,
    )
    if reconstructed_binding != binding:
        raise ValueError("source binding changed during review")
    neural_state, parameter_count = _verify_source_checkpoint(
        source_spec, binding
    )
    if result.get("neural_parameter_count") != parameter_count:
        raise ValueError("gate neural parameter count differs from source")
    replayed = replay_gate_evidence(
        dataset=spec.dataset,
        seed_bundle_id=spec.seed_bundle,
        neural_state=neural_state,
        expected_neural_tensor_sha256=binding[
            "neural_checkpoint_tensor_sha256"
        ],
        expected_prior_file_sha256=binding[
            "acil_checkpoint_file_sha256"
        ],
        registered_batches=registered_batches,
        fit_fallback=fallback,
    )
    if not isinstance(replayed, Mapping) or set(replayed) != set(_FAMILIES):
        raise ValueError("gate replay did not return the exact mask grid")
    for family in _FAMILIES:
        _assert_replay_equal(persisted[family], replayed[family])
    return persisted


def _invalid_report(
    *,
    grid: list[dict[str, object]],
    failures: list[dict[str, object]],
    authority=None,
) -> dict[str, object]:
    payload = authority.payload if authority is not None else {}
    return {
        "schema": _SCHEMA,
        "status": "invalid",
        "verdict": "revise",
        "job_count": 6,
        "grid": grid,
        "failures": failures,
        "uncertainty": None,
        "multibundle_adjudication": None,
        "integrity": {
            "valid": False,
            "source_tree_sha256": payload.get("source_tree_sha256"),
            "config_sha256": payload.get("config_sha256"),
            "method_freeze_sha256": (
                authority.method_freeze_sha256
                if authority is not None
                else None
            ),
            "git_available": False,
            "git_commit": None,
            "test_access": False,
        },
    }


def review_final_gate(
    *,
    output_root: Path,
    authority_record: Path,
    extension_report: Path,
    freeze_record: Path,
    prototype_output_root: Path,
    extension_output_root: Path,
) -> dict[str, object]:
    """Re-audit all six final jobs, replay inference, and apply frozen gates."""

    try:
        authority = verify_gate_authority(
            authority_record=authority_record,
            extension_report=extension_report,
            freeze_record=freeze_record,
            prototype_output_root=prototype_output_root,
            extension_output_root=extension_output_root,
        )
        specs = build_gate_specs(Path(output_root), authority)
        if len(specs) != 6:
            raise ValueError("final gate must contain exactly six jobs")
    except Exception as exc:
        return _invalid_report(
            grid=[],
            failures=[
                {
                    "phase": "authority",
                    "job_id": None,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
        )

    grid, failures = _preflight_grid(specs)
    if failures or len(grid) != 6 or not all(
        row.get("verified") is True for row in grid
    ):
        return _invalid_report(
            grid=grid, failures=failures, authority=authority
        )

    evidence_grid: dict[tuple[int, str, str], CaseEvidence] = {}
    summaries: dict[tuple[int, str, str], Mapping[str, object]] = {}
    gate_cache: dict[str, object] = {}
    for spec in specs:
        try:
            if spec.dataset not in gate_cache:
                gate_cache[spec.dataset] = _load_gate_windows(
                    authority, spec.dataset
                )
            evidence = _audit_gate_job(
                spec,
                gate_windows=gate_cache[spec.dataset],
                prototype_output_root=Path(prototype_output_root),
                extension_output_root=Path(extension_output_root),
            )
            for family in _FAMILIES:
                key = (spec.seed_bundle, spec.dataset, family)
                evidence_grid[key] = evidence[family]
                summaries[key] = summarize_case_evidence(evidence[family])
        except Exception as exc:
            failures.append(
                {
                    "phase": "artifact_replay",
                    "job_id": spec.job_id,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    if failures or len(evidence_grid) != 18:
        return _invalid_report(
            grid=grid, failures=failures, authority=authority
        )
    try:
        uncertainty = analyze_extension_uncertainty(evidence_grid)
        adjudication = adjudicate_multibundle(summaries, uncertainty)
        verdict = adjudication.get("verdict")
        adjudication_integrity = adjudication.get("integrity")
        if (
            verdict not in {"kill", "proceed"}
            or not isinstance(adjudication_integrity, Mapping)
            or adjudication_integrity.get("valid") is not True
        ):
            raise ValueError(
                "complete gate evidence produced an invalid adjudication"
            )
    except Exception as exc:
        return _invalid_report(
            grid=grid,
            failures=[
                {
                    "phase": "adjudication",
                    "job_id": None,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
            authority=authority,
        )

    payload = authority.payload
    return {
        "schema": _SCHEMA,
        "status": "verified_complete",
        "verdict": verdict,
        "job_count": 6,
        "grid": grid,
        "failures": [],
        "uncertainty": uncertainty,
        "multibundle_adjudication": adjudication,
        "integrity": {
            "valid": True,
            "source_tree_sha256": payload["source_tree_sha256"],
            "config_sha256": payload["config_sha256"],
            "method_freeze_sha256": authority.method_freeze_sha256,
            "git_available": False,
            "git_commit": None,
            "test_access": False,
        },
    }


__all__ = [
    "_assert_replay_equal",
    "_audit_persisted_cell",
    "review_final_gate",
]
