"""Independent artifact/hash audit and frozen prototype adjudication."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from safetensors.torch import load_file as load_safetensors
import torch

from experiments.acil_innovation_v1.batching import (
    EvaluationBatch,
    build_evaluation_batch,
)
from experiments.acil_innovation_v1.preprocessing import (
    FitFallback,
    observation_statistics,
)

from .adjudicate import adjudicate_prototype
from .data_access import load_permitted_windows, permitted_fit_fallback
from .evaluation import CaseEvidence, summarize_case_evidence
from .freeze import verify_freeze_record
from .job_runtime import (
    mask_identity_sha256,
    scientific_array_sha256,
    scientific_tensor_sha256,
    verify_completed_job,
)
from .job import JobSpec
from .launcher import build_stage_specs
from .multibundle import adjudicate_multibundle
from .protocol import load_protocol
from .replay import replay_and_verify_tune_evidence
from .training_runtime import build_neural_expert, model_parameter_count
from .uncertainty import analyze_extension_uncertainty


_RESULT_FIELDS = {
    "job_id",
    "status",
    "scientific_identity",
    "neural_parameter_count",
    "neural_checkpoint",
    "prior_checkpoint",
    "training",
    "cells",
}
_NEURAL_CHECKPOINT_FIELDS = {
    "file",
    "file_sha256",
    "tensor_sha256",
    "best_epoch",
    "best_source_dev_nmae",
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
_TRAINING_FIELDS = {
    "epochs_completed",
    "optimizer_updates",
    "best_epoch",
    "best_source_dev_nmae",
    "epoch_records",
}
_EPOCH_RECORD_FIELDS = {
    "epoch",
    "physical_batches",
    "optimizer_updates",
    "mean_loss",
    "source_dev_absolute_error_sum",
    "source_dev_absolute_truth_sum",
    "source_dev_nmae",
    "selected_as_best",
}
_K3_SCALE_BACKEND_RTOL = 2e-7


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_job_artifact_path(
    job_directory: Path,
    reported_name: object,
    *,
    expected_name: str,
) -> Path:
    """Resolve one fixed artifact without allowing symlinks or path escape."""

    directory = Path(job_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("job artifact directory must be a regular directory")
    if not isinstance(reported_name, str) or reported_name != expected_name:
        raise ValueError("reported artifact filename differs from the fixed filename")
    path = directory / expected_name
    if path.is_symlink():
        raise ValueError("job artifact symlink is forbidden")
    if not path.is_file():
        raise ValueError("job artifact must be a regular file")
    try:
        resolved_directory = directory.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("job artifact cannot be resolved") from exc
    if resolved.parent != resolved_directory:
        raise ValueError("job artifact escaped its fixed directory")
    return resolved


def load_evidence_npz(path: Path) -> CaseEvidence:
    expected = {item.name for item in fields(CaseEvidence)}
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("evidence NPZ fields drifted")
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("object evidence arrays are forbidden")
    return CaseEvidence(**arrays)


def verify_registered_evidence(
    evidence: CaseEvidence,
    *,
    registered: EvaluationBatch,
    fit_fallback: FitFallback,
) -> None:
    """Bind persisted case rows to the exact registered batch and K3 operands."""

    if not isinstance(evidence, CaseEvidence):
        raise TypeError("evidence must be CaseEvidence")
    if not isinstance(registered, EvaluationBatch):
        raise TypeError("registered must be EvaluationBatch")
    truth = registered.truth
    observed = registered.observed
    windows, flows, times = (int(value) for value in truth.shape)
    cases = windows * flows
    if evidence.case_count != cases:
        raise ValueError("evidence case count differs from the registered batch")
    expected_window = np.repeat(np.arange(windows, dtype=np.int64), flows)
    expected_flow = np.tile(np.arange(flows, dtype=np.int64), windows)
    if not np.array_equal(evidence.window_index, expected_window):
        raise ValueError("evidence window order differs from the registered batch")
    if not np.array_equal(evidence.flow_index, expected_flow):
        raise ValueError("evidence flow order differs from the registered batch")

    positions = torch.arange(
        times, dtype=torch.long, device=observed.device
    ).reshape(1, 1, times)
    positions = positions.expand_as(observed)
    sorted_observed = torch.where(
        observed, positions, torch.full_like(positions, times)
    ).sort(dim=-1).values
    expected_middle = (
        sorted_observed[..., 1].detach().cpu().numpy().reshape(cases)
    )
    if not np.array_equal(evidence.middle_anchor_index, expected_middle):
        raise ValueError("evidence middle-anchor index differs from the registered mask")

    target = ~observed
    expected_count = (
        target.sum(dim=-1).detach().cpu().numpy().astype(np.int64).reshape(cases)
    )
    if not np.array_equal(evidence.target_count, expected_count):
        raise ValueError("evidence target count differs from the registered mask")
    expected_truth_sum = (
        torch.where(target, truth.abs(), torch.zeros_like(truth))
        .double()
        .sum(dim=-1)
        .cpu()
        .numpy()
        .reshape(cases)
    )
    if not np.allclose(
        evidence.truth_sum, expected_truth_sum, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("evidence truth sum differs from the registered target")
    expected_scale = (
        observation_statistics(
            registered.model_input, observed, fit_fallback
        )
        .std.detach()
        .float()
        .cpu()
        .numpy()
        .reshape(cases)
        .astype(np.float64)
    )
    # CUDA and CPU fp32 reductions differed by at most 1.69e-7 in the
    # registered prototype artifacts; reject changes beyond that envelope.
    if not np.allclose(
        evidence.k3_scale,
        expected_scale,
        rtol=_K3_SCALE_BACKEND_RTOL,
        atol=0.0,
    ):
        raise ValueError("evidence K3 scale differs from registered observations")


def verify_neural_state(
    dataset: str,
    state: Mapping[str, torch.Tensor],
    *,
    reported_parameter_count: object,
) -> None:
    """Strictly bind a checkpoint to the frozen dataset-specific architecture."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("neural checkpoint state must be a nonempty mapping")
    expected_model = build_neural_expert(dataset)
    expected = expected_model.state_dict()
    if set(state) != set(expected):
        raise ValueError("neural checkpoint tensor schema drifted")
    for name in expected:
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
        ):
            raise ValueError("neural checkpoint tensor schema drifted")
    expected_count = model_parameter_count(expected_model)
    if (
        isinstance(reported_parameter_count, bool)
        or not isinstance(reported_parameter_count, int)
        or reported_parameter_count != expected_count
    ):
        raise ValueError("neural parameter count differs from the frozen architecture")
    try:
        expected_model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise ValueError("neural checkpoint tensor schema drifted") from exc


def verify_training_record(
    training: object,
    *,
    checkpoint: object,
) -> None:
    """Bind the complete fixed training budget and source-dev selection."""

    if not isinstance(training, Mapping) or set(training) != _TRAINING_FIELDS:
        raise ValueError("training record schema drifted")
    if (
        training.get("epochs_completed") != 20
        or isinstance(training.get("epochs_completed"), bool)
        or training.get("optimizer_updates") != 320
        or isinstance(training.get("optimizer_updates"), bool)
    ):
        raise ValueError("training budget drifted")
    best_epoch = training.get("best_epoch")
    best_metric = training.get("best_source_dev_nmae")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 0 <= best_epoch < 20
        or isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not math.isfinite(float(best_metric))
        or float(best_metric) < 0.0
    ):
        raise ValueError("training best source-dev selection is invalid")
    records = training.get("epoch_records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("training epoch records are incomplete")

    parsed_metrics: list[float] = []
    selected: list[int] = []
    for expected_epoch, raw in enumerate(records):
        if not isinstance(raw, Mapping) or set(raw) != _EPOCH_RECORD_FIELDS:
            raise ValueError("training epoch record schema drifted")
        if (
            raw.get("epoch") != expected_epoch
            or isinstance(raw.get("epoch"), bool)
            or raw.get("physical_batches") != 64
            or isinstance(raw.get("physical_batches"), bool)
            or raw.get("optimizer_updates") != 16
            or isinstance(raw.get("optimizer_updates"), bool)
        ):
            raise ValueError("training epoch budget or order drifted")
        numeric: dict[str, float] = {}
        for name in (
            "mean_loss",
            "source_dev_absolute_error_sum",
            "source_dev_absolute_truth_sum",
            "source_dev_nmae",
        ):
            value = raw.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("training epoch source-dev values are invalid")
            numeric[name] = float(value)
        if (
            numeric["mean_loss"] < 0.0
            or numeric["source_dev_absolute_error_sum"] < 0.0
            or numeric["source_dev_absolute_truth_sum"] <= 0.0
            or numeric["source_dev_nmae"] < 0.0
            or not math.isclose(
                numeric["source_dev_nmae"],
                numeric["source_dev_absolute_error_sum"]
                / numeric["source_dev_absolute_truth_sum"],
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("training epoch source-dev ratio is invalid")
        marker = raw.get("selected_as_best")
        if not isinstance(marker, bool):
            raise ValueError("training selected epoch marker must be boolean")
        if marker:
            selected.append(expected_epoch)
        parsed_metrics.append(numeric["source_dev_nmae"])

    expected_best = min(range(20), key=lambda epoch: (parsed_metrics[epoch], epoch))
    if (
        selected != [best_epoch]
        or best_epoch != expected_best
        or not math.isclose(
            float(best_metric),
            parsed_metrics[best_epoch],
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("training best epoch differs from source-dev argmin")
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != _NEURAL_CHECKPOINT_FIELDS
        or checkpoint.get("best_epoch") != best_epoch
        or isinstance(checkpoint.get("best_epoch"), bool)
        or isinstance(checkpoint.get("best_source_dev_nmae"), bool)
        or not isinstance(
            checkpoint.get("best_source_dev_nmae"), (int, float)
        )
        or not math.isclose(
            float(checkpoint["best_source_dev_nmae"]),
            float(best_metric),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("training/checkpoint source-dev selection differs")


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON artifact {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


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
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


@dataclass(frozen=True, slots=True)
class AuditedTrainingJob:
    spec: JobSpec
    result: dict[str, object]
    manifest: dict[str, object]
    evidence: Mapping[str, CaseEvidence]
    result_file_sha256: str
    manifest_file_sha256: str


def audit_training_job(
    spec: JobSpec,
    *,
    freeze: Mapping[str, object],
    tune_windows_cache: dict[str, object] | None = None,
) -> AuditedTrainingJob:
    """Audit one exact tune-stage training job down to registered case rows."""

    if not isinstance(spec, JobSpec):
        raise TypeError("spec must be JobSpec")
    if not isinstance(freeze, Mapping):
        raise TypeError("freeze must be a mapping")
    cache = {} if tune_windows_cache is None else tune_windows_cache
    directory = spec.job_directory
    if not verify_completed_job(spec):
        raise ValueError(f"{spec.dataset}: formal job is missing")
    result_path = safe_job_artifact_path(
        directory, "result.json", expected_name="result.json"
    )
    manifest_path = safe_job_artifact_path(
        directory, "manifest.json", expected_name="manifest.json"
    )
    result = _json_object(result_path)
    manifest = _json_object(manifest_path)
    if set(result) != _RESULT_FIELDS:
        raise ValueError(f"{spec.dataset}: result schema drifted")
    neural = result.get("neural_checkpoint")
    prior_checkpoint = result.get("prior_checkpoint")
    cells = result.get("cells")
    if (
        not isinstance(neural, dict)
        or set(neural) != _NEURAL_CHECKPOINT_FIELDS
        or not isinstance(prior_checkpoint, dict)
        or set(prior_checkpoint) != {"file_sha256"}
        or not isinstance(cells, dict)
        or set(cells) != set(load_protocol().mask_families)
    ):
        raise ValueError(f"{spec.dataset}: nested result schema drifted")
    verify_training_record(result.get("training"), checkpoint=neural)
    if manifest.get("result_sha256") != file_sha256(result_path):
        raise ValueError(f"{spec.dataset}: result file hash mismatch")
    if manifest.get("source_tree_sha256") != freeze.get("source_tree_sha256"):
        raise ValueError(f"{spec.dataset}: source freeze mismatch")
    if manifest.get("config_sha256") != freeze.get("config_sha256"):
        raise ValueError(f"{spec.dataset}: config freeze mismatch")
    data_identities = freeze.get("data_identities")
    if not isinstance(data_identities, Mapping):
        raise ValueError("freeze data identities are absent")
    dataset_data = data_identities.get(spec.dataset)
    if not isinstance(dataset_data, Mapping):
        raise ValueError(f"{spec.dataset}: freeze data identity is absent")
    if manifest.get("data_sha256") != dataset_data.get("data_sha256"):
        raise ValueError(f"{spec.dataset}: data freeze mismatch")

    if prior_checkpoint["file_sha256"] != spec.scientific_identity[
        "acil_checkpoint_file_sha256"
    ]:
        raise ValueError(f"{spec.dataset}: prior checkpoint identity mismatch")
    weights_path = safe_job_artifact_path(
        directory,
        neural["file"],
        expected_name="neural_best.safetensors",
    )
    if (
        file_sha256(weights_path) != neural["file_sha256"]
        or neural["file_sha256"]
        != manifest.get("neural_checkpoint_file_sha256")
    ):
        raise ValueError(f"{spec.dataset}: neural checkpoint file hash mismatch")
    state = load_safetensors(str(weights_path), device="cpu")
    if (
        scientific_tensor_sha256(state) != neural["tensor_sha256"]
        or neural["tensor_sha256"]
        != manifest.get("neural_checkpoint_tensor_sha256")
    ):
        raise ValueError(f"{spec.dataset}: neural tensor hash mismatch")
    verify_neural_state(
        spec.dataset,
        state,
        reported_parameter_count=result["neural_parameter_count"],
    )

    if spec.dataset not in cache:
        cache[spec.dataset] = load_permitted_windows(spec.dataset, "tune")
    audited_evidence: dict[str, CaseEvidence] = {}
    registered_batches: dict[str, EvaluationBatch] = {}
    fit_fallback = permitted_fit_fallback(spec.dataset)
    for family in load_protocol().mask_families:
        cell = cells[family]
        if not isinstance(cell, dict) or set(cell) != _CELL_FIELDS:
            raise ValueError(f"{spec.dataset}/{family}: cell schema drifted")
        evidence_path = safe_job_artifact_path(
            directory,
            cell["evidence_file"],
            expected_name=f"evidence_{family}.npz",
        )
        if file_sha256(evidence_path) != cell["evidence_file_sha256"]:
            raise ValueError(
                f"{spec.dataset}/{family}: evidence file hash mismatch"
            )
        evidence = load_evidence_npz(evidence_path)
        if (
            scientific_array_sha256(evidence.as_npz_dict())
            != cell["evidence_content_sha256"]
        ):
            raise ValueError(
                f"{spec.dataset}/{family}: evidence content hash mismatch"
            )
        recomputed_summary = summarize_case_evidence(evidence)
        if not _scientifically_equal(recomputed_summary, cell["summary"]):
            raise ValueError(
                f"{spec.dataset}/{family}: summary differs from evidence"
            )
        registered = build_evaluation_batch(
            cache[spec.dataset],
            seed_bundle=spec.seed_bundle,
            family=family,
        )
        verify_registered_evidence(
            evidence,
            registered=registered,
            fit_fallback=fit_fallback,
        )
        expected_hashes = {
            "mask_sha256": mask_identity_sha256(registered.mask_sha256),
            "oracle_q_sha256": mask_identity_sha256(
                registered.oracle_q_sha256
            ),
            "oracle_e_sha256": mask_identity_sha256(
                registered.oracle_e_sha256
            ),
        }
        if any(
            cell.get(name) != value for name, value in expected_hashes.items()
        ):
            raise ValueError(
                f"{spec.dataset}/{family}: registered mask identity mismatch"
            )
        audited_evidence[family] = evidence
        registered_batches[family] = registered
    replay_and_verify_tune_evidence(
        dataset=spec.dataset,
        seed_bundle_id=spec.seed_bundle,
        neural_state=state,
        expected_neural_tensor_sha256=neural["tensor_sha256"],
        expected_prior_file_sha256=prior_checkpoint["file_sha256"],
        registered_batches=registered_batches,
        fit_fallback=fit_fallback,
        persisted=audited_evidence,
    )
    return AuditedTrainingJob(
        spec=spec,
        result=result,
        manifest=manifest,
        evidence=audited_evidence,
        result_file_sha256=file_sha256(result_path),
        manifest_file_sha256=file_sha256(manifest_path),
    )


def review_prototype(
    *,
    output_root: Path,
    freeze_record: Path,
) -> dict[str, object]:
    freeze = verify_freeze_record(freeze_record)
    specs = build_stage_specs("prototype", Path(output_root), freeze)
    cache: dict[str, object] = {}
    audited = [
        audit_training_job(spec, freeze=freeze, tune_windows_cache=cache)
        for spec in specs
    ]
    artifacts = [
        {"result": item.result, "manifest": item.manifest}
        for item in audited
    ]
    return adjudicate_prototype(artifacts)


def build_extension_report(
    *,
    prototype_report: Mapping[str, object],
    audited_jobs: list[AuditedTrainingJob],
    freeze: Mapping[str, object],
) -> dict[str, object]:
    """Assemble the exact six-job tune review and frozen uncertainty report."""

    if not isinstance(prototype_report, Mapping):
        raise TypeError("prototype_report must be a mapping")
    integrity = prototype_report.get("integrity")
    if (
        prototype_report.get("verdict") != "proceed"
        or not isinstance(integrity, Mapping)
        or integrity.get("valid") is not True
    ):
        raise ValueError("extension review requires audited prototype proceed")
    common = integrity.get("common_identity")
    if not isinstance(common, Mapping) or (
        common.get("source_tree_sha256") != freeze.get("source_tree_sha256")
        or common.get("config_sha256") != freeze.get("config_sha256")
    ):
        raise ValueError("prototype and extension freeze identity differ")
    if not isinstance(audited_jobs, list) or any(
        not isinstance(item, AuditedTrainingJob) for item in audited_jobs
    ):
        raise TypeError("audited_jobs must contain AuditedTrainingJob values")

    expected = {
        (bundle, dataset)
        for bundle in (1, 2, 3)
        for dataset in ("abilene", "geant")
    }
    by_key: dict[tuple[int, str], AuditedTrainingJob] = {}
    for item in audited_jobs:
        key = (item.spec.seed_bundle, item.spec.dataset)
        expected_stage = "prototype" if item.spec.seed_bundle == 1 else "extension"
        if (
            key in by_key
            or key not in expected
            or item.spec.stage != expected_stage
            or item.spec.source_tree_sha256 != freeze.get("source_tree_sha256")
            or item.spec.config_sha256 != freeze.get("config_sha256")
            or set(item.evidence) != set(load_protocol().mask_families)
        ):
            raise ValueError("audited extension job grid or identity drifted")
        by_key[key] = item
    if set(by_key) != expected:
        raise ValueError("extension review requires the exact six-job grid")

    evidence_grid = {
        (bundle, dataset, family): by_key[(bundle, dataset)].evidence[family]
        for bundle in (1, 2, 3)
        for dataset in ("abilene", "geant")
        for family in load_protocol().mask_families
    }
    uncertainty = analyze_extension_uncertainty(evidence_grid)
    cell_summaries = {
        key: summarize_case_evidence(evidence)
        for key, evidence in evidence_grid.items()
    }
    multibundle = adjudicate_multibundle(cell_summaries, uncertainty)
    if (
        not isinstance(multibundle, Mapping)
        or multibundle.get("schema")
        != "anchorcv-v1:multibundle-adjudication:v1"
        or multibundle.get("verdict") not in {"proceed", "revise", "kill"}
    ):
        raise RuntimeError("multi-bundle adjudication contract drifted")
    verdict = str(multibundle["verdict"])
    checkpoint_bindings = []
    for bundle in (1, 2, 3):
        for dataset in ("abilene", "geant"):
            item = by_key[(bundle, dataset)]
            neural = item.result["neural_checkpoint"]
            prior = item.result["prior_checkpoint"]
            training = item.result["training"]
            if not all(
                isinstance(value, Mapping)
                for value in (neural, prior, training)
            ):
                raise ValueError("audited checkpoint binding schema drifted")
            checkpoint_bindings.append(
                {
                    "dataset": dataset,
                    "seed_bundle": bundle,
                    "source_training_stage": item.spec.stage,
                    "source_training_job_id": item.spec.job_id,
                    "source_result_sha256": item.result_file_sha256,
                    "source_manifest_sha256": item.manifest_file_sha256,
                    "neural_checkpoint_file_sha256": neural.get(
                        "file_sha256"
                    ),
                    "neural_checkpoint_tensor_sha256": neural.get(
                        "tensor_sha256"
                    ),
                    "acil_checkpoint_file_sha256": prior.get("file_sha256"),
                    "best_epoch": training.get("best_epoch"),
                    "best_source_dev_nmae": training.get(
                        "best_source_dev_nmae"
                    ),
                }
            )
    return {
        "schema": "anchorcv-v1:extension-review:v1",
        "status": "verified_complete",
        "verdict": verdict,
        "confirmation_authorized": verdict == "proceed",
        "job_count": 6,
        "checkpoint_bindings": checkpoint_bindings,
        "grid": [
            {
                "dataset": dataset,
                "job_id": by_key[(bundle, dataset)].spec.job_id,
                "seed_bundle": bundle,
                "stage": by_key[(bundle, dataset)].spec.stage,
            }
            for bundle in (1, 2, 3)
            for dataset in ("abilene", "geant")
        ],
        "integrity": {
            "valid": True,
            "source_tree_sha256": freeze.get("source_tree_sha256"),
            "config_sha256": freeze.get("config_sha256"),
            "git_available": False,
            "git_commit": None,
            "test_access": False,
        },
        "prototype": dict(prototype_report),
        "uncertainty": uncertainty,
        "multibundle_adjudication": multibundle,
        "interpretation": (
            "the pre-registered multi-bundle scientific and uncertainty "
            "gates must all pass before branch-sealed confirmation is "
            "authorized"
        ),
    }


def review_extension(
    *,
    prototype_output_root: Path,
    extension_output_root: Path,
    freeze_record: Path,
) -> dict[str, object]:
    """Re-audit prototype plus extension artifacts and analyze all bundles."""

    freeze = verify_freeze_record(freeze_record)
    prototype_report = review_prototype(
        output_root=prototype_output_root,
        freeze_record=freeze_record,
    )
    specs = (
        *build_stage_specs("prototype", Path(prototype_output_root), freeze),
        *build_stage_specs("extension", Path(extension_output_root), freeze),
    )
    cache: dict[str, object] = {}
    audited = [
        audit_training_job(spec, freeze=freeze, tune_windows_cache=cache)
        for spec in specs
    ]
    return build_extension_report(
        prototype_report=prototype_report,
        audited_jobs=audited,
        freeze=freeze,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = review_prototype(
        output_root=args.output_root,
        freeze_record=args.freeze_record,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    print(
        json.dumps(
            {"verdict": report["verdict"], "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuditedTrainingJob",
    "audit_training_job",
    "build_parser",
    "build_extension_report",
    "file_sha256",
    "load_evidence_npz",
    "review_extension",
    "review_prototype",
    "safe_job_artifact_path",
    "verify_neural_state",
    "verify_registered_evidence",
    "verify_training_record",
]
