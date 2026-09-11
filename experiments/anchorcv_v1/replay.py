"""Independent deterministic replay of persisted AnchorCV evidence."""

from __future__ import annotations

from collections.abc import Mapping
import re

import torch
from torch import Tensor

from experiments.acil_innovation_v1.batching import EvaluationBatch
from experiments.acil_innovation_v1.preprocessing import FitFallback
from experiments.acil_innovation_v1.registries import seed_bundle
from experiments.sc2_ari_v1.initialization import load_frozen_acil

from .evaluation import CaseEvidence
from .job_runtime import (
    evaluate_evaluation_batch,
    scientific_tensor_sha256,
)
from .protocol import load_protocol
from .training_runtime import build_neural_expert, seed_everything


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_registered_grid(
    registered_batches: Mapping[str, EvaluationBatch],
    *,
    dataset: str,
    seed_bundle_id: int,
    expected_cohort: str,
) -> tuple[tuple[str, EvaluationBatch], ...]:
    protocol = load_protocol()
    families = protocol.mask_families
    if not isinstance(registered_batches, Mapping) or (
        set(registered_batches) != set(families)
    ):
        raise ValueError("replay requires the exact frozen mask families")
    ordered: list[tuple[str, EvaluationBatch]] = []
    for family in families:
        batch = registered_batches[family]
        if not isinstance(batch, EvaluationBatch):
            raise TypeError("registered replay values must be EvaluationBatch")
        if (
            batch.dataset != dataset
            or batch.cohort != expected_cohort
            or batch.seed_bundle != seed_bundle_id
            or batch.family != family
        ):
            raise ValueError(
                "registered replay batch dataset/cohort/bundle/family drifted"
            )
        ordered.append((family, batch))
    return tuple(ordered)


def replay_registered_evidence(
    *,
    dataset: str,
    seed_bundle_id: int,
    neural_state: Mapping[str, Tensor],
    expected_neural_tensor_sha256: str,
    expected_prior_file_sha256: str,
    registered_batches: Mapping[str, EvaluationBatch],
    fit_fallback: FitFallback,
    expected_cohort: str,
) -> dict[str, CaseEvidence]:
    """Replay one registered three-family grid on the fixed formal CUDA lane.

    This low-level entry point is intentionally device-free. Callers establish
    the cohort authority and pass already registered batches; the tune and gate
    wrappers below remove even the cohort choice from their respective reviews.
    """

    protocol = load_protocol()
    if dataset not in protocol.datasets:
        raise ValueError("dataset must be exactly abilene or geant")
    if (
        isinstance(seed_bundle_id, bool)
        or not isinstance(seed_bundle_id, int)
        or seed_bundle_id not in {1, 2, 3}
    ):
        raise ValueError("seed bundle must be exactly 1, 2, or 3")
    if expected_cohort not in {"tune", "gate"}:
        raise ValueError("replay cohort must be fixed to tune or gate")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "formal evidence replay requires the registered CUDA math-SDPA lane"
        )
    neural_sha = _require_sha256(
        expected_neural_tensor_sha256,
        label="expected neural tensor identity",
    )
    prior_sha = _require_sha256(
        expected_prior_file_sha256,
        label="expected prior file identity",
    )
    if not isinstance(neural_state, Mapping) or not neural_state:
        raise ValueError("strict-loaded neural state must be nonempty")
    if scientific_tensor_sha256(neural_state) != neural_sha:
        raise ValueError("strict-loaded neural tensor identity mismatch")
    ordered_batches = _require_registered_grid(
        registered_batches,
        dataset=dataset,
        seed_bundle_id=seed_bundle_id,
        expected_cohort=expected_cohort,
    )
    if not isinstance(fit_fallback, FitFallback):
        raise TypeError("fit_fallback must be FitFallback")

    device = torch.device("cuda")
    seeds = seed_bundle(seed_bundle_id)
    seed_everything(seeds.model, deterministic=True)
    neural = build_neural_expert(dataset)
    try:
        neural.load_state_dict(dict(neural_state), strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "strict-loaded neural state differs from the frozen architecture"
        ) from exc
    neural.requires_grad_(False)
    neural.eval().to(device)
    if neural.training or any(
        parameter.requires_grad for parameter in neural.parameters()
    ):
        raise RuntimeError("replayed neural expert must be frozen in eval mode")

    prior, prior_record = load_frozen_acil(
        dataset, seed_bundle_id, device=device
    )
    if prior_record.file_sha256 != prior_sha:
        raise ValueError("replayed ACIL prior identity mismatch")
    if prior.training or any(
        parameter.requires_grad for parameter in prior.parameters()
    ):
        raise RuntimeError("replayed ACIL expert must be frozen in eval mode")

    replayed: dict[str, CaseEvidence] = {}
    for family, batch in ordered_batches:
        replayed[family] = evaluate_evaluation_batch(
            prior=prior,
            neural=neural,
            batch=batch,
            fit_fallback=fit_fallback,
            device=device,
            chunk_size=8,
        )
    torch.cuda.synchronize(device)
    return replayed


def replay_tune_evidence(
    *,
    dataset: str,
    seed_bundle_id: int,
    neural_state: Mapping[str, Tensor],
    expected_neural_tensor_sha256: str,
    expected_prior_file_sha256: str,
    registered_batches: Mapping[str, EvaluationBatch],
    fit_fallback: FitFallback,
) -> dict[str, CaseEvidence]:
    """Replay formal tune evidence without exposing a cohort or device choice."""

    return replay_registered_evidence(
        dataset=dataset,
        seed_bundle_id=seed_bundle_id,
        neural_state=neural_state,
        expected_neural_tensor_sha256=expected_neural_tensor_sha256,
        expected_prior_file_sha256=expected_prior_file_sha256,
        registered_batches=registered_batches,
        fit_fallback=fit_fallback,
        expected_cohort="tune",
    )


def replay_gate_evidence(
    *,
    dataset: str,
    seed_bundle_id: int,
    neural_state: Mapping[str, Tensor],
    expected_neural_tensor_sha256: str,
    expected_prior_file_sha256: str,
    registered_batches: Mapping[str, EvaluationBatch],
    fit_fallback: FitFallback,
) -> dict[str, CaseEvidence]:
    """Replay formal gate evidence without exposing a cohort or device choice."""

    return replay_registered_evidence(
        dataset=dataset,
        seed_bundle_id=seed_bundle_id,
        neural_state=neural_state,
        expected_neural_tensor_sha256=expected_neural_tensor_sha256,
        expected_prior_file_sha256=expected_prior_file_sha256,
        registered_batches=registered_batches,
        fit_fallback=fit_fallback,
        expected_cohort="gate",
    )


def _array_bits_equal(left: object, right: object) -> bool:
    left_array = left
    right_array = right
    if not hasattr(left_array, "dtype") or not hasattr(right_array, "dtype"):
        return False
    if (
        left_array.dtype != right_array.dtype
        or left_array.shape != right_array.shape
    ):
        return False
    left_bytes = left_array.view("u1").reshape(-1)
    right_bytes = right_array.view("u1").reshape(-1)
    return bool((left_bytes == right_bytes).all())


def verify_replayed_evidence(
    *,
    persisted: Mapping[str, CaseEvidence],
    replayed: Mapping[str, CaseEvidence],
) -> None:
    """Require exact field bits between persisted and independently replayed rows."""

    families = load_protocol().mask_families
    if (
        not isinstance(persisted, Mapping)
        or not isinstance(replayed, Mapping)
        or set(persisted) != set(families)
        or set(replayed) != set(families)
    ):
        raise ValueError("evidence comparison requires the exact frozen mask families")
    field_names = tuple(CaseEvidence.__dataclass_fields__)
    for family in families:
        persisted_item = persisted[family]
        replayed_item = replayed[family]
        if not isinstance(persisted_item, CaseEvidence) or not isinstance(
            replayed_item, CaseEvidence
        ):
            raise TypeError("persisted and replayed values must be CaseEvidence")
        for name in field_names:
            if not _array_bits_equal(
                getattr(persisted_item, name),
                getattr(replayed_item, name),
            ):
                raise ValueError(
                    f"{family}/{name}: replay differs bitwise from persisted evidence"
                )


def replay_and_verify_tune_evidence(
    *,
    dataset: str,
    seed_bundle_id: int,
    neural_state: Mapping[str, Tensor],
    expected_neural_tensor_sha256: str,
    expected_prior_file_sha256: str,
    registered_batches: Mapping[str, EvaluationBatch],
    fit_fallback: FitFallback,
    persisted: Mapping[str, CaseEvidence],
) -> None:
    """Replay and fail closed before a tune artifact can enter adjudication."""

    replayed = replay_tune_evidence(
        dataset=dataset,
        seed_bundle_id=seed_bundle_id,
        neural_state=neural_state,
        expected_neural_tensor_sha256=expected_neural_tensor_sha256,
        expected_prior_file_sha256=expected_prior_file_sha256,
        registered_batches=registered_batches,
        fit_fallback=fit_fallback,
    )
    verify_replayed_evidence(persisted=persisted, replayed=replayed)


__all__ = [
    "replay_and_verify_tune_evidence",
    "replay_gate_evidence",
    "replay_registered_evidence",
    "replay_tune_evidence",
    "verify_replayed_evidence",
]
