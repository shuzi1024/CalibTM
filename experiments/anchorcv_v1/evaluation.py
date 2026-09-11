"""Within-instance anchor cross-validation and diagnostic oracle evidence."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.preprocessing import FitFallback

from .metrics import (
    binary_auroc,
    flow_target_regret,
    hard_anchorcv_decision,
    oracle_gap_capture,
    ratio_of_sums_nmae,
    relative_improvement,
    select_expert_predictions,
    spearman_correlation,
)
from .model import PriorFreeMaskNativeExpert
from .objective import drop_observed_anchor, predict_raw


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _owned_array(value: object, *, dtype: np.dtype, label: str) -> np.ndarray:
    try:
        result = np.array(value, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an array") from exc
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CaseEvidence:
    """Per-window/per-flow evidence; hidden-target fields are diagnostic only."""

    window_index: IntArray
    flow_index: IntArray
    middle_anchor_index: IntArray
    p_error_sum: FloatArray
    n_error_sum: FloatArray
    hard_error_sum: FloatArray
    oracle_error_sum: FloatArray
    truth_sum: FloatArray
    target_count: IntArray
    k3_scale: FloatArray
    p_anchor_absolute_error: FloatArray
    n_anchor_absolute_error: FloatArray
    p_anchor_normalized_error: FloatArray
    n_anchor_normalized_error: FloatArray
    loo_score: FloatArray
    target_regret: FloatArray
    hard_winner_is_n: BoolArray
    oracle_winner_is_n: BoolArray

    def __post_init__(self) -> None:
        integer_names = {
            "window_index",
            "flow_index",
            "middle_anchor_index",
            "target_count",
        }
        boolean_names = {"hard_winner_is_n", "oracle_winner_is_n"}
        values: dict[str, np.ndarray] = {}
        for item in fields(self):
            raw = getattr(self, item.name)
            if item.name in integer_names:
                source = np.asarray(raw)
                if source.dtype.kind not in {"i", "u"}:
                    raise ValueError(f"{item.name} must be an integer array")
                value = _owned_array(raw, dtype=np.dtype("<i8"), label=item.name)
            elif item.name in boolean_names:
                source = np.asarray(raw)
                if source.dtype != np.dtype(np.bool_):
                    raise ValueError(f"{item.name} must be a boolean array")
                value = _owned_array(raw, dtype=np.dtype(np.bool_), label=item.name)
            else:
                value = _owned_array(raw, dtype=np.dtype("<f8"), label=item.name)
            values[item.name] = value
            object.__setattr__(self, item.name, value)

        if any(value.ndim != 1 for value in values.values()):
            raise ValueError("all evidence arrays must be one-dimensional")
        case_count = int(values["window_index"].size)
        if case_count < 1 or any(
            value.size != case_count for value in values.values()
        ):
            raise ValueError("all evidence arrays must align to a nonempty case axis")
        finite_names = set(values) - integer_names - boolean_names
        if any(not np.isfinite(values[name]).all() for name in finite_names):
            raise ValueError("floating evidence arrays must be finite")
        nonnegative_names = {
            "p_error_sum",
            "n_error_sum",
            "hard_error_sum",
            "oracle_error_sum",
            "truth_sum",
            "k3_scale",
            "p_anchor_absolute_error",
            "n_anchor_absolute_error",
            "p_anchor_normalized_error",
            "n_anchor_normalized_error",
        }
        if any(np.any(values[name] < 0.0) for name in nonnegative_names):
            raise ValueError("error, truth, scale, and anchor arrays must be nonnegative")
        if np.any(values["target_count"] <= 0):
            raise ValueError("target_count must be strictly positive")
        if np.any(values["k3_scale"] <= 0.0):
            raise ValueError("k3_scale must be strictly positive")
        if any(
            np.any(values[name] < 0)
            for name in ("window_index", "flow_index", "middle_anchor_index")
        ):
            raise ValueError("window, flow, and middle-anchor indices must be nonnegative")

        expected_p_normalized = (
            values["p_anchor_absolute_error"] / values["k3_scale"]
        )
        expected_n_normalized = (
            values["n_anchor_absolute_error"] / values["k3_scale"]
        )
        if not np.allclose(
            values["p_anchor_normalized_error"],
            expected_p_normalized,
            rtol=1e-12,
            atol=1e-12,
        ) or not np.allclose(
            values["n_anchor_normalized_error"],
            expected_n_normalized,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("normalized anchor errors differ from raw errors / K3 scale")
        expected_loo = expected_p_normalized - expected_n_normalized
        if not np.allclose(values["loo_score"], expected_loo, rtol=1e-12, atol=1e-12):
            raise ValueError("loo_score differs from the normalized middle-anchor error difference")
        if not np.array_equal(values["hard_winner_is_n"], values["loo_score"] > 0.0):
            raise ValueError("hard winner differs from positive LOO score")
        if not np.array_equal(values["oracle_winner_is_n"], values["target_regret"] > 0.0):
            raise ValueError("oracle winner differs from positive target regret")
        expected_target_regret = (
            values["p_error_sum"] - values["n_error_sum"]
        ) / (values["target_count"] * values["k3_scale"])
        if not np.allclose(
            values["target_regret"],
            expected_target_regret,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("target regret differs from normalized target errors")
        expected_hard = np.where(
            values["hard_winner_is_n"],
            values["n_error_sum"],
            values["p_error_sum"],
        )
        expected_oracle = np.minimum(values["p_error_sum"], values["n_error_sum"])
        if not np.allclose(values["hard_error_sum"], expected_hard, rtol=1e-12, atol=1e-12):
            raise ValueError("hard error sums differ from deployable decisions")
        if not np.allclose(values["oracle_error_sum"], expected_oracle, rtol=1e-12, atol=1e-12):
            raise ValueError("oracle error sums differ from per-case best expert")

    @property
    def case_count(self) -> int:
        return int(self.window_index.size)

    def as_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            item.name: np.asarray(getattr(self, item.name))
            for item in fields(self)
        }


@dataclass(frozen=True, slots=True)
class DeployableAnchorDecision:
    """Routing operands computed without any missing-target truth."""

    p_prediction: FloatArray
    n_prediction: FloatArray
    middle_anchor_index: IntArray
    k3_scale: FloatArray
    p_anchor_absolute_error: FloatArray
    n_anchor_absolute_error: FloatArray
    p_anchor_normalized_error: FloatArray
    n_anchor_normalized_error: FloatArray
    loo_score: FloatArray
    winner_is_n: BoolArray


def compute_anchor_decision(
    *,
    prior: ACILBase,
    neural: PriorFreeMaskNativeExpert,
    model_input: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
) -> DeployableAnchorDecision:
    """Compute the fair sorted-middle routing decision from observed data only."""

    if not isinstance(prior, ACILBase):
        raise TypeError("prior must be ACILBase")
    if not isinstance(neural, PriorFreeMaskNativeExpert):
        raise TypeError("neural must be PriorFreeMaskNativeExpert")
    if prior.training or neural.training:
        raise ValueError("both experts must be in evaluation mode")
    if model_input.ndim != 3 or observed.shape != model_input.shape:
        raise ValueError("model input and observed must share [B,F,T]")
    if observed.dtype is not torch.bool:
        raise TypeError("observed must be boolean")
    if model_input.device != observed.device:
        raise ValueError("model input and observed must share a device")
    dropped = drop_observed_anchor(observed, rank=1)

    with torch.no_grad():
        p_full = prior(model_input, observed, fit_fallback)
        n_full = predict_raw(neural, model_input, observed, fit_fallback)
        if not torch.allclose(
            p_full.statistics.mean,
            n_full.prepared.mean,
            rtol=0.0,
            atol=0.0,
        ) or not torch.allclose(
            p_full.statistics.std,
            n_full.prepared.std,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("experts disagree on observation-only normalization")
        submask_input = torch.where(
            dropped.submask,
            model_input,
            torch.full(
                (),
                float("nan"),
                dtype=model_input.dtype,
                device=model_input.device,
            ),
        )
        p_sub = prior(submask_input, dropped.submask, fit_fallback)
        n_sub = predict_raw(neural, submask_input, dropped.submask, fit_fallback)
        if not torch.allclose(
            p_sub.statistics.std,
            n_sub.prepared.std,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("K=2 experts disagree on internal normalization scale")
        anchor_value = model_input.gather(
            -1, dropped.anchor_index.unsqueeze(-1)
        )
        if not torch.isfinite(anchor_value).all().item():
            raise ValueError("observed middle anchor must be finite")
        p_anchor = p_sub.prediction.gather(
            -1, dropped.anchor_index.unsqueeze(-1)
        )
        n_anchor = n_sub.prediction.gather(
            -1, dropped.anchor_index.unsqueeze(-1)
        )
        p_raw = (p_anchor - anchor_value).abs()
        n_raw = (n_anchor - anchor_value).abs()
        common_scale = p_full.statistics.std

    batch, flows, times = (int(value) for value in model_input.shape)
    cases = batch * flows
    p_raw_array = np.array(
        p_raw.detach().float().cpu().numpy().reshape(cases),
        dtype=np.float64,
        copy=True,
    )
    n_raw_array = np.array(
        n_raw.detach().float().cpu().numpy().reshape(cases),
        dtype=np.float64,
        copy=True,
    )
    common_scale_array = np.array(
        common_scale.detach().float().cpu().numpy().reshape(cases),
        dtype=np.float64,
        copy=True,
    )
    # Derive persisted normalized evidence from the persisted raw operands so
    # the artifact invariant is exactly recomputable, not merely close to a
    # separately rounded fp32 division performed on device.
    p_normalized_array = p_raw_array / common_scale_array
    n_normalized_array = n_raw_array / common_scale_array
    decision = hard_anchorcv_decision(
        p_normalized_array.reshape(cases, 1),
        n_normalized_array.reshape(cases, 1),
    )
    return DeployableAnchorDecision(
        p_prediction=np.array(
            p_full.prediction.detach().float().cpu().numpy().reshape(cases, times),
            dtype=np.float64,
            copy=True,
        ),
        n_prediction=np.array(
            n_full.prediction.detach().float().cpu().numpy().reshape(cases, times),
            dtype=np.float64,
            copy=True,
        ),
        middle_anchor_index=np.array(
            dropped.anchor_index.detach().cpu().numpy().reshape(cases),
            dtype=np.int64,
            copy=True,
        ),
        k3_scale=common_scale_array,
        p_anchor_absolute_error=p_raw_array,
        n_anchor_absolute_error=n_raw_array,
        p_anchor_normalized_error=np.array(
            p_normalized_array, dtype=np.float64, copy=True
        ),
        n_anchor_normalized_error=np.array(
            n_normalized_array, dtype=np.float64, copy=True
        ),
        loo_score=np.array(decision.score, dtype=np.float64, copy=True),
        winner_is_n=np.array(decision.winner_is_n, dtype=np.bool_, copy=True),
    )


def _target_error_sums(
    prediction: np.ndarray,
    truth: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    difference = np.zeros_like(truth, dtype=np.float64)
    np.subtract(prediction, truth, out=difference, where=target)
    return np.abs(difference).sum(axis=1, dtype=np.float64)


def evaluate_tensor_batch(
    *,
    prior: ACILBase,
    neural: PriorFreeMaskNativeExpert,
    model_input: Tensor,
    truth: Tensor,
    observed: Tensor,
    fit_fallback: FitFallback,
    window_offset: int = 0,
) -> CaseEvidence:
    """Evaluate fixed K=3 targets after a truth-independent middle-anchor route."""

    if (
        isinstance(window_offset, bool)
        or not isinstance(window_offset, int)
        or window_offset < 0
    ):
        raise ValueError("window_offset must be a nonnegative integer")
    if truth.shape != model_input.shape or observed.shape != truth.shape:
        raise ValueError("model input, truth, and observed must share [B,F,T]")
    if observed.dtype is not torch.bool:
        raise TypeError("observed must be boolean")
    if not torch.isfinite(truth).all().item() or (truth < 0.0).any().item():
        raise ValueError("truth must be finite and nonnegative")
    if not (truth.device == model_input.device == observed.device):
        raise ValueError("model input, truth, and observed must share a device")
    route = compute_anchor_decision(
        prior=prior,
        neural=neural,
        model_input=model_input,
        observed=observed,
        fit_fallback=fit_fallback,
    )

    batch, flows, times = (int(value) for value in truth.shape)
    p_prediction = route.p_prediction
    n_prediction = route.n_prediction
    truth_values = truth.detach().float().cpu().numpy().reshape(-1, times)
    target = (~observed).detach().cpu().numpy().reshape(-1, times)
    target_evidence = flow_target_regret(
        p_prediction, n_prediction, truth_values, target
    )
    hard_prediction = select_expert_predictions(
        p_prediction, n_prediction, route.winner_is_n
    )
    oracle_prediction = select_expert_predictions(
        p_prediction, n_prediction, target_evidence.oracle_winner_is_n
    )
    normalized_target_regret = target_evidence.regret / (
        target_evidence.target_count * route.k3_scale
    )
    return CaseEvidence(
        window_index=np.repeat(
            np.arange(window_offset, window_offset + batch, dtype=np.int64),
            flows,
        ),
        flow_index=np.tile(np.arange(flows, dtype=np.int64), batch),
        middle_anchor_index=route.middle_anchor_index,
        p_error_sum=target_evidence.p_error_sum,
        n_error_sum=target_evidence.n_error_sum,
        hard_error_sum=_target_error_sums(
            hard_prediction, truth_values, target
        ),
        oracle_error_sum=_target_error_sums(
            oracle_prediction, truth_values, target
        ),
        truth_sum=target_evidence.truth_sum,
        target_count=target_evidence.target_count,
        k3_scale=route.k3_scale,
        p_anchor_absolute_error=route.p_anchor_absolute_error,
        n_anchor_absolute_error=route.n_anchor_absolute_error,
        p_anchor_normalized_error=route.p_anchor_normalized_error,
        n_anchor_normalized_error=route.n_anchor_normalized_error,
        loo_score=route.loo_score,
        target_regret=normalized_target_regret,
        hard_winner_is_n=route.winner_is_n,
        oracle_winner_is_n=target_evidence.oracle_winner_is_n,
    )


def concatenate_case_evidence(*parts: CaseEvidence) -> CaseEvidence:
    if not parts or any(not isinstance(part, CaseEvidence) for part in parts):
        raise ValueError("one or more CaseEvidence parts are required")
    return CaseEvidence(
        **{
            item.name: np.concatenate(
                [np.asarray(getattr(part, item.name)) for part in parts],
                axis=0,
            )
            for item in fields(CaseEvidence)
        }
    )


def summarize_case_evidence(evidence: CaseEvidence) -> dict[str, Any]:
    """Summarize one dataset/bundle/mask cell without hiding undefined metrics."""

    if not isinstance(evidence, CaseEvidence):
        raise TypeError("evidence must be CaseEvidence")
    p_nmae = ratio_of_sums_nmae(evidence.p_error_sum, evidence.truth_sum)
    n_nmae = ratio_of_sums_nmae(evidence.n_error_sum, evidence.truth_sum)
    hard_nmae = ratio_of_sums_nmae(evidence.hard_error_sum, evidence.truth_sum)
    oracle_nmae = ratio_of_sums_nmae(evidence.oracle_error_sum, evidence.truth_sum)
    best_name = "n" if n_nmae < p_nmae else "p"
    best_nmae = min(p_nmae, n_nmae)
    result: dict[str, Any] = {
        "case_count": evidence.case_count,
        "target_count": int(evidence.target_count.sum(dtype=np.int64)),
        "absolute_truth_sum": float(
            math.fsum(float(value) for value in evidence.truth_sum)
        ),
        "p_absolute_error_sum": float(
            math.fsum(float(value) for value in evidence.p_error_sum)
        ),
        "n_absolute_error_sum": float(
            math.fsum(float(value) for value in evidence.n_error_sum)
        ),
        "hard_absolute_error_sum": float(
            math.fsum(float(value) for value in evidence.hard_error_sum)
        ),
        "oracle_absolute_error_sum": float(
            math.fsum(float(value) for value in evidence.oracle_error_sum)
        ),
        "p_nmae": p_nmae,
        "n_nmae": n_nmae,
        "hard_nmae": hard_nmae,
        "oracle_nmae": oracle_nmae,
        "best_single_expert": best_name,
        "best_single_nmae": best_nmae,
        "oracle_relative_improvement": relative_improvement(
            candidate=oracle_nmae, baseline=best_nmae
        ),
        "hard_relative_improvement": relative_improvement(
            candidate=hard_nmae, baseline=best_nmae
        ),
        "neural_selection_rate": float(evidence.hard_winner_is_n.mean()),
        "oracle_neural_rate": float(evidence.oracle_winner_is_n.mean()),
        "loo_winner_auroc": None,
        "loo_target_regret_spearman": None,
        "oracle_gap_capture": None,
    }
    try:
        result["loo_winner_auroc"] = binary_auroc(
            evidence.oracle_winner_is_n, evidence.loo_score
        )
    except ValueError as exc:
        result["loo_winner_auroc_undefined_reason"] = str(exc)
    try:
        result["loo_target_regret_spearman"] = spearman_correlation(
            evidence.loo_score, evidence.target_regret
        )
    except ValueError as exc:
        result["loo_target_regret_spearman_undefined_reason"] = str(exc)
    try:
        result["oracle_gap_capture"] = oracle_gap_capture(
            candidate=hard_nmae,
            best_single=best_nmae,
            oracle=oracle_nmae,
        )
    except ValueError as exc:
        result["oracle_gap_capture_undefined_reason"] = str(exc)
    return result


__all__ = [
    "CaseEvidence",
    "DeployableAnchorDecision",
    "concatenate_case_evidence",
    "compute_anchor_decision",
    "evaluate_tensor_batch",
    "summarize_case_evidence",
]
