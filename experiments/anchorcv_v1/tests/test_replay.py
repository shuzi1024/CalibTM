from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.acil_innovation_v1.batching import EvaluationBatch
from experiments.acil_innovation_v1.preprocessing import FitFallback
from experiments.acil_innovation_v1.registries import seed_bundle
from experiments.anchorcv_v1.evaluation import CaseEvidence
from experiments.anchorcv_v1.protocol import load_protocol
from experiments.anchorcv_v1.replay import (
    replay_and_verify_tune_evidence,
    replay_tune_evidence,
    verify_replayed_evidence,
)


def _evidence(
    *,
    p_error: float = 2.0,
    n_error: float = 1.0,
    p_anchor: float = 2.0,
    n_anchor: float = 1.0,
) -> CaseEvidence:
    scale = 1.0
    target_count = 4
    winner_is_n = p_anchor > n_anchor
    oracle_is_n = p_error > n_error
    return CaseEvidence(
        window_index=np.array([0], dtype=np.int64),
        flow_index=np.array([0], dtype=np.int64),
        middle_anchor_index=np.array([3], dtype=np.int64),
        p_error_sum=np.array([p_error]),
        n_error_sum=np.array([n_error]),
        hard_error_sum=np.array([n_error if winner_is_n else p_error]),
        oracle_error_sum=np.array([min(p_error, n_error)]),
        truth_sum=np.array([10.0]),
        target_count=np.array([target_count], dtype=np.int64),
        k3_scale=np.array([scale]),
        p_anchor_absolute_error=np.array([p_anchor]),
        n_anchor_absolute_error=np.array([n_anchor]),
        p_anchor_normalized_error=np.array([p_anchor / scale]),
        n_anchor_normalized_error=np.array([n_anchor / scale]),
        loo_score=np.array([(p_anchor - n_anchor) / scale]),
        target_regret=np.array(
            [(p_error - n_error) / (target_count * scale)]
        ),
        hard_winner_is_n=np.array([winner_is_n], dtype=np.bool_),
        oracle_winner_is_n=np.array([oracle_is_n], dtype=np.bool_),
    )


def _grid(evidence: CaseEvidence | None = None) -> dict[str, CaseEvidence]:
    item = _evidence() if evidence is None else evidence
    return {family: item for family in load_protocol().mask_families}


def _batch(family: str) -> EvaluationBatch:
    truth = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., (0, 3, 6)] = True
    model_input = torch.where(
        observed, truth, torch.full_like(truth, float("nan"))
    )
    return EvaluationBatch(
        dataset="abilene",
        cohort="tune",
        seed_bundle=1,
        family=family,
        window_indices=(0,),
        absolute_starts=(0,),
        mask_identities=tuple(),
        mask_sha256=(("a" * 64,),),
        oracle_q_sha256=(("b" * 64,),),
        oracle_e_sha256=(("c" * 64,),),
        truth=truth,
        model_input=model_input,
        observed=observed,
        target=~observed,
        controlled_gap=torch.zeros_like(observed),
        oracle_q=torch.zeros_like(observed),
        oracle_e=~observed,
        oracle_support_q=None,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("family", "tampered"),
    (
        ("random", _evidence(p_error=3.0)),
        ("internal_block", _evidence(n_error=0.5)),
        ("two_burst", _evidence(p_anchor=3.0)),
    ),
)
def test_verify_replayed_evidence_rejects_any_expert_or_anchor_drift(
    family: str,
    tampered: CaseEvidence,
) -> None:
    persisted = _grid()
    replayed = _grid()
    replayed[family] = tampered

    with pytest.raises(ValueError, match=rf"{family}.*bitwise"):
        verify_replayed_evidence(persisted=persisted, replayed=replayed)


def test_verify_replayed_evidence_requires_and_accepts_all_three_families() -> None:
    persisted = _grid()

    verify_replayed_evidence(persisted=persisted, replayed=_grid())
    replayed = _grid()
    replayed.pop("two_burst")
    with pytest.raises(ValueError, match="exact frozen mask families"):
        verify_replayed_evidence(persisted=persisted, replayed=replayed)


def test_replay_tune_evidence_uses_fixed_cuda_math_lane_and_all_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    expected_tensor_sha = "a" * 64
    expected_prior_sha = "b" * 64

    class FakeModel:
        training = False

        def load_state_dict(self, state, *, strict):
            events.append(("load", state, strict))
            return self

        def requires_grad_(self, enabled):
            events.append(("grad", enabled))
            return self

        def eval(self):
            events.append("eval")
            return self

        def to(self, device):
            events.append(("to", device.type))
            return self

        def parameters(self):
            return ()

    monkeypatch.setattr(
        "experiments.anchorcv_v1.replay.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.replay.seed_everything",
        lambda value, deterministic: events.append(
            ("seed", value, deterministic)
        ),
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.replay.scientific_tensor_sha256",
        lambda _state: expected_tensor_sha,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.replay.build_neural_expert",
        lambda dataset: events.append(("model", dataset)) or FakeModel(),
    )
    fake_prior = SimpleNamespace(training=False, parameters=lambda: ())
    monkeypatch.setattr(
        "experiments.anchorcv_v1.replay.load_frozen_acil",
        lambda dataset, bundle, *, device: (
            events.append(("prior", dataset, bundle, device.type))
            or fake_prior,
            SimpleNamespace(file_sha256=expected_prior_sha),
        ),
    )

    def fake_evaluate(*, prior, neural, batch, fit_fallback, device, chunk_size):
        events.append(
            (
                "evaluate",
                batch.family,
                prior is fake_prior,
                isinstance(neural, FakeModel),
                fit_fallback,
                device.type,
                chunk_size,
            )
        )
        return _evidence()

    monkeypatch.setattr(
        "experiments.anchorcv_v1.replay.evaluate_evaluation_batch",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "experiments.anchorcv_v1.replay.torch.cuda.synchronize",
        lambda device: events.append(("sync", device.type)),
    )
    batches = {
        family: _batch(family) for family in load_protocol().mask_families
    }
    fallback = FitFallback(mean=1.0, std=2.0)
    state = {"weight": torch.ones(1)}

    replayed = replay_tune_evidence(
        dataset="abilene",
        seed_bundle_id=1,
        neural_state=state,
        expected_neural_tensor_sha256=expected_tensor_sha,
        expected_prior_file_sha256=expected_prior_sha,
        registered_batches=batches,
        fit_fallback=fallback,
    )

    assert tuple(replayed) == load_protocol().mask_families
    assert (
        "seed",
        seed_bundle(1).model,
        True,
    ) in events
    assert ("grad", False) in events
    assert [event[1] for event in events if event[0] == "evaluate"] == list(
        load_protocol().mask_families
    )
    assert all(
        event[-2:] == ("cuda", 8)
        for event in events
        if event[0] == "evaluate"
    )


def test_replay_tune_evidence_fails_closed_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiments.anchorcv_v1.replay.torch.cuda.is_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        replay_tune_evidence(
            dataset="abilene",
            seed_bundle_id=1,
            neural_state={"weight": torch.ones(1)},
            expected_neural_tensor_sha256="a" * 64,
            expected_prior_file_sha256="b" * 64,
            registered_batches={
                family: _batch(family)
                for family in load_protocol().mask_families
            },
            fit_fallback=FitFallback(mean=1.0, std=2.0),
        )


@pytest.mark.parametrize("tamper", ("checkpoint", "prior", "evidence"))
def test_replay_and_verify_binds_checkpoint_prior_and_persisted_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_replay(**kwargs):
        captured.append(kwargs)
        if tamper == "checkpoint":
            assert kwargs["expected_neural_tensor_sha256"] == "wrong"
        if tamper == "prior":
            assert kwargs["expected_prior_file_sha256"] == "wrong"
        if tamper == "evidence":
            replayed = _grid()
            replayed["random"] = _evidence(p_error=3.0)
            return replayed
        return _grid()

    monkeypatch.setattr(
        "experiments.anchorcv_v1.replay.replay_tune_evidence",
        fake_replay,
    )
    checkpoint_sha = "wrong" if tamper == "checkpoint" else "a" * 64
    prior_sha = "wrong" if tamper == "prior" else "b" * 64

    if tamper == "evidence":
        with pytest.raises(ValueError, match="random.*bitwise"):
            replay_and_verify_tune_evidence(
                dataset="abilene",
                seed_bundle_id=1,
                neural_state={"weight": torch.ones(1)},
                expected_neural_tensor_sha256=checkpoint_sha,
                expected_prior_file_sha256=prior_sha,
                registered_batches={
                    family: _batch(family)
                    for family in load_protocol().mask_families
                },
                fit_fallback=FitFallback(mean=1.0, std=2.0),
                persisted=_grid(),
            )
    else:
        replay_and_verify_tune_evidence(
            dataset="abilene",
            seed_bundle_id=1,
            neural_state={"weight": torch.ones(1)},
            expected_neural_tensor_sha256=checkpoint_sha,
            expected_prior_file_sha256=prior_sha,
            registered_batches={
                family: _batch(family)
                for family in load_protocol().mask_families
            },
            fit_fallback=FitFallback(mean=1.0, std=2.0),
            persisted=_grid(),
        )
    assert len(captured) == 1
    assert captured[0]["expected_neural_tensor_sha256"] == checkpoint_sha
    assert captured[0]["expected_prior_file_sha256"] == prior_sha
