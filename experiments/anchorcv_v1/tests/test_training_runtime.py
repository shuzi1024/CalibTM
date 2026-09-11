from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.acil_innovation_v1.batching import EvaluationBatch, TrainingBatch
from experiments.acil_innovation_v1.preprocessing import FitFallback
from experiments.anchorcv_v1.training_runtime import (
    build_neural_expert,
    compute_batch_loss,
    evaluate_neural_error_sums,
    model_parameter_count,
    seed_everything,
)


def _training_batch() -> TrainingBatch:
    truth = torch.arange(1, 1 + 2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7)
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., (0, 3, 6)] = True
    return TrainingBatch(
        dataset="abilene",
        cohort="fit",
        seed_bundle=1,
        epoch=0,
        batch_index=0,
        epoch_order_positions=(0, 1),
        window_indices=(0, 1),
        absolute_starts=(0, 50),
        families=("random", "random"),
        mask_identities=tuple(),
        mask_sha256=tuple(),
        oracle_q_sha256=tuple(),
        oracle_e_sha256=tuple(),
        truth=truth,
        model_input=torch.where(observed, truth, torch.full_like(truth, float("nan"))),
        observed=observed,
        target=~observed,
        controlled_gap=torch.zeros_like(observed),
        oracle_q=torch.zeros_like(observed),
        oracle_e=~observed,
        oracle_support_q=None,  # type: ignore[arg-type]
    )


def _evaluation_batch() -> EvaluationBatch:
    training = _training_batch()
    return EvaluationBatch(
        dataset=training.dataset,
        cohort="source_dev",
        seed_bundle=training.seed_bundle,
        family="random",
        window_indices=training.window_indices,
        absolute_starts=training.absolute_starts,
        mask_identities=tuple(),
        mask_sha256=tuple(),
        oracle_q_sha256=tuple(),
        oracle_e_sha256=tuple(),
        truth=training.truth,
        model_input=training.model_input,
        observed=training.observed,
        target=training.target,
        controlled_gap=training.controlled_gap,
        oracle_q=training.oracle_q,
        oracle_e=training.oracle_e,
        oracle_support_q=None,  # type: ignore[arg-type]
    )


def test_seed_everything_recreates_model_bits() -> None:
    seed_everything(41001, deterministic=True)
    first = build_neural_expert("abilene", num_flows_override=3, time_steps_override=7, tiny=True)
    first_state = {name: value.detach().clone() for name, value in first.state_dict().items()}

    seed_everything(41001, deterministic=True)
    second = build_neural_expert("abilene", num_flows_override=3, time_steps_override=7, tiny=True)

    assert first_state.keys() == second.state_dict().keys()
    for name, value in second.state_dict().items():
        torch.testing.assert_close(value, first_state[name], rtol=0.0, atol=0.0)


def test_deterministic_seed_forces_math_attention_backend() -> None:
    seed_everything(41001, deterministic=True)

    assert torch.backends.cuda.math_sdp_enabled()
    assert not torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()


def test_formal_architecture_and_parameter_counts_are_dataset_bound() -> None:
    seed_everything(1, deterministic=True)
    abilene = build_neural_expert("abilene")
    geant = build_neural_expert("geant")

    assert abilene.num_flows == 144
    assert geant.num_flows == 462
    assert abilene.time_steps == geant.time_steps == 50
    assert model_parameter_count(geant) - model_parameter_count(abilene) == (462 - 144) * 256
    assert 3_000_000 < model_parameter_count(abilene) < 5_000_000


def test_compute_batch_loss_is_finite_and_uses_frozen_rank_schedule() -> None:
    seed_everything(3, deterministic=True)
    model = build_neural_expert(
        "abilene", num_flows_override=3, time_steps_override=7, tiny=True
    )
    batch = _training_batch()

    loss, details = compute_batch_loss(
        model,
        batch,
        fit_fallback=FitFallback(mean=10.0, std=5.0),
        device=torch.device("cpu"),
    )

    assert torch.isfinite(loss)
    assert details.drop_rank == 1
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_source_dev_error_sums_use_only_fixed_missing_complement() -> None:
    seed_everything(5, deterministic=True)
    model = build_neural_expert(
        "abilene", num_flows_override=3, time_steps_override=7, tiny=True
    ).eval()
    batch = _evaluation_batch()

    result = evaluate_neural_error_sums(
        model,
        batch,
        fit_fallback=FitFallback(mean=10.0, std=5.0),
        device=torch.device("cpu"),
        chunk_size=1,
    )

    expected_truth = float(batch.truth.masked_select(batch.target).abs().sum())
    assert result["absolute_truth_sum"] == pytest.approx(expected_truth)
    assert result["target_count"] == int(batch.target.sum())
    assert np.isfinite(result["absolute_error_sum"])
    assert result["absolute_error_sum"] >= 0.0
    assert result["nmae"] == pytest.approx(
        result["absolute_error_sum"] / result["absolute_truth_sum"]
    )
