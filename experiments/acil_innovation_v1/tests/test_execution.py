from __future__ import annotations

import inspect
import sys
import pytest
import torch


def _inputs(flows: int = 2):
    truth = torch.arange(1.0, float(flows * 50 + 1)).reshape(1, flows, 50)
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., [0, 24, 49]] = True
    model_input = truth.clone()
    model_input[~observed] = float("nan")
    return truth, model_input, observed


def test_job_resolution_is_exact_and_cannot_inject_an_unplanned_identity() -> None:
    from experiments.acil_innovation_v1.execution import resolve_planned_job
    from experiments.acil_innovation_v1.jobs import planned_jobs

    expected = planned_jobs("stage_h")[2]
    assert resolve_planned_job("stage_h", expected.job_id) == expected
    with pytest.raises(ValueError, match="planned"):
        resolve_planned_job("stage_h", "0" * 64)
    with pytest.raises(ValueError, match="stage"):
        resolve_planned_job("not_a_stage", expected.job_id)


def test_resolver_rejects_future_formal_jobs_outside_active_v1_manifest() -> None:
    from experiments.acil_innovation_v1.execution import resolve_planned_job
    from experiments.acil_innovation_v1.jobs import planned_jobs

    future = planned_jobs("formal_gate")[0]
    with pytest.raises(ValueError, match="active manifest"):
        resolve_planned_job(future.stage, future.job_id)


def test_execute_job_preflights_upstream_dependency_before_creating_writer(
    tmp_path, monkeypatch
) -> None:
    from experiments.acil_innovation_v1 import adjudication, execution
    from experiments.acil_innovation_v1.jobs import planned_jobs

    job = planned_jobs("stage_h")[0]
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    writer_created = []
    monkeypatch.setattr(execution, "_active_freeze", lambda: ("a" * 64, "b" * 64))
    monkeypatch.setattr(execution.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(execution.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        adjudication,
        "load_stage_adjudication",
        lambda stage, output_root: {"stage": stage, "verdict": "proceed"},
    )
    monkeypatch.setattr(
        execution,
        "_load_acil_dependency",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("dependency invalid")),
    )
    monkeypatch.setattr(
        execution,
        "begin_job",
        lambda **kwargs: writer_created.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="dependency invalid"):
        execution.execute_job(job.stage, job.job_id, tmp_path)
    assert writer_created == []
    assert list(tmp_path.iterdir()) == []


def test_execute_job_requires_frozen_cublas_determinism_before_gpu_or_writer(
    tmp_path, monkeypatch
) -> None:
    from experiments.acil_innovation_v1 import execution
    from experiments.acil_innovation_v1.jobs import planned_jobs

    job = planned_jobs("stage0_acil_tune")[0]
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(execution, "_active_freeze", lambda: ("a" * 64, "b" * 64))
    monkeypatch.setattr(execution.torch.cuda, "is_available", lambda: True)
    writer_created = []
    monkeypatch.setattr(
        execution, "begin_job", lambda **kwargs: writer_created.append(kwargs)
    )

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        execution.execute_job(job.stage, job.job_id, tmp_path)
    assert writer_created == []
    assert list(tmp_path.iterdir()) == []


def test_worker_cli_exposes_only_job_identity_and_output_root() -> None:
    from experiments.acil_innovation_v1 import worker

    assert tuple(inspect.signature(worker.main).parameters) == ()
    destinations = {
        action.dest for action in worker._build_parser()._actions if action.dest != "help"
    }
    assert destinations == {"stage", "job_id", "output_root"}


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    (("succeeded", 0), ("algorithmic_failure", 20)),
)
def test_worker_exit_code_distinguishes_algorithmic_failure(
    tmp_path, monkeypatch, status: str, expected_exit: int
) -> None:
    from experiments.acil_innovation_v1 import worker
    from experiments.acil_innovation_v1.jobs import planned_jobs

    job = planned_jobs("stage0_acil_tune")[0]
    result_directory = tmp_path / job.stage / job.job_id
    monkeypatch.setattr(worker, "run_planned_job", lambda *_args: result_directory)
    monkeypatch.setattr(
        worker,
        "load_job_result",
        lambda **_kwargs: {"status": status},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--stage",
            job.stage,
            "--job-id",
            job.job_id,
            "--output-root",
            str(tmp_path),
        ],
    )

    assert worker.main() == expected_exit


def test_acil_worker_objective_accepts_nan_only_missing_model_payload() -> None:
    from experiments.acil_innovation_v1.acil import ACILBase
    from experiments.acil_innovation_v1.execution import acil_training_loss
    from experiments.acil_innovation_v1.preprocessing import FitFallback

    torch.manual_seed(401)
    truth, model_input, observed = _inputs()
    loss = acil_training_loss(
        ACILBase(),
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=FitFallback(mean=25.0, std=10.0),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_oracle_worker_objective_scores_only_e_and_not_q() -> None:
    from experiments.acil_innovation_v1.acil import ACILBase
    from experiments.acil_innovation_v1.execution import oracle_training_loss
    from experiments.acil_innovation_v1.oracle_model import (
        OracleSupportQ,
        TruthQDeepSets,
    )
    from experiments.acil_innovation_v1.preprocessing import FitFallback
    from experiments.acil_innovation_v1.training import normalized_target_mae

    torch.manual_seed(402)
    truth, model_input, observed = _inputs()
    q_indices = torch.full(truth.shape[:2], 10, dtype=torch.int64)
    q = OracleSupportQ(
        indices=q_indices,
        values=truth.gather(-1, q_indices.unsqueeze(-1)),
    )
    e = ~observed
    e.scatter_(-1, q_indices.unsqueeze(-1), False)
    model = TruthQDeepSets(ACILBase())
    fallback = FitFallback(mean=25.0, std=10.0)
    loss = oracle_training_loss(
        model,
        model_input=model_input,
        truth=truth,
        observed=observed,
        support_q=q,
        evaluation_target=e,
        fit_fallback=fallback,
    )
    prediction = model(model_input, observed, q, fallback)
    statistics = model.acil(model_input, observed, fallback).statistics
    expected = normalized_target_mae(prediction, truth, e, statistics.std)
    assert loss.item() == pytest.approx(expected.item())
    changed_q_truth = truth.clone()
    changed_q_truth.scatter_(-1, q_indices.unsqueeze(-1), 1e9)
    # Q truth used by the model remains compact/fixed; changing the loss-only
    # truth at Q cannot affect an E-only objective.
    changed = oracle_training_loss(
        model,
        model_input=model_input,
        truth=changed_q_truth,
        observed=observed,
        support_q=q,
        evaluation_target=e,
        fit_fallback=fallback,
    )
    assert changed.item() == pytest.approx(loss.item())


def test_deployable_residual_objective_scores_full_unobserved_complement() -> None:
    from experiments.acil_innovation_v1.acil import ACILBase
    from experiments.acil_innovation_v1.execution import residual_training_loss
    from experiments.acil_innovation_v1.models import QueryResidualModel
    from experiments.acil_innovation_v1.preprocessing import FitFallback
    from experiments.acil_innovation_v1.training import normalized_target_mae

    torch.manual_seed(403)
    truth, model_input, observed = _inputs()
    fallback = FitFallback(mean=25.0, std=10.0)
    model = QueryResidualModel("deepsets", ACILBase())
    loss = residual_training_loss(
        model,
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=fallback,
    )
    prediction = model(model_input, observed, fallback)
    scale = model.acil(model_input, observed, fallback).statistics.std
    expected = normalized_target_mae(prediction, truth, ~observed, scale)
    assert loss.item() == pytest.approx(expected.item())
