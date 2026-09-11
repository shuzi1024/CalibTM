from __future__ import annotations

import inspect


def test_stage_h_job_grid_is_exact_and_method_independent_of_masks() -> None:
    from experiments.acil_innovation_v1.jobs import planned_jobs

    jobs = planned_jobs("stage_h")
    assert len(jobs) == 6
    assert {(job.dataset, job.seed_bundle) for job in jobs} == {
        (dataset, seed) for dataset in ("abilene", "geant") for seed in (1, 2, 3)
    }
    assert {job.method for job in jobs} == {"truth_q_deepsets"}
    assert all(not hasattr(job, "mask_family") for job in jobs)


def test_formal_runner_has_no_data_or_split_override() -> None:
    from experiments.acil_innovation_v1.run_job import run_planned_job

    parameters = inspect.signature(run_planned_job).parameters
    assert tuple(parameters) == ("stage", "job_id", "output_root")


def test_mixed_mask_schedule_is_balanced_and_method_free() -> None:
    from experiments.acil_innovation_v1.jobs import training_family

    families = [training_family(epoch=e, epoch_order_position=p, seed_bundle=2)
                for e in range(20) for p in range(512)]
    counts = {name: families.count(name) for name in set(families)}
    assert set(counts) == {"random", "internal_block", "two_burst"}
    assert max(counts.values()) - min(counts.values()) == 1

