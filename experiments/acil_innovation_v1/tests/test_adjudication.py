from __future__ import annotations

import inspect
import pytest


def _record(
    method: str,
    *,
    seed: int,
    dataset: str,
    mask: str,
    start: int,
    flow: int,
    error: float,
    truth: float = 100.0,
    target_hash: str = "a" * 64,
):
    return {
        "identity": {
            "method": method,
            "seed_bundle": seed,
            "dataset": dataset,
            "mask_family": mask,
            "window_start": start,
            "flow": flow,
            "oracle": False,
        },
        "absolute_error_sum": error,
        "absolute_truth_sum": truth,
        "squared_error_sum": error * error,
        "squared_truth_sum": truth * truth,
        "target_count": 47,
        "target_set_sha256": target_hash,
    }


def _method_grid(
    errors,
    *,
    seeds=(1, 2, 3),
    datasets=("abilene", "geant"),
):
    rows = []
    for method, values in errors.items():
        for seed in seeds:
            for dataset in datasets:
                for mask in ("random", "internal_block", "two_burst"):
                    if isinstance(values, dict):
                        value = values.get((dataset, mask), values.get(mask))
                    else:
                        value = values
                    rows.append(
                        _record(
                            method,
                            seed=seed,
                            dataset=dataset,
                            mask=mask,
                            start=0,
                            flow=0,
                            error=float(value),
                        )
                    )
    return rows


def _patch_positive_bootstrap(monkeypatch, module) -> None:
    from experiments.acil_innovation_v1.statistics import BootstrapResult

    def bootstrap(*, stage, records, candidate, comparator, pooling_masks):
        point = module.paired_contrast(
            records,
            candidate=candidate,
            comparator=comparator,
            pooling_masks=pooling_masks,
        ).improvement
        return BootstrapResult(
            candidate=candidate,
            comparator=comparator,
            point_estimate=point,
            ci_lower=1e-3,
            ci_upper=point + 0.01,
            draws=10000,
            draws_sha256="a" * 64,
            draw_plan_sha256="b" * 64,
        )

    monkeypatch.setattr(module, "_bootstrap_contrast", bootstrap)


def test_paired_contrast_uses_ratio_of_sums_and_rejects_target_drift() -> None:
    from experiments.acil_innovation_v1.adjudication import paired_contrast

    records = [
        _record("base", seed=1, dataset="abilene", mask="internal_block", start=0,
                flow=0, error=20.0, truth=100.0),
        _record("candidate", seed=1, dataset="abilene", mask="internal_block", start=0,
                flow=0, error=15.0, truth=100.0),
        _record("base", seed=1, dataset="geant", mask="internal_block", start=0,
                flow=0, error=80.0, truth=900.0),
        _record("candidate", seed=1, dataset="geant", mask="internal_block", start=0,
                flow=0, error=75.0, truth=900.0),
    ]
    result = paired_contrast(
        records,
        candidate="candidate",
        comparator="base",
        pooling_masks=("internal_block",),
    )
    # Ratio-of-sums: 1 - (90/1000)/(100/1000), not a mean of cell NMAEs.
    assert result.improvement == pytest.approx(0.10)
    assert result.candidate_nmae == pytest.approx(0.09)
    assert result.comparator_nmae == pytest.approx(0.10)
    assert result.paired_record_count == 2

    records[-1]["target_set_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="target"):
        paired_contrast(
            records,
            candidate="candidate",
            comparator="base",
            pooling_masks=("internal_block",),
        )


def test_paired_contrast_rejects_missing_duplicate_and_denominator_drift() -> None:
    from experiments.acil_innovation_v1.adjudication import paired_contrast

    base = _record("base", seed=1, dataset="abilene", mask="two_burst", start=0,
                   flow=0, error=20.0)
    candidate = _record("candidate", seed=1, dataset="abilene", mask="two_burst",
                        start=0, flow=0, error=19.0)
    with pytest.raises(ValueError, match="paired"):
        paired_contrast(
            [base], candidate="candidate", comparator="base",
            pooling_masks=("two_burst",)
        )
    with pytest.raises(ValueError, match="duplicate"):
        paired_contrast(
            [base, base, candidate], candidate="candidate", comparator="base",
            pooling_masks=("two_burst",)
        )
    candidate["absolute_truth_sum"] = 99.0
    with pytest.raises(ValueError, match="denominator"):
        paired_contrast(
            [base, candidate], candidate="candidate", comparator="base",
            pooling_masks=("two_burst",)
        )


def test_strongest_comparator_and_positive_seed_rule_are_exact() -> None:
    from experiments.acil_innovation_v1.adjudication import (
        positive_seed_count,
        strongest_comparator,
    )

    records = []
    for seed, candidate_error in ((1, 8.0), (2, 9.0), (3, 12.0)):
        for dataset in ("abilene", "geant"):
            for mask in ("internal_block", "two_burst"):
                records.extend(
                    [
                        _record("acil", seed=seed, dataset=dataset, mask=mask,
                                start=0, flow=0, error=11.0),
                        _record("u0", seed=seed, dataset=dataset, mask=mask,
                                start=0, flow=0, error=10.0),
                        _record("full", seed=seed, dataset=dataset, mask=mask,
                                start=0, flow=0, error=candidate_error),
                    ]
                )
    selected = strongest_comparator(
        records,
        candidates=("acil", "u0"),
        pooling_masks=("internal_block", "two_burst"),
    )
    assert selected == "u0"
    count, effects = positive_seed_count(
        records,
        candidate="full",
        comparator="u0",
        pooling_masks=("internal_block", "two_burst"),
        seed_bundles=(1, 2, 3),
    )
    assert count == 2
    assert effects[1] == pytest.approx(0.2)
    assert effects[2] == pytest.approx(0.1)
    assert effects[3] == pytest.approx(-0.2)


def test_stage0_and_stage_h_gate_decisions_apply_every_registered_clause() -> None:
    from experiments.acil_innovation_v1.adjudication import decide_discovery_gate

    common = {
        "main": 0.06,
        "per_mask": {"internal_block": 0.04, "two_burst": 0.07},
        "ci_lower": 0.01,
        "positive_seed_count": 2,
    }
    stage0 = decide_discovery_gate("stage0_acil_tune", **common)
    assert stage0.verdict == "proceed"
    assert all(stage0.checks.values())

    stage_h = decide_discovery_gate("stage_h", random_effect=-0.005, **common)
    assert stage_h.verdict == "proceed"
    failed = decide_discovery_gate(
        "stage_h", random_effect=-0.02, **common
    )
    assert failed.verdict == "kill"
    assert failed.checks["random_no_harm"] is False

    with pytest.raises(ValueError, match="random"):
        decide_discovery_gate("stage_h", **common)


def test_public_discovery_adjudicator_has_no_grid_or_threshold_override() -> None:
    from experiments.acil_innovation_v1.adjudication import (
        adjudicate_discovery_stage,
        adjudicate_stage,
        load_stage_adjudication,
        write_stage_adjudication,
    )

    assert tuple(inspect.signature(adjudicate_discovery_stage).parameters) == (
        "stage",
        "output_root",
    )
    assert tuple(inspect.signature(adjudicate_stage).parameters) == (
        "stage",
        "output_root",
    )
    assert tuple(inspect.signature(write_stage_adjudication).parameters) == (
        "stage",
        "output_root",
    )
    assert tuple(inspect.signature(load_stage_adjudication).parameters) == (
        "stage",
        "output_root",
    )


def test_stage_i_gate_has_exact_proceed_revise_and_kill_regions() -> None:
    from experiments.acil_innovation_v1.adjudication import _decide_stage_i_gate

    passing = {
        "main": 0.015,
        "global_over_local": 0.005,
        "per_dataset": {"abilene": -0.005, "geant": 0.02},
        "ci_lower": 1e-12,
        "positive_seed_count": 2,
        "unshuffled_main_gain": 0.015,
        "derangement_gain_loss": 0.005,
        "derangement_fraction": 0.30,
    }
    decision = _decide_stage_i_gate(**passing)
    assert decision.verdict == "proceed"
    assert all(decision.checks.values())

    revise = _decide_stage_i_gate(
        **{
            **passing,
            "main": 0.01,
            "unshuffled_main_gain": 0.01,
            "global_over_local": 0.0,
        }
    )
    assert revise.verdict == "revise"
    assert revise.checks["main_minimum"] is False

    below_revision = _decide_stage_i_gate(
        **{**passing, "main": 0.004999, "unshuffled_main_gain": 0.004999}
    )
    assert below_revision.verdict == "kill"

    broken_derangement = _decide_stage_i_gate(
        **{**passing, "derangement_fraction": 0.299999}
    )
    assert broken_derangement.verdict == "kill"
    assert broken_derangement.checks["derangement_fraction"] is False


def test_full_tune_gate_separates_method_result_from_pretraining_claim() -> None:
    from experiments.acil_innovation_v1.adjudication import (
        _decide_full_tune_gate,
        _pretraining_claim,
    )

    operands = {
        "main": 0.02,
        "over_deepsets": 0.005,
        "pretraining_attribution": 0.006,
        "random_effect": -0.005,
        "ci_lower": 1e-12,
        "positive_seed_count": 2,
    }
    method = _decide_full_tune_gate(**operands)
    assert method.verdict == "proceed"
    assert all(method.checks.values())
    assert _pretraining_claim(
        effect=operands["pretraining_attribution"], ci_lower=0.01
    ) == "no_claim"
    assert _pretraining_claim(effect=0.01, ci_lower=1e-12) == "claim_supported"
    assert _pretraining_claim(effect=0.02, ci_lower=0.0) == "no_claim"

    harmed = _decide_full_tune_gate(
        **{**operands, "pretraining_attribution": -0.005001}
    )
    assert harmed.verdict == "kill"
    assert harmed.checks["pretraining_no_harm_floor"] is False


def test_formal_gate_applies_cells_external_no_harm_and_both_dataset_rules() -> None:
    from experiments.acil_innovation_v1.adjudication import _decide_formal_gate

    operands = {
        "main": 0.02,
        "per_dataset": {"abilene": -0.005, "geant": 0.03},
        "ci_lower": 1e-12,
        "positive_seed_count": 2,
        "random_effect": -0.005,
        "cell_wins": 4,
        "external_effect": -0.005,
        "external_per_dataset": {"abilene": -0.005, "geant": 0.01},
    }
    passing = _decide_formal_gate(**operands)
    assert passing.verdict == "proceed"
    assert all(passing.checks.values())

    for replacement, failed_check in (
        ({"cell_wins": 3}, "dataset_mask_cell_wins"),
        (
            {"external_per_dataset": {"abilene": -0.005001, "geant": 0.01}},
            "external_abilene_no_harm",
        ),
        (
            {"per_dataset": {"abilene": -0.005001, "geant": 0.03}},
            "abilene_no_harm",
        ),
    ):
        failed = _decide_formal_gate(**{**operands, **replacement})
        assert failed.verdict == "kill"
        assert failed.checks[failed_check] is False


def test_active_freeze_mismatch_is_integrity_kill_and_absence_is_blocked(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module

    monkeypatch.setattr(
        module,
        "_active_freeze_hashes",
        lambda: (_ for _ in ()).throw(FileNotFoundError("no freeze")),
        raising=False,
    )
    blocked = module.adjudicate_stage("stage_i", tmp_path)
    assert blocked["verdict"] == "blocked"
    assert blocked["reason"] == "active_freeze_missing"

    monkeypatch.setattr(
        module,
        "_active_freeze_hashes",
        lambda: (_ for _ in ()).throw(ValueError("anchor drift")),
        raising=False,
    )
    killed = module.adjudicate_stage("stage_i", tmp_path)
    assert killed["verdict"] == "kill"
    assert killed["reason"] == "active_freeze_integrity_failure"

    with pytest.raises(ValueError, match="unsupported"):
        module.adjudicate_stage("formal_gate", tmp_path)


def test_final_adjudication_artifact_has_fixed_path_and_recomputes_before_load(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module

    report = {
        "schema": "acil-innovation-v1:stage-adjudication:v2",
        "protocol": "acil-innovation-v1",
        "stage": "stage_i",
        "verdict": "proceed",
        "job_results": [{"result_sha256": "a" * 64}],
    }
    monkeypatch.setattr(module, "adjudicate_stage", lambda stage, output_root: report)
    path = module.write_stage_adjudication("stage_i", tmp_path)
    assert path == tmp_path / "_adjudication" / "stage_i" / "adjudication.json"
    assert module.load_stage_adjudication("stage_i", tmp_path) == report

    changed = {**report, "verdict": "kill"}
    monkeypatch.setattr(module, "adjudicate_stage", lambda stage, output_root: changed)
    with pytest.raises(ValueError, match="no longer matches"):
        module.load_stage_adjudication("stage_i", tmp_path)


def test_stage_i_recomputes_derangement_gain_loss_and_fraction_from_raw_sums(
    monkeypatch,
) -> None:
    import experiments.acil_innovation_v1.adjudication as module

    _patch_positive_bootstrap(monkeypatch, module)
    records = _method_grid({"acil": 10.0, "local_loo": 9.7, "global_loo": 9.0})
    deranged = _method_grid({"global_loo_deranged": 9.8})
    result = module._adjudicate_stage_i_records(records, deranged)
    assert result["verdict"] == "proceed"
    assert result["derangement"]["unshuffled_improvement_over_acil"] == pytest.approx(0.1)
    assert result["derangement"]["deranged_improvement_over_acil"] == pytest.approx(0.02)
    assert result["derangement"]["gain_loss"] == pytest.approx(0.08)
    assert result["derangement"]["fraction"] == pytest.approx(0.8)
    assert result["checks"]["derangement_fraction"] is True


def test_full_tune_selects_strongest_pool_specific_comparator_and_claim(
    monkeypatch,
) -> None:
    import experiments.acil_innovation_v1.adjudication as module

    _patch_positive_bootstrap(monkeypatch, module)
    records = _method_grid(
        {
            "acil": 10.0,
            "full_u0": 9.5,
            "loo_deepsets": 9.2,
            "full_scratch": {"random": 9.5, "internal_block": 9.1, "two_burst": 9.1},
            "full_gpt2": {"random": 9.4, "internal_block": 8.5, "two_burst": 8.5},
        }
    )
    result = module._adjudicate_full_tune_records(records)
    assert result["verdict"] == "proceed"
    assert result["main_comparator"] == "full_u0"
    assert result["random_comparator"] == "full_u0"
    assert result["over_deepsets"]["improvement"] > 0.005
    assert result["pretraining_attribution"]["claim"] == "claim_supported"


def test_formal_private_arithmetic_uses_dynamic_external_registry_and_four_cells(
    monkeypatch,
) -> None:
    import experiments.acil_innovation_v1.adjudication as module

    _patch_positive_bootstrap(monkeypatch, module)
    formal = module.load_protocol_config()["gates"]["formal_gate"]
    external_candidates = tuple(
        formal["effect_contrasts"]["external_no_harm"]["comparator_candidates"]
    )
    candidate_errors = {
        ("abilene", "random"): 9.0,
        ("abilene", "internal_block"): 9.0,
        ("abilene", "two_burst"): 9.0,
        ("geant", "random"): 9.0,
        ("geant", "internal_block"): 9.54,
        ("geant", "two_burst"): 9.54,
    }
    errors = {
        "acil": 10.0,
        "full_u0": 9.5,
        "full_gpt2": candidate_errors,
        external_candidates[0]: 9.6,
        external_candidates[1]: 9.55,
    }
    records = _method_grid(errors, seeds=(4, 5, 6))
    result = module._adjudicate_formal_records(records)
    assert result["verdict"] == "proceed"
    assert result["main_comparator"] == "full_u0"
    assert result["dataset_mask_cell_wins"] == 4
    assert result["external_comparator"] == external_candidates[1]
    assert set(result["external_per_dataset_comparators"].values()) == {
        external_candidates[1]
    }


def test_report_binds_verified_freeze_hash_and_stale_job_freeze_is_failure(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module
    from experiments.acil_innovation_v1.jobs import PlannedJob

    hashes = ("a" * 64, "b" * 64, "c" * 64)
    collect_stage_evidence = module._collect_stage_evidence
    monkeypatch.setattr(module, "_active_freeze_hashes", lambda: hashes)
    monkeypatch.setattr(
        module,
        "_collect_stage_evidence",
        lambda **kwargs: {
            "records": (),
            "deranged_records": (),
            "missing": [{"error": "missing_job_directory"}],
            "failures": [],
            "job_results": (),
            "infrastructure_attempts": (),
            "successful_jobs": 0,
        },
    )
    report = module.adjudicate_stage("stage_i", tmp_path)
    assert report["verdict"] == "blocked"
    assert report["manifest_sha256"] == hashes[0]
    assert report["provenance_sha256"] == hashes[1]
    assert report["freeze_sha256"] == hashes[2]
    monkeypatch.setattr(module, "_collect_stage_evidence", collect_stage_evidence)

    job = PlannedJob(
        stage="stage_i",
        method="global_loo",
        dataset="abilene",
        seed_bundle=1,
        job_id="d" * 64,
    )
    (tmp_path / "stage_i" / job.job_id).mkdir(parents=True)
    monkeypatch.setattr(module, "planned_jobs", lambda stage: (job,))
    monkeypatch.setattr(
        module,
        "load_job_result",
        lambda **kwargs: {
            "manifest_sha256": "e" * 64,
            "provenance_sha256": hashes[1],
            "status": "succeeded",
        },
    )
    stale = module._collect_stage_evidence(
        stage="stage_i",
        output_root=tmp_path,
        manifest_sha256=hashes[0],
        provenance_sha256=hashes[1],
    )
    assert stale["failures"][0]["status"] == "integrity_failure"
    assert "active freeze" in stale["failures"][0]["error"]


def test_infrastructure_attempts_are_semantically_validated_without_reading_partials(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module
    from pathlib import Path
    from experiments.acil_innovation_v1.tests.test_retry import _initial_archive

    job, attempt = _initial_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module,
        "_active_freeze_hashes",
        lambda: ("b" * 64, "c" * 64, "d" * 64),
    )
    partial = attempt / "partial.bin"
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == partial:
            raise AssertionError("adjudication read a partial metric/loss file")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    rows, failures = module._inventory_infrastructure_attempts(
        stage=job.stage, output_root=tmp_path, jobs=(job,)
    )
    assert failures == []
    assert len(rows) == 1
    assert rows[0]["incident"]["state"] == "failed"
    partial_rows = [row for row in rows[0]["files"] if row["path"] == "partial.bin"]
    assert partial_rows == [
        {"bytes": len(b"first failure"), "path": "partial.bin", "sha256": None}
    ]
    assert len(rows[0]["attempt_sha256"]) == 64


def test_infrastructure_semantic_chain_rejects_incident_command_tamper(
    monkeypatch, tmp_path
) -> None:
    import json
    import experiments.acil_innovation_v1.adjudication as module
    from experiments.acil_innovation_v1.tests.test_retry import _initial_archive

    job, attempt = _initial_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module,
        "_active_freeze_hashes",
        lambda: ("b" * 64, "c" * 64, "d" * 64),
    )
    incident_path = attempt / "incident.json"
    incident = json.loads(incident_path.read_text(encoding="ascii"))
    incident["command"] = ["python", "forged.py"]
    incident_path.write_text(
        json.dumps(incident, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    rows, failures = module._inventory_infrastructure_attempts(
        stage=job.stage, output_root=tmp_path, jobs=(job,)
    )
    assert rows == []
    assert len(failures) == 1
    assert failures[0]["status"] == "integrity_failure"
    assert "semantic" in failures[0]["error"] or "command" in failures[0]["error"]


def test_launch_failed_retry_tail_is_valid_operational_evidence_not_method_failure(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module
    import experiments.acil_innovation_v1.retry as retry
    from experiments.acil_innovation_v1.tests.test_retry import (
        _initial_archive,
        _patch_freeze,
    )

    job, _ = _initial_archive(tmp_path, monkeypatch)
    _patch_freeze(monkeypatch, retry)
    monkeypatch.setattr(
        module,
        "_active_freeze_hashes",
        lambda: ("b" * 64, "c" * 64, "d" * 64),
    )

    def fail_to_launch(*args, **kwargs):
        raise OSError("synthetic launch failure")

    monkeypatch.setattr(retry._subprocess, "Popen", fail_to_launch)
    summary_path = retry.retry_infrastructure_job(
        job.stage, job.job_id, tmp_path, "0"
    )
    assert '"state":"launch_failed"' in summary_path.read_text(encoding="ascii")

    rows, failures = module._inventory_infrastructure_attempts(
        stage=job.stage, output_root=tmp_path, jobs=(job,)
    )
    assert failures == []
    assert rows[0]["retry_tail"]["state"] == "launch_failed"
    report = module.adjudicate_stage(job.stage, tmp_path)
    assert report["verdict"] == "blocked"
    assert report["failures"] == []


def test_tempfail_without_orphan_is_audited_but_stage_remains_blocked(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module
    import experiments.acil_innovation_v1.retry as retry
    from experiments.acil_innovation_v1.tests.test_retry import (
        _initial_archive,
        _patch_freeze,
    )

    job, _ = _initial_archive(tmp_path, monkeypatch)
    _patch_freeze(monkeypatch, retry)
    monkeypatch.setattr(
        module,
        "_active_freeze_hashes",
        lambda: ("b" * 64, "c" * 64, "d" * 64),
    )

    class NoOrphanProcess:
        def __init__(self, command, **kwargs):
            kwargs["stdout"].write(b"temporary failure before begin_job\n")
            kwargs["stdout"].flush()
            self.returncode = 75

        def wait(self):
            return self.returncode

    monkeypatch.setattr(retry._subprocess, "Popen", NoOrphanProcess)
    retry.retry_infrastructure_job(job.stage, job.job_id, tmp_path, "0")
    rows, failures = module._inventory_infrastructure_attempts(
        stage=job.stage, output_root=tmp_path, jobs=(job,)
    )
    assert failures == []
    assert rows[0]["retry_tail"]["state"] == "failed_without_orphan"
    assert rows[0]["retry_tail"]["failure_kind"] == "ex_tempfail"
    report = module.adjudicate_stage(job.stage, tmp_path)
    assert report["verdict"] == "blocked"
    assert report["failures"] == []


def test_algorithmic_failure_is_kill_even_if_another_job_is_missing(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module

    monkeypatch.setattr(
        module, "_active_freeze_hashes", lambda: ("a" * 64, "b" * 64, "c" * 64)
    )
    monkeypatch.setattr(
        module,
        "_direct_upstream_binding",
        lambda stage, output_root: {"stage": "stage_i", "verdict": "proceed"},
    )
    monkeypatch.setattr(
        module,
        "_collect_stage_evidence",
        lambda **kwargs: {
            "records": (),
            "deranged_records": (),
            "missing": [{"error": "missing_job_directory"}],
            "failures": [{"status": "algorithmic_failure", "failure": {"type": "NaN"}}],
            "job_results": (),
            "infrastructure_attempts": (),
            "successful_jobs": 0,
        },
    )
    result = module.adjudicate_stage("full_tune", tmp_path)
    assert result["verdict"] == "kill"
    assert result["reason"] == "failed_or_invalid_registered_job"


def test_inprogress_only_is_blocked_and_is_reported_in_inventory(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module
    from experiments.acil_innovation_v1.jobs import PlannedJob

    job = PlannedJob(
        stage="stage_i",
        method="global_loo",
        dataset="abilene",
        seed_bundle=1,
        job_id="a" * 64,
    )
    inprogress = tmp_path / "stage_i" / f".{job.job_id}.inprogress"
    inprogress.mkdir(parents=True)
    monkeypatch.setattr(module, "planned_jobs", lambda stage: (job,))
    evidence = module._collect_stage_evidence(
        stage="stage_i",
        output_root=tmp_path,
        manifest_sha256="b" * 64,
        provenance_sha256="c" * 64,
    )
    assert evidence["failures"] == []
    assert evidence["missing"] == [
        {"job": job.to_json(), "error": "job_still_in_progress"}
    ]
    assert evidence["inprogress_jobs"] == (
        {
            "job": job.to_json(),
            "path": f"stage_i/.{job.job_id}.inprogress",
            "state": "in_progress_without_final",
        },
    )


def test_final_plus_inprogress_is_integrity_kill_not_a_success(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module
    from experiments.acil_innovation_v1.jobs import PlannedJob

    job = PlannedJob(
        stage="full_tune",
        method="full_gpt2",
        dataset="geant",
        seed_bundle=3,
        job_id="d" * 64,
    )
    stage = tmp_path / "full_tune"
    (stage / job.job_id).mkdir(parents=True)
    (stage / f".{job.job_id}.inprogress").mkdir()
    monkeypatch.setattr(module, "planned_jobs", lambda name: (job,))
    evidence = module._collect_stage_evidence(
        stage="full_tune",
        output_root=tmp_path,
        manifest_sha256="e" * 64,
        provenance_sha256="f" * 64,
    )
    assert evidence["successful_jobs"] == 0
    assert evidence["missing"] == []
    assert evidence["failures"] == [
        {
            "job": job.to_json(),
            "status": "integrity_failure",
            "error": "final_and_inprogress_both_exist",
        }
    ]
    assert evidence["inprogress_jobs"][0]["state"] == "in_progress_alongside_final"


def test_public_adjudicator_never_proceeds_with_inprogress_inventory(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module

    monkeypatch.setattr(
        module, "_active_freeze_hashes", lambda: ("a" * 64, "b" * 64, "c" * 64)
    )
    monkeypatch.setattr(
        module,
        "_direct_upstream_binding",
        lambda stage, output_root: {"stage": "stage_h", "verdict": "proceed"},
    )
    monkeypatch.setattr(
        module,
        "_collect_stage_evidence",
        lambda **kwargs: {
            "records": (),
            "deranged_records": (),
            "missing": [],
            "failures": [],
            "job_results": (),
            "infrastructure_attempts": (),
            "inprogress_jobs": (
                {"job": {"job_id": "d" * 64}, "state": "in_progress_without_final"},
            ),
            "successful_jobs": 0,
        },
    )
    result = module.adjudicate_stage("stage_i", tmp_path)
    assert result["verdict"] == "blocked"
    assert result["reason"] == "registered_job_in_progress"
    assert len(result["inprogress_jobs"]) == 1


def test_downstream_requires_verified_direct_upstream_proceed_artifact(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module

    monkeypatch.setattr(
        module, "_active_freeze_hashes", lambda: ("a" * 64, "b" * 64, "c" * 64)
    )
    monkeypatch.setattr(
        module,
        "_direct_upstream_binding",
        lambda stage, output_root: (_ for _ in ()).throw(
            FileNotFoundError("upstream adjudication absent")
        ),
        raising=False,
    )
    missing = module.adjudicate_stage("stage_i", tmp_path)
    assert missing["verdict"] == "blocked"
    assert missing["reason"] == "direct_upstream_adjudication_missing"

    monkeypatch.setattr(
        module,
        "_direct_upstream_binding",
        lambda stage, output_root: (_ for _ in ()).throw(
            ValueError("upstream artifact drift")
        ),
        raising=False,
    )
    invalid = module.adjudicate_stage("stage_i", tmp_path)
    assert invalid["verdict"] == "kill"
    assert invalid["reason"] == "direct_upstream_integrity_failure"

    binding = {
        "stage": "stage_h",
        "path": "_adjudication/stage_h/adjudication.json",
        "sha256": "d" * 64,
        "verdict": "kill",
    }
    monkeypatch.setattr(
        module, "_direct_upstream_binding", lambda stage, output_root: binding
    )
    stopped = module.adjudicate_stage("stage_i", tmp_path)
    assert stopped["verdict"] == "blocked"
    assert stopped["reason"] == "direct_upstream_did_not_proceed"
    assert stopped["direct_upstream"] == binding


def test_direct_upstream_binding_hashes_the_revalidated_canonical_artifact(
    monkeypatch, tmp_path
) -> None:
    import experiments.acil_innovation_v1.adjudication as module
    from experiments.acil_innovation_v1.result_io import write_canonical_json_exclusive

    upstream = {
        "schema": "acil-innovation-v1:stage-adjudication:v2",
        "stage": "stage_h",
        "verdict": "proceed",
        "freeze_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "provenance_sha256": "c" * 64,
    }
    root = tmp_path / "_adjudication" / "stage_h"
    root.mkdir(parents=True)
    artifact = write_canonical_json_exclusive(root / "adjudication.json", upstream)
    monkeypatch.setattr(
        module, "load_stage_adjudication", lambda stage, output_root: upstream
    )
    binding = module._direct_upstream_binding("stage_i", tmp_path)
    assert binding == {
        "stage": "stage_h",
        "path": "_adjudication/stage_h/adjudication.json",
        "sha256": artifact.sha256,
        "verdict": "proceed",
        "freeze_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "provenance_sha256": "c" * 64,
    }


def test_stage0_has_no_direct_upstream(monkeypatch, tmp_path) -> None:
    import experiments.acil_innovation_v1.adjudication as module

    monkeypatch.setattr(
        module, "_active_freeze_hashes", lambda: ("a" * 64, "b" * 64, "c" * 64)
    )
    monkeypatch.setattr(
        module,
        "_collect_stage_evidence",
        lambda **kwargs: {
            "records": (),
            "deranged_records": (),
            "missing": [{"error": "missing_job_directory"}],
            "failures": [],
            "job_results": (),
            "infrastructure_attempts": (),
            "inprogress_jobs": (),
            "successful_jobs": 0,
        },
    )
    result = module.adjudicate_stage("stage0_acil_tune", tmp_path)
    assert result["direct_upstream"] is None
