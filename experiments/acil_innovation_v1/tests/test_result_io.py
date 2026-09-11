from __future__ import annotations

import copy
from dataclasses import replace
import inspect
import json

import pytest
import torch


_REGISTERED_MODEL_FIXTURES: dict[tuple[str, str, str, int], torch.nn.Module] = {}


def _metric_row(identity: dict[str, object], error: float = 1.0) -> dict[str, object]:
    return {
        "identity": identity,
        "absolute_error_sum": error,
        "absolute_truth_sum": 10.0,
        "squared_error_sum": error * error,
        "squared_truth_sum": 100.0,
        "target_count": 47,
        "target_set_sha256": "a" * 64,
    }


def _checkpoint_identity(
    job,
    *,
    manifest_sha256: str,
    provenance_sha256: str,
    base_checkpoint: dict[str, object] | None = None,
) -> dict[str, object]:
    from experiments.acil_innovation_v1.config import protocol_config_sha256

    identity: dict[str, object] = {
        "protocol": "acil-innovation-v1",
        "config_sha256": protocol_config_sha256(),
        "stage": job.stage,
        "job_id": job.job_id,
        "method": job.method,
        "dataset": job.dataset,
        "seed_bundle": job.seed_bundle,
        "epoch": 3,
        "source_dev_nmae": 0.25,
        "manifest_sha256": manifest_sha256,
        "provenance_sha256": provenance_sha256,
    }
    if base_checkpoint is not None:
        identity["base_checkpoint"] = dict(base_checkpoint)
    return identity


def _dependency_binding(
    source_job,
    checkpoint,
    *,
    source_method: str | None = None,
) -> dict[str, object]:
    binding: dict[str, object] = {
        "source_stage": source_job.stage,
        "source_job_id": source_job.job_id,
        "checkpoint_identity_sha256": checkpoint.identity_sha256,
        "checkpoint_file_sha256": checkpoint.file_sha256,
        "checkpoint_tensor_sha256": checkpoint.tensor_sha256,
    }
    if source_method is not None:
        binding["source_method"] = source_method
    return binding


def _matching_job(stage: str, method: str, dataset: str, seed_bundle: int):
    from experiments.acil_innovation_v1.jobs import planned_jobs

    matches = tuple(
        job
        for job in planned_jobs(stage)
        if job.method == method
        and job.dataset == dataset
        and job.seed_bundle == seed_bundle
    )
    assert len(matches) == 1
    return matches[0]


def _registered_model_fixture(job) -> torch.nn.Module:
    """Reuse real registered architectures while keeping dependency tests small."""

    from experiments.acil_innovation_v1.model_factory import build_registered_model

    key = (job.stage, job.method, job.dataset, job.seed_bundle)
    model = _REGISTERED_MODEL_FIXTURES.get(key)
    if model is None:
        model = build_registered_model(job)
        if job.stage != "stage0_acil_tune":
            source = _matching_job(
                "stage0_acil_tune", "acil", job.dataset, job.seed_bundle
            )
            source_model = _registered_model_fixture(source)
            acil = getattr(model, "acil", None)
            assert acil is not None
            acil.load_state_dict(source_model.state_dict(), strict=True)
        _REGISTERED_MODEL_FIXTURES[key] = model
    return model


def _training_payload(job, *, best_epoch: int = 3) -> dict[str, object]:
    from experiments.acil_innovation_v1.config import protocol_config_sha256

    epochs = []
    for epoch in range(20):
        target_nmae = 0.25 if epoch == best_epoch else 0.5 + epoch / 100.0
        error = target_nmae * 100.0
        nmae = error / 100.0
        epochs.append(
            {
                "epoch": epoch,
                "physical_batches": 64,
                "optimizer_updates": 16,
                "mean_loss": 1.0 / (epoch + 1),
                "source_dev_absolute_error_sum": error,
                "source_dev_absolute_truth_sum": 100.0,
                "source_dev_nmae": nmae,
                "selected_as_best": epoch == best_epoch,
            }
        )
    return {
        "schema": "acil-innovation-v1:training-history:v1",
        "protocol": "acil-innovation-v1",
        "config_sha256": protocol_config_sha256(),
        "job": job.to_json(),
        "fit_fallback": {"mean": 1.0, "std": 2.0},
        "epochs_completed": 20,
        "optimizer_updates": 320,
        "best_epoch": best_epoch,
        "best_source_dev_nmae": 0.25,
        "epochs": epochs,
    }


def _finalize_synthetic_job(
    *,
    job,
    output_root,
    manifest_sha256: str,
    provenance_sha256: str,
    identity: dict[str, object],
    dependencies: dict[str, dict[str, object]] | None = None,
):
    from experiments.acil_innovation_v1.checkpoint import save_checkpoint
    from experiments.acil_innovation_v1.result_io import (
        begin_job,
        write_canonical_json_exclusive,
    )

    writer = begin_job(
        stage=job.stage,
        job_id=job.job_id,
        output_root=output_root,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )
    checkpoint = save_checkpoint(
        _registered_model_fixture(job),
        identity=identity,
        directory=writer.checkpoint_directory,
    )
    writer.register_checkpoint(checkpoint)
    training = write_canonical_json_exclusive(
        writer.staging_directory / "training.json", _training_payload(job)
    )
    writer.register_artifact("training", training.path)
    if job.stage == "stage_i" and job.method == "global_loo":
        plans = write_canonical_json_exclusive(
            writer.staging_directory / "derangement_plans.json", {"plans": []}
        )
        writer.register_artifact("derangement_plans", plans.path)
        records = write_canonical_json_exclusive(
            writer.staging_directory / "derangement_records.jsonl", {"records": []}
        )
        writer.register_artifact("derangement_records", records.path)
    if dependencies is not None:
        artifact = write_canonical_json_exclusive(
            writer.staging_directory / "dependencies.json",
            {
                "schema": "acil-innovation-v1:job-dependencies:v1",
                "job": job.to_json(),
                "dependencies": dependencies,
            },
        )
        writer.register_artifact("dependencies", artifact.path)
    final = writer.succeed()
    return final, checkpoint


def test_public_job_api_has_no_data_split_or_cohort_override() -> None:
    from experiments.acil_innovation_v1.result_io import (
        JobResultWriter,
        begin_job,
        load_job_result,
    )

    assert tuple(inspect.signature(begin_job).parameters) == (
        "stage",
        "job_id",
        "output_root",
        "manifest_sha256",
        "provenance_sha256",
    )
    assert tuple(inspect.signature(load_job_result).parameters) == (
        "stage",
        "job_id",
        "output_root",
    )
    forbidden = {"data", "data_path", "split", "cohort", "test_path"}
    assert forbidden.isdisjoint(inspect.signature(begin_job).parameters)
    assert tuple(inspect.signature(JobResultWriter.register_records).parameters) == (
        "self",
        "records",
        "expected_identities",
    )
    assert tuple(inspect.signature(JobResultWriter.register_artifact).parameters) == (
        "self",
        "name",
        "path",
    )


def test_training_payload_requires_complete_lane_raw_argmin_and_checkpoint_match() -> None:
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import _validate_training_payload

    job = planned_jobs("stage0_acil_tune")[0]
    identity = _checkpoint_identity(
        job, manifest_sha256="1" * 64, provenance_sha256="2" * 64
    )
    payload = _training_payload(job)
    assert _validate_training_payload(
        payload, job=job, checkpoint_identity=identity
    ) == payload

    mutations = {}
    value = copy.deepcopy(payload)
    value["schema"] = "wrong"
    mutations["schema"] = value
    value = copy.deepcopy(payload)
    value["job"] = {**job.to_json(), "seed_bundle": 2}
    mutations["job"] = value
    value = copy.deepcopy(payload)
    value["config_sha256"] = "0" * 64
    mutations["config"] = value
    value = copy.deepcopy(payload)
    value["fit_fallback"]["std"] = 0.0
    mutations["fallback"] = value
    value = copy.deepcopy(payload)
    value["epochs"] = value["epochs"][:-1]
    mutations["20 epochs"] = value
    value = copy.deepcopy(payload)
    value["optimizer_updates"] = 319
    mutations["320"] = value
    value = copy.deepcopy(payload)
    value["epochs"][4]["epoch"] = 5
    mutations["epoch registry"] = value
    value = copy.deepcopy(payload)
    value["epochs"][4]["physical_batches"] = 63
    mutations["64 physical"] = value
    value = copy.deepcopy(payload)
    value["epochs"][4]["optimizer_updates"] = 15
    mutations["16 optimizer"] = value
    value = copy.deepcopy(payload)
    value["epochs"][4]["mean_loss"] = float("inf")
    mutations["finite"] = value
    value = copy.deepcopy(payload)
    value["epochs"][4]["source_dev_nmae"] += 0.01
    mutations["raw source-dev"] = value
    value = copy.deepcopy(payload)
    value["epochs"][4]["selected_as_best"] = True
    mutations["selected_as_best"] = value

    for expected, invalid in mutations.items():
        with pytest.raises(ValueError, match=expected):
            _validate_training_payload(
                invalid, job=job, checkpoint_identity=identity
            )

    tie = copy.deepcopy(payload)
    tie["epochs"][4]["source_dev_absolute_error_sum"] = 25.0
    tie["epochs"][4]["source_dev_nmae"] = 0.25
    tie["epochs"][3]["selected_as_best"] = False
    tie["epochs"][4]["selected_as_best"] = True
    tie["best_epoch"] = 4
    tied_identity = {**identity, "epoch": 4}
    with pytest.raises(ValueError, match="earliest|argmin"):
        _validate_training_payload(
            tie, job=job, checkpoint_identity=tied_identity
        )

    with pytest.raises(ValueError, match="checkpoint.*epoch"):
        _validate_training_payload(
            payload, job=job, checkpoint_identity={**identity, "epoch": 4}
        )
    with pytest.raises(ValueError, match="checkpoint.*source-dev"):
        _validate_training_payload(
            payload,
            job=job,
            checkpoint_identity={**identity, "source_dev_nmae": 0.3},
        )


def test_success_requires_registered_training_artifact(tmp_path) -> None:
    from experiments.acil_innovation_v1.checkpoint import save_checkpoint
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import begin_job

    job = planned_jobs("stage0_acil_tune")[0]
    writer = begin_job(
        stage=job.stage,
        job_id=job.job_id,
        output_root=tmp_path,
        manifest_sha256="3" * 64,
        provenance_sha256="4" * 64,
    )
    checkpoint = save_checkpoint(
        _registered_model_fixture(job),
        identity=_checkpoint_identity(
            job, manifest_sha256="3" * 64, provenance_sha256="4" * 64
        ),
        directory=writer.checkpoint_directory,
    )
    writer.register_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="training"):
        writer.succeed()


def test_load_revalidates_training_semantics_not_only_artifact_hash(
    tmp_path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.result_io as result_io
    from experiments.acil_innovation_v1.jobs import planned_jobs

    monkeypatch.setattr(result_io, "_expected_record_identities", lambda _job: ())
    job = planned_jobs("stage0_acil_tune")[0]
    final, _ = _finalize_synthetic_job(
        job=job,
        output_root=tmp_path,
        manifest_sha256="7" * 64,
        provenance_sha256="8" * 64,
        identity=_checkpoint_identity(
            job, manifest_sha256="7" * 64, provenance_sha256="8" * 64
        ),
    )
    training_path = final / "training.json"
    training = json.loads(training_path.read_text(encoding="ascii"))
    training["epochs"][4]["selected_as_best"] = True
    training_bytes = (
        json.dumps(training, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    training_path.write_bytes(training_bytes)
    result_path = final / "result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))
    result["artifacts"]["training"].update(
        {
            "sha256": result_io.regular_file_sha256(training_path),
            "bytes": len(training_bytes),
        }
    )
    result_path.write_bytes(
        (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
    )
    with pytest.raises(ValueError, match="selected_as_best"):
        result_io.load_job_result(
            stage=job.stage, job_id=job.job_id, output_root=tmp_path
        )


def test_public_result_api_rejects_future_formal_stages(tmp_path) -> None:
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import begin_job, load_job_result

    job = planned_jobs("formal_acil")[0]
    with pytest.raises(ValueError, match="active manifest"):
        begin_job(
            stage=job.stage,
            job_id=job.job_id,
            output_root=tmp_path,
            manifest_sha256="5" * 64,
            provenance_sha256="6" * 64,
        )
    with pytest.raises(ValueError, match="active manifest"):
        load_job_result(stage=job.stage, job_id=job.job_id, output_root=tmp_path)


def test_job_record_carriers_are_exact_and_derived_diagnostic_is_not_headline() -> None:
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import _carried_methods

    assert _carried_methods(planned_jobs("stage0_acil_tune")[0]) == (
        "linear_fill",
        "acil",
    )
    stage_i = {job.method: job for job in planned_jobs("stage_i")}
    assert _carried_methods(stage_i["local_loo"]) == ("acil", "local_loo")
    assert _carried_methods(stage_i["global_loo"]) == ("global_loo",)
    assert "global_loo_deranged" not in {
        method
        for stage in (
            "stage0_acil_tune",
            "stage_h",
            "stage_i",
            "full_tune",
        )
        for job in planned_jobs(stage)
        for method in _carried_methods(job)
    }


def test_canonical_json_and_jsonl_are_exclusive_and_hash_verified(tmp_path) -> None:
    from experiments.acil_innovation_v1.result_io import (
        regular_file_sha256,
        write_canonical_json_exclusive,
        write_canonical_jsonl_exclusive,
    )

    json_path = tmp_path / "value.json"
    artifact = write_canonical_json_exclusive(json_path, {"z": 1, "a": "中"})
    assert json_path.read_bytes() == b'{"a":"\\u4e2d","z":1}\n'
    assert regular_file_sha256(json_path, expected_sha256=artifact.sha256) == artifact.sha256
    with pytest.raises(FileExistsError):
        write_canonical_json_exclusive(json_path, {"different": True})

    jsonl_path = tmp_path / "rows.jsonl"
    rows = write_canonical_jsonl_exclusive(jsonl_path, ({"b": 2}, {"a": 1}))
    assert jsonl_path.read_bytes() == b'{"b":2}\n{"a":1}\n'
    assert rows.count == 2
    with pytest.raises(ValueError, match="hash"):
        regular_file_sha256(jsonl_path, expected_sha256="0" * 64)

    link = tmp_path / "link.jsonl"
    link.symlink_to(jsonl_path)
    with pytest.raises(ValueError, match="regular"):
        regular_file_sha256(link)


def test_metric_jsonl_is_complete_unique_and_canonically_ordered(tmp_path) -> None:
    from experiments.acil_innovation_v1.result_io import (
        validate_metric_records_file,
        write_metric_records_exclusive,
    )

    first = {
        "method": "m",
        "seed_bundle": 1,
        "dataset": "abilene",
        "mask_family": "random",
        "window_start": 0,
        "flow": 0,
        "oracle": False,
    }
    second = {**first, "flow": 1}
    path = tmp_path / "records.jsonl"
    artifact = write_metric_records_exclusive(
        path,
        (_metric_row(second, 2.0), _metric_row(first, 1.0)),
        expected_identities=(first, second),
    )
    assert artifact.count == 2
    decoded = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["identity"] for row in decoded] == [first, second]
    validated = validate_metric_records_file(
        path,
        expected_sha256=artifact.sha256,
        expected_identities=(first, second),
    )
    assert validated == artifact

    missing_path = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="missing"):
        write_metric_records_exclusive(
            missing_path,
            (_metric_row(first),),
            expected_identities=(first, second),
        )
    assert not missing_path.exists()

    duplicate_path = tmp_path / "duplicate.jsonl"
    with pytest.raises(ValueError, match="duplicate"):
        write_metric_records_exclusive(
            duplicate_path,
            (_metric_row(first), _metric_row(first)),
            expected_identities=(first, second),
        )
    assert not duplicate_path.exists()


def test_job_record_target_hash_is_recomputed_from_deterministic_registry() -> None:
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.registries import ordered_window_starts
    from experiments.acil_innovation_v1.result_io import _validate_job_record_targets

    job = planned_jobs("stage0_acil_tune")[0]
    identity = {
        "method": "linear_fill",
        "seed_bundle": job.seed_bundle,
        "dataset": job.dataset,
        "mask_family": "random",
        "window_start": ordered_window_starts(job.dataset, "tune")[0],
        "flow": 0,
        "oracle": False,
    }
    with pytest.raises(ValueError, match="target-set hash"):
        _validate_job_record_targets((_metric_row(identity),), job=job)


def test_success_finalizes_exact_job_directory_and_binds_artifacts(
    tmp_path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.result_io as result_io
    from experiments.acil_innovation_v1.checkpoint import save_checkpoint
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import begin_job, load_job_result

    monkeypatch.setattr(result_io, "_expected_record_identities", lambda _job: ())
    job = planned_jobs("stage0_acil_tune")[0]
    writer = begin_job(
        stage=job.stage,
        job_id=job.job_id,
        output_root=tmp_path,
        manifest_sha256="b" * 64,
        provenance_sha256="c" * 64,
    )
    assert writer.staging_directory == tmp_path / job.stage / f".{job.job_id}.inprogress"
    checkpoint_identity = {
        **_checkpoint_identity(
            job,
            manifest_sha256="b" * 64,
            provenance_sha256="c" * 64,
        )
    }
    checkpoint = save_checkpoint(
        _registered_model_fixture(job),
        identity=checkpoint_identity,
        directory=writer.checkpoint_directory,
    )
    writer.register_checkpoint(checkpoint)
    from experiments.acil_innovation_v1.result_io import write_canonical_json_exclusive

    training = write_canonical_json_exclusive(
        writer.staging_directory / "training.json", _training_payload(job)
    )
    registered = writer.register_artifact("training", training.path)
    assert registered == {
        "path": "training.json",
        "sha256": training.sha256,
        "bytes": training.bytes,
    }
    final = writer.succeed()

    assert final == tmp_path / job.stage / job.job_id
    assert final.is_dir()
    assert not writer.staging_directory.exists()
    payload = load_job_result(stage=job.stage, job_id=job.job_id, output_root=tmp_path)
    assert payload["status"] == "succeeded"
    assert payload["checkpoint"]["metadata"].startswith("checkpoints/")
    assert payload["checkpoint"]["identity"] == checkpoint_identity
    assert payload["checkpoint"]["identity_sha256"] == checkpoint.identity_sha256
    assert payload["checkpoint"]["file_sha256"] == checkpoint.file_sha256
    assert payload["checkpoint"]["tensor_sha256"] == checkpoint.tensor_sha256
    assert payload["records"]["jsonl"] == "records.jsonl"
    assert payload["records"]["count"] == 0
    assert payload["artifacts"] == {"training": registered}

    with pytest.raises(FileExistsError):
        begin_job(
            stage=job.stage,
            job_id=job.job_id,
            output_root=tmp_path,
            manifest_sha256="b" * 64,
            provenance_sha256="c" * 64,
        )

    (final / "records.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="hash|record"):
        load_job_result(stage=job.stage, job_id=job.job_id, output_root=tmp_path)


def test_checkpoint_registration_rejects_wrong_job_or_hash(tmp_path) -> None:
    from experiments.acil_innovation_v1.checkpoint import save_checkpoint
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import begin_job

    job = planned_jobs("stage0_acil_tune")[0]
    writer = begin_job(
        stage=job.stage,
        job_id=job.job_id,
        output_root=tmp_path,
        manifest_sha256="b" * 64,
        provenance_sha256="c" * 64,
    )
    wrong = save_checkpoint(
        torch.nn.Linear(1, 1),
        identity={
            **_checkpoint_identity(
                job,
                manifest_sha256="b" * 64,
                provenance_sha256="c" * 64,
            ),
            "job_id": "f" * 64,
        },
        directory=writer.checkpoint_directory,
    )
    with pytest.raises(ValueError, match="checkpoint identity"):
        writer.register_checkpoint(wrong)
    correct = save_checkpoint(
        torch.nn.Linear(1, 1),
        identity=_checkpoint_identity(
            job,
            manifest_sha256="b" * 64,
            provenance_sha256="c" * 64,
        ),
        directory=writer.checkpoint_directory,
    )
    with pytest.raises(ValueError, match="hash"):
        writer.register_checkpoint(replace(correct, file_sha256="0" * 64))


def test_named_artifacts_are_unique_regular_and_inside_job_staging(tmp_path) -> None:
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import (
        begin_job,
        write_canonical_json_exclusive,
    )

    job = planned_jobs("stage0_acil_tune")[0]
    writer = begin_job(
        stage=job.stage,
        job_id=job.job_id,
        output_root=tmp_path,
        manifest_sha256="1" * 64,
        provenance_sha256="2" * 64,
    )
    artifact = write_canonical_json_exclusive(
        writer.staging_directory / "training.json", {"epochs": 20}
    )
    writer.register_artifact("training", artifact.path)
    with pytest.raises(FileExistsError):
        writer.register_artifact("training", artifact.path)
    outside = write_canonical_json_exclusive(tmp_path / "outside.json", {"x": 1})
    with pytest.raises(ValueError, match="inside"):
        writer.register_artifact("outside", outside.path)
    link = writer.staging_directory / "linked.json"
    link.symlink_to(outside.path)
    with pytest.raises(ValueError, match="regular"):
        writer.register_artifact("linked", link)
    with pytest.raises(ValueError, match="frozen job grid"):
        writer.register_records((), ({"method": "invented"},))
    assert not (writer.staging_directory / "records.jsonl").exists()


def test_algorithmic_failure_atomically_finalizes_and_retains_partial_files(tmp_path) -> None:
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import (
        begin_job,
        load_job_result,
        write_canonical_json_exclusive,
    )

    job = planned_jobs("stage0_acil_tune")[0]
    writer = begin_job(
        stage=job.stage,
        job_id=job.job_id,
        output_root=tmp_path,
        manifest_sha256="d" * 64,
        provenance_sha256="e" * 64,
    )
    write_canonical_json_exclusive(
        writer.staging_directory / "partial-diagnostic.json",
        {"last_finite_epoch": 2},
    )
    final = writer.fail_algorithmically(
        failure_type="nonfinite_loss",
        message="loss became NaN at the registered step",
    )

    assert final == tmp_path / job.stage / job.job_id
    assert (final / "started.json").is_file()
    assert (final / "partial-diagnostic.json").is_file()
    payload = load_job_result(stage=job.stage, job_id=job.job_id, output_root=tmp_path)
    assert payload["status"] == "algorithmic_failure"
    assert payload["failure"] == {
        "type": "nonfinite_loss",
        "message": "loss became NaN at the registered step",
    }
    with pytest.raises(RuntimeError, match="finalized"):
        writer.succeed()


class _UnsupportedRenameAt2:
    def __init__(self, error: int, *, status: int = -1, before_return=None) -> None:
        self.error = error
        self.status = status
        self.before_return = before_return
        self.argtypes = None
        self.restype = None

    def __call__(self, *_args) -> int:
        import ctypes

        if self.before_return is not None:
            self.before_return()
        ctypes.set_errno(self.error)
        return self.status


class _LibcWithUnsupportedRenameAt2:
    def __init__(self, error: int) -> None:
        self.renameat2 = _UnsupportedRenameAt2(error)


def _force_unsupported_renameat2(monkeypatch, result_io, error: int) -> None:
    fake = _LibcWithUnsupportedRenameAt2(error)
    monkeypatch.setattr(result_io.ctypes, "CDLL", lambda *_args, **_kwargs: fake)


def test_directory_publish_succeeds_on_the_workspace_filesystem(tmp_path) -> None:
    import experiments.acil_innovation_v1.result_io as result_io

    source = tmp_path / "workspace-source"
    destination = tmp_path / "workspace-destination"
    source.mkdir()
    (source / "payload").write_bytes(b"workspace-filesystem\n")
    source_identity = (source.lstat().st_dev, source.lstat().st_ino)

    result_io._rename_directory_noreplace(source, destination)

    assert not source.exists()
    assert (destination / "payload").read_bytes() == b"workspace-filesystem\n"
    assert (destination.lstat().st_dev, destination.lstat().st_ino) == source_identity


@pytest.mark.parametrize("error_name", ("EINVAL", "ENOSYS", "EOPNOTSUPP"))
def test_directory_publish_falls_back_when_renameat2_is_unsupported(
    tmp_path, monkeypatch, error_name: str
) -> None:
    import errno

    import experiments.acil_innovation_v1.result_io as result_io

    source = tmp_path / f"source-{error_name.lower()}"
    destination = tmp_path / f"destination-{error_name.lower()}"
    source.mkdir()
    (source / "result.json").write_bytes(b"complete-result\n")
    source_identity = (source.lstat().st_dev, source.lstat().st_ino)
    _force_unsupported_renameat2(
        monkeypatch, result_io, getattr(errno, error_name)
    )

    result_io._rename_directory_noreplace(source, destination)

    assert not source.exists()
    assert destination.is_dir() and not destination.is_symlink()
    assert (destination / "result.json").read_bytes() == b"complete-result\n"
    assert (destination.lstat().st_dev, destination.lstat().st_ino) == source_identity


@pytest.mark.parametrize(
    "destination_kind", ("empty_dir", "complete_dir", "file", "symlink")
)
def test_fallback_rejects_a_destination_visible_at_preflight(
    tmp_path, monkeypatch, destination_kind: str
) -> None:
    import errno

    import experiments.acil_innovation_v1.result_io as result_io

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "source-marker").write_text("source", encoding="ascii")
    if destination_kind in {"empty_dir", "complete_dir"}:
        destination.mkdir()
        if destination_kind == "complete_dir":
            (destination / "result.json").write_text("complete", encoding="ascii")
    elif destination_kind == "file":
        destination.write_text("existing", encoding="ascii")
    else:
        target = tmp_path / "symlink-target"
        target.mkdir()
        (target / "target-marker").write_text("target", encoding="ascii")
        destination.symlink_to(target, target_is_directory=True)
    before = destination.lstat()
    _force_unsupported_renameat2(monkeypatch, result_io, errno.EINVAL)

    with pytest.raises(FileExistsError, match="already exists"):
        result_io._rename_directory_noreplace(source, destination)

    after = destination.lstat()
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )
    assert (source / "source-marker").read_text(encoding="ascii") == "source"
    if destination_kind == "file":
        assert destination.read_text(encoding="ascii") == "existing"
    elif destination_kind == "complete_dir":
        assert (destination / "result.json").read_text(encoding="ascii") == "complete"
    elif destination_kind == "symlink":
        assert destination.is_symlink()
        assert (destination / "target-marker").read_text(encoding="ascii") == "target"


def test_fallback_propagates_rename_error_without_partial_publish(
    tmp_path, monkeypatch
) -> None:
    import errno

    import experiments.acil_innovation_v1.result_io as result_io

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "marker").write_text("intact", encoding="ascii")
    _force_unsupported_renameat2(monkeypatch, result_io, errno.ENOSYS)

    def fail_rename(_source, _destination) -> None:
        raise OSError(errno.EXDEV, "cross-device publish rejected")

    monkeypatch.setattr(result_io.os, "rename", fail_rename)
    with pytest.raises(OSError) as captured:
        result_io._rename_directory_noreplace(source, destination)

    assert captured.value.errno == errno.EXDEV
    assert (source / "marker").read_text(encoding="ascii") == "intact"
    assert not destination.exists()


def test_two_protocol_controlled_publishers_cannot_replace_a_complete_winner(
    tmp_path, monkeypatch
) -> None:
    import errno
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import experiments.acil_innovation_v1.result_io as result_io

    sources = (tmp_path / "source-a", tmp_path / "source-b")
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "marker").write_text(str(index), encoding="ascii")
    destination = tmp_path / "destination"
    _force_unsupported_renameat2(monkeypatch, result_io, errno.EOPNOTSUPP)
    real_rename = result_io.os.rename
    both_checked_destination = threading.Barrier(2)

    def synchronized_rename(source, target) -> None:
        both_checked_destination.wait(timeout=5)
        real_rename(source, target)

    monkeypatch.setattr(result_io.os, "rename", synchronized_rename)

    def publish(source):
        try:
            result_io._rename_directory_noreplace(source, destination)
        except FileExistsError:
            return "exists"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(publish, sources))

    assert sorted(outcomes) == ["exists", "published"]
    winning_marker = (destination / "marker").read_text(encoding="ascii")
    assert winning_marker in {"0", "1"}
    losing_source = sources[1 - int(winning_marker)]
    assert (losing_source / "marker").read_text(encoding="ascii") == str(
        1 - int(winning_marker)
    )


def test_fallback_rejects_a_symlink_that_appears_before_rename(
    tmp_path, monkeypatch
) -> None:
    import errno

    import experiments.acil_innovation_v1.result_io as result_io

    source = tmp_path / "source"
    source.mkdir()
    (source / "marker").write_text("source", encoding="ascii")
    destination = tmp_path / "destination"
    target = tmp_path / "target"
    target.mkdir()
    (target / "marker").write_text("target", encoding="ascii")
    _force_unsupported_renameat2(monkeypatch, result_io, errno.EINVAL)
    real_rename = result_io.os.rename

    def inject_symlink_then_rename(source_path, destination_path) -> None:
        destination.symlink_to(target, target_is_directory=True)
        real_rename(source_path, destination_path)

    monkeypatch.setattr(result_io.os, "rename", inject_symlink_then_rename)
    with pytest.raises(FileExistsError, match="already exists"):
        result_io._rename_directory_noreplace(source, destination)

    assert destination.is_symlink()
    assert (target / "marker").read_text(encoding="ascii") == "target"
    assert (source / "marker").read_text(encoding="ascii") == "source"


def test_directory_publish_rejects_a_symlink_source_before_any_syscall(
    tmp_path, monkeypatch
) -> None:
    import errno

    import experiments.acil_innovation_v1.result_io as result_io

    actual = tmp_path / "actual"
    actual.mkdir()
    source = tmp_path / "source"
    source.symlink_to(actual, target_is_directory=True)
    destination = tmp_path / "destination"
    _force_unsupported_renameat2(monkeypatch, result_io, errno.ENOSYS)

    with pytest.raises(ValueError, match="source.*regular directory"):
        result_io._rename_directory_noreplace(source, destination)

    assert source.is_symlink()
    assert not destination.exists()


@pytest.mark.parametrize(
    ("status", "reported_errno"),
    ((-1, 0), (1, 5)),
)
def test_directory_publish_rejects_ambiguous_renameat2_status_without_fallback(
    tmp_path, monkeypatch, status: int, reported_errno: int
) -> None:
    import experiments.acil_innovation_v1.result_io as result_io

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "marker").write_text("source", encoding="ascii")
    operation = _UnsupportedRenameAt2(
        reported_errno,
        status=status,
    )
    fake_libc = type("FakeLibc", (), {"renameat2": operation})()
    monkeypatch.setattr(
        result_io.ctypes, "CDLL", lambda *_args, **_kwargs: fake_libc
    )
    fallback_called = False

    def forbidden_fallback(_source, _destination) -> None:
        nonlocal fallback_called
        fallback_called = True
        raise AssertionError("ambiguous renameat2 status entered fallback")

    monkeypatch.setattr(result_io.os, "rename", forbidden_fallback)
    with pytest.raises(RuntimeError, match="ambiguous.*renameat2|renameat2.*ambiguous"):
        result_io._rename_directory_noreplace(source, destination)

    assert not fallback_called
    assert (source / "marker").read_text(encoding="ascii") == "source"
    assert not destination.exists()


def test_directory_publish_propagates_unambiguous_renameat2_error_without_fallback(
    tmp_path, monkeypatch
) -> None:
    import errno

    import experiments.acil_innovation_v1.result_io as result_io

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    operation = _UnsupportedRenameAt2(errno.EIO)
    fake_libc = type("FakeLibc", (), {"renameat2": operation})()
    monkeypatch.setattr(
        result_io.ctypes, "CDLL", lambda *_args, **_kwargs: fake_libc
    )
    monkeypatch.setattr(
        result_io.os,
        "rename",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("unambiguous renameat2 error entered fallback")
        ),
    )

    with pytest.raises(OSError) as captured:
        result_io._rename_directory_noreplace(source, destination)

    assert captured.value.errno == errno.EIO
    assert source.is_dir()
    assert not destination.exists()


def test_fallback_rechecks_source_inode_before_portable_rename(
    tmp_path, monkeypatch
) -> None:
    import errno

    import experiments.acil_innovation_v1.result_io as result_io

    source = tmp_path / "source"
    displaced = tmp_path / "displaced-source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "marker").write_text("original", encoding="ascii")
    real_rename = result_io.os.rename

    def replace_source() -> None:
        real_rename(source, displaced)
        source.mkdir()
        (source / "marker").write_text("replacement", encoding="ascii")

    operation = _UnsupportedRenameAt2(
        errno.EINVAL,
        before_return=replace_source,
    )
    fake_libc = type("FakeLibc", (), {"renameat2": operation})()
    monkeypatch.setattr(
        result_io.ctypes, "CDLL", lambda *_args, **_kwargs: fake_libc
    )

    with pytest.raises(ValueError, match="source identity"):
        result_io._rename_directory_noreplace(source, destination)

    assert (source / "marker").read_text(encoding="ascii") == "replacement"
    assert (displaced / "marker").read_text(encoding="ascii") == "original"
    assert not destination.exists()


@pytest.mark.parametrize("changed_parent", ("source", "destination"))
def test_fallback_rechecks_parent_inode_before_portable_rename(
    tmp_path, monkeypatch, changed_parent: str
) -> None:
    import errno

    import experiments.acil_innovation_v1.result_io as result_io

    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "source"
    destination = destination_parent / "destination"
    source.mkdir()
    (source / "marker").write_text("source", encoding="ascii")
    changed = source_parent if changed_parent == "source" else destination_parent
    displaced_parent = tmp_path / f"old-{changed_parent}-parent"
    real_rename = result_io.os.rename

    def replace_parent_but_preserve_source_inode() -> None:
        real_rename(changed, displaced_parent)
        changed.mkdir()
        if changed_parent == "source":
            real_rename(displaced_parent / "source", source)

    operation = _UnsupportedRenameAt2(
        errno.EINVAL,
        before_return=replace_parent_but_preserve_source_inode,
    )
    fake_libc = type("FakeLibc", (), {"renameat2": operation})()
    monkeypatch.setattr(
        result_io.ctypes, "CDLL", lambda *_args, **_kwargs: fake_libc
    )

    with pytest.raises(ValueError, match=f"{changed_parent} parent identity"):
        result_io._rename_directory_noreplace(source, destination)

    assert (source / "marker").read_text(encoding="ascii") == "source"
    assert not destination.exists()


def test_directory_publish_detects_destination_inode_drift_after_rename(
    tmp_path, monkeypatch
) -> None:
    import errno

    import experiments.acil_innovation_v1.result_io as result_io

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    displaced = tmp_path / "displaced-published-directory"
    source.mkdir()
    (source / "marker").write_text("source", encoding="ascii")
    _force_unsupported_renameat2(monkeypatch, result_io, errno.EINVAL)
    real_rename = result_io.os.rename

    def replace_published_destination(source_path, destination_path) -> None:
        real_rename(source_path, destination_path)
        real_rename(destination_path, displaced)
        destination.mkdir()

    monkeypatch.setattr(result_io.os, "rename", replace_published_destination)
    with pytest.raises(ValueError, match="published destination"):
        result_io._rename_directory_noreplace(source, destination)

    assert (displaced / "marker").read_text(encoding="ascii") == "source"
    assert destination.is_dir() and list(destination.iterdir()) == []


def test_directory_publish_documents_the_protocol_controlled_race_boundary() -> None:
    import experiments.acil_innovation_v1.result_io as result_io

    documentation = inspect.getdoc(result_io._rename_directory_noreplace) or ""
    assert "protocol-controlled single publisher" in documentation
    assert "non-cooperating" in documentation
    assert "empty directory" in documentation


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_config",
        "missing_source_dev",
        "wrong_manifest",
        "negative_source_dev",
        "extra_field",
    ),
)
def test_checkpoint_registration_requires_exact_freeze_bound_identity(
    tmp_path, mutation: str
) -> None:
    from experiments.acil_innovation_v1.checkpoint import save_checkpoint
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import begin_job

    manifest_sha256 = "1" * 64
    provenance_sha256 = "2" * 64
    job = planned_jobs("stage0_acil_tune")[0]
    writer = begin_job(
        stage=job.stage,
        job_id=job.job_id,
        output_root=tmp_path,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )
    identity = _checkpoint_identity(
        job,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )
    if mutation == "missing_config":
        del identity["config_sha256"]
    elif mutation == "missing_source_dev":
        del identity["source_dev_nmae"]
    elif mutation == "wrong_manifest":
        identity["manifest_sha256"] = "3" * 64
    elif mutation == "negative_source_dev":
        identity["source_dev_nmae"] = -0.01
    elif mutation == "extra_field":
        identity["unregistered"] = "value"
    checkpoint = save_checkpoint(
        torch.nn.Linear(2, 1),
        identity=identity,
        directory=writer.checkpoint_directory,
    )
    with pytest.raises(ValueError, match="checkpoint identity"):
        writer.register_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "mutation",
    ("missing_base", "wrong_source_job", "missing_tensor_hash", "extra_field"),
)
def test_checkpoint_registration_requires_exact_registry_base_binding(
    tmp_path, mutation: str
) -> None:
    from experiments.acil_innovation_v1.checkpoint import save_checkpoint
    from experiments.acil_innovation_v1.jobs import planned_jobs
    from experiments.acil_innovation_v1.result_io import begin_job

    manifest_sha256 = "4" * 64
    provenance_sha256 = "5" * 64
    job = planned_jobs("stage_h")[0]
    source = _matching_job(
        "stage0_acil_tune", "acil", job.dataset, job.seed_bundle
    )
    base = {
        "source_stage": source.stage,
        "source_job_id": source.job_id,
        "checkpoint_identity_sha256": "6" * 64,
        "checkpoint_file_sha256": "7" * 64,
        "checkpoint_tensor_sha256": "8" * 64,
    }
    identity = _checkpoint_identity(
        job,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        base_checkpoint=base,
    )
    if mutation == "missing_base":
        del identity["base_checkpoint"]
    elif mutation == "wrong_source_job":
        identity["base_checkpoint"]["source_job_id"] = "9" * 64
    elif mutation == "missing_tensor_hash":
        del identity["base_checkpoint"]["checkpoint_tensor_sha256"]
    elif mutation == "extra_field":
        identity["base_checkpoint"]["source_method"] = "acil"
    writer = begin_job(
        stage=job.stage,
        job_id=job.job_id,
        output_root=tmp_path,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )
    checkpoint = save_checkpoint(
        torch.nn.Linear(2, 1),
        identity=identity,
        directory=writer.checkpoint_directory,
    )
    with pytest.raises(ValueError, match="checkpoint identity|base checkpoint|dependency"):
        writer.register_checkpoint(checkpoint)


def test_full_u0_requires_exact_dependency_artifact_and_source_checkpoints(
    tmp_path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.result_io as result_io
    from experiments.acil_innovation_v1.jobs import planned_jobs

    # Keep this integrity test small without changing the production registry.
    monkeypatch.setattr(result_io, "_expected_record_identities", lambda _job: ())
    manifest_sha256 = "a" * 64
    provenance_sha256 = "b" * 64
    candidate = planned_jobs("full_tune")[0]
    acil_job = _matching_job(
        "stage0_acil_tune", "acil", candidate.dataset, candidate.seed_bundle
    )
    _, acil_checkpoint = _finalize_synthetic_job(
        job=acil_job,
        output_root=tmp_path,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        identity=_checkpoint_identity(
            acil_job,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
        ),
    )
    acil_binding = _dependency_binding(acil_job, acil_checkpoint)
    loo_job = _matching_job(
        "stage_i", "global_loo", candidate.dataset, candidate.seed_bundle
    )
    _, loo_checkpoint = _finalize_synthetic_job(
        job=loo_job,
        output_root=tmp_path,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        identity=_checkpoint_identity(
            loo_job,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            base_checkpoint=acil_binding,
        ),
    )
    loo_binding = _dependency_binding(
        loo_job, loo_checkpoint, source_method="global_loo"
    )
    final, _ = _finalize_synthetic_job(
        job=candidate,
        output_root=tmp_path,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        identity=_checkpoint_identity(
            candidate,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            base_checkpoint=acil_binding,
        ),
        dependencies={
            "acil_base": acil_binding,
            "loo_deepsets": loo_binding,
        },
    )
    payload = result_io.load_job_result(
        stage=candidate.stage,
        job_id=candidate.job_id,
        output_root=tmp_path,
    )
    assert final.is_dir()
    assert payload["artifacts"]["dependencies"]["path"] == "dependencies.json"


def test_full_u0_rejects_missing_dependency_artifact(tmp_path, monkeypatch) -> None:
    import experiments.acil_innovation_v1.result_io as result_io
    from experiments.acil_innovation_v1.jobs import planned_jobs

    monkeypatch.setattr(result_io, "_expected_record_identities", lambda _job: ())
    manifest_sha256 = "c" * 64
    provenance_sha256 = "d" * 64
    candidate = planned_jobs("full_tune")[0]
    acil_job = _matching_job(
        "stage0_acil_tune", "acil", candidate.dataset, candidate.seed_bundle
    )
    _, acil_checkpoint = _finalize_synthetic_job(
        job=acil_job,
        output_root=tmp_path,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        identity=_checkpoint_identity(
            acil_job,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
        ),
    )
    acil_binding = _dependency_binding(acil_job, acil_checkpoint)
    with pytest.raises(ValueError, match="dependencies|dependency"):
        _finalize_synthetic_job(
            job=candidate,
            output_root=tmp_path,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            identity=_checkpoint_identity(
                candidate,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
                base_checkpoint=acil_binding,
            ),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_binding",
        "extra_binding",
        "wrong_source_job",
        "wrong_hash",
        "wrong_acil_base",
    ),
)
def test_full_u0_rejects_dependency_artifact_registry_drift(
    tmp_path, monkeypatch, mutation: str
) -> None:
    import experiments.acil_innovation_v1.result_io as result_io
    from experiments.acil_innovation_v1.jobs import planned_jobs

    monkeypatch.setattr(result_io, "_expected_record_identities", lambda _job: ())
    manifest_sha256 = "d" * 64
    provenance_sha256 = "e" * 64
    candidate = planned_jobs("full_tune")[0]
    acil_job = _matching_job(
        "stage0_acil_tune", "acil", candidate.dataset, candidate.seed_bundle
    )
    _, acil_checkpoint = _finalize_synthetic_job(
        job=acil_job,
        output_root=tmp_path,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        identity=_checkpoint_identity(
            acil_job,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
        ),
    )
    acil_binding = _dependency_binding(acil_job, acil_checkpoint)
    loo_job = _matching_job(
        "stage_i", "global_loo", candidate.dataset, candidate.seed_bundle
    )
    _, loo_checkpoint = _finalize_synthetic_job(
        job=loo_job,
        output_root=tmp_path,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
        identity=_checkpoint_identity(
            loo_job,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            base_checkpoint=acil_binding,
        ),
    )
    loo_binding = _dependency_binding(
        loo_job, loo_checkpoint, source_method="global_loo"
    )
    dependencies = {
        "acil_base": acil_binding,
        "loo_deepsets": loo_binding,
    }
    if mutation == "missing_binding":
        del dependencies["loo_deepsets"]
    elif mutation == "extra_binding":
        dependencies["invented"] = dict(loo_binding)
    elif mutation == "wrong_source_job":
        dependencies["loo_deepsets"] = {
            **loo_binding,
            "source_job_id": "0" * 64,
        }
    elif mutation == "wrong_hash":
        dependencies["loo_deepsets"] = {
            **loo_binding,
            "checkpoint_tensor_sha256": "0" * 64,
        }
    elif mutation == "wrong_acil_base":
        dependencies["acil_base"] = {
            **acil_binding,
            "checkpoint_tensor_sha256": "0" * 64,
        }
    with pytest.raises(ValueError, match="dependencies|dependency"):
        _finalize_synthetic_job(
            job=candidate,
            output_root=tmp_path,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            identity=_checkpoint_identity(
                candidate,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
                base_checkpoint=acil_binding,
            ),
            dependencies=dependencies,
        )


def test_job_without_registry_dependency_rejects_dependency_artifact(
    tmp_path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.result_io as result_io
    from experiments.acil_innovation_v1.jobs import planned_jobs

    monkeypatch.setattr(result_io, "_expected_record_identities", lambda _job: ())
    manifest_sha256 = "e" * 64
    provenance_sha256 = "f" * 64
    job = planned_jobs("stage0_acil_tune")[0]
    with pytest.raises(ValueError, match="unregistered dependencies"):
        _finalize_synthetic_job(
            job=job,
            output_root=tmp_path,
            manifest_sha256=manifest_sha256,
            provenance_sha256=provenance_sha256,
            identity=_checkpoint_identity(
                job,
                manifest_sha256=manifest_sha256,
                provenance_sha256=provenance_sha256,
            ),
            dependencies={},
        )
