from __future__ import annotations

from pathlib import Path

import pytest
import torch


def _job(stage: str, method: str):
    from experiments.acil_innovation_v1.jobs import planned_jobs

    matches = tuple(job for job in planned_jobs(stage) if job.method == method)
    assert matches
    return matches[0]


def _descriptor(path: str) -> dict[str, object]:
    return {"path": path, "sha256": "a" * 64, "bytes": 0}


def _checkpoint_descriptor() -> dict[str, object]:
    return {
        "metadata": "checkpoints/identity.json",
        "weights": "checkpoints/identity.safetensors",
    }


def _touch_inventory(
    root: Path,
    *,
    artifacts: dict[str, dict[str, object]],
    include_result: bool,
) -> None:
    root.mkdir()
    checkpoint = root / "checkpoints"
    checkpoint.mkdir()
    for relative in (
        "started.json",
        "records.jsonl",
        "checkpoints/identity.json",
        "checkpoints/identity.safetensors",
        *(str(value["path"]) for value in artifacts.values()),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    if include_result:
        (root / "result.json").touch()


def test_expected_named_artifacts_are_exact_per_active_job() -> None:
    from experiments.acil_innovation_v1.checkpoint_validation import (
        expected_named_artifact_paths,
    )

    assert expected_named_artifact_paths(_job("stage0_acil_tune", "acil")) == {
        "training": "training.json"
    }
    assert expected_named_artifact_paths(
        _job("stage_h", "truth_q_deepsets")
    ) == {"training": "training.json"}
    assert expected_named_artifact_paths(_job("stage_i", "local_loo")) == {
        "training": "training.json"
    }
    assert expected_named_artifact_paths(_job("stage_i", "global_loo")) == {
        "derangement_plans": "derangement_plans.json",
        "derangement_records": "derangement_records.jsonl",
        "training": "training.json",
    }
    for method in ("full_u0", "full_scratch", "full_gpt2"):
        assert expected_named_artifact_paths(_job("full_tune", method)) == {
            "dependencies": "dependencies.json",
            "training": "training.json",
        }


def test_exact_named_artifact_registry_rejects_missing_extra_or_wrong_path() -> None:
    from experiments.acil_innovation_v1.checkpoint_validation import (
        validate_named_artifact_registry,
    )

    job = _job("stage_i", "global_loo")
    exact = {
        "derangement_plans": _descriptor("derangement_plans.json"),
        "derangement_records": _descriptor("derangement_records.jsonl"),
        "training": _descriptor("training.json"),
    }
    validate_named_artifact_registry(job, exact)

    for changed in (
        {key: value for key, value in exact.items() if key != "training"},
        {**exact, "alternate_checkpoint": _descriptor("alternate.json")},
        {**exact, "training": _descriptor("history-selected-after-tune.json")},
    ):
        with pytest.raises(ValueError, match="artifact"):
            validate_named_artifact_registry(job, changed)


@pytest.mark.parametrize("include_result", (False, True))
def test_succeeded_job_inventory_is_exact_and_plain(
    tmp_path: Path, include_result: bool
) -> None:
    from experiments.acil_innovation_v1.checkpoint_validation import (
        validate_succeeded_job_inventory,
    )

    job = _job("stage0_acil_tune", "acil")
    artifacts = {"training": _descriptor("training.json")}
    root = tmp_path / "job"
    _touch_inventory(root, artifacts=artifacts, include_result=include_result)
    validate_succeeded_job_inventory(
        root,
        job=job,
        checkpoint=_checkpoint_descriptor(),
        artifacts=artifacts,
        finalized=include_result,
    )

    (root / "unregistered-best-after-tune.safetensors").touch()
    with pytest.raises(ValueError, match="inventory|extra"):
        validate_succeeded_job_inventory(
            root,
            job=job,
            checkpoint=_checkpoint_descriptor(),
            artifacts=artifacts,
            finalized=include_result,
        )


def test_succeeded_job_inventory_rejects_missing_or_symlinked_checkpoint(
    tmp_path: Path,
) -> None:
    from experiments.acil_innovation_v1.checkpoint_validation import (
        validate_succeeded_job_inventory,
    )

    job = _job("stage0_acil_tune", "acil")
    artifacts = {"training": _descriptor("training.json")}
    root = tmp_path / "job"
    _touch_inventory(root, artifacts=artifacts, include_result=True)
    weights = root / "checkpoints" / "identity.safetensors"
    weights.unlink()
    with pytest.raises(ValueError, match="inventory|missing"):
        validate_succeeded_job_inventory(
            root,
            job=job,
            checkpoint=_checkpoint_descriptor(),
            artifacts=artifacts,
            finalized=True,
        )
    weights.symlink_to(root / "training.json")
    with pytest.raises(ValueError, match="plain|symlink|regular"):
        validate_succeeded_job_inventory(
            root,
            job=job,
            checkpoint=_checkpoint_descriptor(),
            artifacts=artifacts,
            finalized=True,
        )


def test_registered_checkpoint_schema_accepts_acil_and_rejects_arbitrary_linear() -> None:
    from experiments.acil_innovation_v1.acil import ACILBase
    from experiments.acil_innovation_v1.checkpoint_validation import (
        validate_registered_state_schema,
    )

    job = _job("stage0_acil_tune", "acil")
    validate_registered_state_schema(job, ACILBase().state_dict())
    with pytest.raises(ValueError, match="registered.*architecture|state"):
        validate_registered_state_schema(job, torch.nn.Linear(2, 1).state_dict())


@pytest.mark.parametrize("mutation", ("missing", "extra", "wrong_shape"))
def test_registered_checkpoint_schema_rejects_key_or_shape_drift(
    mutation: str,
) -> None:
    from experiments.acil_innovation_v1.acil import ACILBase
    from experiments.acil_innovation_v1.checkpoint_validation import (
        validate_registered_state_schema,
    )

    job = _job("stage0_acil_tune", "acil")
    state = {name: value.clone() for name, value in ACILBase().state_dict().items()}
    first = next(iter(state))
    if mutation == "missing":
        del state[first]
    elif mutation == "extra":
        state["unregistered.weight"] = torch.zeros(1)
    else:
        state[first] = state[first].reshape(-1)[:1]
    with pytest.raises(ValueError, match="registered.*architecture|state"):
        validate_registered_state_schema(job, state)


def test_embedded_acil_state_must_equal_bound_source_tensor_for_tensor() -> None:
    from experiments.acil_innovation_v1.acil import ACILBase
    from experiments.acil_innovation_v1.checkpoint_validation import (
        validate_embedded_acil_state,
    )
    from experiments.acil_innovation_v1.oracle_model import TruthQDeepSets

    job = _job("stage_h", "truth_q_deepsets")
    source = {name: value.clone() for name, value in ACILBase().state_dict().items()}
    model = TruthQDeepSets(ACILBase())
    model.acil.load_state_dict(source, strict=True)
    candidate = {name: value.clone() for name, value in model.state_dict().items()}
    validate_embedded_acil_state(job, candidate, source)

    name = next(iter(source))
    changed = {key: value.clone() for key, value in candidate.items()}
    changed[f"acil.{name}"].reshape(-1)[0] += 1.0
    with pytest.raises(ValueError, match="embedded ACIL"):
        validate_embedded_acil_state(job, changed, source)


def test_full_u0_embedded_check_requires_only_acil_not_external_loo_state() -> None:
    from experiments.acil_innovation_v1.acil import ACILBase
    from experiments.acil_innovation_v1.checkpoint_validation import (
        validate_embedded_acil_state,
    )

    job = _job("full_tune", "full_u0")
    source = {name: value.clone() for name, value in ACILBase().state_dict().items()}
    candidate = {f"acil.{name}": value.clone() for name, value in source.items()}
    candidate["candidate_only.weight"] = torch.ones(1)
    # LOO-DeepSets is an external carried comparator/dependency, not candidate state.
    validate_embedded_acil_state(job, candidate, source)


def test_registered_model_factory_rejects_future_formal_job() -> None:
    from experiments.acil_innovation_v1.checkpoint_validation import (
        build_registered_checkpoint_model,
    )

    with pytest.raises(ValueError, match="active manifest"):
        build_registered_checkpoint_model(_job("formal_acil", "acil"))
