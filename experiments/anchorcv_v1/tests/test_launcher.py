from __future__ import annotations

from pathlib import Path

import pytest

from experiments.anchorcv_v1.launcher import (
    authorize_extension,
    build_stage_specs,
    choose_gpu,
    job_command,
    run_grid,
)
from experiments.anchorcv_v1.protocol import fingerprint


def _freeze() -> dict[str, object]:
    return {
        "source_tree_sha256": "a" * 64,
        "config_sha256": fingerprint(),
        "git_available": False,
        "git_commit": None,
    }


def test_prototype_and_extension_grids_are_exact_and_outcome_independent(tmp_path: Path) -> None:
    prototype = build_stage_specs("prototype", tmp_path / "p", _freeze())
    extension = build_stage_specs("extension", tmp_path / "e", _freeze())

    assert [(item.dataset, item.seed_bundle) for item in prototype] == [
        ("abilene", 1),
        ("geant", 1),
    ]
    assert [(item.dataset, item.seed_bundle) for item in extension] == [
        ("abilene", 2),
        ("geant", 2),
        ("abilene", 3),
        ("geant", 3),
    ]
    assert all(item.stage == "prototype" for item in prototype)
    assert all(item.stage == "extension" for item in extension)


def test_job_command_contains_no_cohort_mask_split_or_test_override(tmp_path: Path) -> None:
    spec = build_stage_specs("prototype", tmp_path, _freeze())[0]

    command = job_command(spec, freeze_record=tmp_path / "freeze.json")
    encoded = " ".join(command).lower()

    assert "--dataset" in command
    assert "--seed-bundle" in command
    assert "--stage" in command
    assert "--device" not in command
    assert "--cohort" not in command
    assert "--mask-family" not in command
    assert "--split" not in command
    assert "--test" not in encoded


def test_choose_gpu_assigns_one_job_per_free_gpu() -> None:
    assert choose_gpu((0, 1, 2, 3), {0, 2}) == 1
    assert choose_gpu((0, 1), {0, 1}) is None
    with pytest.raises(ValueError, match="unique"):
        choose_gpu((0, 0), set())


@pytest.mark.parametrize("stage", ["formal", "smoke", ""])
def test_unknown_stage_is_rejected(tmp_path: Path, stage: str) -> None:
    with pytest.raises(ValueError, match="prototype or extension"):
        build_stage_specs(stage, tmp_path, _freeze())


def test_invalid_freeze_identity_is_rejected(tmp_path: Path) -> None:
    broken = _freeze()
    broken["git_available"] = True

    with pytest.raises(ValueError, match="git_available"):
        build_stage_specs("prototype", tmp_path, broken)


def test_grid_rejects_manifest_only_instead_of_counting_it_complete(
    tmp_path: Path,
) -> None:
    spec = build_stage_specs("prototype", tmp_path / "results", _freeze())[0]
    spec.job_directory.mkdir(parents=True)
    (spec.job_directory / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="partial"):
        run_grid(
            (spec,),
            freeze_record=tmp_path / "freeze.json",
            gpus=(0,),
        )


def test_extension_authority_is_recomputed_from_exact_prototype_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    report = {
        "verdict": "proceed",
        "integrity": {
            "valid": True,
            "common_identity": {
                "source_tree_sha256": "a" * 64,
                "config_sha256": fingerprint(),
            },
        },
    }
    monkeypatch.setattr(
        "experiments.anchorcv_v1.review.review_prototype",
        lambda **kwargs: calls.append(kwargs) or report,
    )

    authorized = authorize_extension(
        prototype_output_root=tmp_path / "prototype",
        freeze_record=tmp_path / "freeze.json",
        freeze=_freeze(),
    )

    assert authorized is report
    assert calls == [
        {
            "output_root": tmp_path / "prototype",
            "freeze_record": tmp_path / "freeze.json",
        }
    ]
