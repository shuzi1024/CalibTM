from __future__ import annotations

import pytest

from experiments.acil_innovation_v1.registries import (
    cohort_spec,
    dataset_spec,
    ordered_window_starts,
    seed_bundle,
    stage_spec,
    window_schedule_sha256,
)


def test_dataset_and_cohort_boundaries_match_the_audited_registry():
    abilene = dataset_spec("abilene")
    geant = dataset_spec("geant")

    assert abilene.flows == 144
    assert abilene.permitted_splits == {"train": (0, 33884), "val": (33884, 41134)}
    assert geant.flows == 462
    assert geant.permitted_splits == {"train": (0, 7572), "val": (7572, 9172)}

    expected = {
        ("abilene", "fit"): ((0, 33335), "train", 512, 0, 33285),
        ("abilene", "source_dev"): ((33384, 33884), "train", 10, 33384, 33834),
        ("abilene", "tune"): ((33884, 37484), "val", 72, 33884, 37434),
        ("abilene", "gate"): ((37534, 41134), "val", 72, 37534, 41084),
        ("geant", "fit"): ((0, 7023), "train", 512, 0, 6973),
        ("geant", "source_dev"): ((7072, 7572), "train", 10, 7072, 7522),
        ("geant", "tune"): ((7572, 8322), "val", 15, 7572, 8272),
        ("geant", "gate"): ((8372, 9172), "val", 16, 8372, 9122),
    }
    for identity, (bounds, split, count, first, last) in expected.items():
        spec = cohort_spec(*identity)
        starts = ordered_window_starts(*identity)
        assert spec.bounds == bounds
        assert spec.split == split
        assert len(starts) == count
        assert starts[0] == first
        assert starts[-1] == last
        assert all(spec.start <= start and start + 50 <= spec.stop for start in starts)
        assert len(window_schedule_sha256(*identity)) == 64

    assert ordered_window_starts("abilene", "fit")[:9] == (
        0,
        65,
        130,
        195,
        260,
        325,
        390,
        455,
        521,
    )
    assert ordered_window_starts("geant", "fit")[:8] == (
        0,
        13,
        27,
        40,
        54,
        68,
        81,
        95,
    )


def test_gap_cohorts_are_registered_but_never_scheduled():
    for dataset, cohort, bounds in (
        ("abilene", "train_gap", (33335, 33384)),
        ("abilene", "validation_gap", (37484, 37534)),
        ("geant", "train_gap", (7023, 7072)),
        ("geant", "validation_gap", (8322, 8372)),
    ):
        spec = cohort_spec(dataset, cohort)
        assert spec.is_gap is True
        assert spec.bounds == bounds
        with pytest.raises(ValueError, match="gap"):
            ordered_window_starts(dataset, cohort)


def test_seed_bundle_registry_is_exact_and_closed():
    for bundle in range(1, 7):
        spec = seed_bundle(bundle)
        assert spec.bundle == bundle
        assert spec.model == 41000 + bundle
        assert spec.data_order == 51000 + bundle
        assert spec.training_mask == 61000 + bundle
        assert spec.evaluation_mask == 71000 + bundle
    for invalid in (0, 7, True, "1"):
        with pytest.raises((TypeError, ValueError)):
            seed_bundle(invalid)


def test_stage_grids_are_exact_and_training_has_no_mask_axis():
    expected = {
        "stage0_acil_tune": (["linear_fill", "acil"], [1, 2, 3], 6, 36),
        "stage_h": (["acil", "truth_q_deepsets"], [1, 2, 3], 6, 36),
        "stage_i": (["acil", "local_loo", "global_loo"], [1, 2, 3], 12, 54),
        "full_tune": (
            ["acil", "full_u0", "loo_deepsets", "full_scratch", "full_gpt2"],
            [1, 2, 3],
            18,
            90,
        ),
        "formal_acil": (["acil"], [4, 5, 6], 6, 0),
        "formal_gate": (
            [
                "linear_fill",
                "acil",
                "ari_llm",
                "loo_deepsets",
                "full_u0",
                "full_scratch",
                "full_gpt2",
                "imputeformer",
            ],
            [4, 5, 6],
            36,
            144,
        ),
    }
    for name, (methods, seeds, learned_fits, result_cells) in expected.items():
        spec = stage_spec(name)
        assert list(spec.methods) == methods
        assert list(spec.seed_bundles) == seeds
        assert spec.datasets == ("abilene", "geant")
        assert spec.training_has_mask_axis is False
        assert spec.learned_fit_count == learned_fits
        assert spec.result_cell_count == result_cells


def test_stage_capabilities_keep_training_and_evaluation_cohorts_separate():
    for stage in ("stage0_acil_tune", "stage_h", "stage_i", "full_tune"):
        spec = stage_spec(stage)
        assert spec.train_cohorts == ("fit", "source_dev")
        assert spec.eval_cohort == "tune"
    formal = stage_spec("formal_gate")
    assert formal.train_cohorts == ("fit", "source_dev")
    assert formal.eval_cohort == "gate"
    assert formal.reuse_methods == ("acil",)
    assert formal.dependency_stages == ("formal_acil",)
    assert len(formal.checkpoint_hash_bindings) == 1
    assert formal.checkpoint_hash_bindings[0].source_stage == "formal_acil"
    assert formal.checkpoint_hash_bindings[0].source_method == "acil"

    formal_acil = stage_spec("formal_acil")
    assert formal_acil.train_cohorts == ("fit", "source_dev")
    assert formal_acil.eval_cohort is None
    assert formal_acil.role == "training_only_dependency"
    assert formal_acil.methods == ("acil",)
    assert formal_acil.learned_methods == ("acil",)
    assert formal_acil.result_cell_count == 0


def test_all_reused_and_embedded_base_checkpoints_are_exactly_bound():
    exact_hashes = (
        "checkpoint_identity_sha256",
        "checkpoint_file_sha256",
        "checkpoint_tensor_sha256",
    )
    expected = {
        "stage_h": {
            ("stage0_acil_tune", "acil"): (
                "acil",
                "truth_q_deepsets.acil_base",
            ),
        },
        "stage_i": {
            ("stage0_acil_tune", "acil"): (
                "acil",
                "local_loo.acil_base",
                "global_loo.acil_base",
            ),
        },
        "full_tune": {
            ("stage0_acil_tune", "acil"): (
                "acil",
                "full_u0.acil_base",
                "full_scratch.acil_base",
                "full_gpt2.acil_base",
            ),
            ("stage_i", "global_loo"): ("loo_deepsets",),
        },
        "formal_gate": {
            ("formal_acil", "acil"): (
                "acil",
                "loo_deepsets.acil_base",
                "full_u0.acil_base",
                "full_scratch.acil_base",
                "full_gpt2.acil_base",
            ),
        },
    }
    for stage_name, wanted in expected.items():
        spec = stage_spec(stage_name)
        actual = {
            (binding.source_stage, binding.source_method): binding.consumer_roles
            for binding in spec.checkpoint_hash_bindings
        }
        assert actual == wanted
        assert spec.dependency_stages == tuple(
            dict.fromkeys(source_stage for source_stage, _ in wanted)
        )
        for binding in spec.checkpoint_hash_bindings:
            assert binding.match_fields == ("dataset", "seed_bundle")
            assert binding.selection == "source_stage_best_source_dev_checkpoint"
            assert binding.required_exact_matches == exact_hashes

    full_alias = stage_spec("full_tune").checkpoint_hash_bindings[1]
    assert full_alias.relationship == "method_label_rename_exact_same_checkpoint"
    assert stage_spec("stage_h").reuse_methods == ("acil",)
    assert stage_spec("stage_i").reuse_methods == ("acil",)
    assert stage_spec("full_tune").reuse_methods == ("acil", "loo_deepsets")


def test_result_carriers_cover_each_evaluation_method_exactly_once():
    expected = {
        "stage0_acil_tune": {"acil": ("linear_fill", "acil")},
        "stage_h": {"truth_q_deepsets": ("acil", "truth_q_deepsets")},
        "stage_i": {
            "local_loo": ("acil", "local_loo"),
            "global_loo": ("global_loo",),
        },
        "full_tune": {
            "full_u0": ("acil", "loo_deepsets", "full_u0"),
            "full_scratch": ("full_scratch",),
            "full_gpt2": ("full_gpt2",),
        },
        "formal_acil": {"acil": ()},
        "formal_gate": {
            "full_u0": ("linear_fill", "acil", "full_u0"),
            "loo_deepsets": ("loo_deepsets",),
            "full_scratch": ("full_scratch",),
            "full_gpt2": ("full_gpt2",),
            "ari_llm": ("ari_llm",),
            "imputeformer": ("imputeformer",),
        },
    }
    for stage_name, carriers in expected.items():
        spec = stage_spec(stage_name)
        assert dict(spec.result_carriers) == carriers
        flattened = tuple(
            method for methods in spec.result_carriers.values() for method in methods
        )
        if spec.eval_cohort is None:
            assert flattened == ()
        else:
            assert len(flattened) == len(set(flattened))
            assert set(flattened) == set(spec.methods)
