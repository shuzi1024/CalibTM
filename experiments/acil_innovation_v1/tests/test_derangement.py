from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch


def test_derangement_builder_and_apply_signatures_have_no_hidden_authority() -> None:
    from experiments.acil_innovation_v1.derangement import (
        DerangementPlan,
        build_derangement,
    )

    assert tuple(inspect.signature(build_derangement).parameters) == (
        "dataset",
        "seed_bundle",
        "mask_family",
    )
    assert tuple(inspect.signature(DerangementPlan.apply).parameters) == (
        "self",
        "innovations",
    )
    forbidden = {"method", "outcome", "truth", "path", "prediction", "metric"}
    assert forbidden.isdisjoint(inspect.signature(build_derangement).parameters)


def test_derangement_identity_rng_and_hash_match_frozen_golden_values() -> None:
    from experiments.acil_innovation_v1.derangement import build_derangement

    first = build_derangement("abilene", 2, "internal_block")
    second = build_derangement("abilene", 2, "internal_block")

    assert tuple(first.identity.__dataclass_fields__) == (
        "protocol",
        "dataset",
        "cohort",
        "window_schedule_sha256",
        "seed_bundle",
        "mask_family",
        "evaluation_mask_seed",
        "registered_window_count",
        "flow_count",
    )
    assert first.identity.protocol == "acil-innovation-v1"
    assert first.identity.dataset == "abilene"
    assert first.identity.cohort == "tune"
    assert first.identity.seed_bundle == 2
    assert first.identity.mask_family == "internal_block"
    assert first.identity.evaluation_mask_seed == 71002
    assert first.identity.registered_window_count == 72
    assert first.identity.flow_count == 144
    assert first.permutation.dtype == np.dtype("<i8")
    assert first.permutation.shape == (72 * 144,)
    assert not first.permutation.flags.writeable
    assert np.array_equal(first.permutation, second.permutation)
    assert first.permutation[:16].tolist() == [
        3193,
        3076,
        7375,
        1285,
        4468,
        1858,
        6901,
        38,
        9373,
        10009,
        611,
        465,
        2976,
        3277,
        8934,
        6689,
    ]
    assert first.permutation_sha256 == (
        "3aaa8ade71cb5a3c596aafec50ce22690a95475357904b135225b694fe86100d"
    )


@pytest.mark.parametrize("dataset", ("abilene", "geant"))
def test_permutation_is_complete_without_replacement(dataset: str) -> None:
    from experiments.acil_innovation_v1.derangement import build_derangement

    plan = build_derangement(dataset, 3, "two_burst")
    count = plan.identity.registered_window_count * plan.identity.flow_count

    assert np.array_equal(np.sort(plan.permutation), np.arange(count, dtype="<i8"))
    # Raw PCG64DXSM permutations follow the config literally; they are not
    # outcome-dependent rejection samples that force every index to move.
    if dataset == "geant":
        assert int(np.sum(plan.permutation == np.arange(count))) == 1


def test_seed_bundle_and_mask_family_change_the_fixed_permutation() -> None:
    from experiments.acil_innovation_v1.derangement import build_derangement

    base = build_derangement("geant", 1, "random")
    different_seed = build_derangement("geant", 2, "random")
    different_mask = build_derangement("geant", 1, "internal_block")

    assert base.permutation_sha256 != different_seed.permutation_sha256
    assert base.permutation_sha256 != different_mask.permutation_sha256
    assert not np.array_equal(base.permutation, different_seed.permutation)
    assert not np.array_equal(base.permutation, different_mask.permutation)


def test_apply_jointly_permutes_wf_scalar_values_and_preserves_multiset() -> None:
    from experiments.acil_innovation_v1.derangement import build_derangement

    plan = build_derangement("geant", 1, "random")
    windows = plan.identity.registered_window_count
    flows = plan.identity.flow_count
    innovations = torch.arange(windows * flows, dtype=torch.float32).reshape(
        windows, flows, 1
    )
    original = innovations.clone()

    result = plan.apply(innovations)
    expected = innovations.reshape(-1).index_select(
        0, torch.from_numpy(np.array(plan.permutation, copy=True))
    ).reshape_as(innovations)

    assert result.shape == innovations.shape
    assert result.dtype == innovations.dtype
    assert result.device.type == "cpu"
    assert result.data_ptr() != innovations.data_ptr()
    assert torch.equal(result, expected)
    assert torch.equal(innovations, original)
    assert torch.equal(torch.sort(result.reshape(-1)).values, innovations.reshape(-1))

    with pytest.raises(ValueError, match="shape"):
        plan.apply(innovations[:1])
    with pytest.raises(TypeError, match="tensor"):
        plan.apply(innovations.numpy())


def test_derangement_rejects_non_stage_i_identity_values() -> None:
    from experiments.acil_innovation_v1.derangement import build_derangement

    with pytest.raises(ValueError, match="seed bundle"):
        build_derangement("geant", 4, "random")
    with pytest.raises(ValueError, match="mask family"):
        build_derangement("geant", 1, "adversarial")
    with pytest.raises(ValueError, match="dataset"):
        build_derangement("unknown", 1, "random")
