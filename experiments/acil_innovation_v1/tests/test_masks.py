from __future__ import annotations

import inspect

import numpy as np
import pytest

from experiments.acil_innovation_v1.masks import (
    MaskBundle,
    build_evaluation_mask,
    build_mixed_training_mask,
    deployable_target,
    factory_mask_sha256,
    factory_oracle_partition,
    loo_observed,
    mask_sha256,
    middle_anchor_index,
    mixed_training_family,
    oracle_partition,
)
from experiments.acil_innovation_v1.registries import ordered_window_starts


FAMILIES = ("random", "internal_block", "two_burst")


def _evaluation_mask(family="random", *, dataset="abilene", cohort="tune", flow=0):
    return build_evaluation_mask(
        dataset=dataset,
        cohort=cohort,
        window_start=ordered_window_starts(dataset, cohort)[0],
        flow_index=flow,
        family=family,
        seed_bundle=1,
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_evaluation_masks_are_deterministic_k3_and_own_fixed_targets(family):
    first = _evaluation_mask(family)
    second = _evaluation_mask(family)

    assert first.identity.family == family
    assert first.identity.observed_count == 3
    assert first.observed.dtype == np.bool_
    assert first.observed.shape == (50,)
    assert int(first.observed.sum()) == 3
    assert int(first.target.sum()) == 47
    assert np.array_equal(first.target, ~first.observed)
    assert np.array_equal(first.observed, second.observed)
    assert mask_sha256(first) == mask_sha256(second)
    assert not first.observed.flags.writeable
    assert not first.target.flags.writeable


def test_factory_hash_is_bound_to_immutable_generated_content_and_public_hash_revalidates():
    generated = _evaluation_mask("random")

    assert factory_mask_sha256(generated) == mask_sha256(generated)
    with pytest.raises(ValueError):
        generated.observed.setflags(write=True)

    manual_equivalent = MaskBundle(
        identity=generated.identity,
        observed=generated.observed,
        target=generated.target,
        controlled_gap=generated.controlled_gap,
    )
    with pytest.raises(ValueError, match="factory"):
        factory_mask_sha256(manual_equivalent)
    assert mask_sha256(manual_equivalent) == mask_sha256(generated)
    trusted_partition = factory_oracle_partition(generated)
    public_partition = oracle_partition(generated)
    assert trusted_partition.mask_sha256 == factory_mask_sha256(generated)
    assert public_partition.mask_sha256 == mask_sha256(generated)
    assert np.array_equal(trusted_partition.support, public_partition.support)
    assert np.array_equal(trusted_partition.evaluation, public_partition.evaluation)
    assert trusted_partition.support_sha256 == public_partition.support_sha256
    assert trusted_partition.evaluation_sha256 == public_partition.evaluation_sha256
    with pytest.raises(ValueError, match="factory"):
        factory_oracle_partition(manual_equivalent)

    changed_observed = np.zeros(50, dtype=np.bool_)
    replacement = next(
        indices
        for indices in ((0, 1, 2), (3, 4, 5), (6, 7, 8))
        if not np.array_equal(np.flatnonzero(generated.observed), indices)
    )
    changed_observed[list(replacement)] = True
    tampered = MaskBundle(
        identity=generated.identity,
        observed=changed_observed,
        target=~changed_observed,
        controlled_gap=np.zeros(50, dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="observed mask content drifted"):
        mask_sha256(tampered)


def test_structured_controlled_gaps_are_missing_and_have_exact_lengths():
    internal = _evaluation_mask("internal_block")
    burst = _evaluation_mask("two_burst")
    random = _evaluation_mask("random")

    assert int(internal.controlled_gap.sum()) == 8
    assert 6 <= int(burst.controlled_gap.sum()) <= 16
    assert int(random.controlled_gap.sum()) == 0
    for bundle in (internal, burst, random):
        assert not np.any(bundle.controlled_gap & bundle.observed)


def test_k3_has_exactly_one_strict_two_sided_loo_anchor():
    for family in FAMILIES:
        bundle = _evaluation_mask(family)
        anchors = np.flatnonzero(bundle.observed)
        middle = middle_anchor_index(bundle)
        reduced = loo_observed(bundle)

        assert middle == int(anchors[1])
        assert anchors[0] < middle < anchors[2]
        assert int(reduced.sum()) == 2
        assert reduced[anchors[0]] and reduced[anchors[2]]
        assert not reduced[middle]
        assert not reduced.flags.writeable


def test_mixed_training_family_schedule_is_exact_balanced_and_method_free():
    signature = inspect.signature(build_mixed_training_mask)
    assert "family" not in signature.parameters
    assert "method" not in signature.parameters

    expected_extra = {1: "random", 2: "internal_block", 3: "two_burst"}
    for bundle in (1, 2, 3):
        counts = {family: 0 for family in FAMILIES}
        for global_position in range(20 * 512):
            epoch, position = divmod(global_position, 512)
            counts[mixed_training_family(epoch, position, bundle)] += 1
        assert sorted(counts.values()) == [3413, 3413, 3414]
        assert counts[expected_extra[bundle]] == 3414

    first_start = ordered_window_starts("geant", "fit")[0]
    built = build_mixed_training_mask(
        dataset="geant",
        window_start=first_start,
        flow_index=0,
        seed_bundle=2,
        epoch=7,
        epoch_order_position=11,
    )
    assert built.identity.family == mixed_training_family(7, 11, 2)
    assert built.identity.purpose == "training"
    assert built.identity.replica == 7


@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_q_and_e_are_identity_fixed_disjoint_and_not_deployable(family):
    bundle = _evaluation_mask(family)
    first = oracle_partition(bundle)
    second = oracle_partition(bundle)
    u = deployable_target(bundle)

    assert int(first.support.sum()) == 1
    assert int(first.evaluation.sum()) == 46
    assert not np.any(first.support & first.evaluation)
    assert np.array_equal(first.support | first.evaluation, u)
    assert np.array_equal(first.support, second.support)
    assert first.support_sha256 == second.support_sha256
    assert first.evaluation_sha256 == second.evaluation_sha256
    assert int(u.sum()) == 47
    assert not u.flags.writeable

    if family != "random":
        assert not np.any(first.support & bundle.controlled_gap)


def test_oracle_identity_and_builders_accept_no_truth_or_method_operand():
    assert "truth" not in inspect.signature(oracle_partition).parameters
    assert "method" not in inspect.signature(oracle_partition).parameters
    assert "truth" not in inspect.signature(build_evaluation_mask).parameters
    assert "method" not in inspect.signature(build_evaluation_mask).parameters


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (build_evaluation_mask, {"family": "unknown"}),
        (build_evaluation_mask, {"seed_bundle": 7}),
        (build_evaluation_mask, {"cohort": "gate", "window_start": 0}),
    ],
)
def test_mask_builders_fail_closed_on_unregistered_identity(builder, kwargs):
    base = {
        "dataset": "abilene",
        "cohort": "tune",
        "window_start": ordered_window_starts("abilene", "tune")[0],
        "flow_index": 0,
        "family": "random",
        "seed_bundle": 1,
    }
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        builder(**base)
