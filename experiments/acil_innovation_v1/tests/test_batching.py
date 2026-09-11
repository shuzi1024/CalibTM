from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from experiments.acil_innovation_v1.data import RegisteredWindows
from experiments.acil_innovation_v1.masks import (
    build_evaluation_mask,
    build_mixed_training_mask,
    mask_sha256,
    mixed_training_family,
    oracle_partition,
)
from experiments.acil_innovation_v1.registries import (
    dataset_spec,
    ordered_window_starts,
)
from experiments.acil_innovation_v1.training import epoch_window_order


def _synthetic_registered_windows(dataset: str, cohort: str) -> RegisteredWindows:
    starts = ordered_window_starts(dataset, cohort)
    flows = dataset_spec(dataset).flows
    window_axis = np.arange(len(starts), dtype=np.float32)[:, None, None]
    flow_axis = np.arange(flows, dtype=np.float32)[None, :, None]
    time_axis = np.arange(50, dtype=np.float32)[None, None, :]
    values = np.ascontiguousarray(
        window_axis * np.float32(10_000.0)
        + flow_axis * np.float32(50.0)
        + time_axis,
        dtype="<f4",
    )
    values.setflags(write=False)
    return RegisteredWindows(
        dataset=dataset,
        cohort=cohort,
        absolute_starts=starts,
        values=values,
    )


@pytest.fixture(scope="module")
def fit_windows() -> RegisteredWindows:
    return _synthetic_registered_windows("abilene", "fit")


@pytest.fixture(scope="module")
def evaluation_windows() -> RegisteredWindows:
    return _synthetic_registered_windows("abilene", "source_dev")


def test_batch_builders_expose_no_order_mask_method_or_device_override() -> None:
    from experiments.acil_innovation_v1.batching import (
        build_evaluation_batch,
        build_training_batch,
    )

    assert tuple(inspect.signature(build_training_batch).parameters) == (
        "windows",
        "seed_bundle",
        "epoch",
        "batch_index",
    )
    assert tuple(inspect.signature(build_evaluation_batch).parameters) == (
        "windows",
        "seed_bundle",
        "family",
    )


def test_training_batch_follows_epoch_order_and_scalar_masks_bitwise(
    fit_windows: RegisteredWindows,
) -> None:
    from experiments.acil_innovation_v1.batching import build_training_batch

    seed = 2
    epoch = 7
    batch_index = 3
    batch = build_training_batch(
        fit_windows,
        seed_bundle=seed,
        epoch=epoch,
        batch_index=batch_index,
    )
    positions = tuple(range(24, 32))
    order = epoch_window_order(seed_bundle=seed, epoch=epoch)
    indices = tuple(int(order[position]) for position in positions)
    starts = tuple(fit_windows.absolute_starts[index] for index in indices)

    assert batch.epoch_order_positions == positions
    assert batch.window_indices == indices
    assert batch.absolute_starts == starts
    assert batch.families == tuple(
        mixed_training_family(epoch, position, seed) for position in positions
    )
    expected_truth = torch.from_numpy(
        np.ascontiguousarray(fit_windows.values[list(indices)])
    )
    assert torch.equal(batch.truth, expected_truth)
    assert batch.truth.data_ptr() != batch.model_input.data_ptr()
    assert batch.truth.device.type == batch.model_input.device.type == "cpu"
    assert torch.isfinite(batch.truth).all()
    assert torch.isnan(batch.model_input.masked_select(~batch.observed)).all()
    assert torch.equal(
        batch.model_input.masked_select(batch.observed),
        batch.truth.masked_select(batch.observed),
    )
    assert not torch.any(batch.oracle_q & batch.oracle_e)
    assert torch.equal(batch.oracle_q | batch.oracle_e, batch.target)
    assert torch.equal(
        batch.oracle_q.sum(dim=-1),
        torch.ones_like(batch.oracle_support_q.indices),
    )
    assert torch.equal(
        batch.oracle_e.sum(dim=-1),
        torch.full_like(batch.oracle_support_q.indices, 46),
    )
    assert torch.equal(
        batch.oracle_support_q.support_mask(batch.observed), batch.oracle_q
    )
    assert torch.equal(
        batch.oracle_support_q.values,
        batch.truth.gather(-1, batch.oracle_support_q.indices.unsqueeze(-1)),
    )

    for row, (position, start) in enumerate(zip(positions, starts)):
        for flow in range(dataset_spec("abilene").flows):
            scalar = build_mixed_training_mask(
                dataset="abilene",
                window_start=start,
                flow_index=flow,
                seed_bundle=seed,
                epoch=epoch,
                epoch_order_position=position,
            )
            partition = oracle_partition(scalar)
            assert batch.mask_identities[row][flow] == scalar.identity
            assert batch.mask_sha256[row][flow] == mask_sha256(scalar)
            assert batch.oracle_q_sha256[row][flow] == partition.support_sha256
            assert batch.oracle_e_sha256[row][flow] == partition.evaluation_sha256
            assert np.array_equal(batch.observed[row, flow].numpy(), scalar.observed)
            assert np.array_equal(batch.target[row, flow].numpy(), scalar.target)
            assert np.array_equal(
                batch.controlled_gap[row, flow].numpy(), scalar.controlled_gap
            )
            assert np.array_equal(batch.oracle_q[row, flow].numpy(), partition.support)
            assert np.array_equal(
                batch.oracle_e[row, flow].numpy(), partition.evaluation
            )


def test_training_batch_uses_factory_hash_without_public_revalidation(
    monkeypatch, fit_windows: RegisteredWindows
) -> None:
    from experiments.acil_innovation_v1 import batching as batching_module

    def forbidden_public_revalidation(*args, **kwargs):
        raise AssertionError("batching must consume the factory-bound hash")

    monkeypatch.setattr(
        batching_module,
        "scalar_mask_sha256",
        forbidden_public_revalidation,
        raising=False,
    )
    monkeypatch.setattr(
        batching_module,
        "oracle_partition",
        forbidden_public_revalidation,
        raising=False,
    )
    batch = batching_module.build_training_batch(
        fit_windows, seed_bundle=1, epoch=0, batch_index=0
    )

    assert len(batch.mask_sha256) == 8
    assert all(len(row) == 144 for row in batch.mask_sha256)
    assert int(batch.oracle_q.sum()) == 8 * 144
    assert int(batch.oracle_e.sum()) == 8 * 144 * 46


def test_evaluation_batch_uses_registered_masks_q_and_e_bitwise(
    evaluation_windows: RegisteredWindows,
) -> None:
    from experiments.acil_innovation_v1.batching import build_evaluation_batch

    seed = 3
    family = "two_burst"
    batch = build_evaluation_batch(
        evaluation_windows,
        seed_bundle=seed,
        family=family,
    )

    assert batch.absolute_starts == evaluation_windows.absolute_starts
    assert batch.window_indices == tuple(range(len(evaluation_windows.absolute_starts)))
    assert batch.family == family
    assert batch.truth.data_ptr() != batch.model_input.data_ptr()
    assert batch.truth.device.type == batch.model_input.device.type == "cpu"
    assert torch.isnan(batch.model_input.masked_select(~batch.observed)).all()
    assert torch.equal(
        batch.model_input.masked_select(batch.observed),
        batch.truth.masked_select(batch.observed),
    )
    assert torch.equal(batch.target, ~batch.observed)
    assert not torch.any(batch.oracle_q & batch.oracle_e)
    assert torch.equal(batch.oracle_q | batch.oracle_e, batch.target)
    assert torch.equal(
        batch.oracle_q.sum(dim=-1),
        torch.ones_like(batch.oracle_support_q.indices),
    )
    assert torch.equal(
        batch.oracle_e.sum(dim=-1),
        torch.full_like(batch.oracle_support_q.indices, 46),
    )
    assert torch.equal(
        batch.oracle_support_q.support_mask(batch.observed), batch.oracle_q
    )
    assert torch.equal(
        batch.oracle_support_q.values,
        batch.truth.gather(-1, batch.oracle_support_q.indices.unsqueeze(-1)),
    )

    for row, start in enumerate(batch.absolute_starts):
        for flow in range(dataset_spec("abilene").flows):
            scalar = build_evaluation_mask(
                dataset="abilene",
                cohort="source_dev",
                window_start=start,
                flow_index=flow,
                family=family,
                seed_bundle=seed,
            )
            partition = oracle_partition(scalar)
            assert batch.mask_identities[row][flow] == scalar.identity
            assert batch.mask_sha256[row][flow] == mask_sha256(scalar)
            assert batch.oracle_q_sha256[row][flow] == partition.support_sha256
            assert batch.oracle_e_sha256[row][flow] == partition.evaluation_sha256
            assert np.array_equal(batch.observed[row, flow].numpy(), scalar.observed)
            assert np.array_equal(batch.target[row, flow].numpy(), scalar.target)
            assert np.array_equal(
                batch.controlled_gap[row, flow].numpy(), scalar.controlled_gap
            )
            assert np.array_equal(batch.oracle_q[row, flow].numpy(), partition.support)
            assert np.array_equal(
                batch.oracle_e[row, flow].numpy(), partition.evaluation
            )


def test_batching_rejects_wrong_cohort_schedule_and_batch_index(
    fit_windows: RegisteredWindows,
    evaluation_windows: RegisteredWindows,
) -> None:
    from experiments.acil_innovation_v1.batching import (
        build_evaluation_batch,
        build_training_batch,
    )

    with pytest.raises(ValueError, match="fit"):
        build_training_batch(
            evaluation_windows, seed_bundle=1, epoch=0, batch_index=0
        )
    with pytest.raises(ValueError, match="evaluation"):
        build_evaluation_batch(fit_windows, seed_bundle=1, family="random")
    with pytest.raises(ValueError, match="batch index"):
        build_training_batch(fit_windows, seed_bundle=1, epoch=0, batch_index=64)

    forged = RegisteredWindows(
        dataset=evaluation_windows.dataset,
        cohort=evaluation_windows.cohort,
        absolute_starts=tuple(reversed(evaluation_windows.absolute_starts)),
        values=evaluation_windows.values,
    )
    with pytest.raises(ValueError, match="schedule"):
        build_evaluation_batch(forged, seed_bundle=1, family="random")
