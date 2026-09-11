from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch


def test_epoch_order_is_deterministic_complete_and_method_free() -> None:
    from experiments.acil_innovation_v1.training import epoch_window_order

    assert "method" not in inspect.signature(epoch_window_order).parameters
    first = epoch_window_order(seed_bundle=2, epoch=7)
    second = epoch_window_order(seed_bundle=2, epoch=7)
    assert np.array_equal(first, second)
    assert sorted(map(int, first)) == list(range(512))
    assert not np.array_equal(first, epoch_window_order(seed_bundle=2, epoch=8))


def test_normalized_target_mae_uses_only_fixed_target() -> None:
    from experiments.acil_innovation_v1.training import normalized_target_mae

    truth = torch.tensor([[[10.0, 20.0, 30.0]]])
    prediction = torch.tensor([[[1000.0, 18.0, 34.0]]])
    target = torch.tensor([[[False, True, True]]])
    scale = torch.tensor([[[2.0]]])
    assert normalized_target_mae(prediction, truth, target, scale).item() == pytest.approx(1.5)


def test_acil_loss_pairs_k3_and_middle_dropped_k2_equally() -> None:
    from experiments.acil_innovation_v1.acil import ACILBase
    from experiments.acil_innovation_v1.preprocessing import FitFallback
    from experiments.acil_innovation_v1.training import acil_paired_objective

    torch.manual_seed(301)
    truth = torch.arange(1.0, 8.0).reshape(1, 1, 7)
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., [0, 3, 6]] = True
    loss, details = acil_paired_objective(
        ACILBase(), truth, observed, FitFallback(mean=4.0, std=2.0)
    )
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(
        0.5 * (details.k3_loss.item() + details.k2_loss.item())
    )
    assert details.k3_target_count == 4
    assert details.k2_target_count == 5


def test_acil_separated_objective_never_requires_missing_truth_in_model_input() -> None:
    from experiments.acil_innovation_v1.acil import ACILBase
    from experiments.acil_innovation_v1.preprocessing import FitFallback
    from experiments.acil_innovation_v1.training import (
        acil_paired_objective_separated,
    )

    torch.manual_seed(302)
    truth = torch.arange(1.0, 8.0).reshape(1, 1, 7)
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[..., [0, 3, 6]] = True
    model_input = truth.clone()
    model_input[~observed] = float("nan")
    loss, details = acil_paired_objective_separated(
        ACILBase(),
        model_input=model_input,
        truth=truth,
        observed=observed,
        fit_fallback=FitFallback(mean=4.0, std=2.0),
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(details.k3_loss)
    assert torch.isfinite(details.k2_loss)


def test_checkpoint_selection_pools_raw_three_mask_sums() -> None:
    from experiments.acil_innovation_v1.training import choose_checkpoint_epoch

    rows = [
        {"epoch": 0, "mask_family": "random", "ae": 1.0, "truth": 1.0},
        {"epoch": 0, "mask_family": "internal_block", "ae": 1.0, "truth": 99.0},
        {"epoch": 0, "mask_family": "two_burst", "ae": 1.0, "truth": 100.0},
        {"epoch": 1, "mask_family": "random", "ae": 0.2, "truth": 1.0},
        {"epoch": 1, "mask_family": "internal_block", "ae": 2.0, "truth": 99.0},
        {"epoch": 1, "mask_family": "two_burst", "ae": 2.0, "truth": 100.0},
    ]
    # Epoch 0 wins pooled ratio (3/200 < 4.2/200), despite epoch 1 winning random.
    assert choose_checkpoint_epoch(rows) == 0


def test_model_method_mapping_is_frozen() -> None:
    from experiments.acil_innovation_v1.training import residual_model_kind

    assert residual_model_kind("local_loo") == "local"
    assert residual_model_kind("global_loo") == "deepsets"
    assert residual_model_kind("loo_deepsets") == "deepsets"
    assert residual_model_kind("full_u0") == "full_u0"
    assert residual_model_kind("full_scratch") == "gpt2_scratch"
    assert residual_model_kind("full_gpt2") == "gpt2_set"
    with pytest.raises(ValueError):
        residual_model_kind("dynamic_orbit")


class _ToyOptimModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = torch.nn.Linear(1, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.backbone = torch.nn.Linear(4, 4)
        self.output = torch.nn.Linear(4, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.input(value))
        return self.output(torch.tanh(self.backbone(hidden)))


def test_optimizer_partition_pins_pretrained_lr_and_no_decay() -> None:
    from experiments.acil_innovation_v1.training import build_protocol_optimizer

    model = _ToyOptimModel()
    pretrained = tuple(model.backbone.parameters())
    optimizer, audit = build_protocol_optimizer(
        model, pretrained_parameters=pretrained
    )

    assert set(audit) == {
        "new_decay",
        "new_no_decay",
        "pretrained_decay",
        "pretrained_no_decay",
    }
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        tuple(model.parameters())
    )
    by_name = {group["group_name"]: group for group in optimizer.param_groups}
    assert by_name["new_decay"]["lr"] == pytest.approx(3e-4)
    assert by_name["pretrained_decay"]["lr"] == pytest.approx(1e-5)
    assert by_name["new_decay"]["weight_decay"] == pytest.approx(0.01)
    assert by_name["new_no_decay"]["weight_decay"] == 0.0
    assert by_name["pretrained_no_decay"]["weight_decay"] == 0.0
    assert "norm.weight" in audit["new_no_decay"]
    assert "backbone.bias" in audit["pretrained_no_decay"]


def test_fixed_training_lane_runs_20_epochs_320_updates_and_keeps_earlier_tie() -> None:
    from experiments.acil_innovation_v1.training import (
        SourceDevResult,
        run_protocol_training,
    )

    model = torch.nn.Linear(1, 1, bias=False)

    def epoch_batches(epoch: int):
        assert 0 <= epoch < 20
        for offset in range(0, 512, 8):
            value = torch.arange(offset, offset + 8, dtype=torch.float32)
            yield value.reshape(-1, 1) / 512.0

    def loss_fn(active: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return active(batch).square().mean() + 0.01 * active.weight.square().mean()

    def source_dev(active: torch.nn.Module, epoch: int) -> SourceDevResult:
        del active
        # Epochs 3 and 4 tie. The earlier epoch must remain selected.
        value = 1.0 if epoch in {3, 4} else 10.0 + abs(epoch - 3)
        rows = tuple(
            {
                "epoch": epoch,
                "mask_family": family,
                "ae": value,
                "truth": 100.0,
            }
            for family in ("random", "internal_block", "two_burst")
        )
        return SourceDevResult(rows=rows)

    result = run_protocol_training(
        model,
        pretrained_parameters=(),
        epoch_batches=epoch_batches,
        compute_loss=loss_fn,
        evaluate_source_dev=source_dev,
        device_type="cpu",
    )
    assert result.epochs_completed == 20
    assert result.optimizer_updates == 320
    assert result.best_epoch == 3
    assert len(result.epochs) == 20
    assert sum(row.physical_batches for row in result.epochs) == 1280
    assert all(math.isfinite(row.mean_loss) for row in result.epochs)
    assert [row.epoch for row in result.epochs if row.selected_as_best] == [3]
    assert result.best_state
    assert all(tensor.device.type == "cpu" for tensor in result.best_state.values())
