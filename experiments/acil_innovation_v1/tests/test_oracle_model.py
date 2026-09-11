from __future__ import annotations

import inspect

import pytest
import torch

from experiments.acil_innovation_v1.acil import ACILBase
from experiments.acil_innovation_v1.models import QueryResidualModel
from experiments.acil_innovation_v1.oracle_model import (
    OracleSupportQ,
    TruthQDeepSets,
)
from experiments.acil_innovation_v1.preprocessing import FitFallback


def _fixture():
    observed = torch.zeros(1, 3, 7, dtype=torch.bool)
    observed[0, 0, [0, 3, 6]] = True
    observed[0, 1, [1, 2, 5]] = True
    observed[0, 2, [2, 4, 5]] = True
    values = torch.full((1, 3, 7), float("nan"))
    values[0, 0, [0, 3, 6]] = torch.tensor([2.0, 5.0, 11.0])
    values[0, 1, [1, 2, 5]] = torch.tensor([3.0, 7.0, 8.0])
    values[0, 2, [2, 4, 5]] = torch.tensor([4.0, 6.0, 9.0])
    support = OracleSupportQ(
        indices=torch.tensor([[2, 4, 6]]),
        values=torch.tensor([[[4.25], [7.5], [10.0]]]),
    )
    return values, observed, support, FitFallback(mean=5.0, std=2.0)


def _schema(module: torch.nn.Module):
    return tuple((name, tuple(value.shape)) for name, value in module.named_parameters())


def test_truth_q_forward_boundary_contains_only_compact_support_not_full_truth() -> None:
    assert tuple(inspect.signature(TruthQDeepSets.forward).parameters) == (
        "self",
        "values",
        "observed",
        "support_q",
        "fit_fallback",
    )
    assert tuple(OracleSupportQ.__dataclass_fields__) == ("indices", "values")
    forbidden = ("truth", "target", "endpoint", "relation", "flow_id", "topology")
    assert not any(
        token in name.lower()
        for name in inspect.signature(TruthQDeepSets.forward).parameters
        for token in forbidden
    )


def test_support_q_is_exactly_one_missing_entry_per_flow_and_disjoint_from_observed() -> None:
    values, observed, support, _ = _fixture()
    mask = support.support_mask(observed)

    assert mask.shape == values.shape
    assert torch.equal(mask.sum(dim=-1), torch.ones(1, 3, dtype=torch.long))
    assert not torch.any(mask & observed)

    with pytest.raises(ValueError, match="missing"):
        OracleSupportQ(
            indices=torch.tensor([[0, 2, 4]]),
            values=support.values,
        ).support_mask(observed)
    with pytest.raises(ValueError, match="shape"):
        OracleSupportQ(
            indices=torch.tensor([0, 2, 4]),
            values=support.values,
        ).support_mask(observed)


def test_oracle_innovation_is_exact_q_truth_minus_acil_q_over_observed_scale() -> None:
    torch.manual_seed(120)
    values, observed, support, fallback = _fixture()
    model = TruthQDeepSets(ACILBase()).eval()

    features = model.prepare_features(values, observed, support, fallback)

    q_base = features.full_result.prediction.gather(
        -1, support.indices.unsqueeze(-1)
    )
    assert torch.equal(
        features.innovation_raw,
        support.values - q_base,
    )
    assert torch.allclose(
        features.innovation_normalized,
        features.innovation_raw / features.full_result.statistics.std,
    )


def test_e_payload_perturbation_cannot_change_oracle_prediction() -> None:
    torch.manual_seed(121)
    values, observed, support, fallback = _fixture()
    model = TruthQDeepSets(ACILBase()).eval()
    support_mask = support.support_mask(observed)
    evaluation_e = (~observed) & (~support_mask)
    changed = values.clone()
    changed[evaluation_e] = torch.linspace(-1e12, 1e12, int(evaluation_e.sum()))
    changed[support_mask] = -7e11  # Q payload in values is also forbidden/ignored.

    with torch.no_grad():
        first = model(values, observed, support, fallback)
        second = model(changed, observed, support, fallback)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.all(first >= 0)
    assert torch.equal(
        first.contiguous().view(torch.int32).masked_select(observed),
        values.contiguous().view(torch.int32).masked_select(observed),
    )


def test_oracle_uses_deepsets_compatible_flow_query_modules_and_is_permutation_equivariant() -> None:
    torch.manual_seed(122)
    values, observed, support, fallback = _fixture()
    oracle = TruthQDeepSets(ACILBase()).eval()
    torch.manual_seed(122)
    deployable = QueryResidualModel("deepsets", ACILBase()).eval()
    assert _schema(oracle.flow_encoder) == _schema(deployable.flow_encoder)
    assert _schema(oracle.query_encoder) == _schema(deployable.query_encoder)
    assert _schema(oracle.decoder) == _schema(deployable.decoder)

    permutation = torch.tensor([2, 0, 1])
    permuted_support = OracleSupportQ(
        indices=support.indices[:, permutation],
        values=support.values[:, permutation],
    )
    with torch.no_grad():
        baseline = oracle(values, observed, support, fallback)
        permuted = oracle(
            values[:, permutation],
            observed[:, permutation],
            permuted_support,
            fallback,
        )
    assert torch.allclose(permuted, baseline[:, permutation], rtol=1e-5, atol=1e-6)
