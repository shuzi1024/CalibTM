from __future__ import annotations

import torch


def test_all_feature_variants_are_exactly_parameter_matched() -> None:
    from experiments.acil_mechanism_v1.model import new_acil

    models = {
        method: new_acil(method, model_seed=41001)
        for method in ("full", "value_only", "no_anchor")
    }
    counts = {
        method: sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        for method, model in models.items()
    }

    assert counts == {
        "full": 5475,
        "value_only": 5475,
        "no_anchor": 5475,
    }


def test_method_changes_only_registered_feature_mask() -> None:
    from experiments.acil_mechanism_v1.model import new_acil

    full = new_acil("full", model_seed=41001)
    value = new_acil("value_only", model_seed=41001)
    no_anchor = new_acil("no_anchor", model_seed=41001)

    assert full.extractor.feature_set == "full"
    assert value.extractor.feature_set == "value_only"
    assert no_anchor.extractor.feature_set == "no_anchor_values"
    full_state = full.state_dict()
    assert all(
        torch.equal(full_state[name], value.state_dict()[name])
        and torch.equal(full_state[name], no_anchor.state_dict()[name])
        for name in full_state
    )
