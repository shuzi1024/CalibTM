from __future__ import annotations

import torch


def test_tensor_digest_is_ordered_and_bit_sensitive() -> None:
    from experiments.sc_acil_v1.smoke import tensor_digest

    first = tensor_digest({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
    reordered = tensor_digest({"b": torch.tensor([2.0]), "a": torch.tensor([1.0])})
    changed = tensor_digest({"a": torch.tensor([1.0]), "b": torch.tensor([3.0])})
    assert first == reordered
    assert first != changed
    assert len(first) == 64

