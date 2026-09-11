from __future__ import annotations

import json


def test_manifest_binds_protocol_jobs_sources_and_permitted_arrays() -> None:
    from experiments.sc_acil_v1.freeze import build_manifest

    first = build_manifest()
    second = build_manifest()
    assert first == second
    assert first["protocol"] == "sc-acil-v1"
    assert first["git_available"] is False
    assert first["git_commit"] is None
    assert len(first["protocol_sha256"]) == 64
    assert len(first["source_tree_sha256"]) == 64
    assert len(first["manifest_sha256"]) == 64
    assert len(first["jobs"]) == 18
    assert set(first["permitted_parent_semantic_sha256"]) == {
        "abilene/train",
        "abilene/val",
        "geant/train",
        "geant/val",
    }
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":")).encode("ascii")
    assert b"_test.npz" not in encoded.lower()
    assert "data_paths" not in first


def test_window_rows_are_flow_summed_and_preserve_ratio_of_sums_operands() -> None:
    import torch

    from experiments.sc_acil_v1.evidence import window_metric_rows

    truth = torch.tensor([[[1.0, 2.0, 3.0]], [[2.0, 4.0, 8.0]]])
    prediction = torch.tensor([[[1.0, 1.0, 5.0]], [[1.0, 4.0, 10.0]]])
    observed = torch.tensor([[[True, False, False]], [[True, False, False]]])
    rows = window_metric_rows(
        method="sc_acil",
        seed_bundle=4,
        dataset="abilene",
        mask_family="random",
        absolute_starts=(100, 150),
        truth=truth,
        prediction=prediction,
        observed=observed,
        mask_registry_sha256="a" * 64,
    )
    assert len(rows) == 2
    assert rows[0]["absolute_error_sum"] == 3.0
    assert rows[0]["absolute_truth_sum"] == 5.0
    assert rows[0]["target_count"] == 2
    assert rows[1]["absolute_error_sum"] == 2.0
    assert rows[1]["absolute_truth_sum"] == 12.0
