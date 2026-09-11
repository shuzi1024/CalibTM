from __future__ import annotations

from experiments.anchorcv_v1.smoke import build_parser


def test_smoke_parser_has_no_cohort_mask_split_or_test_authority() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}

    assert destinations == {
        "help",
        "dataset",
        "seed_bundle",
        "device",
        "freeze_record",
        "output",
    }
    assert not {"cohort", "mask_family", "split", "test_path"} & destinations
