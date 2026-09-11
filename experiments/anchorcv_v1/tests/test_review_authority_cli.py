from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from experiments.anchorcv_v1 import create_gate_authority
from experiments.anchorcv_v1 import gate_identity
from experiments.anchorcv_v1 import review
from experiments.anchorcv_v1 import run_extension_review


def _destinations(parser: object) -> set[str]:
    return {action.dest for action in parser._actions}  # type: ignore[attr-defined]


def test_extension_review_parser_exposes_only_fixed_artifact_paths() -> None:
    destinations = _destinations(run_extension_review.build_parser())

    assert destinations == {
        "help",
        "prototype_output_root",
        "extension_output_root",
        "freeze_record",
        "output",
    }
    assert not any(
        forbidden in destination
        for destination in destinations
        for forbidden in ("dataset", "cohort", "split", "test", "data")
    )


def test_gate_authority_parser_exposes_only_rereview_inputs() -> None:
    parser = create_gate_authority.build_parser()
    destinations = _destinations(parser)

    assert destinations == {
        "help",
        "prototype_output_root",
        "extension_output_root",
        "extension_report",
        "freeze_record",
        "output",
    }
    assert not any(
        forbidden in destination
        for destination in destinations
        for forbidden in ("dataset", "cohort", "split", "test", "data")
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--prototype-output-root",
                "/tmp/prototype",
                "--extension-output-root",
                "/tmp/extension",
                "--extension-report",
                "/tmp/report.json",
                "--freeze-record",
                "/tmp/freeze.json",
                "--output",
                "/tmp/authority.json",
                "--authority",
                "/tmp/unverified-authority.json",
            ]
        )


def test_extension_review_main_forwards_exact_paths_and_writes_finite_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prototype_root = tmp_path / "prototype"
    extension_root = tmp_path / "extension"
    freeze_record = tmp_path / "freeze.json"
    output = tmp_path / "reviews" / "extension.json"
    report = {
        "verdict": "proceed",
        "nested": {"z": 2, "a": 1.25},
        "label": "证据",
    }
    calls: list[dict[str, Path]] = []

    def fake_review_extension(**kwargs: Path) -> dict[str, object]:
        calls.append(dict(kwargs))
        return report

    monkeypatch.setattr(review, "review_extension", fake_review_extension)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_extension_review",
            "--prototype-output-root",
            str(prototype_root),
            "--extension-output-root",
            str(extension_root),
            "--freeze-record",
            str(freeze_record),
            "--output",
            str(output),
        ],
    )

    assert run_extension_review.main() == 0
    assert calls == [
        {
            "prototype_output_root": prototype_root,
            "extension_output_root": extension_root,
            "freeze_record": freeze_record,
        }
    ]
    assert output.read_text(encoding="utf-8") == (
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    assert output.stat().st_mode & 0o222 == 0
    assert capsys.readouterr().out == (
        json.dumps(
            {"verdict": "proceed", "output": str(output)},
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def test_extension_review_rejects_nonfinite_report_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "review.json"
    monkeypatch.setattr(
        review,
        "review_extension",
        lambda **kwargs: {"verdict": "proceed", "bad": float("nan")},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_extension_review",
            "--prototype-output-root",
            str(tmp_path / "prototype"),
            "--extension-output-root",
            str(tmp_path / "extension"),
            "--freeze-record",
            str(tmp_path / "freeze.json"),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(ValueError):
        run_extension_review.main()
    assert not output.exists()


def test_extension_review_refuses_existing_output_without_rereview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "review.json"
    output.write_text("keep-me\n", encoding="utf-8")

    def forbidden_review(**kwargs: Path) -> dict[str, object]:
        raise AssertionError("an existing immutable output must fail before review")

    monkeypatch.setattr(review, "review_extension", forbidden_review)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_extension_review",
            "--prototype-output-root",
            str(tmp_path / "prototype"),
            "--extension-output-root",
            str(tmp_path / "extension"),
            "--freeze-record",
            str(tmp_path / "freeze.json"),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(FileExistsError, match="already exists"):
        run_extension_review.main()
    assert output.read_text(encoding="utf-8") == "keep-me\n"


def test_gate_authority_main_forwards_all_rereview_paths_then_uses_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prototype_root = tmp_path / "prototype"
    extension_root = tmp_path / "extension"
    extension_report = tmp_path / "extension-report.json"
    freeze_record = tmp_path / "freeze.json"
    output = tmp_path / "authority" / "gate.json"
    method_hash = "a" * 64

    class FakeAuthority:
        method_freeze_sha256 = method_hash

    authority = FakeAuthority()
    factory_calls: list[dict[str, Path]] = []
    writer_calls: list[tuple[object, Path]] = []

    def fake_factory(**kwargs: Path) -> FakeAuthority:
        factory_calls.append(dict(kwargs))
        return authority

    def fake_writer(value: object, path: Path) -> None:
        writer_calls.append((value, path))

    monkeypatch.setattr(
        gate_identity,
        "create_verified_gate_authority",
        fake_factory,
    )
    monkeypatch.setattr(gate_identity, "write_gate_authority", fake_writer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_gate_authority",
            "--prototype-output-root",
            str(prototype_root),
            "--extension-output-root",
            str(extension_root),
            "--extension-report",
            str(extension_report),
            "--freeze-record",
            str(freeze_record),
            "--output",
            str(output),
        ],
    )

    assert create_gate_authority.main() == 0
    assert factory_calls == [
        {
            "extension_report": extension_report,
            "freeze_record": freeze_record,
            "prototype_output_root": prototype_root,
            "extension_output_root": extension_root,
        }
    ]
    assert writer_calls == [(authority, output)]
    assert capsys.readouterr().out == (
        json.dumps(
            {
                "method_freeze_sha256": method_hash,
                "output": str(output),
            },
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def test_gate_authority_refuses_existing_output_before_expensive_rereview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "gate-authority.json"
    output.write_text("keep-me\n", encoding="utf-8")

    def forbidden_factory(**kwargs: Path) -> object:
        raise AssertionError("existing immutable authority must fail before re-review")

    def forbidden_writer(value: object, path: Path) -> None:
        raise AssertionError("existing immutable authority must not reach writer")

    monkeypatch.setattr(
        gate_identity,
        "create_verified_gate_authority",
        forbidden_factory,
    )
    monkeypatch.setattr(
        gate_identity,
        "write_gate_authority",
        forbidden_writer,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_gate_authority",
            "--prototype-output-root",
            str(tmp_path / "prototype"),
            "--extension-output-root",
            str(tmp_path / "extension"),
            "--extension-report",
            str(tmp_path / "extension-report.json"),
            "--freeze-record",
            str(tmp_path / "freeze.json"),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(FileExistsError, match="already exists"):
        create_gate_authority.main()
    assert output.read_text(encoding="utf-8") == "keep-me\n"
