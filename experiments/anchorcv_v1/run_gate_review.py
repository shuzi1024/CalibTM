"""Write the one-shot AnchorCV final-gate review as an immutable JSON file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

from .gate_review import review_final_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--authority-record", required=True, type=Path)
    parser.add_argument("--extension-report", required=True, type=Path)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--prototype-output-root", required=True, type=Path)
    parser.add_argument("--extension-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _exclusive_readonly_json(path: Path, value: object) -> None:
    try:
        encoded = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("final review must be finite JSON") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("final-review parent must be a regular directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short final-review write")
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    args = build_parser().parse_args()
    report = review_final_gate(
        output_root=args.output_root,
        authority_record=args.authority_record,
        extension_report=args.extension_report,
        freeze_record=args.freeze_record,
        prototype_output_root=args.prototype_output_root,
        extension_output_root=args.extension_output_root,
    )
    if not isinstance(report, Mapping):
        raise TypeError("final-gate review must return a mapping")
    verdict = report.get("verdict")
    if verdict not in {"revise", "kill", "proceed"}:
        raise ValueError("final-gate verdict is invalid")
    _exclusive_readonly_json(args.output, report)
    print(
        json.dumps(
            {"output": str(args.output), "verdict": verdict},
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
