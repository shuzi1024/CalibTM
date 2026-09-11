"""Re-audit the fixed six-job extension grid and persist its immutable report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-output-root", required=True, type=Path)
    parser.add_argument("--extension-output-root", required=True, type=Path)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _reject_existing_output(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"cannot inspect extension-review output: {path}") from exc
    raise FileExistsError(
        f"extension-review output already exists; overwrite is forbidden: {path}"
    )


def main() -> int:
    args = build_parser().parse_args()
    _reject_existing_output(args.output)

    # Delayed import keeps parser inspection independent from artifact/data code.
    from . import review

    report = review.review_extension(
        prototype_output_root=args.prototype_output_root,
        extension_output_root=args.extension_output_root,
        freeze_record=args.freeze_record,
    )
    encoded = json.dumps(
        report,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "output": str(args.output),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
