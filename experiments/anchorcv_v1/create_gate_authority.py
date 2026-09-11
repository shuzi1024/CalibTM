"""Create a final-gate authority only after exact artifact re-review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-output-root", required=True, type=Path)
    parser.add_argument("--extension-output-root", required=True, type=Path)
    parser.add_argument("--extension-report", required=True, type=Path)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _reject_existing_output(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"cannot inspect gate-authority output: {path}") from exc
    raise FileExistsError(
        f"gate-authority output already exists; overwrite is forbidden: {path}"
    )


def main() -> int:
    args = build_parser().parse_args()
    _reject_existing_output(args.output)

    # Delayed import keeps parser inspection independent from artifact/data code.
    from . import gate_identity

    authority = gate_identity.create_verified_gate_authority(
        extension_report=args.extension_report,
        freeze_record=args.freeze_record,
        prototype_output_root=args.prototype_output_root,
        extension_output_root=args.extension_output_root,
    )
    gate_identity.write_gate_authority(authority, args.output)
    print(
        json.dumps(
            {
                "method_freeze_sha256": authority.method_freeze_sha256,
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
