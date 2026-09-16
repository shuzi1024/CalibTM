"""Compile the separate Chinese working draft with the existing Tectonic binary."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_COMPILER = ROOT / "analysis/overnight_20260915/runtime_deps/tex/tectonic"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=os.environ.get("TECTONIC", DEFAULT_COMPILER))
    parser.add_argument("--fetch", action="store_true", help="Allow Tectonic to fetch missing TeX packages into its cache")
    args = parser.parse_args()
    compiler = Path(args.compiler).expanduser().resolve()
    if not compiler.is_file():
        parser.error(f"Tectonic not found: {compiler}; pass --compiler /path/to/tectonic")
    build = HERE / "build"
    build.mkdir(exist_ok=True)
    command = [str(compiler), "--untrusted", "--keep-logs", "--keep-intermediates"]
    if not args.fetch:
        command.append("--only-cached")
    command += ["--outdir", str(build), str(HERE / "manuscript.tex")]
    subprocess.run(command, cwd=HERE, check=True)
    log = (build / "manuscript.log").read_text(errors="replace")
    if "Missing character:" in log:
        raise RuntimeError("Missing glyphs found in build/manuscript.log")
    pdf = build / "manuscript.pdf"
    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "compiler": str(compiler),
        "command": command,
        "pdf": str(pdf),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "purpose": "Chinese working draft in IEEEtran conference layout",
    }
    (build / "BUILD_INFO.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(pdf)


if __name__ == "__main__":
    main()
