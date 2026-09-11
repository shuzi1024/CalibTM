from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        head = path.read_text(errors="ignore")[:200]
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def exists_and_ready(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        return any(path.iterdir())
    return path.stat().st_size > 0 and not is_lfs_pointer(path)


def main() -> int:
    checks = {
        "requirements.txt": exists_and_ready(PROJECT_ROOT / "requirements.txt"),
        "GPT2": exists_and_ready(PROJECT_ROOT / "GPT2"),
        "datasets/net_traffic/Abilene/abilene.csv": exists_and_ready(
            PROJECT_ROOT / "datasets" / "net_traffic" / "Abilene" / "abilene.csv"
        ),
        "datasets/net_traffic/Abilene/abilene_result.csv": exists_and_ready(
            PROJECT_ROOT / "datasets" / "net_traffic" / "Abilene" / "abilene_result.csv"
        ),
        "datasets/net_traffic/GEANT/geant.csv": exists_and_ready(
            PROJECT_ROOT / "datasets" / "net_traffic" / "GEANT" / "geant.csv"
        ),
        "datasets/net_traffic/wsdream/wsdream.csv": exists_and_ready(
            PROJECT_ROOT / "datasets" / "net_traffic" / "wsdream" / "wsdream.csv"
        ),
    }

    for name, ok in checks.items():
        print(f"{'OK' if ok else 'MISSING':8} {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
