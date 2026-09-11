import argparse
import csv
import gzip
import os
import pickle
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree as ET

import numpy as np

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "downloads" / "raw"
DATA_ROOT = PROJECT_ROOT / "datasets" / "net_traffic"
HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))

ABILENE_URL = "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/AbileneTM-all.tar"
GEANT_CANDIDATE_URLS = (
    "https://raw.githubusercontent.com/duchuyle108/SDN-TMprediction/main/dataset/geant_flat_tms.csv",
    "https://raw.githubusercontent.com/duchuyle108/SDN-TMprediction/master/dataset/geant_flat_tms.csv",
    "https://app.box.com/shared/static/shzgaxnt36org6dmu9q228kzk28numue?dl=1",
)
WSDREAM_CANDIDATE_URLS = (
    "https://zenodo.org/records/1133476/files/wsdream_dataset2.zip?download=1",
    "https://zenodo.org/records/1133476/files/wsdream_dataset1.zip?download=1",
    "https://raw.githubusercontent.com/rzhu3/WS-DREAM/master/data/rtMatrix.txt",
    "https://raw.githubusercontent.com/LanternYing/Dataset/master/QoS/rtMatrix.txt",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        head = path.read_text(errors="ignore")[:200]
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def download_url(url: str, dest: Path) -> Path:
    ensure_dir(dest.parent)
    req = urllib.request.Request(url, headers={"User-Agent": "ARI-LLM-setup/1.0"})
    with urllib.request.urlopen(req) as resp:
        target = dest
        filename = resp.headers.get_filename()
        if filename:
            target = dest.parent / filename
        with open(target, "wb") as f:
            shutil.copyfileobj(resp, f)
    return target


def try_download(urls: Iterable[str], dest_dir: Path) -> Optional[Path]:
    ensure_dir(dest_dir)
    for idx, url in enumerate(urls, start=1):
        try:
            return download_url(url, dest_dir / f"download_{idx}")
        except (urllib.error.URLError, urllib.error.HTTPError):
            continue
    return None


def save_csv(matrix: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(matrix.tolist())


def prepare_abilene(force: bool = False) -> Path:
    out_dir = DATA_ROOT / "Abilene"
    csv_path = out_dir / "abilene.csv"
    aux_path = out_dir / "abilene_result.csv"
    if csv_path.exists() and aux_path.exists() and not force and not is_lfs_pointer(csv_path):
        return csv_path

    raw_dir = RAW_ROOT / "abilene"
    ensure_dir(raw_dir)
    tar_path = raw_dir / "AbileneTM-all.tar"
    if force or not tar_path.exists():
        tar_path = download_url(ABILENE_URL, tar_path)

    extract_dir = raw_dir / "AbileneTM-all"
    ensure_dir(extract_dir)
    if force or not any(extract_dir.iterdir()):
        with tarfile.open(tar_path) as tar:
            tar.extractall(extract_dir)

    for gz_path in extract_dir.rglob("*.gz"):
        target = gz_path.with_suffix("")
        if force or not target.exists():
            with gzip.open(gz_path, "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    rows = []
    tm_files = sorted(
        p for p in extract_dir.rglob("*") if p.is_file() and p.name.startswith("X") and p.suffix != ".gz"
    )
    for tm_file in tm_files:
        with open(tm_file) as f:
            for line in f:
                values = np.fromstring(line, sep=" ")
                if values.size == 0 or values.size % 5 != 0:
                    continue
                matrix = values.reshape(-1, 5)
                if matrix.shape[0] >= 144:
                    rows.append(matrix[:144, 0])

    if not rows:
        raise RuntimeError("Failed to parse Abilene traffic matrices.")

    matrix = np.asarray(rows, dtype=np.float64)
    save_csv(matrix, csv_path)
    save_csv(matrix, aux_path)
    return csv_path


def maybe_extract_archive(path: Path, out_dir: Path) -> Path:
    if zipfile.is_zipfile(path):
        ensure_dir(out_dir)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
        return out_dir
    return path


def normalize_geant_csv(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise RuntimeError("GEANT dataset is not a 2D matrix.")
    if arr.shape[1] >= 529:
        return arr[:, :529]
    if arr.shape[1] >= 300:
        return arr
    raise RuntimeError(f"GEANT dataset has too few columns: {arr.shape[1]}")


def parse_geant_pickle_folder(folder: Path) -> np.ndarray:
    def numeric_key(path: Path) -> int:
        stem = path.stem[1:] if path.stem.startswith("t") else path.stem
        return int(stem)

    rows = []
    for pkl_path in sorted(folder.rglob("t*.pkl"), key=numeric_key):
        with open(pkl_path, "rb") as f:
            rows.append(np.asarray(pickle.load(f), dtype=np.float64).reshape(-1))
    if not rows:
        raise RuntimeError("No GEANT pickle snapshots were found.")
    return np.asarray(rows, dtype=np.float64)


def parse_geant_xml_folder(folder: Path) -> np.ndarray:
    matrices = []
    for xml_path in sorted(folder.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        demand_map = {}
        for elem in root.iter():
            if not elem.tag.lower().endswith("demand"):
                continue
            src = elem.attrib.get("source") or elem.attrib.get("src")
            dst = elem.attrib.get("target") or elem.attrib.get("dst")
            value = elem.attrib.get("demandValue") or elem.attrib.get("value") or elem.text
            if src is None or dst is None or value is None:
                continue
            try:
                demand_map[(src, dst)] = float(value)
            except ValueError:
                continue
        if not demand_map:
            continue
        nodes = sorted({src for src, _ in demand_map} | {dst for _, dst in demand_map})
        if len(nodes) != 23:
            continue
        matrices.append([demand_map.get((src, dst), 0.0) for src in nodes for dst in nodes])
    if not matrices:
        raise RuntimeError("Failed to parse GEANT matrices from XML.")
    return np.asarray(matrices, dtype=np.float64)


def prepare_geant(force: bool = False) -> Path:
    out_dir = DATA_ROOT / "GEANT"
    csv_path = out_dir / "geant.csv"
    if csv_path.exists() and not force and not is_lfs_pointer(csv_path):
        return csv_path

    raw_dir = RAW_ROOT / "geant"
    download = try_download(GEANT_CANDIDATE_URLS, raw_dir)
    if download is None:
        raise RuntimeError("Failed to download GEANT dataset from candidate URLs.")

    extracted = maybe_extract_archive(download, raw_dir / "extracted")
    candidates = [extracted] if extracted.is_file() else sorted(extracted.rglob("*.csv"))
    for candidate in candidates:
        try:
            arr = np.loadtxt(candidate, delimiter=",")
            arr = normalize_geant_csv(arr)
            save_csv(arr, csv_path)
            return csv_path
        except Exception:
            continue

    for parser in (parse_geant_pickle_folder, parse_geant_xml_folder):
        try:
            arr = normalize_geant_csv(parser(extracted))
            save_csv(arr, csv_path)
            return csv_path
        except Exception:
            continue

    raise RuntimeError("Downloaded GEANT asset could not be converted to geant.csv.")


def download_wsdream_raw(force: bool = False) -> Optional[Path]:
    raw_dir = RAW_ROOT / "wsdream"
    ensure_dir(raw_dir)
    if not force:
        existing = list(raw_dir.iterdir())
        if existing:
            return existing[0]
    download = try_download(WSDREAM_CANDIDATE_URLS, raw_dir)
    if download is None:
        return None
    if zipfile.is_zipfile(download):
        extracted = raw_dir / "extracted"
        ensure_dir(extracted)
        with zipfile.ZipFile(download) as zf:
            zf.extractall(extracted)
        return extracted
    return download


def download_gpt2(force: bool = False) -> Path:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is not installed; cannot download gpt2.")
    target = PROJECT_ROOT / "GPT2"
    if target.exists() and any(target.iterdir()) and not force:
        return target
    ensure_dir(target)
    ensure_dir(HF_HOME / "hub")
    ensure_dir(HF_HOME / "transformers")
    snapshot_download(
        repo_id="gpt2",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "pytorch_model.bin",
            "model.safetensors",
        ],
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ARI-LLM local datasets and GPT2.")
    parser.add_argument("--datasets", nargs="*", choices=["abilene", "geant", "wsdream"], default=[])
    parser.add_argument("--gpt2", action="store_true", help="Download gpt2 into ./GPT2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.gpt2:
        print(f"Prepared GPT2 at {download_gpt2(force=args.force)}")

    for dataset in args.datasets:
        if dataset == "abilene":
            print(f"Prepared Abilene at {prepare_abilene(force=args.force)}")
        elif dataset == "geant":
            print(f"Prepared GEANT at {prepare_geant(force=args.force)}")
        elif dataset == "wsdream":
            path = download_wsdream_raw(force=args.force)
            if path is None:
                raise RuntimeError("Failed to download any WS-DREAM source asset.")
            print(
                f"Downloaded raw WS-DREAM asset to {path}; the repo-specific "
                "wsdream.csv preprocessing rule is not present in this codebase."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
