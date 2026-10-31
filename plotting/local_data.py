"""Portable local input selection and allowlisted figure provenance."""
import hashlib
import json
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")


def source_path(data_dir, relative):
    """Resolve an existing input inside the selected dataset directory."""
    root = Path(data_dir).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("input filenames must be relative to the data directory")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing local plot input {relative.as_posix()!r}; supply it in --data-dir. "
            "Create datasets.json from assets/datasets.example.json and provide your own measurements. "
            "Plotting never downloads data."
        )
    return path


def load_manifest(data_dir=ASSETS):
    with source_path(data_dir, "datasets.json").open(encoding="utf-8") as stream:
        datasets = json.load(stream)
    if not isinstance(datasets, dict):
        raise ValueError("datasets.json must contain an object keyed by plot name")
    return datasets


def provenance(data_dir, paths):
    """Record relative filenames and content hashes, never original locations."""
    return [{"file": str(relative),
             "sha256": hashlib.sha256(source_path(data_dir, relative).read_bytes()).hexdigest()}
            for relative in dict.fromkeys(paths)]
