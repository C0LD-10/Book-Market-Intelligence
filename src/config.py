"""
Config loading utility.

Every path in config.yaml is stored relative to the project root. This
module resolves them to absolute paths at load time so that scripts, tests,
and notebooks all agree on "where things are" regardless of the current
working directory they were launched from.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml and resolve every entry under `paths:` to an
    absolute path rooted at the project directory.

    Parameters
    ----------
    config_path : path to config.yaml. Defaults to <project_root>/config.yaml.

    Returns
    -------
    dict with the same structure as config.yaml, plus `paths` values
    replaced by absolute path strings.
    """
    config_path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    resolved_paths = {}
    dir_keys = {"model_dir", "reports_dir", "figures_dir"}
    for key, rel_path in cfg["paths"].items():
        resolved = str((PROJECT_ROOT / rel_path).resolve())
        if key in dir_keys:
            resolved += os.sep  # guarantee callers can safely do f"{path}filename.ext"
        resolved_paths[key] = resolved
    cfg["paths"] = resolved_paths

    # Ensure output directories exist (idempotent).
    for key in ("model_dir", "reports_dir", "figures_dir"):
        os.makedirs(cfg["paths"][key], exist_ok=True)
    os.makedirs(Path(cfg["paths"]["clean_data"]).parent, exist_ok=True)

    return cfg


if __name__ == "__main__":
    import json

    print(json.dumps(load_config(), indent=2))
