from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _as_yaml_block(data: Dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, sort_keys=False)
    except Exception:
        return json.dumps(data, indent=2)


def write_run_notebook(
    run_dir: Path,
    notes: str | None,
    cfg_train: Dict[str, Any],
    cfg_model: Dict[str, Any],
    cfg_dataset: Dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    commit = _get_git_commit()

    notes_text = notes.strip() if notes else "Add your run notes here."

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MILD Run Notes\n",
                f"\n- Timestamp: `{ts}`\n",
                f"- Git commit: `{commit}`\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [notes_text + "\n"],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Configs\n",
                "\n### Train\n",
                "```yaml\n",
                _as_yaml_block(cfg_train),
                "```\n",
                "\n### Model\n",
                "```yaml\n",
                _as_yaml_block(cfg_model),
                "```\n",
                "\n### Dataset\n",
                "```yaml\n",
                _as_yaml_block(cfg_dataset),
                "```\n",
            ],
        },
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path = run_dir / "notes.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
