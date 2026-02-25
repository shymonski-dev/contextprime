#!/usr/bin/env python3
"""Utility script to download model assets into the repo-local cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil

from loguru import logger

try:  # Delayed import so the script can explain missing deps clearly
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # pragma: no cover - handled by runtime messaging
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

try:  # pragma: no cover - optional dependency
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None

def _ensure_dependencies() -> None:
    missing = []
    if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
        missing.append("transformers/torch")
    if snapshot_download is None:
        missing.append("huggingface_hub")
    if missing:
        logger.error(
            "Missing dependencies: %s. Run 'pip install torch transformers huggingface_hub' and retry.",
            ", ".join(missing),
        )
        raise SystemExit(1)


def _resolve_base_dir(explicit_dir: str | None) -> Path:
    if explicit_dir:
        base = Path(explicit_dir).expanduser()
    else:
        from contextprime.core.config import get_settings

        settings = get_settings()
        base = Path(settings.paths.models_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _download_mono_t5(base_dir: Path, model_name: str, force: bool = False) -> None:
    target_dir = base_dir / model_name.replace("/", "_")

    has_snapshot = (target_dir / "config.json").exists()
    if target_dir.exists() and any(target_dir.iterdir()) and has_snapshot and not force:
        logger.info("MonoT5 weights already present at %s; skipping download", target_dir)
        return

    if target_dir.exists() and force:
        logger.info("Force flag supplied; clearing %s", target_dir)
        shutil.rmtree(target_dir)

    logger.info("Downloading %s into %s", model_name, target_dir)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.success("Downloaded MonoT5 weights to %s", target_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model assets for the RAG stack")
    parser.add_argument("--models-dir", help="Base directory for storing model weights", default=None)
    parser.add_argument(
        "--mono-t5-model",
        help="MonoT5 model identifier to download",
        default="castorini/monot5-base-msmarco-10k",
    )
    parser.add_argument(
        "--force",
        help="Re-download even if the target directory already exists",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_dependencies()
    base_dir = _resolve_base_dir(args.models_dir)
    try:
        _download_mono_t5(base_dir=base_dir, model_name=args.mono_t5_model, force=args.force)
    except Exception as err:  # pragma: no cover - conversion to exit code
        logger.error("Failed to download models: %s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
