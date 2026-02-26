"""
Mutation Logger

Appends a JSONL record for every change made to a note:
    {"timestamp": "...", "file": "...", "action": "backlink_added", "detail": {...}}

Also sets up root Python logging for the rag_engine package.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import RAGConfig


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for rag_engine with console + rotating file handler."""
    root = logging.getLogger("rag_engine")
    root.setLevel(level)
    root.propagate = False

    if root.handlers:
        return  # already configured

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file
    RAGConfig.setup_dirs()
    fh = logging.handlers.RotatingFileHandler(
        RAGConfig.DATA_DIR / "rag_engine.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


class MutationLogger:
    """Append-only JSONL audit trail of all file mutations."""

    def __init__(self) -> None:
        RAGConfig.setup_dirs()
        self._path = RAGConfig.MUTATION_LOG_PATH

    def log(self, file_path: str, action: str, detail: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file": file_path,
            "action": action,
            "detail": detail,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Convenience wrappers
    def log_indexed(self, file_path: str, chunk_count: int) -> None:
        self.log(file_path, "indexed", {"chunk_count": chunk_count})

    def log_backlinks_added(self, file_path: str, targets: list) -> None:
        self.log(file_path, "backlinks_added", {"targets": targets})

    def log_tags_assigned(self, file_path: str, tags: list) -> None:
        self.log(file_path, "tags_assigned", {"tags": tags})

    def log_entities_linked(self, file_path: str, entities: list) -> None:
        self.log(file_path, "entities_linked", {"entities": entities})

    def log_deleted(self, file_path: str) -> None:
        self.log(file_path, "deleted", {})
