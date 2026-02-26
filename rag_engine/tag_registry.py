"""
Tag Registry Manager

Maintains a JSON file:
    {
        "tags": {
            "mindfulness": {"count": 3, "documents": ["path/a.md", "path/b.md", ...]},
            ...
        }
    }

Rules:
    • Prefer (reuse) existing tags over creating new ones
    • A new tag is accepted only if it appears in ≥ 2 documents OR is explicitly strong
    • Tags are lowercase, hyphenated
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from .config import RAGConfig

logger = logging.getLogger(__name__)


def _normalise_tag(tag: str) -> str:
    """Lowercase, strip special chars, replace spaces with hyphens."""
    tag = tag.lower().strip()
    tag = re.sub(r"[^a-z0-9\-]", "-", tag)
    tag = re.sub(r"-{2,}", "-", tag).strip("-")
    return tag


# Tags that are too generic to be useful
_BLOCKLIST: Set[str] = {
    "thoughts", "life", "general", "misc", "journal",
    "entry", "note", "notes", "today", "day",
}


class TagRegistry:
    """
    Persistent tag registry.  Thread-safe for single-process use.
    """

    def __init__(self) -> None:
        RAGConfig.setup_dirs()
        self._path = RAGConfig.TAG_REGISTRY_PATH
        self._data: Dict[str, Dict] = {}   # tag → {count, documents: []}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = raw.get("tags", {})
            except Exception as e:
                logger.warning("Could not load tag registry: %s", e)
                self._data = {}

    def save(self) -> None:
        self._path.write_text(
            json.dumps({"tags": self._data}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def all_tags(self) -> Dict[str, int]:
        """Return {tag: document_count}."""
        return {t: d["count"] for t, d in self._data.items()}

    def assign_tags_to_document(
        self,
        file_path: str,
        candidate_tags: List[str],
        new_tags_proposed: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Given candidates (from LLM), decide which to accept and register.
        Returns the final accepted tag list for the document.
        """
        accepted: List[str] = []

        all_candidates = list(candidate_tags) + (new_tags_proposed or [])
        for raw_tag in all_candidates:
            tag = _normalise_tag(raw_tag)
            if not tag or tag in _BLOCKLIST:
                continue

            if tag in self._data:
                # Existing tag — always accept
                accepted.append(tag)
                self._register(tag, file_path)
            else:
                # New tag — stage it; accept only after ≥ 2 docs mention it
                # We use a "pending" count stored with count=0 docs initially
                self._stage(tag, file_path)
                # Accept immediately if explicitly proposed (strong signal from LLM)
                if new_tags_proposed and raw_tag in new_tags_proposed:
                    # Still guard against garbage
                    if len(tag) >= 3:
                        accepted.append(tag)

        # Promote staged tags that now meet the threshold
        for tag in list(self._data.keys()):
            if self._data[tag]["count"] >= RAGConfig.TAG_MIN_DOCUMENTS:
                if file_path in self._data[tag]["documents"] and tag not in accepted:
                    accepted.append(tag)

        # Cap
        accepted = accepted[: RAGConfig.MAX_TAGS_PER_DOC]
        self.save()
        return accepted

    def remove_document(self, file_path: str) -> None:
        """Remove file from all tag document lists; decrement counts."""
        for tag in list(self._data.keys()):
            docs = self._data[tag]["documents"]
            if file_path in docs:
                docs.remove(file_path)
                self._data[tag]["count"] = len(docs)
        self.save()

    def get_tags_for_document(self, file_path: str) -> List[str]:
        return [t for t, d in self._data.items() if file_path in d["documents"]]

    # ── Internal ─────────────────────────────────────────────────────────

    def _register(self, tag: str, file_path: str) -> None:
        entry = self._data.setdefault(tag, {"count": 0, "documents": []})
        if file_path not in entry["documents"]:
            entry["documents"].append(file_path)
            entry["count"] = len(entry["documents"])

    def _stage(self, tag: str, file_path: str) -> None:
        """Register the tag but it may not qualify yet (count < threshold)."""
        self._register(tag, file_path)
