"""
Entity Registry Manager

Stores known entities and their aliases.

JSON layout:
    {
        "entities": {
            "Rohit Sharma": {
                "type": "PERSON",
                "aliases": ["Sharma sir", "rohit"],
                "documents": ["path/a.md"],
                "create_note": true
            },
            ...
        }
    }

Responsibilities:
    • Normalise incoming entity surface forms to canonical names
    • Trigger optional auto-creation of canonical entity notes
    • Expose tags derived from entities (used by TagRegistry)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from .config import RAGConfig

logger = logging.getLogger(__name__)


def _entity_to_tag(name: str) -> str:
    """Convert a canonical entity name to a valid tag slug."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", "-", name).strip("-")
    return name


class EntityRegistry:
    """Persistent entity + alias registry."""

    def __init__(self) -> None:
        RAGConfig.setup_dirs()
        self._path = RAGConfig.ENTITY_REGISTRY_PATH
        self._data: Dict[str, Dict] = {}   # canonical → {type, aliases, documents, create_note}
        self._alias_map: Dict[str, str] = {}   # lowercase alias → canonical
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = raw.get("entities", {})
                self._rebuild_alias_map()
            except Exception as e:
                logger.warning("Could not load entity registry: %s", e)

    def save(self) -> None:
        self._path.write_text(
            json.dumps({"entities": self._data}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _rebuild_alias_map(self) -> None:
        self._alias_map = {}
        for canonical, info in self._data.items():
            self._alias_map[canonical.lower()] = canonical
            for alias in info.get("aliases", []):
                self._alias_map[alias.lower()] = canonical

    # ── Public API ───────────────────────────────────────────────────────

    def process_extracted_entities(
        self,
        extracted: List[Dict],
        file_path: str,
    ) -> List[str]:
        """
        Accept entity dicts from LLM (see prompts.py schema).
        Register them, update aliases, return list of canonical names found.
        """
        canonicals: List[str] = []

        for item in extracted:
            surface = item.get("surface", "").strip()
            canonical = item.get("canonical", surface).strip()
            entity_type = item.get("type", "OTHER")
            create_note = item.get("create_note", False)

            if not canonical:
                continue

            existing = self._resolve(surface) or self._resolve(canonical)

            if existing:
                canonical = existing
            else:
                # New entity
                self._data[canonical] = {
                    "type": entity_type,
                    "aliases": [],
                    "documents": [],
                    "create_note": create_note,
                }
                self._rebuild_alias_map()

            # Add alias if different from canonical
            if surface.lower() != canonical.lower():
                aliases = self._data[canonical].setdefault("aliases", [])
                if surface not in aliases:
                    aliases.append(surface)
                    self._alias_map[surface.lower()] = canonical

            # Register document
            docs = self._data[canonical].setdefault("documents", [])
            if file_path not in docs:
                docs.append(file_path)

            canonicals.append(canonical)

        self.save()
        return list(set(canonicals))

    def get_entity_tags(self, entities: List[str]) -> List[str]:
        """Convert canonical entity names to tag slugs."""
        return [_entity_to_tag(e) for e in entities]

    def get_entities_for_document(self, file_path: str) -> List[str]:
        return [c for c, d in self._data.items() if file_path in d.get("documents", [])]

    def remove_document(self, file_path: str) -> None:
        for canonical in self._data:
            docs = self._data[canonical].get("documents", [])
            if file_path in docs:
                docs.remove(file_path)
        self.save()

    def get_notes_to_create(self) -> List[Dict]:
        """Return entities that need a canonical entity note created."""
        return [
            {"canonical": c, "type": d["type"]}
            for c, d in self._data.items()
            if d.get("create_note") and len(d.get("documents", [])) >= 1
        ]

    def registry_json(self) -> str:
        """Compact JSON for use in LLM prompts."""
        compact = {
            alias: canonical
            for alias, canonical in self._alias_map.items()
        }
        return json.dumps(compact, ensure_ascii=False)

    # ── Internal ─────────────────────────────────────────────────────────

    def _resolve(self, surface: str) -> Optional[str]:
        """Return canonical name for a surface form, or None."""
        return self._alias_map.get(surface.lower())
