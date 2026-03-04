"""
RAG Orchestrator

Central coordinator that wires all components together and is called by:
    1. The filesystem watcher (on file events)
    2. The CLI (for bulk reindexing)

Process for a single file (2 LLM calls total):
    ┌──────────────────────────────────────────┐
    │  Read + chunk markdown                  │
    │  Call 1: extract people + assign tags   │
    │  Embed chunks                           │
    │  Upsert into Chroma                     │
    │  Call 2: batched backlink judgment       │
    │  Write frontmatter + backlinks section  │
    │  Log all mutations                      │
    └──────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List

from .config import RAGConfig
from .chunker import MarkdownChunker, Chunk
from .embedder import get_embedder
from .vector_store import VectorStoreManager
from .tag_registry import TagRegistry
from .entity_registry import EntityRegistry
from .backlink_engine import BacklinkEngine
from .markdown_editor import MarkdownEditor
from .llm_interface import get_llm
from .prompts import ENTITY_AND_TAG_SYSTEM, ENTITY_AND_TAG_USER
from .logger import MutationLogger, setup_logging

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Stateful coordinator.  Create once, reuse for the lifetime of the watcher.
    """

    def __init__(self) -> None:
        setup_logging()
        RAGConfig.setup_dirs()

        self._watcher_handler = None  # set after watcher is created
        self._chunker   = MarkdownChunker()
        self._embedder  = get_embedder()
        self._vs        = VectorStoreManager()
        self._tags      = TagRegistry()
        self._entities  = EntityRegistry()
        self._backlinks = BacklinkEngine(self._vs)
        self._llm       = get_llm()
        self._mlog      = MutationLogger()

        logger.info("RAG Orchestrator initialised — vault: %s", RAGConfig.VAULT_DIR)

    # ── Public surface ────────────────────────────────────────────────────

    def process_file(self, file_path: str) -> None:
        """
        Full pipeline for a single markdown file.
        Idempotent: safe to call multiple times on the same file.
        """
        path = Path(file_path)
        if not path.exists() or path.suffix != ".md":
            logger.warning("Skipping non-existent or non-markdown file: %s", file_path)
            return

        logger.info("Processing: %s", path.name)

        # 1. Chunk
        chunks = self._chunker.chunk_file(file_path)
        if not chunks:
            logger.info("No chunks produced for %s — skipped", path.name)
            return

        # 2. Entity extraction + tag assignment — one merged LLM call
        combined_text = "\n\n".join(c.text for c in chunks)[:1500]
        entities, tags = self._extract_entities_and_tags(
            title=path.stem,
            text=combined_text,
            file_path=file_path,
        )

        # 4. Embed
        emb_array = self._embedder.embed_passages([c.text for c in chunks])
        embeddings = emb_array.tolist()

        # 5. Upsert into Chroma (delete old, insert new)
        self._vs.delete_file(file_path)
        self._vs.upsert_chunks(
            chunks,
            embeddings,
            tags=[tags] * len(chunks),
            entities=[entities] * len(chunks),
        )
        self._mlog.log_indexed(file_path, len(chunks))

        # 6. Backlink decisions
        decisions = self._backlinks.compute_backlinks(
            source_file=file_path,
            chunks=chunks,
            embeddings=embeddings,
        )

        # 7. Write back to markdown (frontmatter + backlinks)
        editor = MarkdownEditor(file_path)
        fm_changed = editor.update_frontmatter(tags=tags, entities=entities)
        bl_changed = editor.inject_backlinks(decisions)

        if fm_changed or bl_changed:
            self._safe_write(editor, file_path)

        if fm_changed:
            self._mlog.log_tags_assigned(file_path, tags)
            self._mlog.log_entities_linked(file_path, entities)

        if bl_changed:
            targets = [d.target_title for d in decisions]
            self._mlog.log_backlinks_added(file_path, targets)
            logger.info("Backlinks added: %s → %s", path.name, targets)

        # 8. Auto-create entity notes (if configured)
        if RAGConfig.CREATE_ENTITY_NOTES:
            self._maybe_create_entity_notes()

        logger.info("Done: %s  [%d chunks, %d tags, %d backlinks]",
                    path.name, len(chunks), len(tags), len(decisions))

    def set_watcher_handler(self, handler) -> None:
        """Inject the watcher event handler so writes can be suppressed."""
        self._watcher_handler = handler

    def _safe_write(self, editor: "MarkdownEditor", file_path: str) -> None:
        """Save *editor* while suppressing the watchdog event for *file_path*."""
        handler = self._watcher_handler
        if handler:
            handler.suppress(file_path)
        try:
            editor.save()
        finally:
            if handler:
                time.sleep(0.5)  # wait for watchdog event to fire before unsuppressing
                handler.unsuppress(file_path)

    def delete_file(self, file_path: str) -> None:
        """Remove all vectors for a deleted file."""
        deleted = self._vs.delete_file(file_path)
        self._tags.remove_document(file_path)
        self._entities.remove_document(file_path)
        self._mlog.log_deleted(file_path)
        logger.info("Deleted %d chunks for %s", deleted, Path(file_path).name)

    def reindex_vault(self) -> None:
        """Re-embed every markdown file in the vault. Useful after model change."""
        ignored = RAGConfig.VAULT_IGNORE_DIRS
        entries = [
            md for md in RAGConfig.VAULT_DIR.rglob("*.md")
            if not any(part in ignored for part in md.parts)
            and not md.name.startswith(".")
            and not md.name.startswith("~")
        ]
        logger.info("Reindexing %d files in vault…", len(entries))
        for md in entries:
            self.process_file(str(md))
        logger.info("Reindex complete.")

    def reason_file(self, file_path: str) -> str:
        """
        Reasoning-only pipeline for a single file.

        Assumes the file is already embedded in the vector store.
        Performs: entity extraction, tag assignment, backlink computation,
        and markdown mutation.  Does NOT chunk or embed.

        Returns the LLM model identifier used (for recording in FSM store),
        or an empty string if the LLM was not reached.

        Called exclusively by ReasoningScheduler after the document has been
        advanced to READY_FOR_REASONING by FSMStore.mark_ready_for_reasoning().
        """
        path = Path(file_path)
        if not path.exists() or path.suffix != ".md":
            logger.warning("reason_file: skipping non-existent or non-md: %s", file_path)
            return ""

        logger.info("Reasoning: %s", path.name)

        # Re-read and chunk the document to extract combined text for LLM prompts.
        # We do NOT re-embed; we only need the text for entity/tag/backlink prompts.
        chunks = self._chunker.chunk_file(file_path)
        if not chunks:
            logger.info("reason_file: no chunks for %s — skipped", path.name)
            return ""

        combined_text = "\n\n".join(c.text for c in chunks)[:1500]

        # Read existing frontmatter state so the LLM can preserve it
        editor = MarkdownEditor(file_path)
        existing_tags    = editor.read_frontmatter_list("tags")
        existing_people  = editor.read_frontmatter_list("entities")
        existing_backlinks = editor.read_existing_backlink_titles()

        # 1. Entity extraction + tag assignment — one merged LLM call
        entities, tags = self._extract_entities_and_tags(
            title=path.stem,
            text=combined_text,
            file_path=file_path,
            existing_tags=existing_tags,
            existing_people=existing_people,
        )

        # 2. Re-embed chunks for backlink similarity queries
        #    (embeddings already in Chroma but we need the vectors to query against)
        emb_array = self._embedder.embed_passages([c.text for c in chunks])
        embeddings = emb_array.tolist()

        # 3. Backlink decisions
        decisions = self._backlinks.compute_backlinks(
            source_file=file_path,
            chunks=chunks,
            embeddings=embeddings,
            existing_backlink_titles=existing_backlinks,
        )

        # 4. Write back to markdown
        fm_changed = editor.update_frontmatter(tags=tags, entities=entities)
        bl_changed = editor.inject_backlinks(decisions)

        if fm_changed or bl_changed:
            self._safe_write(editor, file_path)

        if fm_changed:
            self._mlog.log_tags_assigned(file_path, tags)
            self._mlog.log_entities_linked(file_path, entities)

        if bl_changed:
            targets = [d.target_title for d in decisions]
            self._mlog.log_backlinks_added(file_path, targets)
            logger.info("Backlinks added: %s → %s", path.name, targets)

        # 5. Auto-create entity notes
        if RAGConfig.CREATE_ENTITY_NOTES:
            self._maybe_create_entity_notes()

        logger.info(
            "Reasoned: %s  [%d tags, %d backlinks]",
            path.name, len(tags), len(decisions),
        )

        # Return model name used by LLM interface for FSM versioning
        try:
            return self._llm._last_model_used  # type: ignore[attr-defined]
        except AttributeError:
            return "unknown"

    def embed_only_file(self, file_path: str) -> None:
        """
        Embed a single file and store vectors — no LLM calls, no markdown writes.
        Only chunks the file, embeds, and upserts into Chroma.
        Tags and entities are left empty; backlinks are not computed.
        """
        path = Path(file_path)
        if not path.exists() or path.suffix != ".md":
            logger.warning("Skipping: %s", file_path)
            return

        chunks = self._chunker.chunk_file(file_path)
        chunks = [c for c in chunks if c.text and c.text.strip()]
        if not chunks:
            logger.info("No chunks for %s — skipped", path.name)
            return

        emb_array = self._embedder.embed_passages([c.text for c in chunks])
        self._vs.delete_file(file_path)
        self._vs.upsert_chunks(chunks, emb_array.tolist())
        self._mlog.log_indexed(file_path, len(chunks))
        logger.info("Embedded: %s  [%d chunks]", path.name, len(chunks))

    def embed_vault(self) -> None:
        """
        Embed-only reindex of the entire vault.
        No LLM calls. No writes to any markdown file.
        Fast — only limited by embedding speed.
        """
        ignored = RAGConfig.VAULT_IGNORE_DIRS
        entries = [
            md for md in RAGConfig.VAULT_DIR.rglob("*.md")
            if not any(part in ignored for part in md.parts)
            and not md.name.startswith(".")
            and not md.name.startswith("~")
        ]
        logger.info("Embed-only indexing %d files…", len(entries))
        for md in entries:
            self.embed_only_file(str(md))
        logger.info("Embed-only index complete. %d files stored.", len(entries))

    # ── LLM helpers ──────────────────────────────────────────────────────

    def _extract_entities_and_tags(
        self,
        title: str,
        text: str,
        file_path: str,
        existing_tags: list | None = None,
        existing_people: list | None = None,
    ) -> tuple:
        """
        Single LLM call that returns both entities and tags.
        Replaces the former two-call _extract_entities() + _assign_tags() pattern.
        Returns (entity_canonicals: List[str], tags: List[str]).

        *existing_tags* and *existing_people* are passed to the prompt so the
        LLM can preserve them rather than regenerating from scratch.
        """
        tag_registry_json    = json.dumps(self._tags.all_tags, indent=2)
        entity_registry_json = self._entities.registry_json()

        result = self._llm.call_json(
            ENTITY_AND_TAG_SYSTEM,
            ENTITY_AND_TAG_USER.format(
                title=title,
                excerpt=text[:1200],
                existing_tags_json=json.dumps(existing_tags or []),
                existing_people_json=json.dumps(existing_people or []),
                tag_registry_json=tag_registry_json,
                entity_registry_json=entity_registry_json,
            ),
            run_name="entity_and_tag",
        )

        if not result:
            return [], []

        # ── Entities (people only per new prompt schema) ──────────────────
        # New prompt returns 'people' with {surface, canonical}.
        # Inject type=PERSON so entity_registry stores them correctly.
        raw_people = result.get("people", [])
        extracted_entities = [
            {"surface": p.get("surface", ""), "canonical": p.get("canonical", ""), "type": "PERSON"}
            for p in raw_people
            if p.get("canonical") or p.get("surface")
        ]
        entity_canonicals  = self._entities.process_extracted_entities(
            extracted_entities, file_path
        )

        # ── Tags ──────────────────────────────────────────────────────────
        candidate_tags = result.get("tags", [])
        new_tags       = result.get("new_tags_proposed", [])

        # Fold in any tags implied by the resolved entities
        entity_tags    = self._entities.get_entity_tags(entity_canonicals)
        candidate_tags = list(set(candidate_tags + entity_tags))

        final_tags = self._tags.assign_tags_to_document(
            file_path=file_path,
            candidate_tags=candidate_tags,
            new_tags_proposed=new_tags,
        )

        return entity_canonicals, final_tags

    def _maybe_create_entity_notes(self) -> None:
        """Auto-create a stub note for entities flagged for note creation."""
        entity_notes_dir = RAGConfig.VAULT_DIR / "entities"
        entity_notes_dir.mkdir(parents=True, exist_ok=True)
        for entity_info in self._entities.get_notes_to_create():
            canonical = entity_info["canonical"]
            entity_type = entity_info["type"]
            note_path = entity_notes_dir / f"{canonical}.md"
            if note_path.exists():
                continue
            stub = (
                f"---\ntags: [{entity_info['type'].lower()}]\n"
                f"entity_type: {entity_type}\n---\n\n"
                f"# {canonical}\n\n"
                f"*Auto-generated entity note.*\n"
            )
            note_path.write_text(stub, encoding="utf-8")
            logger.info("Created entity note: %s", note_path.name)
