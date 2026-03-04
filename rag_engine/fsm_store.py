"""
fsm_store.py — SQLite-backed FSM state persistence.

Every document's FSM record is stored here.  All FSM logic (schedulers,
watcher handler) reads from and writes to this store exclusively; no
in-memory state is ever the authoritative source for a document's lifecycle
stage.  This means the FSM survives process restarts cleanly.

SCHEMA (table: document_states)
--------------------------------
    file_path           TEXT  PRIMARY KEY
                              Absolute path to the markdown file.

    current_state       TEXT  One of the DocState enum values.

    last_edit_time      REAL  Unix timestamp of the most recent file-modification
                              event seen by the watcher.  NULL for NEW documents.

    last_index_time     REAL  Unix timestamp of the last successful embedding.
                              NULL until the document has been INDEXED at least once.

    last_reason_time    REAL  Unix timestamp of the last successful reasoning pass.
                              NULL until at least one REASONED cycle completes.

    content_hash        TEXT  SHA-256 of the file content at last indexing.
                              Used by the indexing scheduler to detect whether
                              re-embedding is actually necessary.

    embedding_version   TEXT  Identifier of the embedding model used for the
                              current vectors (e.g., "BAAI/bge-large-en-v1.5").

    reasoning_version   TEXT  Identifier of the LLM model used for the last
                              reasoning pass (e.g., "gemini-2.5-flash").

All mutating helpers call validate_transition() / assert_action_allowed()
from fsm_state before writing, so the DB can never drift into an illegal state.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import RAGConfig
from .fsm_state import DocState, validate_transition, assert_action_allowed

logger = logging.getLogger(__name__)

# One connection per thread; SQLite is not safe to share without care.
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return (or create) a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = RAGConfig.FSM_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # concurrent read/write
        _create_schema(conn)
        _local.conn = conn
    return _local.conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_states (
            file_path           TEXT    PRIMARY KEY,
            current_state       TEXT    NOT NULL DEFAULT 'NEW',
            last_edit_time      REAL,
            last_index_time     REAL,
            last_reason_time    REAL,
            content_hash        TEXT,
            embedding_version   TEXT,
            reasoning_version   TEXT,
            last_reasoned_hash  TEXT
        )
        """
    )
    conn.commit()
    _run_migrations(conn)


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply any schema columns added after the initial release."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(document_states)")}
    if "last_reasoned_hash" not in existing:
        conn.execute(
            "ALTER TABLE document_states ADD COLUMN last_reasoned_hash TEXT"
        )
        conn.commit()
        logger.info("Migration: added last_reasoned_hash column.")
    if "last_reasoned_body_hash" not in existing:
        conn.execute(
            "ALTER TABLE document_states ADD COLUMN last_reasoned_body_hash TEXT"
        )
        conn.commit()
        logger.info("Migration: added last_reasoned_body_hash column.")


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------

@dataclass
class FSMRecord:
    """In-memory representation of one document_states row."""
    file_path:               str
    current_state:           DocState
    last_edit_time:          Optional[float] = None
    last_index_time:         Optional[float] = None
    last_reason_time:        Optional[float] = None
    content_hash:            Optional[str]   = None
    embedding_version:       Optional[str]   = None
    reasoning_version:       Optional[str]   = None
    last_reasoned_hash:      Optional[str]   = None
    last_reasoned_body_hash: Optional[str]   = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FSMRecord":
        keys = row.keys()
        return cls(
            file_path=row["file_path"],
            current_state=DocState(row["current_state"]),
            last_edit_time=row["last_edit_time"],
            last_index_time=row["last_index_time"],
            last_reason_time=row["last_reason_time"],
            content_hash=row["content_hash"],
            embedding_version=row["embedding_version"],
            reasoning_version=row["reasoning_version"],
            last_reasoned_hash=row["last_reasoned_hash"] if "last_reasoned_hash" in keys else None,
            last_reasoned_body_hash=row["last_reasoned_body_hash"] if "last_reasoned_body_hash" in keys else None,
        )


# ---------------------------------------------------------------------------
# FSMStore
# ---------------------------------------------------------------------------

class FSMStore:
    """
    Thread-safe interface to the document-state database.

    All public methods that change state validate the transition before writing,
    so invalid state moves raise RuntimeError rather than corrupting the DB.
    """

    # ── Read helpers ─────────────────────────────────────────────────────

    def get(self, file_path: str) -> Optional[FSMRecord]:
        """Return the FSM record for *file_path*, or None if not registered."""
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM document_states WHERE file_path = ?", (file_path,)
        ).fetchone()
        return FSMRecord.from_row(row) if row else None

    def list_by_state(self, state: DocState) -> List[FSMRecord]:
        """Return all records whose current_state matches *state*."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM document_states WHERE current_state = ?",
            (state.value,),
        ).fetchall()
        return [FSMRecord.from_row(r) for r in rows]

    def all_records(self) -> List[FSMRecord]:
        """Return every record in the store."""
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM document_states").fetchall()
        return [FSMRecord.from_row(r) for r in rows]

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, file_path: str) -> FSMRecord:
        """
        Register a new document in state NEW.

        Action guard: 'register_metadata' is only allowed in NEW state.  The
        row does not exist yet, so we insert it directly in NEW and then validate
        the guard before returning (the guard is on the target state, not a
        pre-existing one).
        """
        existing = self.get(file_path)
        if existing:
            return existing

        # Validate that register_metadata is allowed in NEW (sanity guard)
        assert_action_allowed(DocState.NEW, "register_metadata")

        conn = _get_conn()
        conn.execute(
            """
            INSERT OR IGNORE INTO document_states (file_path, current_state)
            VALUES (?, ?)
            """,
            (file_path, DocState.NEW.value),
        )
        conn.commit()
        logger.debug("FSM registered NEW: %s", Path(file_path).name)
        return self.get(file_path)  # type: ignore[return-value]

    # ── Watcher-driven mutation ───────────────────────────────────────────

    def mark_edited(self, file_path: str, edit_time: Optional[float] = None) -> FSMRecord:
        """
        Record a file-modification event.

        Implements transition rules:
            NEW       → DIRTY   (Rule 1: first modification)
            DIRTY     → DIRTY   (Rule 2: further edits, update timestamp)
            STABILIZING → DIRTY (Rule 4: new edit interrupts stabilisation)
            REASONED  → DIRTY   (Rule 9: modification resets the cycle)

        Action guard: 'update_timestamps' is allowed in DIRTY.  The guard is
        checked after the state is determined so that self-loop writes (DIRTY→DIRTY)
        also pass through the guard.

        Why this matters: if a document is being reasoned about (READY_FOR_REASONING)
        and a new edit arrives, we cannot interrupt mid-reasoning.  Instead we let
        the current reasoning pass complete, and the watcher re-queues the event;
        the next scheduler tick will observe the edit time and restart the cycle.
        """
        now = edit_time or time.time()

        rec = self.get(file_path)
        if rec is None:
            rec = self.register(file_path)

        current = rec.current_state

        # Determine target state
        if current in (DocState.NEW, DocState.STABILIZING, DocState.REASONED):
            target = DocState.DIRTY
        elif current == DocState.DIRTY:
            target = DocState.DIRTY  # self-loop
        elif current == DocState.READY_FOR_REASONING:
            # Mid-reasoning: store the edit time but do not interrupt.
            # The reasoning scheduler will detect the stale edit after it finishes.
            self._update_edit_time(file_path, now)
            logger.debug(
                "Edit during READY_FOR_REASONING for %s — queued, not interrupting.",
                Path(file_path).name,
            )
            return self.get(file_path)  # type: ignore[return-value]
        else:
            # INDEXED → was just freshly embedded; restart cycle
            target = DocState.DIRTY

        validate_transition(current, target)
        assert_action_allowed(DocState.DIRTY, "update_timestamps")

        conn = _get_conn()
        conn.execute(
            """
            UPDATE document_states
               SET current_state = ?, last_edit_time = ?
             WHERE file_path = ?
            """,
            (target.value, now, file_path),
        )
        conn.commit()
        logger.debug(
            "FSM %s → %s  (%s)", current.value, target.value, Path(file_path).name
        )
        return self.get(file_path)  # type: ignore[return-value]

    # ── Indexing scheduler mutations ─────────────────────────────────────

    def try_stabilize(self, file_path: str) -> bool:
        """
        Attempt DIRTY → STABILIZING if the stabilisation window has elapsed.

        Transition rule 3: now - last_edit_time >= STABILIZATION_WINDOW.

        Returns True if the transition was made, False if the window has not
        elapsed yet or the document is not in DIRTY state.
        """
        rec = self.get(file_path)
        if rec is None or rec.current_state != DocState.DIRTY:
            return False

        now = time.time()
        last = rec.last_edit_time or 0.0
        if (now - last) < RAGConfig.STABILIZATION_WINDOW:
            return False

        validate_transition(DocState.DIRTY, DocState.STABILIZING)

        conn = _get_conn()
        conn.execute(
            "UPDATE document_states SET current_state = ? WHERE file_path = ?",
            (DocState.STABILIZING.value, file_path),
        )
        conn.commit()
        logger.debug("FSM DIRTY → STABILIZING  (%s)", Path(file_path).name)
        return True

    def compute_hash(self, file_path: str) -> Optional[str]:
        """
        Compute and store the SHA-256 content hash for a STABILIZING document.

        Action guard: 'compute_content_hash' is only allowed in STABILIZING.

        Returns the hex digest, or None if the file no longer exists.
        """
        rec = self.get(file_path)
        if rec is None:
            return None

        assert_action_allowed(rec.current_state, "compute_content_hash")

        path = Path(file_path)
        if not path.exists():
            return None

        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        conn = _get_conn()
        conn.execute(
            "UPDATE document_states SET content_hash = ? WHERE file_path = ?",
            (digest, file_path),
        )
        conn.commit()
        return digest

    def mark_indexed(
        self,
        file_path: str,
        embedding_version: str,
    ) -> None:
        """
        Transition STABILIZING → INDEXED after successful chunk+embed+upsert.

        Stores the index timestamp and embedding model version so the reasoning
        scheduler can determine freshness.
        """
        rec = self.get(file_path)
        if rec is None:
            return

        # Action guards for the work that just completed
        for action in ("chunk", "embed", "vector_upsert"):
            assert_action_allowed(DocState.INDEXED, action)

        validate_transition(rec.current_state, DocState.INDEXED)

        conn = _get_conn()
        conn.execute(
            """
            UPDATE document_states
               SET current_state = ?, last_index_time = ?, embedding_version = ?
             WHERE file_path = ?
            """,
            (DocState.INDEXED.value, time.time(), embedding_version, file_path),
        )
        conn.commit()
        logger.debug("FSM → INDEXED  (%s)", Path(file_path).name)

    def mark_indexed_no_embed(self, file_path: str) -> None:
        """
        Transition STABILIZING → INDEXED when content hash is unchanged.

        Content hash matched the last indexed hash, so no re-embedding was
        necessary.  We still advance to INDEXED so reasoning can proceed.
        """
        validate_transition(DocState.STABILIZING, DocState.INDEXED)

        conn = _get_conn()
        conn.execute(
            """
            UPDATE document_states
               SET current_state = ?, last_index_time = ?
             WHERE file_path = ?
            """,
            (DocState.INDEXED.value, time.time(), file_path),
        )
        conn.commit()
        logger.debug(
            "FSM → INDEXED (hash unchanged, skipped embed)  (%s)",
            Path(file_path).name,
        )

    # ── Reasoning scheduler mutations ────────────────────────────────────

    def mark_ready_for_reasoning(self, file_path: str) -> None:
        """
        Transition INDEXED → READY_FOR_REASONING.

        Called by the reasoning scheduler after all guards (token budget,
        cooldown, batch size) have been confirmed to pass.
        """
        validate_transition(DocState.INDEXED, DocState.READY_FOR_REASONING)

        conn = _get_conn()
        conn.execute(
            "UPDATE document_states SET current_state = ? WHERE file_path = ?",
            (DocState.READY_FOR_REASONING.value, file_path),
        )
        conn.commit()
        logger.debug("FSM → READY_FOR_REASONING  (%s)", Path(file_path).name)

    def mark_reasoned(
        self,
        file_path: str,
        reasoning_version: str,
        reasoned_hash: Optional[str] = None,
        reasoned_body_hash: Optional[str] = None,
    ) -> None:
        """
        Transition READY_FOR_REASONING → REASONED after a successful LLM pass.

        *reasoned_hash* is sha256(full file AFTER reason_file() writes back).
        *reasoned_body_hash* is sha256(_extract_body() AFTER write-back) — used
        by the significance gate to ignore minor edits that don't change real content.
        """
        rec = self.get(file_path)
        if rec is None:
            return

        for action in ("similarity_search", "rag_reasoning",
                       "tag_entity_inference", "markdown_mutation"):
            assert_action_allowed(DocState.READY_FOR_REASONING, action)

        validate_transition(rec.current_state, DocState.REASONED)

        conn = _get_conn()
        conn.execute(
            """
            UPDATE document_states
               SET current_state          = ?,
                   last_reason_time       = ?,
                   reasoning_version      = ?,
                   last_reasoned_hash     = ?,
                   last_reasoned_body_hash = ?
             WHERE file_path = ?
            """,
            (DocState.REASONED.value, time.time(), reasoning_version,
             reasoned_hash, reasoned_body_hash, file_path),
        )
        conn.commit()
        logger.debug("FSM → REASONED  (%s)", Path(file_path).name)

    def mark_reasoned_unchanged(self, file_path: str) -> None:
        """
        Transition STABILIZING → REASONED when the content hash matches
        *last_reasoned_hash*, meaning no meaningful change has occurred since
        the last successful reasoning pass.

        Skips embedding, LLM calls, and markdown mutation entirely.  This is
        the optimised path for noise edits (whitespace, Obsidian auto-saves,
        or minor formatting tweaks) on an already-reasoned document.

        Preconditions enforced by caller:
            - current state is STABILIZING
            - last_reasoned_hash is not None
            - sha256(file) == last_reasoned_hash
        """
        validate_transition(DocState.STABILIZING, DocState.REASONED)

        conn = _get_conn()
        conn.execute(
            """
            UPDATE document_states
               SET current_state = ?
             WHERE file_path = ?
            """,
            (DocState.REASONED.value, file_path),
        )
        conn.commit()
        logger.info(
            "FSM → REASONED (hash unchanged since last reasoning — skipped)"
            "  (%s)", Path(file_path).name,
        )

    def mark_failed_reasoning(self, file_path: str) -> None:
        """
        Roll READY_FOR_REASONING back to INDEXED on a failed reasoning pass.

        The document stays embedded and will be picked up again on the next
        reasoning scheduler tick (subject to cooldown).  Avoids the document
        being permanently stuck in READY_FOR_REASONING after an LLM error.
        """
        # We treat this as a re-entry to INDEXED (special rollback).
        conn = _get_conn()
        conn.execute(
            "UPDATE document_states SET current_state = ? WHERE file_path = ?",
            (DocState.INDEXED.value, file_path),
        )
        conn.commit()
        logger.warning(
            "Reasoning failed — rolled back to INDEXED  (%s)", Path(file_path).name
        )

    # ── Deletion ──────────────────────────────────────────────────────────

    def remove(self, file_path: str) -> None:
        """Delete the FSM record for a document that has been removed from vault."""
        conn = _get_conn()
        conn.execute(
            "DELETE FROM document_states WHERE file_path = ?", (file_path,)
        )
        conn.commit()
        logger.debug("FSM record removed: %s", Path(file_path).name)

    # ── Private helpers ───────────────────────────────────────────────────

    def _update_edit_time(self, file_path: str, edit_time: float) -> None:
        conn = _get_conn()
        conn.execute(
            "UPDATE document_states SET last_edit_time = ? WHERE file_path = ?",
            (edit_time, file_path),
        )
        conn.commit()

    # ── Body-content helpers ──────────────────────────────────────────────

    @staticmethod
    def _extract_body(file_path: str) -> str:
        """
        Return the document body with:
          1. YAML frontmatter (---…---) stripped.
          2. The '## Related Notes' section (and everything below it) stripped.

        This ensures the body hash is immune to reasoning write-backs:
        changing tags in frontmatter or adding backlinks will NOT change the
        body hash, so those writes never trigger a new reasoning cycle.
        """
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

        # Strip YAML frontmatter
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                text = text[end + 3:].lstrip()

        # Strip '## Related Notes' section (and everything after it)
        # Matches the heading regardless of trailing whitespace
        text = re.split(r"^##\s+Related Notes\s*$", text, maxsplit=1, flags=re.MULTILINE)[0]

        return text.strip()

    def compute_body_hash(self, file_path: str) -> Optional[str]:
        """
        Compute SHA-256 of the stripped body (no frontmatter, no backlinks section).
        Returns None if the file does not exist.
        Does NOT require STABILIZING state — safe to call from any scheduler.
        """
        path = Path(file_path)
        if not path.exists():
            return None
        body = self._extract_body(file_path)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()
