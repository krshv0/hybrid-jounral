"""
indexing_scheduler.py — Indexing Scheduler Loop

Runs in a background daemon thread.  On every tick it:

    1.  Advances DIRTY → STABILIZING
            for documents whose last_edit_time is at least STABILIZATION_WINDOW
            seconds in the past.  The edit stream is considered quiet.

    2.  Advances STABILIZING → INDEXED
            by computing (or confirming) the content hash and, if the hash
            differs from the last indexed version, running chunk + embed +
            vector-upsert.  If the hash is unchanged, embedding is skipped and
            the document moves straight to INDEXED (Rule 6).

WHAT THIS LOOP DOES NOT DO
---------------------------
    - Never calls the LLM.
    - Never writes to markdown files.
    - Never advances documents into or past READY_FOR_REASONING.
    - Never directly modifies the vault.

ACTION GUARD CONTRACT
---------------------
    Before embedding, assert_action_allowed(DocState.INDEXED, 'chunk') etc.
    is called via FSMStore.mark_indexed(), which internally invokes the guards.
    This ensures the scheduler cannot accidentally embed a document that is
    still in DIRTY or STABILIZING state.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .config import RAGConfig
from .fsm_state import DocState
from .fsm_store import FSMStore

logger = logging.getLogger(__name__)


class IndexingScheduler:
    """
    Periodic background thread that advances documents through:

        DIRTY  →  STABILIZING  →  INDEXED

    Embedding work is performed only when transitioning into INDEXED.
    Resources (chunker, embedder, vector store) are injected at construction
    so this scheduler shares the same singleton instances as the orchestrator.
    """

    def __init__(
        self,
        fsm_store: FSMStore,
        chunker,
        embedder,
        vector_store,
        mutation_logger,
    ) -> None:
        self._store   = fsm_store
        self._chunker = chunker
        self._emb     = embedder
        self._vs      = vector_store
        self._mlog    = mutation_logger

        self._interval = RAGConfig.INDEXING_INTERVAL
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(
            target=self._run,
            name="IndexingScheduler",
            daemon=True,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "IndexingScheduler started — tick every %.1fs, "
            "stabilisation window %.1fs",
            self._interval,
            RAGConfig.STABILIZATION_WINDOW,
        )

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=10)
        logger.info("IndexingScheduler stopped.")

    # ── Main loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        """
        Tick indefinitely at INDEXING_INTERVAL.

        Why periodic rather than event-driven: the scheduler is the only place
        that may advance documents.  Making it periodic guarantees that state
        transitions cannot race with incoming watcher events, and that the
        load on the embedding model is predictable and bounded.
        """
        while not self._stop_evt.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("IndexingScheduler tick error")
            self._stop_evt.wait(timeout=self._interval)

    def _tick(self) -> None:
        """One scheduling cycle."""
        # Phase A: DIRTY → STABILIZING
        # ----------------------------------------------------------------
        # Walk every DIRTY document and attempt the stabilisation transition.
        # try_stabilize() checks the elapsed time internally (Rule 3).
        dirty_docs = self._store.list_by_state(DocState.DIRTY)
        for rec in dirty_docs:
            self._store.try_stabilize(rec.file_path)

        # Phase B: STABILIZING → INDEXED
        # ----------------------------------------------------------------
        # For each document that is now stabilising, compute the content hash
        # (Rule 5/6).  If the hash has changed, run the full embedding pipeline.
        # If it is unchanged, skip embedding and advance directly (Rule 6).
        stabilizing_docs = self._store.list_by_state(DocState.STABILIZING)
        for rec in stabilizing_docs:
            self._process_stabilizing(rec.file_path)

    # ── Embedding pipeline ───────────────────────────────────────────────

    def _process_stabilizing(self, file_path: str) -> None:
        """
        Evaluate a STABILIZING document and advance it to INDEXED.

        Why hash comparison matters: if a file is saved without content changes
        (e.g., the editor auto-saves whitespace), re-embedding would waste CPU
        and vector-store capacity.  The hash is the gating condition.
        """
        path = Path(file_path)

        # File may have been deleted while stabilising
        if not path.exists():
            self._store.remove(file_path)
            logger.debug("Removed missing file from FSM store: %s", path.name)
            return

        rec = self._store.get(file_path)
        if rec is None:
            return

        # Compute current hash (action guard enforced inside compute_hash)
        current_hash = self._store.compute_hash(file_path)
        if current_hash is None:
            return

        # Rule 10 — hash matches last_reasoned_hash: no meaningful change since
        # the last successful reasoning pass.  The vectors, tags, entities, and
        # backlinks in the vault are already correct.  Return directly to REASONED
        # without re-indexing or re-reasoning.  Guard: last_reasoned_hash must be
        # set (i.e. document has been through at least one full reasoning cycle).
        if (
            rec.last_reasoned_hash is not None
            and current_hash == rec.last_reasoned_hash
        ):
            logger.info(
                "Hash matches last reasoned baseline — restoring REASONED directly  (%s)",
                path.name,
            )
            self._store.mark_reasoned_unchanged(file_path)
            return

        # Rule 6 — hash unchanged since last INDEX but differs from last reasoning
        # baseline (or document was never indexed): skip re-embedding but still
        # advance to INDEXED so reasoning can run.
        if current_hash == rec.content_hash and rec.last_index_time is not None:
            logger.debug(
                "Hash unchanged since last index — skipping embed, advancing to INDEXED  (%s)",
                path.name,
            )
            self._store.mark_indexed_no_embed(file_path)
            return

        # Rule 5 — hash changed (or never indexed): run full embedding pipeline
        self._run_embedding(file_path)

    def _run_embedding(self, file_path: str) -> None:
        """
        Execute chunk → embed → vector-upsert for one document.

        Action guards (chunk, embed, vector_upsert) are enforced inside
        FSMStore.mark_indexed(), which is called at the end after successful
        completion.  If an exception occurs, the document stays in STABILIZING
        and the next tick will retry.
        """
        path = Path(file_path)
        logger.info("Indexing: %s", path.name)

        # 1. Chunk
        chunks = self._chunker.chunk_file(file_path)
        chunks = [c for c in chunks if c.text and c.text.strip()]
        if not chunks:
            logger.info("No chunks produced for %s — skipping", path.name)
            # Advance anyway so it does not loop here forever
            self._store.mark_indexed_no_embed(file_path)
            return

        # 2. Embed
        emb_array = self._emb.embed_passages([c.text for c in chunks])

        # 3. Vector upsert (delete stale vectors first for idempotency)
        self._vs.delete_file(file_path)
        self._vs.upsert_chunks(chunks, emb_array.tolist())

        # 4. Advance FSM state — guards are checked inside mark_indexed
        self._store.mark_indexed(
            file_path,
            embedding_version=RAGConfig.EMBEDDING_MODEL,
        )

        self._mlog.log_indexed(file_path, len(chunks))
        logger.info("Indexed: %s  [%d chunks]", path.name, len(chunks))
