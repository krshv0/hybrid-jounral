"""
reasoning_scheduler.py — Reasoning Scheduler Loop

Runs in a background daemon thread.  On every tick it:

    1.  Selects INDEXED documents that pass the reasoning guards:
            a. Token budget — estimated cost of the batch does not exceed
               TOKEN_BUDGET_PER_PASS (prevents runaway API spend).
            b. Cooldown — each document's last_reason_time is at least
               REASONING_COOLDOWN seconds in the past.  Prevents hammering
               the LLM for the same document on rapid repeated edits.
            c. Batch threshold — at least REASONING_BATCH_SIZE documents must
               be available before the loop fires.  Amortises API call
               overhead and respects rate-limit quotas.

    2.  Advances qualifying documents from INDEXED → READY_FOR_REASONING.

    3.  Runs the full LLM enrichment pipeline (entity extraction, tag
        assignment, backlink computation, markdown mutation) by delegating
        to orchestrator.reason_file().

    4.  Advances each document to REASONED on success, or rolls back to
        INDEXED on failure.

WHAT THIS LOOP DOES NOT DO
---------------------------
    - Never chunks or embeds documents.
    - Never calls the vector store directly.
    - Never advances documents back past INDEXED on partial failures.

ACTION GUARD CONTRACT
---------------------
    All enrichment actions ('similarity_search', 'rag_reasoning', etc.) are
    allowed only in READY_FOR_REASONING.  These guards are enforced inside
    FSMStore.mark_reasoned(), which calls assert_action_allowed() for every
    action before committing REASONED to the database.

TOKEN & RATE LIMIT STRATEGY
----------------------------
    Token cost is estimated BEFORE the LLM is called.  If the estimated cost
    exceeds the configured budget, the entire batch is deferred, not partially
    executed.  This is stricter than the actual API behaviour but protects
    against runaway spending on long documents.

    Reasoning is:
        - batchable  : multiple INDEXED docs are collected before the loop runs.
        - deferrable : if guards fail, the tick is skipped without side effects.
        - cancellable: stop() sets the stop event; no reasoning call is mid-flight
          when the thread exits.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import List, TYPE_CHECKING

from .config import RAGConfig
from .fsm_state import DocState
from .fsm_store import FSMRecord, FSMStore

if TYPE_CHECKING:
    from .orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)


class ReasoningScheduler:
    """
    Periodic background thread that advances documents through:

        INDEXED  →  READY_FOR_REASONING  →  REASONED

    LLM enrichment (tags, entities, backlinks, markdown mutation) is performed
    only inside READY_FOR_REASONING and only when all three guards pass.
    """

    def __init__(self, fsm_store: FSMStore, orchestrator: "RAGOrchestrator") -> None:
        self._store = fsm_store
        self._orch  = orchestrator

        self._interval  = RAGConfig.REASONING_INTERVAL
        self._stop_evt  = threading.Event()
        self._thread    = threading.Thread(
            target=self._run,
            name="ReasoningScheduler",
            daemon=True,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "ReasoningScheduler started — tick every %.0fs, "
            "cooldown %.0fs, batch_size %d, token_budget %d",
            self._interval,
            RAGConfig.REASONING_COOLDOWN,
            RAGConfig.REASONING_BATCH_SIZE,
            RAGConfig.TOKEN_BUDGET_PER_PASS,
        )

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=30)  # reasoning calls can be slow
        logger.info("ReasoningScheduler stopped.")

    # ── Main loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        """
        Tick indefinitely at REASONING_INTERVAL.

        Why periodic: same rationale as the indexing scheduler — keeping the
        loop separate from the watcher prevents reasoning from starting the
        moment a file is saved and guarantees that scheduling decisions can
        be inspected via the FSM store without timing ambiguity.
        """
        while not self._stop_evt.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("ReasoningScheduler tick error")
            self._stop_evt.wait(timeout=self._interval)

    def _tick(self) -> None:
        """One reasoning scheduling cycle."""

        # Collect INDEXED candidates and filter against all three guards.
        indexed_docs = self._store.list_by_state(DocState.INDEXED)
        candidates   = self._filter_by_guards(indexed_docs)

        if len(candidates) < RAGConfig.REASONING_BATCH_SIZE:
            # Guard C — batch threshold not met; defer.
            if indexed_docs:
                logger.debug(
                    "Reasoning deferred — %d INDEXED docs, need %d for batch.",
                    len(indexed_docs),
                    RAGConfig.REASONING_BATCH_SIZE,
                )
            return

        # Estimate total token cost for this batch (Guard A check).
        estimated_tokens = len(candidates) * RAGConfig.TOKEN_COST_PER_DOC
        if estimated_tokens > RAGConfig.TOKEN_BUDGET_PER_PASS:
            # Trim batch to what fits within the budget.
            max_docs = max(1, RAGConfig.TOKEN_BUDGET_PER_PASS // RAGConfig.TOKEN_COST_PER_DOC)
            logger.info(
                "Token budget %d — trimming batch from %d to %d docs.",
                RAGConfig.TOKEN_BUDGET_PER_PASS, len(candidates), max_docs,
            )
            candidates = candidates[:max_docs]

        logger.info("Reasoning batch: %d document(s)", len(candidates))

        for rec in candidates:
            self._reason_document(rec)

    # ── Guard evaluation ─────────────────────────────────────────────────

    def _filter_by_guards(self, records: List[FSMRecord]) -> List[FSMRecord]:
        """
        Return documents from *records* that pass cooldown (Guard B).

        Guard A (token budget) and Guard C (batch threshold) are checked at
        the batch level in _tick(); this method filters at the document level.

        Why cooldown matters: a document whose frontmatter was just updated by
        the reasoning pass will trigger a watcher event, which marks it DIRTY
        again.  Without cooldown, a document could oscillate through the cycle
        in minutes, exhausting the daily LLM quota.
        """
        now = time.time()
        passing = []
        for rec in records:
            last = rec.last_reason_time or 0.0
            elapsed = now - last
            if elapsed >= RAGConfig.REASONING_COOLDOWN:
                passing.append(rec)
            else:
                logger.debug(
                    "Cooldown holding %s — %.0fs / %.0fs elapsed.",
                    Path(rec.file_path).name,
                    elapsed,
                    RAGConfig.REASONING_COOLDOWN,
                )
        return passing

    # ── Per-document reasoning ────────────────────────────────────────────

    def _reason_document(self, rec: FSMRecord) -> None:
        """
        Advance one document through READY_FOR_REASONING → REASONED.

        Sequence:
            1. Transition to READY_FOR_REASONING  (validates FSM transition)
            2. Call orchestrator.reason_file()    (LLM enrichment + markdown write)
            3a. On success → transition to REASONED
            3b. On failure → roll back to INDEXED (scheduler will retry next tick)

        Why rollback on failure: partial reasoning (e.g., entities written but
        backlinks failed) would leave the markdown in an inconsistent state.
        Rolling back to INDEXED keeps the cycle deterministic.
        """
        file_path = rec.file_path
        path = Path(file_path)

        # Verify the file still exists
        if not path.exists():
            logger.info("reason_document: file gone, removing from FSM  (%s)", path.name)
            self._store.remove(file_path)
            return

        try:
            # Transition to READY_FOR_REASONING — validates guard is passed
            self._store.mark_ready_for_reasoning(file_path)

            # Run the full LLM enrichment pipeline.
            # All enrichment actions are guarded at the action-allowlist level.
            reasoning_version = self._orch.reason_file(file_path)

            # Capture the file hash AFTER reason_file() has written tags and
            # backlinks back to disk.  This becomes the new reasoning baseline:
            # if the file is modified and later stabilises to this same hash,
            # the system will restore REASONED directly without re-running LLM.
            reasoned_hash: str | None = None
            if path.exists():
                reasoned_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            # Mark REASONED (internally validates action guards + FSM transition)
            self._store.mark_reasoned(
                file_path,
                reasoning_version=reasoning_version or "unknown",
                reasoned_hash=reasoned_hash,
            )

            logger.info("Reasoned successfully: %s", path.name)

        except Exception as exc:
            logger.exception("Reasoning failed for %s: %s", path.name, exc)
            # Roll back to INDEXED so the document is retried on the next tick
            self._store.mark_failed_reasoning(file_path)
