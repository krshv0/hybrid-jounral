"""
Filesystem Watcher — watchdog-based FSM event source

Watches `RAGConfig.VAULT_DIR` recursively for Markdown file events.

ROLE IN THE FSM ARCHITECTURE
------------------------------
The watcher is the ONLY component that reacts to filesystem events, and it
is also the CHEAPEST component.  Its sole responsibility is to update document
FSM state in the FSMStore.  It NEVER:

    - Chunks documents
    - Embeds documents
    - Calls the LLM
    - Writes to markdown files
    - Directly invokes the orchestrator for processing work

This separation ensures that file events (which can arrive faster than the
embedding model can consume them) never queue up expensive work inline.
Instead, the FSMStore is updated immediately, and the schedulers (IndexingScheduler,
ReasoningScheduler) pick up the work at their own pace.

EVENT → FSM STATE MAPPING
--------------------------
    created  → FSMStore.register() + FSMStore.mark_edited()
    modified → FSMStore.mark_edited()
    deleted  → FSMStore.remove() + RAGOrchestrator.delete_file()
    moved    → old path: remove + delete_file
               new path: register + mark_edited
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from .config import RAGConfig
from .fsm_store import FSMStore

logger = logging.getLogger(__name__)


class _VaultEventHandler(FileSystemEventHandler):
    """
    Translates watchdog filesystem events into FSM state mutations.

    No debounce timer is needed here: the stabilisation window in the
    IndexingScheduler plays the equivalent role, and doing it at the FSM
    level (per-document, time-based) is more precise than a fixed debounce.
    """

    def __init__(self, fsm_store: FSMStore, orchestrator) -> None:
        super().__init__()
        self._store = fsm_store
        self._orch  = orchestrator  # used only for delete_file vector cleanup
        self._writing: set[str] = set()
        self._writing_lock = threading.Lock()

    # ── self-write suppression ────────────────────────────────────────────

    def suppress(self, path: str) -> None:
        """Tell the handler to ignore the next write to *path*."""
        with self._writing_lock:
            self._writing.add(str(path))

    def unsuppress(self, path: str) -> None:
        """Re-enable change tracking for *path*."""
        with self._writing_lock:
            self._writing.discard(str(path))

    def _is_self_write(self, path: str) -> bool:
        with self._writing_lock:
            return str(path) in self._writing

    # ── watchdog callbacks ────────────────────────────────────────────────

    def on_created(self, event: FileCreatedEvent) -> None:
        if not self._is_relevant(event.src_path):
            return
        path = event.src_path
        if self._is_self_write(path):
            logger.debug("Watcher: suppressed created  %s", Path(path).name)
            return
        logger.debug("Watcher: created  %s", Path(path).name)
        # Register if new, then mark as edited so indexing scheduler picks it up
        self._store.register(path)
        self._store.mark_edited(path, edit_time=time.time())

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not self._is_relevant(event.src_path):
            return
        path = event.src_path
        if self._is_self_write(path):
            logger.debug("Watcher: suppressed modified %s", Path(path).name)
            return
        logger.debug("Watcher: modified %s", Path(path).name)
        # FSMStore.mark_edited() handles all valid source states:
        # NEW / DIRTY / STABILIZING / INDEXED / REASONED → DIRTY
        self._store.register(path)   # no-op if already registered
        self._store.mark_edited(path, edit_time=time.time())

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if not self._is_relevant(event.src_path):
            return
        path = event.src_path
        logger.info("Watcher: deleted  %s", Path(path).name)
        # Remove from vector store and FSM store
        try:
            self._orch.delete_file(path)
        except Exception as exc:
            logger.warning("delete_file error for %s: %s", Path(path).name, exc)
        self._store.remove(path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if self._is_relevant(event.src_path):
            logger.info(
                "Watcher: moved    %s → %s",
                Path(event.src_path).name,
                Path(event.dest_path).name,
            )
            try:
                self._orch.delete_file(event.src_path)
            except Exception as exc:
                logger.warning("delete_file (move src) error: %s", exc)
            self._store.remove(event.src_path)

        if self._is_relevant(event.dest_path):
            dest = event.dest_path
            self._store.register(dest)
            self._store.mark_edited(dest, edit_time=time.time())

    # ── Path filter ───────────────────────────────────────────────────────

    @staticmethod
    def _is_relevant(path: str) -> bool:
        p = Path(path)
        if p.suffix != ".md":
            return False
        if p.name.startswith(".") or p.name.startswith("~"):
            return False
        ignored = RAGConfig.VAULT_IGNORE_DIRS
        if any(part in ignored for part in p.parts):
            return False
        return True


class VaultWatcher:
    """
    Starts the watchdog observer and the two scheduler threads.

    Call start(), then join() (blocks) or keep as daemon.

    The three components run concurrently:
        - watchdog observer  : ingests filesystem events, writes FSM state only
        - IndexingScheduler  : advances DIRTY → STABILIZING → INDEXED
        - ReasoningScheduler : advances INDEXED → READY_FOR_REASONING → REASONED
    """

    def __init__(
        self,
        fsm_store: FSMStore,
        orchestrator,
        indexing_scheduler,
        reasoning_scheduler,
    ) -> None:
        self._store   = fsm_store
        self._orch    = orchestrator
        self._idx_sched = indexing_scheduler
        self._rsn_sched = reasoning_scheduler

        self._handler  = _VaultEventHandler(fsm_store, orchestrator)
        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            path=str(RAGConfig.VAULT_DIR),
            recursive=True,
        )

    def start(self) -> None:
        self._observer.start()
        self._idx_sched.start()
        self._rsn_sched.start()
        logger.info(
            "VaultWatcher started — vault: %s  (ignoring: %s)",
            RAGConfig.VAULT_DIR,
            RAGConfig.VAULT_IGNORE_DIRS,
        )

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
        self._idx_sched.stop()
        self._rsn_sched.stop()
        logger.info("VaultWatcher stopped.")

    def join(self) -> None:
        """Block until interrupted (Ctrl-C)."""
        import time as _time
        try:
            while self._observer.is_alive():
                _time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

