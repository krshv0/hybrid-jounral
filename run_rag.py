#!/usr/bin/env python3
"""
run_rag.py — Hybrid Journal RAG Knowledge Base Agent

Entry point for the RAG system.

    python run_rag.py watch       # Start FSM watcher (default)
    python run_rag.py reindex     # Re-embed the entire vault
    python run_rag.py embed       # Embed-only (no LLM, no file writes)
    python run_rag.py query <q>   # Ad-hoc similarity query
"""

import sys
import argparse
import logging
import time
from pathlib import Path

from rag_engine.logger import setup_logging
from rag_engine.orchestrator import RAGOrchestrator
from rag_engine.watcher import VaultWatcher
from rag_engine.retriever import Retriever
from rag_engine.vector_store import VectorStoreManager
from rag_engine.embedder import get_embedder
from rag_engine.config import RAGConfig


# ── Vault bootstrap ───────────────────────────────────────────────────────────

def _bootstrap_vault(store) -> None:
    """
    Register all existing vault .md files with the FSM store if they are not
    already tracked.  Files are registered in the NEW state, then immediately
    marked as edited so the IndexingScheduler can pick them up.

    Files that are already tracked (any state) are left untouched.
    """
    ignored = RAGConfig.VAULT_IGNORE_DIRS
    registered = 0
    for md in RAGConfig.VAULT_DIR.rglob("*.md"):
        if any(part in ignored for part in md.parts):
            continue
        if md.name.startswith(".") or md.name.startswith("~"):
            continue
        rec = store.get(str(md))
        if rec is None:
            store.register(str(md))
            store.mark_edited(str(md), edit_time=md.stat().st_mtime)
            registered += 1
    if registered:
        print(f"  Bootstrap: registered {registered} vault file(s) with FSM store.")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_watch(args) -> None:
    from rag_engine.fsm_store import FSMStore
    from rag_engine.indexing_scheduler import IndexingScheduler
    from rag_engine.reasoning_scheduler import ReasoningScheduler

    RAGConfig.require_vault()
    orch  = RAGOrchestrator()
    store = FSMStore()

    print("Bootstrapping vault…")
    _bootstrap_vault(store)

    idx_sched = IndexingScheduler(
        fsm_store=store,
        chunker=orch._chunker,
        embedder=orch._embedder,
        vector_store=orch._vs,
        mutation_logger=orch._mlog,
    )
    rsn_sched = ReasoningScheduler(
        fsm_store=store,
        orchestrator=orch,
    )

    watcher = VaultWatcher(
        fsm_store=store,
        orchestrator=orch,
        indexing_scheduler=idx_sched,
        reasoning_scheduler=rsn_sched,
    )
    watcher.start()
    orch.set_watcher_handler(watcher._handler)
    print(
        f"✓ FSM watcher running.\n"
        f"  Vault             : {RAGConfig.VAULT_DIR}\n"
        f"  Stabilisation     : {RAGConfig.STABILIZATION_WINDOW}s\n"
        f"  Indexing interval : {RAGConfig.INDEXING_INTERVAL}s\n"
        f"  Reasoning interval: {RAGConfig.REASONING_INTERVAL}s  "
        f"(cooldown {RAGConfig.REASONING_COOLDOWN}s)\n"
        f"Press Ctrl-C to stop."
    )
    watcher.join()


def cmd_reindex(args) -> None:
    RAGConfig.require_vault()
    orch = RAGOrchestrator()
    orch.reindex_vault()
    print("✓ Reindex complete.")


def cmd_embed(args) -> None:
    RAGConfig.require_vault()
    orch = RAGOrchestrator()
    orch.embed_vault()
    print("✓ Embed-only index complete.")


def cmd_query(args) -> None:
    query_text = " ".join(args.query)
    if not query_text:
        print("Usage: python run_rag.py query <your question>")
        return

    print(f"\n🔍 Query: {query_text}")
    print("Loading embedding model (cached after first run)…", flush=True)

    vs = VectorStoreManager()
    retriever = Retriever(vs)
    results = retriever.query(query_text, top_k=args.top_k)

    if not results:
        print("No results found above similarity threshold.")
        return

    print(f"\n{'─' * 60}")
    for i, r in enumerate(results, 1):
        import os
        filename = os.path.basename(r.file_path)
        print(f"\n[{i}] {filename}  (similarity: {r.similarity:.3f})")
        print(f"    Heading : {r.heading}")
        print(f"    Tags    : {', '.join(r.tags) or '—'}")
        print(f"    Entities: {', '.join(r.entities) or '—'}")
        print(f"    Excerpt : {r.text[:200].replace(chr(10), ' ')}…")


def main() -> None:
    setup_logging(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Hybrid Journal RAG Knowledge Base Agent"
    )
    subparsers = parser.add_subparsers(dest="command")

    # watch
    subparsers.add_parser("watch", help="Start filesystem watcher (default)")

    # reindex
    subparsers.add_parser("reindex", help="Re-embed entire vault + run LLM enrichment")

    # embed
    subparsers.add_parser("embed", help="Embed-only index — no LLM calls, no file writes")

    # query
    q_parser = subparsers.add_parser("query", help="Ad-hoc similarity query")
    q_parser.add_argument("query", nargs="+", help="Query text")
    q_parser.add_argument("--top-k", type=int, default=5, dest="top_k")

    args = parser.parse_args()

    if args.command == "reindex":
        cmd_reindex(args)
    elif args.command == "embed":
        cmd_embed(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        # Default: watch
        cmd_watch(args)


if __name__ == "__main__":
    main()
