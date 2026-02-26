"""
fsm_state.py — Document FSM: states, action allowlist, and transition rules.

STATES
------
Each document (markdown file) owns exactly one FSM instance.  The state
represents the document's processing lifecycle stage:

    NEW                 File is known but has never been indexed.  Only
                        metadata registration is allowed here — no heavy work.

    DIRTY               File was modified since the last stable version.
                        Only timestamp bookkeeping is allowed; no embedding or
                        reasoning may start while a document is actively changing.

    STABILIZING         The edit stream has gone quiet for STABILIZATION_WINDOW
                        seconds, but freshness has not yet been confirmed.
                        Only content-hash computation is allowed here, to decide
                        whether actual re-embedding is necessary.

    INDEXED             Content is stable and the vector store is up-to-date.
                        Chunking, embedding, and vector-upsert happened on the
                        way into this state (during the STABILIZING → INDEXED
                        transition).  This is the gate before reasoning.

    READY_FOR_REASONING All reasoning guards have been passed (token budget,
                        cooldown, batch threshold).  The full LLM enrichment
                        pipeline — similarity search, tag/entity inference, and
                        markdown mutation — may execute here.

    REASONED            Backlinks, tags, and entities are written for the current
                        content version.  The document idles until the next edit.

ACTION ALLOWLIST
----------------
Enforces the invariant that expensive work can only start when the document
is in an appropriate state.  Any attempt to run an action outside its listed
state raises RuntimeError before a single token is spent.

TRANSITION TABLE
----------------
Only the pairs listed in VALID_TRANSITIONS are legal.  Any other move raises
RuntimeError, making invalid state evolution impossible to overlook.
"""

from __future__ import annotations

from enum import Enum
from typing import Set


# ---------------------------------------------------------------------------
# 1. State enum
# ---------------------------------------------------------------------------

class DocState(str, Enum):
    """Per-document FSM states — exactly as specified, no additions."""
    NEW                 = "NEW"
    DIRTY               = "DIRTY"
    STABILIZING         = "STABILIZING"
    INDEXED             = "INDEXED"
    READY_FOR_REASONING = "READY_FOR_REASONING"
    REASONED            = "REASONED"


# ---------------------------------------------------------------------------
# 2. Action allowlist — state × action
# ---------------------------------------------------------------------------

# Maps each state to the set of action names that may execute there.
# The guard layer calls assert_action_allowed() before any significant work.
ACTION_ALLOWLIST: dict[DocState, Set[str]] = {
    DocState.NEW: {
        # Only bookkeeping — record that the file exists.
        "register_metadata",
    },
    DocState.DIRTY: {
        # Only timestamp updates — no embedding, no LLM (file is still changing).
        "update_timestamps",
    },
    DocState.STABILIZING: {
        # Hash computation tells us whether content actually changed since last
        # indexing.  No writes, no embedding, no LLM yet.
        "compute_content_hash",
    },
    DocState.INDEXED: {
        # Content is stable.  These three actions happen together when entering
        # INDEXED from STABILIZING; they are not rerun unless content changes.
        "chunk",
        "embed",
        "vector_upsert",
    },
    DocState.READY_FOR_REASONING: {
        # Full enrichment pipeline.  Only fires when token budget, cooldown,
        # and batch-threshold guards all pass.
        "similarity_search",
        "rag_reasoning",
        "tag_entity_inference",
        "markdown_mutation",
    },
    DocState.REASONED: {
        # Idle.  No actions allowed until the next modification restarts the cycle.
    },
}


def assert_action_allowed(state: DocState, action: str) -> None:
    """
    Raise RuntimeError if *action* is not in the allowlist for *state*.

    This is the single enforcement point for the action guard layer.  Call it
    as the very first line of any function that does expensive work, before
    allocating memory, opening files, or making API calls.

    Why blocked: if an action is attempted outside its allowed state, it means
    the scheduler let a document advance further than the FSM permits.  Failing
    loudly here makes the bug obvious rather than silently corrupting state.
    """
    allowed = ACTION_ALLOWLIST.get(state, set())
    if action not in allowed:
        raise RuntimeError(
            f"Action '{action}' is blocked in state {state.value}. "
            f"Allowed in this state: {sorted(allowed) or ['(none — idle only)']}"
        )


# ---------------------------------------------------------------------------
# 3. Transition table
# ---------------------------------------------------------------------------

# Every valid (from_state, to_state) pair is listed explicitly.
# This is the ground truth for the FSM; no other transitions exist.
VALID_TRANSITIONS: Set[tuple[DocState, DocState]] = {
    # Rule 1  — NEW → DIRTY on first file modification
    (DocState.NEW,                DocState.DIRTY),

    # Rule 2  — DIRTY → DIRTY: further edits reset last_edit_time (self-loop)
    (DocState.DIRTY,              DocState.DIRTY),

    # Rule 3  — DIRTY → STABILIZING: edit stream quiet for STABILIZATION_WINDOW
    (DocState.DIRTY,              DocState.STABILIZING),

    # Rule 4  — STABILIZING → DIRTY: any new edit interrupts stabilisation
    (DocState.STABILIZING,        DocState.DIRTY),

    # Rule 5  — STABILIZING → INDEXED: hash differs from last indexed → re-embed
    # Rule 6  — STABILIZING → INDEXED: hash matches → skip embedding (same destination)
    #           Both rules share the same transition arrow; the scheduler decides
    #           internally whether embedding is necessary.
    (DocState.STABILIZING,        DocState.INDEXED),

    # Rule 7a — INDEXED → DIRTY: file edited after embedding but before reasoning
    (DocState.INDEXED,            DocState.DIRTY),

    # Rule 7b — INDEXED → READY_FOR_REASONING: all reasoning guards pass
    (DocState.INDEXED,            DocState.READY_FOR_REASONING),

    # Rule 8a — READY_FOR_REASONING → DIRTY: file edited while reasoning in-flight
    (DocState.READY_FOR_REASONING, DocState.DIRTY),

    # Rule 8b — READY_FOR_REASONING → REASONED: successful LLM enrichment pass
    (DocState.READY_FOR_REASONING, DocState.REASONED),

    # Rule 9  — REASONED → DIRTY: any file modification restarts the cycle
    (DocState.REASONED,           DocState.DIRTY),

    # Extra   — NEW → INDEXED: used by the embed-only bulk path when no prior
    #           DIRTY event was recorded (initial vault scan).
    (DocState.NEW,                DocState.INDEXED),

    # Rule 10 — STABILIZING → REASONED: content hash matches last_reasoned_hash,
    #           meaning no meaningful change has occurred since the last successful
    #           reasoning pass.  Skips re-indexing and re-reasoning entirely.
    #           Guard: last_reasoned_hash is non-None AND current == last_reasoned.
    (DocState.STABILIZING,        DocState.REASONED),
}


def validate_transition(from_state: DocState, to_state: DocState) -> None:
    """
    Raise RuntimeError if (from_state → to_state) is not in VALID_TRANSITIONS.

    Why enforced: unconstrained state mutation is the root cause of race
    conditions and duplicate processing.  Every state change must pass through
    this checkpoint so that the FSM's invariants are always maintained.
    """
    if (from_state, to_state) not in VALID_TRANSITIONS:
        raise RuntimeError(
            f"Invalid FSM transition: {from_state.value} → {to_state.value}. "
            f"This transition is not in the allowed transition table."
        )
