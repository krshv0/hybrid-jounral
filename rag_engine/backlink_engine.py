"""
Backlink Decision Engine

Pipeline:
    1. For each chunk in source file, query Chroma for top-K similar chunks
    2. Deduplicate to best hit per target file
    3. Single batched LLM call to judge all candidates at once
    4. Return a list of BacklinkDecision objects
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .config import RAGConfig
from .chunker import Chunk
from .vector_store import VectorStoreManager
from .llm_interface import get_llm
from .prompts import BACKLINK_BATCH_SYSTEM, BACKLINK_BATCH_USER

logger = logging.getLogger(__name__)


@dataclass
class BacklinkDecision:
    """A confirmed decision to add a backlink from source → target."""
    source_file: str
    target_file: str
    target_title: str
    anchor_text: str
    link_type: str           # concept | continuation | reference | person
    similarity: float
    reason: str


class BacklinkEngine:
    """
    Produces BacklinkDecision objects; does NOT mutate any files.
    File mutation is handled by MarkdownEditor.
    """

    def __init__(self, vector_store: VectorStoreManager) -> None:
        self._vs = vector_store
        self._llm = get_llm()

    def compute_backlinks(
        self,
        source_file: str,
        chunks: List[Chunk],
        embeddings: "list[list[float]]",
        existing_backlink_titles: List[str] | None = None,
    ) -> List[BacklinkDecision]:
        """
        Given the chunks + embeddings for source_file, query Chroma for similar
        notes, deduplicate to the best hit per target file, then make a SINGLE
        batched LLM call to judge all candidates at once.

        This reduces the LLM call count from O(TOP_K * chunks) to exactly 1.
        """
        source_title = Path(source_file).stem
        existing_titles: List[str] = existing_backlink_titles or []

        # Step 1: collect best hit per target file across all chunks
        best_hits: Dict[str, dict] = {}   # target_file → hit dict
        for chunk, emb in zip(chunks, embeddings):
            similar = self._vs.query_similar(
                query_embedding=emb,
                top_k=RAGConfig.TOP_K_CANDIDATES,
                exclude_file=source_file,
            )
            for hit in similar:
                target_file = hit["metadata"]["file_path"]
                existing = best_hits.get(target_file)
                if existing is None or hit["similarity"] > existing["similarity"]:
                    best_hits[target_file] = hit

        # Filter out candidates already linked
        if existing_titles:
            existing_lower = {t.lower() for t in existing_titles}
            best_hits = {
                f: h for f, h in best_hits.items()
                if Path(f).stem.lower() not in existing_lower
            }

        if not best_hits:
            return []

        # Step 2: one batched LLM call for all candidates
        source_excerpt = "\n\n".join(c.text[:300] for c in chunks[:2])[:600]
        return self._llm_judge_batch(
            source_title=source_title,
            source_file=source_file,
            source_excerpt=source_excerpt,
            hits=list(best_hits.values()),
            existing_backlink_titles=existing_titles,
        )

    # ── Internal ─────────────────────────────────────────────────────────

    def _llm_judge_batch(
        self,
        source_title: str,
        source_file: str,
        source_excerpt: str,
        hits: List[dict],
        existing_backlink_titles: List[str] | None = None,
    ) -> List[BacklinkDecision]:
        """
        Single LLM call that judges all candidate hits at once.
        Returns BacklinkDecision objects for candidates where should_link=true.
        """
        existing_titles = existing_backlink_titles or []

        # Build a numbered candidate block for the prompt
        candidates_text = ""
        for i, hit in enumerate(hits):
            target_title = Path(hit["metadata"]["file_path"]).stem
            candidates_text += (
                f"\n[{i}] title: {target_title}\n"
                f"    heading: {hit['metadata'].get('heading', '')}\n"
                f"    excerpt: {hit['document'][:300]}\n"
                f"    similarity: {hit['similarity']:.3f}\n"
            )

        # Format existing backlinks for the prompt
        existing_backlinks_text = (
            "\n".join(f"- [[{t}]]" for t in existing_titles)
            if existing_titles else "(none)"
        )

        user_prompt = BACKLINK_BATCH_USER.format(
            source_title=source_title,
            source_excerpt=source_excerpt,
            existing_backlinks=existing_backlinks_text,
            candidates=candidates_text,
        )

        raw     = self._llm.call(BACKLINK_BATCH_SYSTEM, user_prompt, run_name="backlink_batch")
        results = self._llm.extract_json_array(raw)

        if not results:
            logger.warning("Batch backlink: LLM returned no JSON array for %s", source_title)
            return []

        decisions: List[BacklinkDecision] = []
        for item in results:
            if not item.get("should_link", False):
                continue
            link_type = item.get("link_type", "concept")
            if link_type == "none":
                continue
            try:
                cand_id = int(item.get("candidate_id", -1))
                hit     = hits[cand_id]
            except (TypeError, ValueError, IndexError):
                logger.warning("Batch backlink: invalid candidate_id in %s", item)
                continue

            target_file  = hit["metadata"]["file_path"]
            target_title = Path(target_file).stem
            decisions.append(BacklinkDecision(
                source_file=source_file,
                target_file=target_file,
                target_title=target_title,
                anchor_text=item.get("anchor_text", target_title),
                link_type=link_type,
                similarity=hit["similarity"],
                reason=item.get("reason", ""),
            ))

        logger.info(
            "Batch backlink: %d/%d candidates accepted for %s",
            len(decisions), len(hits), source_title,
        )
        return decisions
