"""
Similarity Retriever

High-level query interface.  Accepts natural language and returns
ranked note chunks, ready for RAG answering or further processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import RAGConfig
from .embedder import get_embedder
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    file_path: str
    heading: str
    text: str
    similarity: float
    tags: List[str]
    entities: List[str]


class Retriever:
    """
    Natural-language → relevant chunks.
    Used both by the backlink engine and for ad-hoc queries.
    """

    def __init__(self, vector_store: VectorStoreManager) -> None:
        self._vs = vector_store
        self._embedder = get_embedder()

    def query(
        self,
        text: str,
        top_k: int = RAGConfig.TOP_K_CANDIDATES,
        exclude_file: str | None = None,
    ) -> List[RetrievedChunk]:
        """
        Embed `text` as a query and return the top_k similar chunks.
        Results are sorted descending by similarity.
        """
        import json

        emb = self._embedder.embed_query(text).tolist()
        hits = self._vs.query_similar(
            query_embedding=emb,
            top_k=top_k,
            exclude_file=exclude_file,
        )

        results = []
        for hit in hits:
            meta = hit["metadata"]
            results.append(RetrievedChunk(
                chunk_id=hit["id"],
                file_path=meta["file_path"],
                heading=meta.get("heading", ""),
                text=hit["document"],
                similarity=hit["similarity"],
                tags=json.loads(meta.get("tags", "[]")),
                entities=json.loads(meta.get("entities", "[]")),
            ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    def query_by_embedding(
        self,
        embedding: List[float],
        top_k: int = RAGConfig.TOP_K_CANDIDATES,
        exclude_file: str | None = None,
    ) -> List[RetrievedChunk]:
        """Query using a pre-computed embedding (avoids double embed)."""
        import json

        hits = self._vs.query_similar(
            query_embedding=embedding,
            top_k=top_k,
            exclude_file=exclude_file,
        )

        results = []
        for hit in hits:
            meta = hit["metadata"]
            results.append(RetrievedChunk(
                chunk_id=hit["id"],
                file_path=meta["file_path"],
                heading=meta.get("heading", ""),
                text=hit["document"],
                similarity=hit["similarity"],
                tags=json.loads(meta.get("tags", "[]")),
                entities=json.loads(meta.get("entities", "[]")),
            ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results
