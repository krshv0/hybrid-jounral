"""
Vector Store Manager — ChromaDB

Responsibilities:
    • Persist embeddings + metadata for every chunk
    • Upsert (add or replace) chunks for a file
    • Delete all chunks belonging to a file
    • Expose similarity search
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .config import RAGConfig
from .chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Metadata stored per document:
        file_path   (str)
        chunk_id    (str)
        heading     (str)
        chunk_type  (str)
        tags        (JSON-serialised list[str])
        entities    (JSON-serialised list[str])
        index       (int)
    """

    def __init__(self) -> None:
        RAGConfig.setup_dirs()
        self._client = chromadb.PersistentClient(
            path=str(RAGConfig.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=RAGConfig.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready — collection '%s' has %d documents",
            RAGConfig.CHROMA_COLLECTION,
            self._col.count(),
        )

    # ── Write operations ─────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: "list[list[float]]",
        tags: Optional[List[List[str]]] = None,
        entities: Optional[List[List[str]]] = None,
    ) -> None:
        """
        Add or replace all chunks for a set of chunks.
        `embeddings` must be a flat list aligned with `chunks`.
        """
        if not chunks:
            return

        tags = tags or [[] for _ in chunks]
        entities = entities or [[] for _ in chunks]

        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [
            {
                "file_path": c.file_path,
                "chunk_id": c.chunk_id,
                "heading": c.heading,
                "chunk_type": c.chunk_type,
                "index": c.index,
                "tags": json.dumps(t),
                "entities": json.dumps(e),
            }
            for c, t, e in zip(chunks, tags, entities)
        ]

        self._col.upsert(
            ids=ids,
            embeddings=[emb for emb in embeddings],
            documents=docs,
            metadatas=metas,
        )
        logger.debug("Upserted %d chunks for %s", len(chunks), chunks[0].file_path)

    def delete_file(self, file_path: str) -> int:
        """Remove all vectors associated with a file. Returns count deleted."""
        results = self._col.get(where={"file_path": file_path})
        ids = results.get("ids", [])
        if ids:
            self._col.delete(ids=ids)
            logger.info("Deleted %d chunks for %s", len(ids), file_path)
        return len(ids)

    # ── Query operations ─────────────────────────────────────────────────

    def query_similar(
        self,
        query_embedding: "list[float]",
        top_k: int = RAGConfig.TOP_K_CANDIDATES,
        exclude_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top_k most similar chunks.
        Optionally exclude all chunks from `exclude_file` (for self-similarity).

        Returns list of dicts:
            {id, document, metadata, distance, similarity}
        """
        where = {"file_path": {"$ne": exclude_file}} if exclude_file else None

        try:
            results = self._col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(1, self._col.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("Chroma query failed: %s", exc)
            return []

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            dist = results["distances"][0][i]
            # Chroma cosine space returns 1-cosine (distance), convert to similarity
            similarity = 1.0 - dist
            if similarity >= RAGConfig.SIMILARITY_THRESHOLD:
                output.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": dist,
                    "similarity": round(similarity, 4),
                })
        return output

    def get_all_file_paths(self) -> List[str]:
        """Return unique file paths currently indexed."""
        results = self._col.get(include=["metadatas"])
        paths = {m["file_path"] for m in results["metadatas"]}
        return list(paths)

    def count(self) -> int:
        return self._col.count()

    def file_is_indexed(self, file_path: str) -> bool:
        results = self._col.get(
            where={"file_path": file_path},
            limit=1,
        )
        return bool(results["ids"])
