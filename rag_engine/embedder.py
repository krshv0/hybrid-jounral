"""
Embedding Engine — BGE-Large (BAAI/bge-large-en-v1.5)

Correct usage:
  • Passage documents  → prefix  "Represent this sentence for searching relevant passages: "
  • Query strings      → prefix  "Represent this question for searching relevant passages: "
  • Use [CLS] token    → transformers hidden_state[:, 0, :]
  • L2-normalise output so cosine similarity == dot product
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import RAGConfig

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Singleton-style wrapper around BGE-Large.
    Thread-safe for read-only inference.
    """

    _instance: "EmbeddingEngine | None" = None

    def __new__(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        print(f"[embedder] Loading {RAGConfig.EMBEDDING_MODEL} — first run downloads ~1.3 GB, please wait…", flush=True)
        # Show HuggingFace download progress bars
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_info()
        self.tokenizer = AutoTokenizer.from_pretrained(RAGConfig.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(RAGConfig.EMBEDDING_MODEL)
        hf_logging.set_verbosity_warning()   # quiet again after load
        self.model.eval()
        self.device = torch.device(RAGConfig.DEVICE)
        self.model.to(self.device)
        self._initialized = True
        print(f"[embedder] Model ready on {self.device}", flush=True)
        logger.info("Embedding model loaded on device: %s", self.device)

    # ── Public API ───────────────────────────────────────────────────────

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        """
        Embed document passages (adds passage prefix).
        Returns shape (N, 1024), float32, L2-normalised.
        """
        prefixed = [RAGConfig.PASSAGE_PREFIX + t for t in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query (adds query prefix).
        Returns shape (1024,), float32, L2-normalised.
        """
        prefixed = RAGConfig.QUERY_PREFIX + text
        return self._encode([prefixed])[0]

    def embed_raw(self, texts: List[str]) -> np.ndarray:
        """
        Embed without any prefix (use for programmatic access).
        Returns shape (N, 1024), float32, L2-normalised.
        """
        return self._encode(texts)

    # ── Internal ─────────────────────────────────────────────────────────

    def _encode(self, texts: List[str]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), RAGConfig.BATCH_SIZE):
            batch = texts[i : i + RAGConfig.BATCH_SIZE]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)

            # CLS token — first token of last hidden state
            cls_embeddings = output.last_hidden_state[:, 0, :]
            cls_embeddings = cls_embeddings.cpu().float().numpy()

            # L2 normalise
            norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)          # avoid div-by-zero
            cls_embeddings = cls_embeddings / norms

            all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)


# Module-level singleton accessor
def get_embedder() -> EmbeddingEngine:
    return EmbeddingEngine()
