"""
Central configuration for the RAG Knowledge Base Engine.
All tunable parameters and paths live here.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


class RAGConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    BASE_DIR         = Path(__file__).parent.parent
    VAULT_DIR        = Path("/Volumes/Vault/Encrypted Vault")
    DATA_DIR         = BASE_DIR / "data"          # index stays in project folder

    # Directories inside VAULT_DIR that should never be indexed
    VAULT_IGNORE_DIRS = {".obsidian", "attachments", "templates", "_templates", ".trash"}
    CHROMA_DIR       = DATA_DIR / "chroma_db"
    TAG_REGISTRY_PATH     = DATA_DIR / "tag_registry.json"
    ENTITY_REGISTRY_PATH  = DATA_DIR / "entity_registry.json"
    MUTATION_LOG_PATH     = DATA_DIR / "mutation_log.jsonl"

    # ── Embedding ──────────────────────────────────────────────────────────
    EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIM    = 1024          # BGE-Large output dimension
    DEVICE           = "cpu"         # "cuda" / "mps" / "cpu"
    BATCH_SIZE       = 16            # batch size for bulk embedding
    PASSAGE_PREFIX   = "Represent this sentence for searching relevant passages: "
    QUERY_PREFIX     = "Represent this question for searching relevant passages: "

    # ── Chunking ───────────────────────────────────────────────────────────
    CHUNK_MIN_TOKENS = 50
    CHUNK_MAX_TOKENS = 350
    CHUNK_TARGET_TOKENS = 200

    # ── Chroma ─────────────────────────────────────────────────────────────
    CHROMA_COLLECTION = "journal_notes"

    # ── Similarity & Backlinks ─────────────────────────────────────────────
    SIMILARITY_THRESHOLD  = 0.72     # cosine similarity (L2-normalised → dot product)
    TOP_K_CANDIDATES      = 8        # candidates sent to LLM for decision
    BACKLINKS_HEADING     = "## Related Notes"

    # ── Tag System ─────────────────────────────────────────────────────────
    TAG_MIN_DOCUMENTS     = 2        # new tag only created when seen in ≥ 2 docs
    TAG_SIMILARITY_MATCH  = 0.82     # threshold to re-use an existing tag
    MAX_TAGS_PER_DOC      = 8

    # ── Entity ─────────────────────────────────────────────────────────────
    ENTITY_TYPES          = ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"]
    CREATE_ENTITY_NOTES   = True     # auto-create canonical entity notes

    # ── LLM ────────────────────────────────────────────────────────────────
    # Ordered fallback chain — tried top-to-bottom on 429 / daily quota exhaustion.
    # All models use the same GEMINI_API_KEY — no extra credentials needed.
    #
    # Model                   Free RPD    Free RPM
    # ─────────────────────   ─────────   ────────
    # gemini-3-flash             20          5
    # gemini-2.5-flash           20          5
    # gemini-2.5-flash-lite     1500         ?
    # gemma-3-27b-it           14400         30
    # gemma-3-12b-it           14400         30
    # gemma-3-4b-it            14400         30
    # gemma-3-1b-it            14400         30
    LLM_FALLBACK_CHAIN = [
        {"provider": "gemini", "model": "gemini-2.5-flash"},        # primary
        {"provider": "gemini", "model": "gemini-2.5-flash-lite"},   # fallback 1 — 1500 RPD
        {"provider": "gemini", "model": "gemma-3-27b-it"},          # fallback 2 — 14400 RPD, cloud
        {"provider": "gemini", "model": "gemma-3-12b-it"},          # fallback 3
        {"provider": "gemini", "model": "gemma-3-4b-it"},           # fallback 4
        {"provider": "gemini", "model": "gemma-3-1b-it"},           # fallback 5 — last resort
    ]
    LLM_API_KEY           = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    LLM_MAX_TOKENS        = 2048
    LLM_TEMPERATURE       = 0.1
    LLM_RPM_CAP           = 5        # Gemini free tier: 5 requests per minute
    LLM_RETRY_BASE_SEC    = 10.0     # wait before retrying the *same* model on RPM 429
    LLM_MAX_RETRIES       = 2        # per-model retry attempts before cascading

    # ── Watcher ────────────────────────────────────────────────────────────
    WATCH_PATTERNS        = ["*.md"]
    WATCH_IGNORE_PATTERNS = [".*", "~*"]


    # ── FSM ────────────────────────────────────────────────────────────────
    # Path to the SQLite database that persists per-document FSM state.
    FSM_DB_PATH           = DATA_DIR / "fsm_state.db"

    # Seconds of file-edit silence before DIRTY → STABILIZING is allowed.
    STABILIZATION_WINDOW  = 200.0   # ~3.5 min — lets you finish a full journaling session

    # How often the indexing scheduler wakes and checks for work (seconds).
    INDEXING_INTERVAL     = 30.0    # no point ticking faster than stabilization window

    # How often the reasoning scheduler wakes and checks for work (seconds).
    REASONING_INTERVAL    = 120.0   # 2 min ticks

    # Minimum seconds between two reasoning passes on the same document.
    # Prevents hammering the LLM for tiny back-to-back edits.
    REASONING_COOLDOWN    = 600.0   # 10 min — one reasoning pass per major editing session

    # Minimum number of INDEXED documents before a reasoning batch starts.
    # Batching amortises LLM call overhead and respects rate limits.
    REASONING_BATCH_SIZE  = 1

    # Estimated token ceiling for a single reasoning batch.
    # If the estimated cost would exceed this, reasoning is deferred.
    TOKEN_BUDGET_PER_PASS = 50_000

    # Approximate tokens per document for budget estimation.
    # Used when exact token counts are unavailable before calling the LLM.
    TOKEN_COST_PER_DOC    = 3_000

    @classmethod
    def setup_dirs(cls):
        """Create all required directories."""
        for d in [cls.DATA_DIR, cls.CHROMA_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def is_vault_available(cls) -> bool:
        """Returns True only when the Cryptomator vault is unlocked and mounted."""
        return cls.VAULT_DIR.exists() and cls.VAULT_DIR.is_dir()

    @classmethod
    def require_vault(cls) -> None:
        """Raise a clear RuntimeError if the vault is locked or not mounted."""
        if not cls.is_vault_available():
            raise RuntimeError(
                f"\n🔒  Vault is locked or not mounted."
                f"\n    Unlock it in Cryptomator first, then retry."
                f"\n    Expected path: {cls.VAULT_DIR}"
            )
