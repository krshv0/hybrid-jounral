# RAG Knowledge Base Engine — Integration Guide

## What this adds

After `img-to-md.py` writes a Markdown file to `journal_vault/entries/`, this engine automatically:

| Step | What happens |
|------|-------------|
| **Chunk** | The note is split into semantic units (headings, paragraphs, blockquotes) |
| **Embed** | Each chunk is embedded with BGE-Large (local, no API cost) |
| **Store** | Vectors land in a local ChromaDB — it mirrors the vault exactly |
| **Entities** | An LLM call extracts and normalises named people/orgs/events |
| **Tags** | An LLM call proposes tags; the tag registry enforces stability |
| **Backlinks** | Similar chunks trigger an LLM judgment; accepted links are written back as `## 🔗 Related Notes` |
| **Log** | Every mutation is appended to `data/mutation_log.jsonl` |

---

## Folder structure

```
hybrid-journal/
├── img-to-md.py              ← existing pipeline (unchanged)
├── run_rag.py                ← RAG entry point
├── requirements.txt          ← updated
├── rag_engine/
│   ├── config.py             ← all tunable parameters
│   ├── embedder.py           ← BGE-Large embedding engine
│   ├── chunker.py            ← semantic markdown chunker
│   ├── vector_store.py       ← ChromaDB manager
│   ├── retriever.py          ← similarity query interface
│   ├── tag_registry.py       ← stable tag system
│   ├── entity_registry.py    ← entity + alias system
│   ├── backlink_engine.py    ← backlink decision pipeline
│   ├── markdown_editor.py    ← safe markdown mutation
│   ├── llm_interface.py      ← provider-agnostic LLM adapter
│   ├── prompts.py            ← all LLM prompt templates
│   ├── watcher.py            ← watchdog filesystem watcher
│   ├── orchestrator.py       ← main coordinator
│   └── logger.py             ← mutation log + Python logging
└── data/
    ├── chroma_db/            ← ChromaDB persisted vectors
    ├── tag_registry.json     ← stable tag registry
    ├── entity_registry.json  ← entity + alias map
    ├── mutation_log.jsonl    ← append-only audit trail
    └── rag_engine.log        ← rotating log file
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your LLM provider in .env
#    The RAG engine reuses the same GEMINI_API_KEY as img-to-md.py
echo "GEMINI_API_KEY=your_key_here" >> .env

# Optional: override LLM provider
# LLM_PROVIDER=ollama
# LLM_MODEL=mistral
```

---

## Running

### Option A — Watcher (recommended)

Runs continuously in the background. Every time `img-to-md.py` writes a new `.md` file, the watcher picks it up within seconds.

```bash
python run_rag.py watch
```

### Option B — Reindex entire vault

```bash
python run_rag.py reindex
```

### Option C — Ad-hoc query

```bash
python run_rag.py query "how do I deal with overthinking?"
python run_rag.py query "what have I written about creativity?" --top-k 8
```

---

## Integrating with img-to-md.py

The simplest integration is to start the watcher **in parallel** with your normal workflow:

```bash
# Terminal 1 — process a journal photo
python img-to-md.py my_page.jpg

# Terminal 2 (running continuously) — automatically picks up the new file
python run_rag.py watch
```

Or call the orchestrator directly at the end of `img-to-md.py`:

```python
# At the bottom of img-to-md.py, after saving the markdown file:
from rag_engine.orchestrator import RAGOrchestrator

rag = RAGOrchestrator()
rag.process_file(str(output_path))
```

---

## Configuration

All parameters live in `rag_engine/config.py`.  Key knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model (local) |
| `DEVICE` | `"cpu"` | Set to `"mps"` (Apple) or `"cuda"` for GPU |
| `SIMILARITY_THRESHOLD` | `0.72` | Min cosine similarity to pass to LLM |
| `TAG_MIN_DOCUMENTS` | `2` | Min docs before a new tag is stabilised |
| `MAX_TAGS_PER_DOC` | `8` | Cap on tags per note |
| `LLM_PROVIDER` | `"gemini"` | `gemini` / `openai` / `anthropic` / `ollama` |
| `CREATE_ENTITY_NOTES` | `True` | Auto-stub notes for recurring entities |
| `WATCHER_DEBOUNCE_SEC` | `2.0` | Seconds to wait after last save before processing |

---

## What gets written to your notes

The engine writes **only** these two sections — it never touches your existing content:

### 1. Frontmatter (tags + entities)

```yaml
---
tags: [mindfulness, second-brain, creativity]
entities: [Rohit Sharma]
---
```

If frontmatter already exists it updates only the `tags` and `entities` lines.

### 2. Related Notes section (appended at end of file)

```markdown
## 🔗 Related Notes
- 💡 [[Overthinking Patterns]] — shared mindfulness thread
- ➡️ [[Creative Output]] — continuation of the idea framework
```

Link types:
- 💡 `concept` — shared conceptual theme
- ➡️ `continuation` — one note extends another
- 📎 `reference` — factual / explicit reference
- 👤 `person` — same person appears in both notes

---

## Local-only guarantee

- **Embeddings**: `BAAI/bge-large-en-v1.5` runs 100 % locally via HuggingFace Transformers.
- **Vector store**: ChromaDB writes to `data/chroma_db/` on disk — no outbound traffic.
- **LLM calls**: The reasoning steps (backlink judgment, tag assignment, entity normalisation) use the same Gemini key already in `.env`.  Swap `LLM_PROVIDER=ollama` to make the whole system fully offline.

---

## Re-embedding after model change

```bash
python run_rag.py reindex
```

This deletes and re-creates all vectors.  Tag/entity registries are preserved.
