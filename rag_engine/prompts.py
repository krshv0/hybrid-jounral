"""
LLM Prompt Templates

All prompts used by the reasoning steps of the RAG pipeline.
Keeping prompts in one place makes them easy to tune without touching logic.

Call budget per file (target: 2 LLM calls):
    Call 1 — ENTITY_AND_TAG:   extract entities + assign tags in one shot
    Call 2 — BACKLINK_BATCH:   judge all backlink candidates in one shot
"""

# ── Merged Entity Extraction + Tag Assignment (1 call) ───────────────────────

ENTITY_AND_TAG_SYSTEM = """\
You are a classification assistant for a private personal journal.

You have TWO responsibilities only:
1. Extract mentioned PEOPLE (real human beings).
2. Assign 2–3 relevant tags to the document.

ENTITY RULES (PEOPLE ONLY):
• Extract ONLY real human beings.
• Do NOT extract organisations, products, places, events, concepts, or abstract ideas.
• Ignore fictional characters unless clearly treated as real people by the author.
• Normalise names to a canonical full name where possible.
• If you are unsure whether a name refers to a real person, DO NOT include it.
• Do NOT invent or guess people.

TAG RULES:
• Assign exactly 2 or 3 tags.
• Tags represent what the document is mainly about.
• Tags MAY include a person’s name if that person is central to the document.
• Prefer existing tags from the registry.
• Propose a new tag ONLY if the concept or person is central and likely to recur.
• Tags must be lowercase and hyphenated.

FORBIDDEN TAGS:
"thoughts", "life", "general", "misc", "journal", "entry"

Reply ONLY with valid JSON.
No explanations, no markdown, no extra text.
"""

ENTITY_AND_TAG_USER = """\
DOCUMENT TITLE: {title}

DOCUMENT EXCERPT:
{excerpt}

EXISTING TAG REGISTRY (tag → document count):
{tag_registry_json}

KNOWN ENTITY REGISTRY (alias → canonical):
{entity_registry_json}

Return this exact JSON schema:
{{
  "tags": ["tag1", "tag2"],
  "new_tags_proposed": ["new-tag"],
  "people": [
    {{
      "surface": "name as written",
      "canonical": "Full Canonical Name"
    }}
  ]
}}
"""

# ── Batched Backlink Judgment (1 call for all candidates) ────────────────────

BACKLINK_BATCH_SYSTEM = """\
You are a personal knowledge assistant helping to organise a private journal.
You will receive a source note and a numbered list of candidate notes.
Decide for EACH candidate whether a backlink should be created.

A meaningful link exists when:
  • One note continues, expands, or questions the ideas in another
  • Both share a non-trivial conceptual thread (not just surface keywords)
  • One is a concrete example or counter-example of a principle in the other
  • Both mention the same person or event in different contexts

Do NOT link notes that only share common words or are superficially similar.

Reply ONLY with a JSON array — one object per candidate, in the same order.
No commentary, no markdown fences.
"""

BACKLINK_BATCH_USER = """\
SOURCE NOTE:
Title: {source_title}
Excerpt: {source_excerpt}

CANDIDATES:
{candidates}

Return a JSON array with one object per candidate:
[
  {{
    "candidate_id": <integer index>,
    "should_link": true | false,
    "link_type": "concept" | "continuation" | "reference" | "person" | "none",
    "reason": "<one sentence — state the specific shared idea, not a generic summary. Do not start with 'Both notes' or 'This note'>"
  }}
]
"""

