# APPROACH.md — SHL Assessment Recommender

## Design Overview

The system has two completely separate phases: **offline indexing** (run once, by the developer) and **online serving** (runs live for every API call). Keeping these separate is the most important architectural decision — it means the API starts in seconds and has no expensive setup on first request.

---

## Offline Phase (scripts/)

**scrape.py** paginates the SHL catalog and saves every Individual Test Solution as JSON. It captures name, URL, description, test type, and competencies. This takes ~5 minutes and runs once.

**build_index.py** converts each catalog entry into a single text string and embeds it with `all-MiniLM-L6-v2` (384-dimensional, free, local). Embeddings are stored in a FAISS `IndexFlatIP` — exact cosine search, no approximation. At 50–200 items, exact search is both fast and more accurate than approximate methods.

---

## Online Phase (app/)

**Retrieval (retriever.py):** On each relevant turn, the user's hiring intent is embedded and searched against the FAISS index. Top-10 results are returned as JSON. The LLM then selects the best 1–10 from that shortlist — it never invents URLs because it can only choose from what the retriever provides.

**Agent (agent.py):** A LangChain OPENAI_FUNCTIONS agent with one tool: `SearchSHLCatalog`. The tool pattern means the agent decides *when* to retrieve (only once it has enough context), not *whether* to retrieve on every turn. `temperature=0` ensures deterministic behavior for eval reproducibility.

**API (main.py):** Stateless FastAPI service. Every `/chat` call receives the full history. The response parser handles two recommendation formats: pipe-delimited lines and JSON blocks.

---

## Stack Justification

| Tool | Why this, not X |
|---|---|
| FAISS (local) | No network call, no rate limit. 100% accurate at this data size. ChromaDB adds complexity without benefit for <200 items. |
| all-MiniLM-L6-v2 | Free, offline, 384-dim vectors are more than adequate for short assessment descriptions. OpenAI embeddings cost money and add latency. |
| Groq llama-3.1-70b | ~300 tok/s — fits easily inside the 30s timeout. Free tier. OpenAI's free tier no longer exists. |
| LangChain (minimal) | Used only for Tool/Agent (decision of when to retrieve) and ChatPromptTemplate. Not used for memory or RAG chains — those would fight the spec's stateless design. |
| Render | Free tier, straightforward `render.yaml` config, supports `$PORT` env var natively. |

---

## Prompt Design

The system prompt enforces four behaviors:
1. **Clarify first** — one question if the query is too vague to act on.
2. **Retrieve and recommend** — only from tool results, 1–10 items.
3. **Refine** — re-call the tool when constraints change; don't start over.
4. **Stay in scope** — politely refuse anything not about SHL assessments.

`end_of_conversation: true` is emitted in the reply text and stripped before returning to the user.

---

## Evaluation Approach

**Hard evals:** Schema compliance tested on every response. URL domain checked against `shl.com`.

**Recall@10:** Replayed public traces through the API. Calculated `recall = hits / len(expected)` per trace, then averaged. Target: ≥0.6 mean Recall@10.

**Behavior probes:** Automated checks for vague-query behavior (no recs on turn 1), off-topic refusal, prompt injection resistance, mid-conversation refinement, and turn cap compliance.

---

## What Didn't Work

- **LangChain memory modules** — fought the stateless design. Removed in favor of passing full history on every call.
- **Embedding only the assessment name** — poor recall for skill-based queries. Fixed by embedding name + description + type + competencies as one string.
- **Parsing recommendations from free-form LLM text** — brittle. Fixed by specifying a strict `Name | Type | URL` format in the prompt and adding a JSON-block fallback.

---

## AI Tool Usage

Claude Sonnet used for: initial scaffolding of FastAPI routes, debugging FAISS index mismatch, and drafting the system prompt. All code reviewed and understood before submission. Every design choice in this document can be defended line-by-line.
