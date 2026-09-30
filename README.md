# Advanced RAG Chatbot

A retrieval-augmented chatbot built with **LangGraph**, **LangChain**, and **Gemini** — a FastAPI
backend plus a small web chat frontend, wired around a self-correcting RAG pipeline instead of a
single naive retrieve-then-generate call.

It answers from documents you upload, a website you point it at, or a live web search fallback,
and it checks its own answers before returning them.

## What makes this "advanced"

A plain RAG chatbot does: embed the question → similarity search → stuff results into a prompt →
generate. This project adds three well-known techniques on top of that, all orchestrated as a
LangGraph state machine:

- **Adaptive routing** — a router decides whether a question even needs the knowledge base
  (skips retrieval entirely for "hello" or "what can you do").
- **Hybrid retrieval + reranking** — dense vector search (Chroma) and sparse keyword search
  (BM25) are fused via reciprocal rank fusion, then the combined results are reranked with a
  local cross-encoder (FlashRank) before anything reaches the LLM.
- **Corrective RAG (CRAG)** — retrieved chunks are relevance-graded by the LLM; if too few are
  relevant, the graph automatically falls back to a live web search (Tavily) instead of
  answering from weak context.
- **Self-correction** — the generated answer is checked for groundedness (is it actually
  supported by the retrieved facts?) and relevance (does it address the question?). If either
  check fails, the graph regenerates with a stricter prompt, bounded by
  `MAX_SELF_CORRECT_RETRIES` so it can't loop forever.
- **Query rewriting** — follow-up questions ("what about its pricing?") are rewritten into
  standalone questions using chat history before retrieval, so pronouns don't break search.

See `backend/app/graph/build.py` for the graph topology and `backend/app/graph/nodes.py` for
what each node does.

## Project layout

```
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app: ingestion + chat endpoints, static frontend hosting
│   │   ├── config.py          Settings, loaded from .env
│   │   ├── llm.py             Gemini chat model factory
│   │   ├── schemas.py         Pydantic request/response models
│   │   ├── graph/             LangGraph state, nodes, and graph assembly
│   │   ├── ingestion/         File loaders (PDF/TXT/MD/DOCX) and website scraper
│   │   ├── retrieval/         Chroma + BM25 hybrid retriever, FlashRank reranker
│   │   └── tools/             Tavily web-search tool wrapper
│   ├── requirements.txt       Pinned to versions verified against this codebase
│   ├── .env.example
│   └── data/                  Created at runtime: Chroma DB + a JSONL doc mirror for BM25
└── frontend/                  Static chat UI (vanilla HTML/CSS/JS), served by FastAPI
    ├── index.html
    ├── style.css
    └── app.js
```

## Setup

**1. Get API keys**
- **Gemini** (required)
- **Tavily** (optional, enables the web-search fallback): https://tavily.com

**2. Install and configure**

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste in GOOGLE_API_KEY (and TAVILY_API_KEY if you have one)
```

**3. Run**

```bash
# from the backend/ folder, with the venv active
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** — that's the chat UI, served by the same FastAPI app.
Interactive API docs are at http://localhost:8000/docs.

> Run uvicorn from inside `backend/` (not the repo root) — the storage paths in `.env` are
> relative to the working directory the app starts from.

The first time you ingest a document or send a message, two things download automatically in
the background: the embedding model (~130MB, via `fastembed`) and the reranker model (a few
tens of MB, via `flashrank`). Both need internet access once; after that they're cached locally
and everything runs offline except the actual Google/Tavily API calls.

## Using it

- **Upload documents** or **add a website** from the sidebar before or during a conversation.
  The "follow links on this site" checkbox does a shallow same-domain crawl (capped by
  `SCRAPE_MAX_PAGES_DEFAULT`/`SCRAPE_MAX_PAGES_LIMIT` in `.env`).
- Ask questions in the chat. Answers stream in token-by-token; retrieved sources appear as tags
  under the response.
- **New chat** starts a fresh conversation thread — it doesn't clear the knowledge base, just
  the chat history.
- If nothing's been ingested and `TAVILY_API_KEY` isn't set, the assistant answers from its own
  general knowledge and says so rather than inventing sources.

## API reference

| Method | Path                    | Purpose                                            |
|--------|-------------------------|-----------------------------------------------------|
| GET    | `/api/health`           | Backend status, active model, web-search availability |
| GET    | `/api/sources`          | List ingested sources and chunk counts             |
| POST   | `/api/ingest/documents` | Multipart upload (`files`), PDF/TXT/MD/DOCX        |
| POST   | `/api/ingest/url`       | JSON `{url, crawl, max_pages}`                     |
| POST   | `/api/chat`             | JSON `{message, thread_id}` → full answer + sources |
| GET    | `/api/chat/stream`      | SSE stream, query params `message` & `thread_id`   |

`thread_id` scopes conversation memory — reuse it across calls for the same conversation, use a
new one to start fresh. Full request/response schemas are on the `/docs` page.

## Extending this for production

This is a reference implementation tuned for clarity and running entirely on free tiers. A few
things worth swapping before relying on it for real traffic:

- **Conversation memory** — `graph/build.py` uses LangGraph's in-process `MemorySaver`, which is
  lost on restart and doesn't scale across multiple server processes. Swap in
  [`SqliteSaver`](https://langchain-ai.github.io/langgraph/reference/checkpoints/) or a
  Postgres-backed checkpointer for anything persistent.
- **Streaming transport** — `/api/chat/stream` uses `GET` + query params so the browser's native
  `EventSource` can consume it with no extra JS. That means the message text is subject to URL
  length limits (fine for chat messages, but worth knowing). For very long inputs, switch to a
  `POST` + `fetch`/`ReadableStream` reader on the frontend.
- **GOOGLE rate limits** — this graph makes multiple LLM calls per turn (route, grade each
  retrieved chunk, generate, groundedness check, relevance check). That's easy to hit on Google's
  free tier during heavy testing. Lower `RETRIEVAL_K` / `MAX_SELF_CORRECT_RETRIES`, or move to a
  paid tier, if you see 429s.
- **BM25 rebuilds** — `retrieval/store.py` rebuilds the in-memory BM25 index from the full
  corpus whenever new documents are ingested. Fine up to a few thousand chunks; for a large
  corpus, move to a persistent hybrid-search backend (e.g. Elasticsearch/OpenSearch, or
  Chroma's own metadata filtering plus a dedicated BM25 service).
- **Auth** — there isn't any. `ALLOWED_ORIGINS` controls CORS, not access; add an auth
  dependency to the routes in `main.py` before exposing this beyond localhost.

## Troubleshooting

- **"GOOGLE_API_KEY is not set"** — `.env` isn't being picked up. Confirm it's at
  `backend/.env` (not the repo root) and that you're running uvicorn from `backend/`.
- **Reranking/embeddings silently not improving results** — check server logs for a Hugging
  Face download failure (blocked network, corporate proxy). The app degrades gracefully
  (skips reranking, falls back to vector-only search) rather than crashing, but you'll get
  better retrieval once the models can download.
- **429 errors from Google** — you've hit the free-tier rate limit; see "Google rate limits" above.
