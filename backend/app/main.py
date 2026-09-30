import json
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.graph.build import rag_graph
from app.ingestion.documents import load_file, split_documents as split_uploaded
from app.ingestion.web import scrape_website, split_documents as split_scraped
from app.retrieval.store import add_documents, has_documents, list_sources
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    IngestUrlRequest,
    SourceChunk,
    SourceInfo,
)

app = FastAPI(title="Advanced RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        groq_model=settings.groq_model,
        web_search_enabled=settings.web_search_enabled,
        documents_ingested=has_documents(),
    )


@app.get("/api/sources", response_model=List[SourceInfo])
def sources():
    return list_sources()


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------

@app.post("/api/ingest/documents", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    total_chunks = 0
    errors: List[str] = []

    for upload in files:
        suffix = Path(upload.filename or "").suffix
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await upload.read())
                tmp_path = tmp.name
            docs = load_file(tmp_path, display_name=upload.filename)
            chunks = split_uploaded(docs)
            total_chunks += add_documents(chunks)
        except Exception as e:
            errors.append(f"{upload.filename}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    status = "ok" if not errors else ("partial" if total_chunks else "error")
    return IngestResponse(status=status, chunks_added=total_chunks, details="; ".join(errors) or None)


@app.post("/api/ingest/url", response_model=IngestResponse)
def ingest_url(payload: IngestUrlRequest):
    try:
        docs = scrape_website(payload.url, crawl=payload.crawl, max_pages=payload.max_pages)
        if not docs:
            raise HTTPException(status_code=422, detail="No readable content found at that URL.")
        chunks = split_scraped(docs)
        added = add_documents(chunks)
        return IngestResponse(status="ok", chunks_added=added, details=f"Pages scraped: {len(docs)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------

def _initial_state(message: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "question": message,
        "documents": [],
        "web_search_needed": False,
        "retries": 0,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        result = rag_graph.invoke(
            _initial_state(payload.message),
            config={"configurable": {"thread_id": payload.thread_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources_out = [
        SourceChunk(source=d.metadata.get("source", "unknown"), title=d.metadata.get("title"), snippet=d.page_content[:220])
        for d in result.get("documents", [])
    ]
    return ChatResponse(
        answer=result.get("generation", ""),
        sources=sources_out,
        route=result.get("route"),
        retries=result.get("retries", 0),
    )


@app.get("/api/chat/stream")
async def chat_stream(message: str = Query(...), thread_id: str = Query("default")):
    """Server-Sent Events stream. Emits `token` events as the answer is
    generated, then one `sources` event and a final `done` event."""

    async def event_generator():
        config = {"configurable": {"thread_id": thread_id}}
        try:
            async for event in rag_graph.astream_events(_initial_state(message), config=config, version="v2"):
                if event.get("event") == "on_chat_model_stream" and "final_answer" in event.get("tags", []):
                    chunk = event["data"]["chunk"]
                    raw_content = getattr(chunk, "content", "") or ""
                    
                    # --- THE FIX: Extract string from list if necessary ---
                    content = ""
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif isinstance(raw_content, list):
                        for part in raw_content:
                            if isinstance(part, str):
                                content += part
                            elif isinstance(part, dict) and "text" in part:
                                content += part["text"]
                    
                    if content:
                        yield {"event": "token", "data": json.dumps({"content": content})}
        except Exception as e:
            yield {"event": "stream_error", "data": json.dumps({"message": str(e)})}
            return

        snapshot = await rag_graph.aget_state(config)
        final_state = snapshot.values if snapshot else {}
        sources_out = [
            {
                "source": d.metadata.get("source", "unknown"),
                "title": d.metadata.get("title"),
                "snippet": d.page_content[:220],
            }
            for d in (final_state.get("documents") or [])
        ]
        yield {"event": "sources", "data": json.dumps({"sources": sources_out, "route": final_state.get("route")})}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_generator())


# ----------------------------------------------------------------------
# Frontend (static files) - mounted last so /api/* routes above take priority
# ----------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
