from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class SourceChunk(BaseModel):
    source: str
    title: Optional[str] = None
    snippet: str
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []
    route: Optional[str] = None
    retries: int = 0


class IngestUrlRequest(BaseModel):
    url: str
    crawl: bool = False
    max_pages: int = Field(default=5, ge=1, le=25)


class IngestResponse(BaseModel):
    status: str  # "ok" | "partial" | "error"
    chunks_added: int
    details: Optional[str] = None


class SourceInfo(BaseModel):
    source: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    gemini_model: str 
    web_search_enabled: bool
    documents_ingested: bool