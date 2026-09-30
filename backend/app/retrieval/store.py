"""
Owns three things:
  1. The Chroma vector store (dense/semantic search, persisted to disk).
  2. A flat JSONL docstore mirroring everything ever ingested (needed
     because BM25Retriever is in-memory and has to be rebuilt from the full
     corpus - Chroma alone can't hand back "every document" cheaply).
  3. A hybrid retriever that fuses vector search + BM25 keyword search via
     LangChain's EnsembleRetriever (reciprocal rank fusion).

BM25 is rebuilt lazily and cached; ingesting new documents just flips a
dirty flag so the next retrieval call rebuilds it once.
"""
import json
import threading
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

_lock = threading.Lock()
_embeddings: Optional[FastEmbedEmbeddings] = None
_vectorstore: Optional[Chroma] = None
_bm25_cache: Optional[BM25Retriever] = None
_bm25_dirty = True

_cached_docs: Optional[List[Document]] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    return _embeddings

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        settings.ensure_dirs()
        _vectorstore = Chroma(
            collection_name="rag_chatbot",
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
    return _vectorstore


def _docstore_path() -> Path:
    settings.ensure_dirs()
    return Path(settings.docstore_path)


def add_documents(documents: List[Document]) -> int:
    if not documents:
        return 0
    global _bm25_dirty, _cached_docs
    with _lock:
        get_vectorstore().add_documents(documents)
        with open(_docstore_path(), "a", encoding="utf-8") as f:
            for doc in documents:
                f.write(
                    json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False)
                    + "\n"
                )
        
        if _cached_docs is not None:
            _cached_docs.extend(documents)
            
        _bm25_dirty = True
    return len(documents)


def load_all_documents() -> List[Document]:
    global _cached_docs
    if _cached_docs is not None:
        return _cached_docs
        
    path = _docstore_path()
    if not path.exists():
        _cached_docs = []
        return _cached_docs
        
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(Document(page_content=obj["page_content"], metadata=obj.get("metadata", {})))
            
    _cached_docs = docs
    return _cached_docs


def list_sources() -> List[dict]:
    counts: dict[str, int] = {}
    for doc in load_all_documents():
        src = doc.metadata.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": k, "chunk_count": v} for k, v in sorted(counts.items())]


def has_documents() -> bool:
    path = _docstore_path()
    return path.exists() and path.stat().st_size > 0


def _get_bm25_retriever(k: int) -> Optional[BM25Retriever]:
    global _bm25_cache, _bm25_dirty
    with _lock:
        if _bm25_cache is None or _bm25_dirty:
            all_docs = load_all_documents()
            _bm25_cache = BM25Retriever.from_documents(all_docs) if all_docs else None
            _bm25_dirty = False
        if _bm25_cache is not None:
            _bm25_cache.k = k
        return _bm25_cache


def get_hybrid_retriever(k: Optional[int] = None):
    """Vector similarity + BM25 keyword search, fused via reciprocal rank
    fusion. Falls back to vector-only if the corpus is too small/empty for
    a meaningful BM25 index."""
    k = k or settings.retrieval_k
    vector_retriever = get_vectorstore().as_retriever(search_kwargs={"k": k})
    bm25 = _get_bm25_retriever(k)
    if bm25 is None:
        return vector_retriever
    return EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.6, 0.4])
