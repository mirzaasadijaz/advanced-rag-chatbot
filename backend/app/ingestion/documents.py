"""
Loads a single uploaded file into LangChain Documents and splits them into
retrieval-sized chunks. Add a new suffix to LOADER_MAP to support more file
types (the loader class just needs a `.load()` method).
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_file(file_path: str, display_name: str | None = None) -> List[Document]:
    """Load a file on disk into Documents, tagging metadata with a clean source name."""
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(LOADER_MAP))}")

    loader = loader_cls(file_path)
    docs = loader.load()
    source_name = display_name or Path(file_path).name
    for doc in docs:
        doc.metadata["source"] = source_name
        doc.metadata.setdefault("title", source_name)
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
