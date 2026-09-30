"""
Reranks retrieved chunks with a HuggingFace Cross-Encoder, a small local 
model that runs on CPU with no API key. On first use, it downloads its model 
weights from Hugging Face (a few tens of MB); if that fails - no internet, a
locked-down network, etc. - we fall back to the pre-rerank ordering instead
of crashing the request.
"""
from typing import List, Optional

from langchain_core.documents import Document

from app.config import settings

_compressor = None
_unavailable = False


def _get_compressor():
    global _compressor, _unavailable
    
    if _unavailable or not settings.enable_reranker:
        return None
        
    if _compressor is None:
        try:
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            # ms-marco-MiniLM-L-6-v2 is a smaller, faster cross-encoder model 
            hf_model = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                model_kwargs={"device": "cpu"}
            )
            _compressor = CrossEncoderReranker(model=hf_model, top_n=settings.rerank_top_n)
        except Exception as e:
            print(f"Reranker unavailable: {e}")
            _unavailable = True
            return None
            
    return _compressor


def maybe_rerank(query: str, documents: List[Document], top_n: Optional[int] = None) -> List[Document]:
    if not documents:
        return documents
    
    # Only rerank the top 10 to save CPU cycles
    candidates = documents[:10]
    top_n = top_n or settings.rerank_top_n
    
    compressor = _get_compressor()
    
    if compressor is None:
        return candidates[:top_n]
        
    try:
        compressor.top_n = top_n
        return list(compressor.compress_documents(candidates, query))
    except Exception:
        # If compression fails during runtime, fall back gracefully
        return candidates[:top_n]