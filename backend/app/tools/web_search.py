"""
Wraps Tavily search as a source of Documents so it can be dropped straight
into the same `documents` list the vector retriever fills - the generate
node doesn't need to know whether a chunk came from the knowledge base or
the live web.
"""
from typing import List

from langchain_core.documents import Document

from app.config import settings

_tool = None


def _get_tool():
    global _tool
    if _tool is None:
        # Correct standard import for Tavily in Langchain
        from langchain_community.tools.tavily_search import TavilySearchResults

        # Initialize the tool. (Note: Make sure TAVILY_API_KEY is in your .env)
        _tool = TavilySearchResults(max_results=4)
    return _tool


def web_search_documents(query: str, max_results: int = 4) -> List[Document]:
    if not settings.web_search_enabled:
        return []
        
    try:
        tool = _get_tool()
        # LangChain tools typically accept either a string or a dict.
        # String is safer for simple string inputs.
        raw = tool.invoke(query)
    except Exception as e:
        return [
            Document(
                page_content=f"Web search failed: {e}",
                metadata={"source": "web_search_error", "title": "Web search error"},
            )
        ]

    # TavilySearchResults usually returns a list of dictionaries directly
    items = raw.get("results", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    
    docs: List[Document] = []
    for item in items[:max_results]:
        if not isinstance(item, dict):
            continue
            
        content = item.get("content", "")
        url = item.get("url", "web")
        title = item.get("title", url)
        
        if content:
            docs.append(Document(page_content=content, metadata={"source": url, "title": title}))
            
    return docs