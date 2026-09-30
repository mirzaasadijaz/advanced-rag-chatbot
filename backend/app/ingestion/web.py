"""
Lightweight website scraper: fetch a page (or shallow-crawl same-domain
links from it), strip boilerplate, and turn the result into Documents ready
for chunking. Deliberately dependency-light (requests + BeautifulSoup only)
instead of a headless browser, so it stays fast and easy to self-host.
"""
from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AdvancedRAGChatbot/1.0; +https://example.local)"}
TIMEOUT_SECONDS = 15


def _clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _same_domain_links(base_url: str, soup: BeautifulSoup) -> List[str]:
    domain = urlparse(base_url).netloc
    links = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        if parsed.netloc == domain and parsed.scheme in ("http", "https"):
            links.add(parsed._replace(fragment="").geturl())
    return list(links)


def scrape_website(url: str, crawl: bool = False, max_pages: int = 5) -> List[Document]:
    """Fetch `url`, and if `crawl=True`, breadth-first follow same-domain
    links up to `max_pages` total pages. Pages that fail to fetch are
    skipped rather than aborting the whole ingestion."""
    max_pages = max(1, min(max_pages, settings.scrape_max_pages_limit))
    visited: set[str] = set()
    to_visit = [url]
    documents: List[Document] = []

    while to_visit and len(visited) < max_pages:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)
        try:
            resp = requests.get(current, headers=HEADERS, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
        except requests.RequestException:
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else current
        text = _clean_text(soup)
        if text:
            documents.append(Document(page_content=text, metadata={"source": current, "title": title}))

        if crawl and len(visited) < max_pages:
            for link in _same_domain_links(current, soup):
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
