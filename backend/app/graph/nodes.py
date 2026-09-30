"""
Node functions for the advanced RAG graph. Each node takes the current
GraphState and returns a partial dict of updates (standard LangGraph style).

The pipeline implements three well-known "advanced RAG" ideas together:
  - Adaptive RAG: a router decides whether a question needs retrieval at all.
  - Corrective RAG (CRAG): retrieved docs are relevance-graded, and if too
    few are relevant, the graph falls back to a live web search.
  - Self-RAG-style correction: the generated answer is checked for
    groundedness and relevance, and regenerated (bounded retries) if it
    fails either check.
"""
import uuid
from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.graph.state import GraphState
from app.llm import get_llm
from app.retrieval.reranker import maybe_rerank
from app.retrieval.store import get_hybrid_retriever, has_documents
from app.tools.web_search import web_search_documents

MAX_CONTEXT_CHARS_FOR_GRADING = 2000
MAX_CONTEXT_CHARS_FOR_HALLUCINATION_CHECK = 6000
HISTORY_WINDOW = 6


# ----------------------------------------------------------------------
# Structured output schemas for the LLM "graders"
# ----------------------------------------------------------------------

class RouteQuery(BaseModel):
    route: Literal["vectorstore", "direct"] = Field(
        description=(
            "'vectorstore' if answering well requires the private knowledge base "
            "(facts, documents, specifics the assistant wouldn't just know); "
            "'direct' for greetings, small talk, or questions about the assistant itself."
        )
    )


class GradeDocument(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, otherwise 'no'."
    )


class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer is grounded in / supported by the given facts, otherwise 'no'."
    )


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer actually addresses the question, otherwise 'no'."
    )


def _format_source(doc: Document) -> str:
    return doc.metadata.get("title") or doc.metadata.get("source", "unknown")


def _extract_text(content: any) -> str:
    """Helper to ensure AIMessage.content is always converted to a clean string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Gemini / LangChain often returns content blocks as a list of dicts/strings
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content or "")


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------

def contextualize_question(state: GraphState) -> dict:
    """Rewrite a follow-up question ('what about pricing?') into a standalone
    question using chat history, so retrieval doesn't need to guess referents."""
    messages = state.get("messages", [])
    
    latest = ""
    if messages:
        latest = _extract_text(messages[-1].content)
    else:
        latest = state.get("question", "")

    history = messages[:-1][-HISTORY_WINDOW:]
    response_id = str(uuid.uuid4())

    if not history:
        return {
            "question": latest,
            "retries": 0,
            "documents": [],
            "web_search_needed": False,
            "response_id": response_id,
        }

    history_text = "\n".join(f"{m.type}: {_extract_text(m.content)}" for m in history)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question, rewrite the latest "
                "question as a standalone question that can be understood without the "
                "chat history. If it is already standalone, return it unchanged. "
                "Return ONLY the rewritten question, nothing else.",
            ),
            ("human", "Chat history:\n{history_text}\n\nLatest question: {latest}\n\nStandalone question:"),
        ]
    )
    chain = prompt | get_llm(temperature=0)
    try:
        result = chain.invoke({"history_text": history_text, "latest": latest})
        extracted = _extract_text(result.content)
        question = (extracted or latest).strip()
    except Exception:
        question = latest

    return {
        "question": question,
        "retries": 0,
        "documents": [],
        "web_search_needed": False,
        "response_id": response_id,
    }


def route_question(state: GraphState) -> dict:
    if not has_documents():
        return {"route": "web_search" if settings.web_search_enabled else "direct"}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You route user questions to the right data source. Decide whether the "
                "question needs the private knowledge base or can be answered directly.",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | get_llm(temperature=0).with_structured_output(RouteQuery)
    try:
        result = chain.invoke({"question": state["question"]})
        return {"route": result.route}
    except Exception:
        return {"route": "vectorstore"}


def retrieve(state: GraphState) -> dict:
    retriever = get_hybrid_retriever(k=settings.retrieval_k)
    docs = retriever.invoke(state["question"])
    docs = maybe_rerank(state["question"], docs, top_n=settings.rerank_top_n)
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    grader = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Grade whether the retrieved document is relevant to the user question. "
                "Give a binary 'yes' or 'no'. A document is relevant if it contains "
                "information that would help answer the question, even partially.",
            ),
            ("human", "Document:\n{document}\n\nQuestion: {question}"),
        ]
    ) | get_llm(temperature=0).with_structured_output(GradeDocument)

    relevant: List[Document] = []
    for doc in state.get("documents", []):
        try:
            result = grader.invoke(
                {"document": doc.page_content[:MAX_CONTEXT_CHARS_FOR_GRADING], "question": state["question"]}
            )
            if result.binary_score == "yes":
                relevant.append(doc)
        except Exception:
            relevant.append(doc)

    return {"documents": relevant, "web_search_needed": len(relevant) == 0}


def web_search(state: GraphState) -> dict:
    if not settings.web_search_enabled:
        return {}
    results = web_search_documents(state["question"])
    existing = state.get("documents", [])
    return {"documents": existing + results, "web_search_needed": False}


def generate(state: GraphState) -> dict:
    retries = state.get("retries", 0)
    docs = state.get("documents", [])

    if docs:
        context = "\n\n".join(f"[Source: {_format_source(d)}]\n{d.page_content}" for d in docs)
        context_block = f"Context:\n{context}\n\n"
    else:
        context_block = ""

    strict_note = ""
    if retries > 0:
        strict_note = (
            "\nIMPORTANT: Your previous answer was not sufficiently grounded in the context "
            "or did not address the question. Re-answer using ONLY the given context. If the "
            "context is genuinely insufficient, say so plainly instead of guessing.\n"
        )

    system = (
        "You are a helpful, accurate assistant. "
        "Cite sources inline like [Source: name] when referencing retrieved context. "
        "Your first priority is to answer based on the provided retrieved context. "
        "If the context is insufficient or missing, use your general knowledge to answer clearly. "
        "Strict Instruction: Output plain text only. Do not use Markdown formatting like bolding (**) or italics (*). "
        "Do not use headings or HTML tags."
        f"{strict_note}"
    )

    history = state.get("messages", [])[:-1][-HISTORY_WINDOW:]

    messages = (
        [SystemMessage(content=system)]
        + history
        + [HumanMessage(content=f"{context_block}Question: {state['question']}")]
    )

    llm = get_llm(temperature=0.3, streaming=True, tags=["final_answer"])
    response = llm.invoke(messages)
    
    # Ensure text is parsed properly out of the model's response
    answer = _extract_text(response.content)

    response_id = state.get("response_id") or str(uuid.uuid4())
    return {"generation": answer, "messages": [AIMessage(content=answer, id=response_id)]}


def grade_generation(state: GraphState) -> dict:
    """Self-RAG-style check: is the answer grounded in the retrieved facts,
    and does it actually address the question?"""
    docs = state.get("documents", [])
    generation = state.get("generation", "")

    # If there were no documents retrieved (direct route or general knowledge),
    # skip hallucination check so it does not fail against empty facts.
    if not docs:
        is_grounded = True
    else:
        context = "\n\n".join(d.page_content for d in docs)[:MAX_CONTEXT_CHARS_FOR_HALLUCINATION_CHECK]
        hallucination_chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Determine whether the ANSWER is grounded in / supported by the FACTS. "
                    "'yes' if every substantive claim is supported, 'no' if it invents "
                    "information not present in the facts.",
                ),
                ("human", "FACTS:\n{context}\n\nANSWER:\n{generation}"),
            ]
        ) | get_llm(temperature=0).with_structured_output(GradeHallucination)

        try:
            is_grounded = hallucination_chain.invoke({"context": context, "generation": generation}).binary_score == "yes"
        except Exception:
            is_grounded = True

    answer_chain = ChatPromptTemplate.from_messages(
        [
            ("system", "Determine whether the ANSWER actually addresses the QUESTION. 'yes' or 'no'."),
            ("human", "QUESTION: {question}\n\nANSWER:\n{generation}"),
        ]
    ) | get_llm(temperature=0).with_structured_output(GradeAnswer)

    try:
        addresses = answer_chain.invoke(
            {"question": state["question"], "generation": generation}
        ).binary_score == "yes"
    except Exception:
        addresses = True

    return {"is_grounded": is_grounded, "addresses_question": addresses}


def increment_retry(state: GraphState) -> dict:
    return {"retries": state.get("retries", 0) + 1}