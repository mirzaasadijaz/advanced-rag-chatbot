from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.config import settings
from app.graph.nodes import (
    contextualize_question,
    generate,
    grade_documents,
    grade_generation,
    increment_retry,
    retrieve,
    route_question,
    web_search,
)
from app.graph.state import GraphState


def _after_route(state: GraphState) -> str:
    return state["route"]  # "vectorstore" | "direct" | "web_search"


def _after_grade_documents(state: GraphState) -> str:
    if state.get("web_search_needed") and settings.web_search_enabled:
        return "web_search"
    return "generate"


def _after_generate(state: GraphState) -> str:
    # Only worth a groundedness check if we actually gave the model context
    # to be grounded in (skip it for the "direct", no-context answers).
    return "grade_generation" if state.get("documents") else "end"


def _after_grade_generation(state: GraphState) -> str:
    if state.get("is_grounded") and state.get("addresses_question"):
        return "end"
    if state.get("retries", 0) >= settings.max_self_correct_retries:
        return "end"  # give up gracefully; return the best-effort answer we have
    return "retry"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("contextualize_question", contextualize_question)
    graph.add_node("route_question", route_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)
    graph.add_node("grade_generation", grade_generation)
    graph.add_node("increment_retry", increment_retry)

    graph.set_entry_point("contextualize_question")
    graph.add_edge("contextualize_question", "route_question")

    graph.add_conditional_edges(
        "route_question",
        _after_route,
        {"vectorstore": "retrieve", "direct": "generate", "web_search": "web_search"},
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        _after_grade_documents,
        {"web_search": "web_search", "generate": "generate"},
    )
    graph.add_edge("web_search", "generate")
    graph.add_conditional_edges(
        "generate",
        _after_generate,
        {"grade_generation": "grade_generation", "end": END},
    )
    graph.add_conditional_edges(
        "grade_generation",
        _after_grade_generation,
        {"end": END, "retry": "increment_retry"},
    )
    graph.add_edge("increment_retry", "generate")

    return graph.compile(checkpointer=MemorySaver())


# Built once at import time and reused across requests; MemorySaver keeps
# per-thread_id conversation state in-process (see README for swapping in a
# persistent checkpointer like SqliteSaver for production).
rag_graph = build_graph()
