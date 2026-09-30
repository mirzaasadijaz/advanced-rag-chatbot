"""
Shared state that flows through every node in the graph.

`messages` uses the `add_messages` reducer, so LangGraph's checkpointer
appends to it turn over turn (this is our conversation memory, keyed by
thread_id). Every other field has no reducer, which means whatever we pass
into `.invoke()` for that key overwrites the checkpointed value - that's why
main.py resets `documents`, `retries`, etc. to fresh values on every call.
"""
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    question: str          # standalone question (post query-rewrite)
    route: str              # "vectorstore" | "direct" | "web_search"

    documents: List[Document]
    web_search_needed: bool

    generation: str
    is_grounded: bool
    addresses_question: bool
    retries: int
    response_id: str  # stable id for this turn's AI message, so retries replace it instead of duplicating
