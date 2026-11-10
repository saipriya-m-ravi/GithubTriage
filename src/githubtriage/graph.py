from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from githubtriage.models import TriageState
from githubtriage.nodes.approval import approval, notify, route_after_approval
from githubtriage.nodes.classify import classify
from githubtriage.nodes.completeness import completeness
from githubtriage.nodes.draft import draft
from githubtriage.nodes.post import post, stale_check


def route_after_classify(state: TriageState) -> Literal["completeness", "draft"]:
    """Only bugs get a completeness check (DESIGN.md R3, decision 1).

    Routers only read the state and return the next node's name: no LLM calls,
    no side effects, same state -> same route. The Literal return type tells
    LangGraph every possible target (checked at compile time, drawn in diagrams).
    """
    if state.classification is not None and state.classification.type == "bug":
        return "completeness"
    return "draft"  # default branch, including a missing classification


def route_after_stale_check(state: TriageState) -> Literal["post", "__end__"]:
    """A stale draft is discarded: end without posting (the maintainer re-triages)."""
    if state.stale_reason is None:
        return "post"
    return "__end__"


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Triage -> draft -> maintainer approval -> post.

    The approval node pauses with interrupt(), which needs a checkpointer to save
    the paused run: memory_checkpointer() in tests, sqlite_checkpointer() for the
    CLI, Postgres in production. Without one, the graph fails at approval.
    """
    builder = StateGraph(TriageState)
    builder.add_node("classify", classify)
    builder.add_node("completeness", completeness)
    builder.add_node("draft", draft)
    builder.add_node("notify", notify)
    builder.add_node("approval", approval)
    builder.add_node("stale_check", stale_check)
    builder.add_node("post", post)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    builder.add_edge("completeness", "draft")
    builder.add_edge("draft", "notify")
    builder.add_edge("notify", "approval")
    builder.add_conditional_edges("approval", route_after_approval)
    builder.add_conditional_edges("stale_check", route_after_stale_check)
    builder.add_edge("post", END)
    return builder.compile(checkpointer=checkpointer)
