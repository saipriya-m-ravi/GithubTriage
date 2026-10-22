from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from githubtriage.models import TriageState
from githubtriage.nodes.classify import classify
from githubtriage.nodes.completeness import completeness
from githubtriage.nodes.draft import draft


def route_after_classify(state: TriageState) -> Literal["completeness", "draft"]:
    """Only bugs get a completeness check (DESIGN.md R3, decision 1).

    Routers only read the state and return the next node's name: no LLM calls,
    no side effects, same state -> same route. The Literal return type tells
    LangGraph every possible target (checked at compile time, drawn in diagrams).
    """
    if state.classification is not None and state.classification.type == "bug":
        return "completeness"
    return "draft"  # default branch, including a missing classification


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(TriageState)
    builder.add_node("classify", classify)
    builder.add_node("completeness", completeness)
    builder.add_node("draft", draft)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    builder.add_edge("completeness", "draft")
    builder.add_edge("draft", END)
    return builder.compile()
