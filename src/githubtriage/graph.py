from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from githubtriage.models import TriageState
from githubtriage.nodes.classify import classify
from githubtriage.nodes.draft import draft


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(TriageState)
    builder.add_node("classify", classify)
    builder.add_node("draft", draft)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "draft")
    builder.add_edge("draft", END)
    return builder.compile()
