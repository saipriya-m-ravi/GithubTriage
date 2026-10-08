from datetime import UTC, datetime

from githubtriage.graph import build_graph
from githubtriage.models import Issue, TriageState

graph = build_graph()
print(graph.get_graph().draw_mermaid())

s = TriageState(
    repo="x/y",
    issue_number=1,
    issue=Issue(
        title="App crashes on login",
        body="Since v2.3 clicking Sign in shows a blank page.",
        author="a",
        updated_at=datetime.now(UTC),
    ),
)
result = graph.invoke(s)
print(f"result is {result}")
for update in graph.stream(s, stream_mode="updates"):
    print(update)
