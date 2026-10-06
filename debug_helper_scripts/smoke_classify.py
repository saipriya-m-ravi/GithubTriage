from datetime import UTC, datetime

from githubtriage.models import Issue, TriageState
from githubtriage.nodes.classify import classify

s = TriageState(repo='x/y', issue_number=1, issue=Issue(title='App crashes on login', body='Since v2.3 clicking Sign in shows a blank page.', author='a', updated_at=datetime.now(UTC)))
print(classify(s))

# Run as below:
# uv run python debug_helper_scripts/smoke_classify.py
