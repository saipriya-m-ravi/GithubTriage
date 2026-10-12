"""Run every sample issue through the real graph (calls Groq) and compare to expected.json.

Usage:
    uv run python debug_helper_scripts/run_samples.py

This is a manual behaviour check, not a pytest test: it calls the real LLM, so it
costs tokens, needs network access, and results can vary slightly between runs.
"""

import json
import logging
from pathlib import Path

from githubtriage.graph import build_graph
from githubtriage.models import TriageState

SAMPLES_DIR = Path(__file__).parent.parent / "samples" / "issues"
MANIFEST = SAMPLES_DIR / "expected.json"


def main() -> None:
    logging.basicConfig(level=logging.WARNING)  # only show problems, not every request

    # 1. Load the answer key: {"bug.json": {"type": "bug", "why": "..."}, ...}
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))

    graph = build_graph()  # compile once, reuse for every sample
    correct = 0

    # 2. For each sample file listed in the answer key...
    for name, entry in sorted(expected.items()):
        path = SAMPLES_DIR / name
        state = TriageState.model_validate_json(path.read_text(encoding="utf-8-sig"))

        # 3. ...run the real graph...
        result = TriageState.model_validate(graph.invoke(state))
        got = result.classification.type if result.classification else "(none)"

        # 4. ...and compare the model's type with the expected type.
        ok = got == entry["type"]
        correct += ok
        mark = "OK  " if ok else "MISS"
        conf = result.classification.confidence if result.classification else 0.0
        reasoning = result.classification.reasoning if result.classification else ""

        print(
            f"{mark} {name:<20} expected={entry['type']:<9} got={got:<9} conf={conf:.2f}"
        )
        print(f"     {reasoning}")
        if not ok:
            print(f"     sample purpose: {entry['why']}")
        for error in result.errors:
            print(f"     error: {error}")

    # 5. Summary.
    print(f"\n{correct}/{len(expected)} matched expected type")


if __name__ == "__main__":
    main()
