"""Run every sample issue through the real graph (calls Groq) and compare to expected.json.

Usage:
    uv run python debug_helper_scripts/run_samples.py

Compare models / settings without editing .env (environment variables override .env):
    in command prompt -> set GROQ_REASONING_EFFORT=low
    $env:GROQ_REASONING_EFFORT="low"; uv run python debug_helper_scripts/run_samples.py
    $env:GROQ_MODEL="qwen/qwen3.8-27b"; uv run python debug_helper_scripts/run_samples.py
    Remove-Item Env:GROQ_REASONING_EFFORT, Env:GROQ_MODEL   # back to .env values

This is a manual behaviour check, not a pytest test: it calls the real LLM, so it
costs tokens, needs network access, and results can vary slightly between runs.
Traces go to LangSmith (if enabled) tagged "samples", with model and reasoning
effort in the metadata, so runs can be compared there.
"""

import json
import logging
import statistics
import time
from pathlib import Path

from langchain_core.tracers.langchain import wait_for_all_tracers

from githubtriage.config import settings
from githubtriage.graph import build_graph
from githubtriage.models import Completeness, TriageState
from githubtriage.nodes.draft import REQUEST_STATUSES
from githubtriage.observability import configure_tracing, run_config

SAMPLES_DIR = Path(__file__).parent.parent / "samples" / "issues"
MANIFEST = SAMPLES_DIR / "expected.json"


def asks(status: str) -> bool:
    """Would the draft ask the reporter for this field?"""
    return status in REQUEST_STATUSES


def compare_completeness(expected: dict, got: Completeness | None, counts: dict):
    """Print one line per field and update counts.

    Two levels: the exact status, and what the reporter experiences — asked or not.
    missing vs vague both ask (different wording), ok vs not_needed both don't.
    """
    if got is None:
        print("     completeness: (none — skipped or failed)")
        counts["unchecked"] += 1
        return
    parts = []
    for field, want in expected.items():
        have = getattr(got, field).status
        counts["fields"] += 1
        counts["exact"] += have == want
        if asks(have) and not asks(want):
            counts["false_nag"] += 1
            flag = " ← FALSE NAG"
        elif asks(want) and not asks(have):
            counts["missed_ask"] += 1
            flag = " ← MISSED ASK"
        else:
            flag = "" if have == want else " (same ask/no-ask)"
        parts.append(f"{field}={have} (want {want}){flag}")
    print("     completeness: " + "; ".join(parts))


def main() -> None:
    logging.basicConfig(level=logging.WARNING)  # only show problems, not every request
    tracing = configure_tracing(settings)  # before any LangChain code runs

    effort = settings.groq_reasoning_effort or "default"
    print(
        f"model={settings.groq_model}  reasoning_effort={effort}  tracing={tracing}\n"
    )

    # 1. Load the answer key: {"bug.json": {"type": "bug", "why": "..."}, ...}
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))

    graph = build_graph()  # compile once, reuse for every sample
    correct = 0
    durations = []
    counts = {"fields": 0, "exact": 0, "false_nag": 0, "missed_ask": 0, "unchecked": 0}

    # 2. For each sample file listed in the answer key...
    for name, entry in sorted(expected.items()):
        path = SAMPLES_DIR / name
        state = TriageState.model_validate_json(path.read_text(encoding="utf-8-sig"))
        config = run_config(
            state.repo,
            state.issue_number,
            model=settings.groq_model,
            source="samples",
            reasoning_effort=settings.groq_reasoning_effort,
        )

        # 3. ...run the real graph, timing it...
        start = time.perf_counter()
        result = TriageState.model_validate(graph.invoke(state, config=config))
        seconds = time.perf_counter() - start
        durations.append(seconds)

        # 4. ...and compare the model's type with the expected type.
        got = result.classification.type if result.classification else "(none)"
        ok = got == entry["type"]
        correct += ok
        mark = "OK  " if ok else "MISS"
        conf = result.classification.confidence if result.classification else 0.0
        reasoning = result.classification.reasoning if result.classification else ""

        print(
            f"{mark} {name:<20} expected={entry['type']:<9} got={got:<9} "
            f"conf={conf:.2f}  {seconds:5.1f}s"
        )
        print(f"     {reasoning}")
        if "completeness" in entry:
            compare_completeness(entry["completeness"], result.completeness, counts)
        if not ok:
            print(f"     sample purpose: {entry['why']}")
        for error in result.errors:
            print(f"     error: {error}")

    # 5. Summary.
    print(f"\n{correct}/{len(expected)} matched expected type")
    if counts["fields"] or counts["unchecked"]:
        print(
            f"completeness: {counts['exact']}/{counts['fields']} fields exact | "
            f"false nags {counts['false_nag']} | missed asks {counts['missed_ask']} | "
            f"bugs not checked {counts['unchecked']}"
        )
    print(
        f"time: total {sum(durations):.1f}s | "
        f"median {statistics.median(durations):.1f}s | "
        f"max {max(durations):.1f}s"
    )

    if tracing:
        wait_for_all_tracers()  # don't exit before traces are uploaded


if __name__ == "__main__":
    main()
