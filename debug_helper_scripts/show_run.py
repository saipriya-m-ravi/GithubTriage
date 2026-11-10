"""Show the history of one triage run, step by step, from the local checkpoints.

Usage (from the project folder — the checkpoint path is relative):
    uv run python debug_helper_scripts/show_run.py                 # list all runs
    uv run python debug_helper_scripts/show_run.py <run id>        # one run's history
    uv run python debug_helper_scripts/show_run.py <run id> --values   # + final values

Read-only: no LLM calls, no GitHub calls. Values are decoded by LangGraph with the
project's allowlisted serializer (the raw SQLite columns are binary msgpack).
"""

import argparse

from githubtriage.checkpointing import all_threads, sqlite_checkpointer, thread_config
from githubtriage.config import settings
from githubtriage.graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", help="run id, e.g. owner/repo#7@2026...Z")
    parser.add_argument(
        "--values", action="store_true", help="print the final state's values"
    )
    args = parser.parse_args()

    with sqlite_checkpointer(settings.checkpoint_path) as saver:
        graph = build_graph(checkpointer=saver)

        if args.run is None:
            for thread in sorted(all_threads(saver)):
                status = graph.get_state(thread_config(thread)).next or ("finished",)
                print(f"{thread}  ->  {status[0]}")
            return

        config = thread_config(args.run)
        history = list(graph.get_state_history(config))  # newest first
        if not history:
            print(f"No run {args.run!r} in {settings.checkpoint_path}")
            return

        # Oldest first: watch the state grow as each node adds its field.
        for snap in reversed(history):
            step = snap.metadata.get("step")
            nxt = ", ".join(snap.next) or "-"
            print(
                f"step {step:>2}  next: {nxt:<13} fields: {', '.join(sorted(snap.values))}"
            )

        if args.values:
            final = graph.get_state(config).values
            print("\n---- final values ----")
            for name in sorted(final):
                print(f"{name}: {final[name]!r}")


if __name__ == "__main__":
    main()
