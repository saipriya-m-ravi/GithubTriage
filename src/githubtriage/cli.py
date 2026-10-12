"""Command-line adapter: read an issue file, run the triage graph, print the result.

Keep this module thin — no triage logic here. The same core (`build_graph`) will be
called by the FastAPI adapter in slice 7.

Exit codes: 0 = triage ran (even if degraded — errors are printed as warnings),
1 = the input could not be read or validated, 2 = wrong usage (set by argparse).
"""

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from githubtriage.graph import build_graph
from githubtriage.models import TriageState

EXIT_OK = 0
EXIT_BAD_INPUT = 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="githubtriage",
        description="Triage a GitHub issue from a JSON file and print a draft reply.",
    )
    parser.add_argument("path", type=Path, help="path to an issue JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON"
    )
    return parser.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # The HTTP client logs every request at INFO; keep it quiet unless debugging.
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)


def _load_state(path: Path) -> TriageState:
    """Read and validate an issue file. Raises OSError or ValidationError."""
    # utf-8-sig = UTF-8 that also strips a leading BOM (added by many Windows tools).
    text = path.read_text(encoding="utf-8-sig")
    return TriageState.model_validate_json(text)


def _print_human(result: TriageState) -> None:
    c = result.classification
    if c is not None:
        print(f"type:       {c.type} ({c.confidence:.2f})")
        print(f"reasoning:  {c.reasoning}")
    print()
    print("---- draft ----")
    print(result.draft or "(no draft)")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    try:
        state = _load_state(args.path)
    except FileNotFoundError:
        print(f"Error: file not found: {args.path}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except ValidationError as e:
        print(f"Error: invalid issue file {args.path}:\n{e}", file=sys.stderr)
        return EXIT_BAD_INPUT

    result = TriageState.model_validate(build_graph().invoke(state))

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_human(result)

    # Degraded runs still produced a draft, so they exit 0 — but say so clearly.
    for error in result.errors:
        print(f"Warning: {error}", file=sys.stderr)

    return EXIT_OK
