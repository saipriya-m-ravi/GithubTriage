"""Score R3 completeness against the hand labels (calls Groq; cached).

    uv run python debug_helper_scripts/run_completeness_eval.py --split dev
    uv run python debug_helper_scripts/run_completeness_eval.py --split dev --show-errors
    uv run python debug_helper_scripts/run_completeness_eval.py --split test

Same rules as run_eval.py: predictions cached per configuration (model,
reasoning effort, completeness prompt version) under DATA_DIR/eval/predictions/;
--show-errors on dev only; complete test runs appended to
evals/pydantic/completeness_test_runs.jsonl.

The labels are an ENRICHED sample (every bug-shaped issue without a version
line was included), so read results per field — never as "x% of pydantic bug
reports are complete".
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from githubtriage.config import settings
from githubtriage.evaluation.completeness_eval import (
    CompletenessPrediction,
    asks,
    config_id,
    predict,
    reports_with_a_false_nag,
    score_all,
)
from githubtriage.evaluation.completeness_labels import FIELDS, load_labels
from githubtriage.evaluation.dataset import IssueText
from githubtriage.evaluation.runner import PredictionStore, TokenBudget, run_predictions

EVALS = Path("evals/pydantic")
SHEET = EVALS / "completeness_labels.csv"
TEST_LOG = EVALS / "completeness_test_runs.jsonl"
TEXTS = settings.data_dir / "eval" / "pydantic_issues.jsonl"
PREDICTIONS = settings.data_dir / "eval" / "predictions"


def pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--max-calls", type=int)
    # 2 for v1-vs-v2 comparisons (decided 2026-11-24): R3 changed its answer on
    # 2 of 4 borderline issues when asked again, so one run mostly measures noise.
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--tokens-per-minute", type=int, default=7000)
    parser.add_argument("--show-errors", action="store_true", help="dev only")
    args = parser.parse_args()

    if args.show_errors and args.split == "test":
        print(
            "--show-errors is dev-only: looking at test errors turns test into dev.",
            file=sys.stderr,
        )
        return 1
    report = load_labels(SHEET)
    if report.problems or report.todo:
        print(
            f"{SHEET}: {report.todo} unlabelled, {len(report.problems)} invalid — "
            "run completeness_labelling.py --check",
            file=sys.stderr,
        )
        return 1
    labels = [label for label in report.labels if label.split == args.split]
    texts = {
        (t.repo, t.number): t
        for t in (
            IssueText.model_validate_json(line)
            for line in TEXTS.read_text(encoding="utf-8").splitlines()
        )
    }

    cfg = config_id()
    store = PredictionStore(PREDICTIONS / f"{cfg}.jsonl", CompletenessPrediction)
    print(f"config: {cfg}\nsplit: {args.split} ({len(labels)} labelled issues)\n")

    run = run_predictions(
        [texts[(label.repo, label.number)] for label in labels],
        args.runs,
        store,
        budget=TokenBudget(args.tokens_per_minute),
        max_calls=args.max_calls,
        predict_fn=predict,
    )
    print(f"new calls: {run.called}, cached: {run.cached}, failed: {len(run.failures)}")
    for failure in run.failures:
        print(f"  FAILED (not scored): {failure}")
    if run.stopped:
        print(f"STOPPED: {run.stopped}")

    runs = [
        [
            (label, p)
            for label in labels
            if (p := store.get(label.repo, label.number, r)) is not None
        ]
        for r in range(1, args.runs + 1)
    ]
    pairs = runs[0]  # the detailed tables below show run 1
    complete = all(len(run_pairs) == len(labels) for run_pairs in runs)
    if not pairs:
        print("\nNothing to score yet.")
        return 1
    if not complete:
        print(
            "\n⚠ INCOMPLETE: some issues have no prediction yet; numbers cover what exists."
        )

    def table(title, subset):
        scores = score_all(subset)
        print(f"\n=== {title} ({len(subset)} issues) ===")
        print(
            f"{'field':12} {'exact':>7} {'should ask':>11} {'asked':>6} "
            f"{'false nags':>11} {'missed asks':>12} {'ask precision':>14} {'ask recall':>11}"
        )
        for field, s in scores.items():
            print(
                f"{field:12} {s.exact:3}/{s.n:<3} {s.should_ask:11} {s.asked:6} "
                f"{s.false_nags:11} {s.missed_asks:12} {pct(s.ask_precision):>14} "
                f"{pct(s.ask_recall):>11}"
            )
        nagged = reports_with_a_false_nag(subset)
        print(
            f"replies asking for something already given: {nagged} of {len(subset)} issues"
        )
        return scores

    if args.runs > 1:
        # What the success criteria are judged on: the mean over runs.
        print(f"\n=== {args.split}: per run, and the mean over {args.runs} runs ===")
        print(
            f"{'run':6} {'version':>8} {'repro':>6} {'logs':>5}   "
            f"{'missed asks':>11}   {'replies with a false nag':>24}   (false nags)"
        )
        per_run = []
        for r, run_pairs in enumerate(runs, start=1):
            s = score_all(run_pairs)
            row = (
                s["version"].false_nags,
                s["repro_steps"].false_nags,
                s["logs"].false_nags,
                sum(x.missed_asks for x in s.values()),
                reports_with_a_false_nag(run_pairs),
            )
            per_run.append(row)
            print(
                f"{r:<6} {row[0]:8} {row[1]:6} {row[2]:5}   {row[3]:11}   "
                f"{row[4]:>18} of {len(run_pairs)}"
            )
        means = [sum(col) / len(col) for col in zip(*per_run, strict=True)]
        print(
            f"{'mean':6} {means[0]:8.1f} {means[1]:6.1f} {means[2]:5.1f}   "
            f"{means[3]:11.1f}   {means[4]:>21.1f}"
        )
        print("(the tables below show run 1)")

    scores = table(f"{args.split}: all labels (run 1)", pairs)
    sure = [(lab, p) for lab, p in pairs if not lab.unsure]
    if len(sure) < len(pairs):
        table(
            f"{args.split}: excluding the {len(pairs) - len(sure)} 'unsure' labels",
            sure,
        )

    print("\nconfusion per field (rows = your label, columns = R3; pooled):")
    for field, s in scores.items():
        print(f"  {field}")
        print(
            f"    {'':12}"
            + "".join(f"{c:>11}" for c in ("missing", "vague", "ok", "not_needed"))
        )
        for lab, row in s.confusion.items():
            if any(row.values()):
                print(f"    {lab:12}" + "".join(f"{row[c]:11}" for c in row))

    preds = [p for _, p in pairs]
    tin, tout = sum(p.input_tokens for p in preds), sum(p.output_tokens for p in preds)
    print(f"\ncalls scored: {len(preds)}; tokens in/out: {tin:,} / {tout:,}")

    if args.show_errors:
        print("\ndisagreements on dev (label → R3; '*' = changes what the reply asks):")
        for label, p in pairs:
            diffs = [
                f"{f}: {getattr(label, f)} → {getattr(p, f)}"
                + ("*" if asks(getattr(label, f)) != asks(getattr(p, f)) else "")
                for f in FIELDS
                if getattr(label, f) != getattr(p, f)
            ]
            if diffs:
                flag = " (unsure)" if label.unsure else ""
                print(f"  #{label.number:<6}{flag}  " + "; ".join(diffs))
                print(
                    f"           https://github.com/{label.repo}/issues/{label.number}"
                )

    if args.split == "test":
        if complete and not run.stopped:
            entry = {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "config": cfg,
                "n": len(pairs),
                **{
                    f: {
                        "exact": s.exact,
                        "false_nags": s.false_nags,
                        "missed_asks": s.missed_asks,
                        "should_ask": s.should_ask,
                        "asked": s.asked,
                    }
                    for f, s in scores.items()
                },
                "replies_with_a_false_nag": reports_with_a_false_nag(pairs),
            }
            with TEST_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"\ntest run logged in {TEST_LOG}")
        else:
            print("\ntest run incomplete: not logged. Re-run to finish it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
