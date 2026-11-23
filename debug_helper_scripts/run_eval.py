"""Run and score the R1 classifier on the pydantic eval set (calls Groq).

    uv run python debug_helper_scripts/run_eval.py --split dev
    uv run python debug_helper_scripts/run_eval.py --split dev --show-errors
    uv run python debug_helper_scripts/run_eval.py --split dev --max-calls 40
    uv run python debug_helper_scripts/run_eval.py --split test

Predictions are cached per configuration (model, reasoning effort, prompt
version) under DATA_DIR/eval/predictions/, so a run stopped by a rate limit or
--max-calls resumes where it stopped, and re-running only re-scores.

The dev/test rule is enforced here, not promised:
- dev: everything, including each misclassified issue (--show-errors);
- test: summary numbers only, and every COMPLETE test run is appended to
  evals/pydantic/test_runs.jsonl (committed), so test-set use is visible.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median, quantiles

from githubtriage.config import settings
from githubtriage.evaluation.dataset import IssueText, ManifestRow
from githubtriage.evaluation.labels import LABEL_TO_TYPE
from githubtriage.evaluation.metrics import (
    TYPES,
    agreement,
    confusion,
    ece,
    reliability,
    score,
    separation,
    spread,
)
from githubtriage.evaluation.runner import (
    PredictionStore,
    TokenBudget,
    config_id,
    predict,
    run_predictions,
)
from githubtriage.evaluation.split import split_key

EVALS = Path("evals/pydantic")
MANIFEST = EVALS / "manifest.jsonl"
FREQUENCIES = EVALS / "frequencies.json"
TEST_LOG = EVALS / "test_runs.jsonl"
TEXTS = settings.data_dir / "eval" / "pydantic_issues.jsonl"
PREDICTIONS = settings.data_dir / "eval" / "predictions"
# USD per 1M tokens (input, output), checked against Groq's pricing page 2026-10-15.
PRICES = {"openai/gpt-oss-120b": (0.15, 0.60), "qwen/qwen3.8-27b": (0.80, 4.00)}


def pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def spread_text(values) -> str:
    s = spread(values)
    if s.mean is None:
        return "n/a"
    if s.low == s.high:
        return pct(s.mean).strip()
    return f"{pct(s.mean).strip()}  [{pct(s.low).strip()} – {pct(s.high).strip()}]"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    # 1 by default (decided 2026-11-11): 200k tokens/day allows ~170 calls, so
    # repeated runs would triple every evaluation. Variation is measured
    # separately when needed (--runs 2 on a subset; run 1 stays cached).
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-calls", type=int, help="stop after N new LLM calls")
    parser.add_argument(
        "--limit",
        type=int,
        help="run AND score only the first N issues (hash order): compare "
        "configurations on the same quick-dev sample",
    )
    parser.add_argument(
        "--tokens-per-minute",
        type=int,
        default=7000,
        help="pace below Groq's 8,000/min limit (default 7000)",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="list misclassified issues (dev only)",
    )
    args = parser.parse_args()

    if args.show_errors and args.split == "test":
        print(
            "--show-errors is dev-only: looking at test errors turns test into dev.",
            file=sys.stderr,
        )
        return 1
    if not TEXTS.exists():
        print(
            f"{TEXTS} is missing: run backfill_eval.py --rebuild first.",
            file=sys.stderr,
        )
        return 1

    rows = [
        ManifestRow.model_validate_json(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]
    # Hash order (fixed, effectively random), not the manifest's by-type order:
    # a partial run (--max-calls 40) is then a mixed sample — a quick dev set —
    # instead of "the first 40 bugs".
    rows = sorted(
        (r for r in rows if r.split == args.split),
        key=lambda r: split_key(r.repo, r.number),
    )
    if args.limit:
        if args.split == "test":
            print(
                "--limit is dev-only: the test measurement uses every issue.",
                file=sys.stderr,
            )
            return 1
        rows = rows[: args.limit]
    texts = {
        (t.repo, t.number): t
        for t in (
            IssueText.model_validate_json(line)
            for line in TEXTS.read_text(encoding="utf-8").splitlines()
        )
    }
    frequencies = json.loads(FREQUENCIES.read_text(encoding="utf-8"))[
        "closed_issues_per_type"
    ]

    cfg = config_id()
    store = PredictionStore(PREDICTIONS / f"{cfg}.jsonl")
    print(
        f"config: {cfg}\nsplit: {args.split} ({len(rows)} issues) x {args.runs} runs\n"
    )

    done = 0

    def predict_with_progress(text, run):
        nonlocal done
        prediction = predict(text, run)
        done += 1
        if done % 10 == 0:
            print(f"  ... {done} new calls", flush=True)
        return prediction

    report = run_predictions(
        [texts[(r.repo, r.number)] for r in rows],
        args.runs,
        store,
        budget=TokenBudget(args.tokens_per_minute),
        max_calls=args.max_calls,
        predict_fn=predict_with_progress,
    )
    print(
        f"new calls: {report.called}, cached: {report.cached}, failed: {len(report.failures)}"
    )
    for failure in report.failures:
        print(f"  FAILED (not scored): {failure}")
    if report.stopped:
        print(f"STOPPED: {report.stopped}")

    # --- Scoring ----------------------------------------------------------------
    per_run, by_issue, complete = [], defaultdict(list), True
    for run in range(1, args.runs + 1):
        pairs = []
        for r in rows:
            p = store.get(r.repo, r.number, run)
            if p is not None:
                pairs.append((r, p))
                by_issue[(r.repo, r.number)].append(p.predicted)
        if len(pairs) < len(rows):
            complete = False
        if pairs:
            per_run.append(pairs)
    if not per_run:
        print("\nNothing to score yet.")
        return 1
    if not complete:
        print(
            "\n⚠ INCOMPLETE: some issues have no prediction yet; numbers cover what exists."
        )

    def scores_for(keep=lambda r: True):
        return [
            score(
                [(r.truth.type, p.predicted) for r, p in pairs if keep(r)], frequencies
            )
            for pairs in per_run
        ]

    overall = scores_for()
    print(f"\n=== {args.split}: {len(per_run)} run(s) — mean [low – high] ===")
    for name, attr in [
        ("coverage", "coverage"),
        ("accuracy on answered", "accuracy_answered"),
        ("macro-F1", "macro_f1"),
        ("frequency-weighted accuracy", "weighted_accuracy"),
    ]:
        print(f"  {name:28} {spread_text(getattr(s, attr) for s in overall)}")
    print(f"  {'same answer in every run':28} {pct(agreement(by_issue)).strip()}")

    print(
        f"\n{'type':9} {'n':>4} {'coverage':>9} {'precision':>10} {'recall':>8}   (mean of runs)"
    )
    for c in TYPES:
        cs = [s.per_class[c] for s in overall]
        m = {
            k: spread([getattr(x, k) for x in cs]).mean
            for k in ("coverage", "precision", "recall")
        }
        print(
            f"{c:9} {cs[0].support:4} {pct(m['coverage']):>9} {pct(m['precision']):>10} {pct(m['recall']):>8}"
        )

    # Option A (2026-11-20): R1 decides what the REPORTER asks. Issues the
    # maintainer kept as filed measure that; relabelled ones needed a diagnosis
    # the text doesn't contain — a known limit, reported separately. Coverage per
    # group shows experiment C: does "unclear" land on the relabelled issues?
    print("\nby kind of answer key:            coverage   accuracy on answered")
    for name, keep in [
        ("kept as filed (R1, option A)", lambda r: not r.truth.relabelled),
        ("relabelled (known limit)", lambda r: r.truth.relabelled),
        ("gold", lambda r: r.truth.trust == "gold"),
        ("silver", lambda r: r.truth.trust == "silver"),
    ]:
        subset = scores_for(keep)
        print(
            f"  {name:28} n={subset[0].n:3}  "
            f"{spread_text(s.coverage for s in subset):>8}   "
            f"{spread_text(s.accuracy_answered for s in subset)}"
        )

    # Baselines on exactly the issues the model was scored on (run 1):
    # numbers from different issue sets can't be compared.
    scored = [r for r, _ in per_run[0]]
    print(f"\nbaselines (the same {len(scored)} issues):")
    majority = max(frequencies, key=frequencies.get)
    base_major = score([(r.truth.type, majority) for r in scored])
    base_form = score(
        [
            (r.truth.type, LABEL_TO_TYPE.get(r.truth.reporter_label, "unclear"))
            for r in scored
        ]
    )
    print(
        f"  always '{majority}':  coverage {pct(base_major.coverage)}, "
        f"accuracy {pct(base_major.accuracy_answered)}, macro-F1 {pct(base_major.macro_f1)}"
    )
    print(
        f"  trust the form: coverage {pct(base_form.coverage)}, "
        f"accuracy {pct(base_form.accuracy_answered)}, macro-F1 {pct(base_form.macro_f1)}"
    )

    pooled = [(r.truth.type, p.predicted) for pairs in per_run for r, p in pairs]
    print(
        f"\nconfusion (rows = answer key, columns = prediction; pooled over {len(per_run)} runs):"
    )
    print(f"{'':10}" + "".join(f"{p:>9}" for p in (*TYPES, "unclear")))
    for t, row in confusion(pooled).items():
        print(f"{t:10}" + "".join(f"{row[p]:9}" for p in (*TYPES, "unclear")))

    # Calibration: can the model's confidence be used in a routing rule?
    answered = [
        (p.confidence, p.predicted == r.truth.type)
        for pairs in per_run
        for r, p in pairs
        if p.predicted != "unclear"
    ]
    if answered:
        print("\ncalibration (answered only; does confidence 0.9 mean 90% right?):")
        print(f"  {'confidence':12} {'n':>4} {'mean conf':>10} {'accuracy':>9}")
        for b in reliability(answered):
            print(
                f"  [{b.low:.2f}, {b.high:.2f}) {b.n:4} {b.mean_confidence:10.2f} "
                f"{b.accuracy:9.2f}"
            )
        right, wrong = separation(answered)
        print(f"  ECE {ece(answered):.3f} (0 = perfect)")
        print(
            "  mean confidence when right "
            + (f"{right:.3f}" if right is not None else "n/a")
            + ", when wrong "
            + (f"{wrong:.3f}" if wrong is not None else "n/a")
            + "  (equal = no threshold can separate them)"
        )

    preds = [p for pairs in per_run for _, p in pairs]
    latencies = sorted(p.latency_s for p in preds)
    tokens_in = sum(p.input_tokens for p in preds)
    tokens_out = sum(p.output_tokens for p in preds)
    print(
        f"\ncalls scored: {len(preds)}; tokens in/out: {tokens_in:,} / {tokens_out:,} "
        f"(mean {(tokens_in + tokens_out) // len(preds):,} per call)"
    )
    if len(latencies) >= 2:
        print(
            f"latency p50 {median(latencies):.1f}s, p95 {quantiles(latencies, n=20)[-1]:.1f}s"
            " (cached calls keep their original latency)"
        )
    if settings.groq_model in PRICES:
        cin, cout = PRICES[settings.groq_model]
        cost = (tokens_in * cin + tokens_out * cout) / 1e6
        print(f"cost: ${cost:.4f} (${1000 * cost / len(preds):.3f} per 1,000 issues)")

    if args.show_errors:
        print("\nmisclassified on dev (any run):")
        for r in rows:
            got = by_issue.get((r.repo, r.number), [])
            if any(p != r.truth.type for p in got):
                title = texts[(r.repo, r.number)].title[:70]
                print(
                    f"  #{r.number:<6} key={r.truth.type:8} got={'/'.join(got):26} "
                    f"{r.truth.trust}{' relabelled' if r.truth.relabelled else ''}  {title}"
                )
                print(f"           https://github.com/{r.repo}/issues/{r.number}")

    if args.split == "test":
        if complete and not report.stopped:
            entry = {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "config": cfg,
                "runs": len(per_run),
                "n": len(rows),
                **{
                    attr: spread(getattr(s, attr) for s in overall).model_dump()
                    for attr in (
                        "coverage",
                        "accuracy_answered",
                        "macro_f1",
                        "weighted_accuracy",
                    )
                },
            }
            with TEST_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"\ntest run logged in {TEST_LOG}")
        else:
            print("\ntest run incomplete: not logged. Re-run to finish it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
