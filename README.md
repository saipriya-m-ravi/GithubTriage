## The project: **GITHUB-TRIAGE** — an agentic GitHub issue triage assistant

A GitHub App that watches a repository. When a new issue is opened, a LangGraph workflow (with one sandboxed agent for reproduction):
1. **Classifies** it (bug / feature / question / docs) and suggests labels and priority
2. **Finds duplicates** by searching past issues and docs (RAG)
3. **Checks completeness** — is there a version, repro steps, logs? If not, drafts a polite "please add X" request
4. **Attempts reproduction** for bugs with a code snippet, inside a sandbox
5. **Drafts a first response** grounded in docs and past issues, with citations
6. **Waits for a maintainer to approve** (human-in-the-loop) before posting anything

### Status
Built in 8 vertical slices, each adding one capability to a system that already runs end to end.

- ✅ **Slice 1:** classify an issue (one structured LLM call with prompt-injection guardrails) → template-based draft reply → CLI.
- ✅ **Slice 2:** completeness check for bugs (conditional routing) → "please add…" requests from fixed, allowlisted wording.
- ✅ **Slice 3:** human approval — the run pauses, survives restarts (SQLite checkpoints), and resumes on approve / edit / reject; posting is idempotent per run (tested through a simulated crash). Replies go to a local outbox until the GitHub API arrives in slice 4.
- 🔄 **Slice 4:** live GitHub issues — triage by `owner/repo#N` or URL, real comments posted only to allowlisted repos (exactly once per run, author-verified marker), and stale drafts discarded if the issue was edited or commented on after triage. Tested against fake and **recorded, anonymised** GitHub responses. **First real evaluation**: 237 closed pydantic issues with a maintainer-based answer key, a frozen dev/test split, and a cached, rate-limit-aware eval runner — see [Results](#results-so-far): **62.5% on the held-out test split (88.9% on issues kept as filed) vs 44.2% for trusting the reporter's form.** *R3 hand-labels are deferred.*
- ⏳ Slices 5–8: duplicate finder (RAG), grounded answers, FastAPI service, reproduction agent.

## Running it

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- A [Groq](https://console.groq.com) API key, and a Groq model that supports tool use / structured outputs

### Setup
```bash
git clone https://github.com/saipriya-m-ravi/GithubTriage.git
cd GithubTriage
uv sync                      # creates .venv and installs the exact locked versions
cp .env.example .env         # Windows PowerShell: Copy-Item .env.example .env
```
Edit `.env`:
```
GROQ_API_KEY=your-key-here
GROQ_MODEL=your-chosen-groq-model
```
`.env` is git-ignored — never commit it.

### Triage an issue, then decide
Triage runs until a maintainer's decision is needed, then **pauses** — the run is saved, so the decision can come later, from a separate command.
```bash
uv run githubtriage triage samples/issues/bug_vague_version.json
```
```
type:       bug (0.99)
reasoning:  The reporter shows a traceback and says the function raises KeyError after upgrade, ...

---- draft ----
Thanks for opening this issue, @reporter11!

We've labelled this as a **bug**. A maintainer will look into it.

To help us investigate, could you add:
- the version of the project you're using (for example, 2.3.0)

_Drafted by GithubTriage — pending maintainer review._

Run: example-org/datekit#111@20261104T061416.125787Z
Waiting for approval: githubtriage approve|edit|reject example-org/datekit#111
```

| Command | What it does |
|---|---|
| `uv run githubtriage show <ref>` | Show the waiting draft, with the classification and completeness notes |
| `uv run githubtriage approve <ref>` | Post the draft as is |
| `uv run githubtriage edit <ref> --body-file reply.md` | Post your edited reply instead |
| `uv run githubtriage reject <ref> [--reason "..."]` | Post nothing |
| `uv run githubtriage retry <ref>` | Approved, but posting failed (e.g. GitHub was down): try posting again |
| `uv run githubtriage triage owner/repo#7` | Triage a **live** GitHub issue (also accepts the issue's URL) |
| `uv run githubtriage triage <source> --json` | Print the full result as JSON |
| add `-v` to any command | Show debug logs |
| `uv run githubtriage --help` | Show usage |

Live triage works anonymously for public repositories (60 GitHub requests/hour); set `GITHUB_TOKEN` (see `.env.example`) for private repositories and 5,000 requests/hour. Pull requests are refused.

**Where approved replies go:** by default to a local outbox (`.githubtriage/outbox/`) — nothing is posted to GitHub. To post real comments, set `POST_MODE=github`, `GITHUB_TOKEN`, and `POST_ALLOWED_REPOS` — **only repositories you own**; posting anywhere else is refused. Each reply carries a hidden marker, so a retry never posts it twice.

**Stale drafts aren't posted:** before posting a reply to a live issue, GithubTriage checks whether the issue was edited, or someone else commented, since it was triaged. If so, the draft is discarded and you're asked to triage again, so a reply never ignores information added in the meantime.

`<ref>` is an issue (`owner/repo#7`, meaning its one waiting draft) or an exact run id. Triaging an issue again **cancels** its waiting draft; after a reply was posted, a new triage may post again. Paused runs are stored in `.githubtriage/checkpoints.sqlite` and "posted" replies in `.githubtriage/outbox/` (relative to where you run the command; set `DATA_DIR` to change it).

Exit codes: `0` success (degraded triage runs included — errors are printed to stderr as warnings), `1` bad input or nothing waiting, `2` wrong usage. Results go to stdout; logs and warnings go to stderr.

### Input format
An issue file is JSON shaped like this (see [samples/issues/](samples/issues/) for 13 examples, including prompt-injection and edge cases):
```json
{
  "repo": "example-org/datekit",
  "issue_number": 101,
  "issue": {
    "title": "parse_date returns March 1st for 2024-02-30 instead of raising",
    "body": "Steps to reproduce ...",
    "author": "reporter1",
    "updated_at": "2026-10-01T09:15:00Z"
  }
}
```

### Run all samples against the real model
```bash
uv run python debug_helper_scripts/run_samples.py
```
Compares each sample's classification with [samples/issues/expected.json](samples/issues/expected.json) and prints `OK` / `MISS` per sample plus a summary. Calls the LLM — costs tokens and results can vary slightly between runs.

### Evaluate on real issues (pydantic/pydantic)
The answer key is committed ([evals/pydantic/manifest.jsonl](evals/pydantic/manifest.jsonl): issue numbers and correct types, no issue text). Issue texts are fetched to a local, gitignored file:
```bash
uv run python debug_helper_scripts/backfill_eval.py --rebuild    # needs GITHUB_TOKEN; fetches the texts once
uv run python debug_helper_scripts/run_eval.py --split dev       # calls the LLM; results are cached
uv run python debug_helper_scripts/run_eval.py --split dev --show-errors
uv run python debug_helper_scripts/run_eval.py --split test      # final measurement only: summary + logged
```
Every prediction is cached per configuration (model, reasoning effort, prompt version), so a run stopped by a rate limit resumes where it stopped and re-scoring is free. `--show-errors` works on dev only; complete test runs are appended to `evals/pydantic/test_runs.jsonl` (created by the first test run).

### Tests and checks
```bash
uv run pytest              # all tests; uses a fake LLM, no network or API key needed
uv run ruff check .        # lint
uv run ruff format .       # format
```
Tests never call GitHub: they use hand-written fakes plus real GitHub responses recorded in `tests/fixtures/github/`. To refresh the recordings (read-only; needs `GITHUB_TOKEN` with access to the sandbox repo named in the script):
```bash
uv run python debug_helper_scripts/record_github_fixtures.py
```
Logins, e-mails and third-party text are anonymised before anything is written.

### Project layout
```
src/githubtriage/
  config.py              settings from .env (pydantic-settings)
  models.py              Issue, Classification, Completeness, Decision, TriageState
  llm.py                 creates the chat model
  issue_text.py          sanitising + truncation of issue text for prompts
  nodes/classify.py      R1: structured LLM classification + guardrails
  nodes/completeness.py  R3: is a bug report complete?
  nodes/draft.py         R5: template-based draft reply ("please add…" from an allowlist)
  nodes/approval.py      R6: notify + pause for the maintainer's decision
  nodes/post.py          R6: stale check + idempotent posting (outbox or GitHub)
  graph.py               LangGraph workflow (code-routed)
  checkpointing.py       SQLite checkpoints, run ids, waiting/stuck runs
  github_client.py       small httpx GitHub REST client (errors, retries, pagination)
  github_issues.py       issue refs/URLs; GitHub JSON → TriageState
  repo_names.py          the one owner/name validation rule
  observability.py       LangSmith tracing + run metadata
  cli.py                 command-line adapter
  evaluation/            eval set + scoring: labels.py (answer key from label history),
                         dataset.py (collection), split.py (dev/test), runner.py
                         (cache, pacing), metrics.py, completeness_labels.py (R3 kit)
tests/                   pytest suite (shared fakes in conftest.py)
tests/fixtures/github/   recorded, anonymised GitHub responses
evals/pydantic/          committed eval data: manifest, frequencies, test-run log, R3 labelling guide
samples/issues/          sample issues + expected.json
debug_helper_scripts/    run_samples, show_run, record_github_fixtures, backfill_eval,
                         run_eval, completeness_labelling
docs/DESIGN.md           design decisions, threat model, eval results
```

## Design

### Stack
| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.13+ | Ecosystem default |
| Orchestration | LangGraph | Explicit state graphs, checkpointing, HITL interrupts built in |
| LLM | Groq via LangChain (`langchain-groq`) | One-line structured output, automatic LangSmith tracing, provider switch is a one-line change |
| Schemas | Pydantic | Structured outputs, validation |
| API | FastAPI | Webhook receiver + approval endpoints |
| Storage | Postgres + pgvector | One DB for vectors, checkpoints, and app data — less to operate |
| Queue | Background worker (arq/Celery, or a simple Postgres job table) | Webhooks must return fast; agents run async |
| Observability | LangSmith | Traces, tokens, cost, latency; LangGraph Studio for debugging |
| UI | FastAPI + Jinja + HTMX | Maintainer approval page + public read-only demo, same service |
| Deploy | Docker → Render / Fly.io / Railway | Cheap, real URL |

### Target architecture
```
GitHub webhook ──> FastAPI ──> job queue ──> LangGraph workflow (code-routed)

   ┌─ Classifier (rule / 1 LLM call) ─┐
   │                                  ├─► duplicate? ─yes─────────────────────────┐
   └─ Duplicate finder (hybrid RAG) ──┘       │no                                 │
                                              ▼                                   ▼
                              Completeness (rule + LLM) ─► complete bug? ─no─► Responder
                                                              │yes          (templates + grounded
                                                              ▼               LLM for questions)
                                                  Repro agent (ReAct, sandbox) ──┘    │
                                                                                      ▼
                                              notify ─► HITL interrupt: maintainer approves (FastAPI page)
                                                                                      │
                                              stale check ─► post comment / apply labels via GitHub API
```
Routing is done by code, not an LLM supervisor — see [docs/DESIGN.md](docs/DESIGN.md) §4 for why.

## Results so far

**R1 classification on real pydantic issues** — `openai/gpt-oss-120b` (low reasoning effort), prompt `classify-v1`, one run.

| Split | Issues | Accuracy | Macro-F1 | On issues the maintainer kept as filed | On issues the maintainer relabelled |
|---|---|---|---|---|---|
| dev (tuning) | 115 | 63.5% | 61.6% | 95.0% (n=60) | 29.1% (n=55) |
| **test (measured once)** | 122 | **62.5%** | **60.0%** | **88.9%** (63 answered of 65) | 33.3% (n=57) |

Baselines on the same test issues: **trusting the reporter's chosen form 44.2%** (and 0 of 60 questions across the whole set — not one question was filed as a question), **always "bug" 25.4%**. The test row is the result to quote; with ~60 issues per group, one run carries roughly ±8 points of uncertainty.

What the numbers mean:
- **The answer key** comes from each issue's label *history*, not its final label: pydantic's issue forms add labels automatically, so a label is trusted (gold) only when a maintainer applied or confirmed it.
- **95% vs 29%.** Most misses are issues *written as bug reports* that a maintainer later diagnosed as intended behaviour (→ question) or a docs gap. R1 decides **what the reporter is asking**; that diagnosis needs knowledge the text doesn't contain, so it's a known limit for the knowledge-based steps (duplicates, docs-grounded answers).
- **Confidence isn't usable for routing:** on test, 0.976 average when right vs 0.952 when wrong (ECE 0.342), so only `unclear` sends an issue to a human.
- **An abstention experiment failed** by criteria fixed in advance (the model never said `unclear`) and wasn't shipped.
- Cost ~$0.22 per 1,000 issues; p50 latency 2.6s. The binding constraint is the provider's rate limit, not money.

**R3 completeness check** (`completeness-v1`) against 50 hand-labelled bug reports (25 dev / 25 test, labelled blind): on the test split, **6 of 25 replies would ask for something the reporter already gave** (mostly repro steps), and **1 missed ask** — R3 errs towards asking. A tighter `logs` wording halved nagging replies on dev but missed its pre-registered bar, so it wasn't adopted. About 1 call in 10 fails to produce structured output (counted, not scored; a retry is planned).

**R2 duplicate finder, Stage 1 (retrieval)** on 97 real duplicate pairs from pydantic (GitHub's "closed as duplicate of", 48 dev / 49 test), searching only issues created *before* each duplicate: the true duplicate is among the **top 5 candidates for 67.3%** of test pairs (vector search). Keyword search and hybrid (RRF) fusion were measured and lost — Postgres's ranking has no IDF, so even with a common-word filter the keyword list mostly added noise near the top.

Full method, caveats and decisions: [docs/DESIGN.md](docs/DESIGN.md) → R1 → "Eval ground truth" onwards, R2 and R3.

### Metrics to be reported
- Label accuracy / macro-F1 vs. maintainer labels — *measured: test 62.5% / 60.0%, 88.9% on issues kept as filed (above)*
- Duplicate detection recall@5
- Response quality score (LLM-as-judge with a rubric, spot-checked by hand)
- Tool-call correctness rate
- p50/p95 latency and cost per issue, before and after optimization — *R1 (test): p50 2.6s, ~$0.22 per 1,000 issues*
- Prompt-injection test pass rate