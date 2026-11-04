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
- ⏳ Slices 4–8: GitHub API, duplicate finder (RAG), grounded answers, FastAPI service, reproduction agent.

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
| `uv run githubtriage triage <file> --json` | Print the full result as JSON |
| add `-v` to any command | Show debug logs |
| `uv run githubtriage --help` | Show usage |

`<ref>` is an issue (`owner/repo#7`, meaning its one waiting draft) or an exact run id. Triaging an issue again **cancels** its waiting draft; after a reply was posted, a new triage may post again. Paused runs are stored in `.githubtriage/checkpoints.sqlite` and "posted" replies in `.githubtriage/outbox/` (relative to where you run the command; set `DATA_DIR` to change it).

Exit codes: `0` success (degraded triage runs included — errors are printed to stderr as warnings), `1` bad input or nothing waiting, `2` wrong usage. Results go to stdout; logs and warnings go to stderr.

### Input format
An issue file is JSON shaped like this (see [samples/issues/](samples/issues/) for 10 examples, including prompt-injection and edge cases):
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

### Tests and checks
```bash
uv run pytest              # all tests; uses a fake LLM, no network or API key needed
uv run ruff check .        # lint
uv run ruff format .       # format
```

### Project layout
```
src/githubtriage/
  config.py          settings from .env (pydantic-settings)
  models.py          Issue, Classification, TriageState
  llm.py             creates the chat model
  nodes/classify.py  R1: structured LLM classification + guardrails
  nodes/draft.py     R5: template-based draft reply
  graph.py           LangGraph workflow
  cli.py             command-line adapter
tests/               pytest suite (shared fakes in conftest.py)
samples/issues/      sample issues + expected.json
docs/DESIGN.md       design decisions, threat model
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

### Metrics to be reported
- Label accuracy / macro-F1 vs. maintainer labels
- Duplicate detection recall@5
- Response quality score (LLM-as-judge with a rubric, spot-checked by hand)
- Tool-call correctness rate
- p50/p95 latency and cost per issue, before and after optimization
- Prompt-injection test pass rate