## The project: **GITHUB-TRIAGE** — an agentic GitHub issue triage assistant

A GitHub App that watches a repository. When a new issue is opened, a LangGraph workflow (with one sandboxed agent for reproduction):
1. **Classifies** it (bug / feature / question / docs) and suggests labels and priority
2. **Finds duplicates** by searching past issues and docs (RAG)
3. **Checks completeness** — is there a version, repro steps, logs? If not, drafts a polite "please add X" request
4. **Attempts reproduction** for bugs with a code snippet, inside a sandbox
5. **Drafts a first response** grounded in docs and past issues, with citations
6. **Waits for a maintainer to approve** (human-in-the-loop) before posting anything

### Stack
| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.13+ | Ecosystem default |
| Orchestration | LangGraph | Explicit state graphs, checkpointing, HITL interrupts built in |
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
- Response quality score (LLM-as-judge with a rubric, spot-checked by you)
- Tool-call correctness rate
- p50/p95 latency and cost per issue, before and after optimization
- Prompt-injection test pass rate