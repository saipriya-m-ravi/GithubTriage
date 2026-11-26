-- 0001: the index of past issues for the duplicate finder (slice 5 module 2).
--
-- Migrations are applied in file-name order, once each, inside a transaction
-- (see db.migrate). Never edit an applied migration: add a new numbered file.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per GitHub issue (pull requests are never stored).
CREATE TABLE issues (
    repo         text        NOT NULL,
    number       integer     NOT NULL,
    title        text        NOT NULL,
    body         text        NOT NULL DEFAULT '',
    state        text        NOT NULL CHECK (state IN ('open', 'closed')),
    -- GitHub's close reason (completed / not_planned / duplicate / reopened…):
    -- lets a reply say "this was discussed in #412 and declined".
    state_reason text,
    labels       text[]      NOT NULL DEFAULT '{}',
    html_url     text        NOT NULL,
    -- created_at drives the time-travel rule: a search for an issue may only
    -- return issues created BEFORE it. The search filters on it inside the
    -- query (before ranking), so it is indexed.
    created_at   timestamptz NOT NULL,
    updated_at   timestamptz NOT NULL,
    closed_at    timestamptz,
    PRIMARY KEY (repo, number),

    -- Keyword search: title weighted above body ('A' > 'B'). A generated
    -- column is always in sync with the text — no code path can forget it.
    search tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', title), 'A')
        || setweight(to_tsvector('english', body), 'B')
    ) STORED
);

CREATE INDEX issues_search_idx ON issues USING gin (search);
CREATE INDEX issues_created_idx ON issues (repo, created_at);

-- One embedding per issue PER MODEL. Vectors from different models can't be
-- compared, so the model name is part of the key: switching models means
-- re-embedding into new rows, never mixing.
--
-- vector(384) = BAAI/bge-small-en-v1.5. A model with another dimension needs
-- a new migration (the HNSW index requires a fixed dimension).
CREATE TABLE issue_embeddings (
    repo        text        NOT NULL,
    number      integer     NOT NULL,
    model       text        NOT NULL,
    embedding   vector(384) NOT NULL,
    -- Hash of the exact text that was embedded: an edited issue is re-embedded
    -- only when its text actually changed.
    text_sha256 text        NOT NULL,
    embedded_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, number, model),
    FOREIGN KEY (repo, number) REFERENCES issues (repo, number) ON DELETE CASCADE
);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
CREATE INDEX issue_embeddings_hnsw_idx
    ON issue_embeddings USING hnsw (embedding vector_cosine_ops);
