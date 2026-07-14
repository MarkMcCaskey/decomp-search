# decomp-search

Local similarity search over decompilation-project functions. Ingest a
project's target assembly (and match metadata), embed every function, and
query for structurally similar functions — e.g. "given this unmatched
function, show me the most similar **matched** functions so I can steal
their source recipe."

Storage/search is [LanceDB](https://lancedb.github.io/lancedb/) (local, no
server). Embeddings are pluggable (each backend gets its own table, so they
coexist for A/B comparison; select with the global `--backend` flag):

- `hashed` (default): deterministic feature-hashed n-grams over a normalized
  instruction-token stream. No API, no model download, fully reproducible.
- `local`: [voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano)
  self-hosted via sentence-transformers (open weights, Apache 2.0; ~340M
  params, runs on MPS/CUDA/CPU; first run downloads the model). Shares an
  embedding space with the larger Voyage 4 API models, so a locally built
  index can later be queried with `voyage-4-large` API embeddings without
  re-indexing.
- `voyage`: voyage-4-nano via the Voyage API (`VOYAGE_API_KEY` env var).
  Same embedding space as `local`.

Normalization keeps the structural signal (mnemonic skeleton, operand
shapes, **branch direction** — `b(back)` is a backedge) and discards what
varies between twins (register numbers, addresses, symbol names).

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -e .          # or: pip install -e '.[voyage]'
```

## Ingest a dtk-based project

Needs the project's built target objects (`build/<VERSION>/obj/**/*.o`) and
optionally a decomp.dev progress report for match percentages:

```sh
.venv/bin/python -m dsearch.cli ingest-dtk ~/etc/melee \
    --project melee --version GALE01 \
    --report 'https://decomp.dev/doldecomp/melee/GALE01.json?mode=report'
```

Ingest is **incremental**: each function's token text is diffed against the
stored row, so re-running only re-embeds new/changed functions (metadata-only
changes like a moved match % reuse the stored vector), and deletes stale
rows. Every embedding batch writes to LanceDB as it finishes, so an
interrupted ingest loses at most one batch — rerun and it resumes. `--full`
forces a re-embed of everything. Multiple games coexist in one index
(`--project kirby ...` etc.). Progress renders as rich bars on a TTY and as
plain flushed lines when redirected to a log.

## Query

```sh
# top matched functions similar to an unmatched one (the twin-finder):
.venv/bin/python -m dsearch.cli find lbHeap_80015900 --min-match 99.5

# unfiltered similarity (see the whole neighborhood):
.venv/bin/python -m dsearch.cli find mpRightWallGetTop --all

# cross-TU only (drop trivial same-file siblings):
.venv/bin/python -m dsearch.cli find mpRightWallGetTop --exclude-self-unit
```

## Eval

`eval/known_pairs.json` holds ground-truth twin pairs found manually during
matching work. `eval` reports recall@k:

```sh
.venv/bin/python -m dsearch.cli eval
```

## Layout

- `dsearch/normalize.py` — objdump text → instruction token stream
- `dsearch/embed.py` — hashed / voyage embedding backends
- `dsearch/ingest_dtk.py` — dtk project adapter (objdump + decomp.dev report)
- `dsearch/sync.py` — incremental sync planning (token diff → embed/reuse/delete)
- `dsearch/db.py` — LanceDB schema/connection
- `dsearch/cli.py` — `ingest-dtk` / `find` / `stats` / `eval`

Adding another project layout = one new `ingest_*.py` adapter that yields
`normalize.Function` records.
