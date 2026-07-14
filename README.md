# decomp-search

Local similarity search over decompilation-project functions. Ingest a
project's target assembly (and match metadata), embed every function, and
query for structurally similar functions — e.g. "given this unmatched
function, show me the most similar **matched** functions so I can steal
their source recipe."

Storage/search is [LanceDB](https://lancedb.github.io/lancedb/) (local, no
server). Embeddings are pluggable:

- `hashed` (default): deterministic feature-hashed n-grams over a normalized
  instruction-token stream. No API, fully reproducible.
- `voyage`: voyage-4-nano via the Voyage API (`VOYAGE_API_KEY` env var).

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

Re-running replaces that project's rows, so multiple games can coexist in
one index (`--project kirby ...` etc.).

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
- `dsearch/db.py` — LanceDB schema/connection
- `dsearch/cli.py` — `ingest-dtk` / `find` / `stats` / `eval`

Adding another project layout = one new `ingest_*.py` adapter that yields
`normalize.Function` records.
