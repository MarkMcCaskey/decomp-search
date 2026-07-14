---
name: decomp-search
description: Local similarity search over decomp-project functions — find matched twins and recipe donors for an unmatched function's asm, whole-function or construct-level (windows). Use when stuck on a match residual, hunting shape twins, or scoping whether a function is unique.
---

# decomp-search

Repo: `~/etc/decomp-search` (own venv). Run everything from that dir:

```sh
.venv/bin/python -m dsearch.cli [--backend hashed|local|voyage] <cmd> ...
```

## Core query — matched donors for a function

```sh
.venv/bin/python -m dsearch.cli find <fn> --min-match 99.5 --exclude-self-unit -k 10
```

- `--all` = no match% filter (see the whole neighborhood)
- Drop `--exclude-self-unit` to include same-TU siblings
- Default backend is `local` (self-hosted voyage-4-nano, semantic;
  DSEARCH_BACKEND env overrides). `--backend hashed` = deterministic
  n-grams — check both when it matters. `find` never runs the model at
  query time; it uses the function's stored vector, so a backend only
  works for projects ingested with it (check `stats`).

## Construct-level query — window twins

A loop/construct buried inside a larger matched function won't surface in
`find` (whole-function vectors). `findw` searches every 32-insn sliding
window of the query fn against all indexed windows and aggregates the best
hit per candidate function (`q@`/`t@` columns = window insn offsets):

```sh
.venv/bin/python -m dsearch.cli --backend hashed findw <fn> --min-match 99.5 -k 10
```

Validated: found lbHeap_80015900 -> MakeColorGenTExp (2x-unroll construct
at t@416) which whole-function search misses. Windows are hashed-backend
only so far. For exact instruction-pattern hunts, targeted scans
(`melee/build/twinscan_*.py`: objdump all target .o's, regex, rank by
match%) remain the sharpest tool — findw for recall, twinscan for proof.

## Picking solve targets

`solvability_sweep.py` ranks every sub-95% function by its best matched
cross-unit neighbor — top entries are "probably solvable by donor". This
found un_80300AF4 (74%->match, melee PR: extern-decl/define-at-bottom
hoist fix).

## Ingest a project (dtk-based)

Needs target objects (`build/<VER>/obj/**/*.o` — dtk split runs natively on
macOS, no wine) + a decomp.dev report:

```sh
.venv/bin/python -m dsearch.cli --backend hashed ingest-dtk <repo_root> \
    --project <name> --version <VER> --windows \
    --report 'https://decomp.dev/<org>/<repo>/<VER>.json?mode=report'
```

Ingest is incremental (token-text diff): re-runs only embed new/changed
functions; every embedding batch persists immediately, so interrupt/resume
is safe (rerun continues). `--full` forces a re-embed. `--windows` also
builds the construct-level index. One table per backend+kind. Progress
prints plain flushed lines when redirected.
Project names in the index: melee, mp4, pikmin2 — reuse these exact names
on re-ingest or you'll create a duplicate project.

## Maintain the eval

When you find a true twin pair manually, add it to `eval/known_pairs.json`,
then check recall: `.venv/bin/python -m dsearch.cli eval`.

## Gotchas

- venv scripts (pip) have stale shebangs after any repo move — use
  `.venv/bin/python -m pip`, never `.venv/bin/pip`.
- `local` backend: transformers must be <5 (pinned). Embedding batches are
  token-budget packed (32k padded tokens); don't run two GPU ingests at
  once on MPS.
- This file is canonical in the repo
  (`.claude/skills/decomp-search/SKILL.md`); `~/.claude/skills/decomp-search/SKILL.md`
  is a symlink to it.
