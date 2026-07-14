"""decomp-search CLI.

  ingest-dtk  ROOT --project NAME [--version GALE01] [--report PATH|URL]
              [--backend hashed|local|voyage]
  find        FN [--project P] [--min-match 99.5] [-k 15] [--all] [--backend B]
  stats       [--backend B]
  eval        [--pairs eval/known_pairs.json] [--backend B]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TextColumn, TimeElapsedColumn)
from rich.table import Table

from . import db as dbmod
from .embed import default_backend, dim_for, embed
from .ingest_dtk import (find_objdump, find_source_file, iter_units,
                         load_report, parse_object)
from .normalize import token_text
from .sync import plan_sync

console = Console()


class Reporter:
    """Progress UI: rich bars on a TTY, plain flushed lines when redirected
    (rich Progress renders nothing until exit if stdout is not a terminal)."""

    def __init__(self):
        self._prog = (Progress(TextColumn("[bold blue]{task.description}"),
                               BarColumn(), MofNCompleteColumn(),
                               TimeElapsedColumn(), console=console)
                      if console.is_terminal else None)

    def __enter__(self):
        if self._prog:
            self._prog.__enter__()
        return self

    def __exit__(self, *exc):
        if self._prog:
            self._prog.__exit__(*exc)

    def task(self, desc: str, total: int):
        if self._prog:
            tid = self._prog.add_task(desc, total=total)
            return lambda done: self._prog.update(tid, completed=done)
        start = time.time()
        state = {"last": 0.0}

        def update(done: int) -> None:
            now = time.time()
            if done < total and now - state["last"] < 5.0:
                return
            state["last"] = now
            elapsed = now - start
            eta = elapsed / done * (total - done) if done else 0.0
            print(f"{desc}: {done}/{total} elapsed {elapsed:.0f}s "
                  f"eta {eta:.0f}s", flush=True)

        return update


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def cmd_ingest_dtk(args) -> None:
    root = Path(args.root).resolve()
    objdump = find_objdump(root)
    report = load_report(args.report) if args.report else {}
    backend = args.backend or default_backend()

    conn = dbmod.connect(args.db)
    table = dbmod.open_or_create(conn, dim_for(backend), backend)

    units = list(iter_units(root, args.version))
    desired: list[dict] = []
    with Reporter() as rep:
        upd = rep.task(f"disassembling {args.project}", len(units))
        for i, (obj, unit) in enumerate(units):
            src = find_source_file(root, unit) or ""
            for fn in parse_object(objdump, obj, unit):
                if fn.size < args.min_insns:
                    continue
                pct, _ = report.get(fn.name, (-1.0, None))
                desired.append({
                    "id": f"{args.project}:{unit}:{fn.name}",
                    "name": fn.name,
                    "project": args.project,
                    "unit": unit,
                    "src_path": src,
                    "n_insns": fn.size,
                    "match_pct": pct,
                    "backend": backend,
                    "tokens": token_text(fn),
                })
            upd(i + 1)

        existing = dbmod.fetch_project(table, args.project)
        plan = plan_sync(existing, desired, full=args.full)
        print(f"sync plan: {plan.unchanged} unchanged, "
              f"{len(plan.rewrite)} metadata-only, {len(plan.embed)} to embed, "
              f"{len(plan.delete_ids)} deletes", flush=True)

        if plan.delete_ids:
            dbmod.delete_ids(table, plan.delete_ids)
        for chunk in _chunks(plan.rewrite, args.chunk):
            table.add(chunk)

        # longest-first: chunks stay length-homogeneous for batch packing,
        # and the memory-heaviest work runs first (fail fast, not at hour 1)
        plan.embed.sort(key=lambda r: len(r["tokens"]), reverse=True)
        upd = rep.task(f"embed+write ({backend})", len(plan.embed))
        done = 0
        for chunk in _chunks(plan.embed, args.chunk):
            vecs = embed([r["tokens"] for r in chunk], backend,
                         progress=lambda d, t: upd(done + d))
            table.add([{**r, "vector": v} for r, v in zip(chunk, vecs)])
            done += len(chunk)
            upd(done)

    console.print(f"[green]{args.project}: {plan.unchanged} kept, "
                  f"{len(plan.rewrite)} refreshed, {len(plan.embed)} embedded, "
                  f"{len(plan.delete_ids)} deleted (backend={backend})[/green]")


def _lookup(table, name: str, project: str | None):
    q = table.search().where(
        f"name = '{name}'" + (f" AND project = '{project}'" if project else ""),
        prefilter=True).limit(2).to_list()
    return q[0] if q else None


def _find(table, row: dict, k: int, min_match: float | None,
          exclude_self_unit: bool) -> list[dict]:
    where = []
    if min_match is not None:
        where.append(f"match_pct >= {min_match}")
    res = (table.search(row["vector"]).metric("cosine")
           .where(" AND ".join(where) if where else None, prefilter=True)
           .limit(k * 10 + 50).to_list())
    out = []
    for r in res:
        if r["id"] == row["id"]:
            continue
        if exclude_self_unit and r["unit"] == row["unit"] \
                and r["project"] == row["project"]:
            continue
        out.append(r)
    return out[:k]


def _print_hits(query_name: str, hits: list[dict]) -> None:
    tbl = Table(title=f"neighbors of {query_name}")
    tbl.add_column("sim", justify="right")
    tbl.add_column("match%", justify="right")
    tbl.add_column("insns", justify="right")
    tbl.add_column("function")
    tbl.add_column("unit")
    for r in hits:
        sim = 1.0 - r["_distance"]
        pct = r["match_pct"]
        pct_s = f"{pct:.2f}" if pct >= 0 else "?"
        style = "green" if pct >= 99.5 else ("yellow" if pct >= 0 else "dim")
        tbl.add_row(f"{sim:.3f}", f"[{style}]{pct_s}[/{style}]",
                    str(r["n_insns"]), r["name"], r["unit"])
    console.print(tbl)


def _open(args):
    conn = dbmod.connect(args.db)
    name = dbmod.table_name(args.backend)
    if name not in conn.table_names():
        console.print(f"[red]no index for backend {args.backend!r} — "
                      f"run ingest-dtk with --backend {args.backend}[/red]")
        sys.exit(1)
    return conn.open_table(name)


def cmd_find(args) -> None:
    table = _open(args)
    row = _lookup(table, args.function, args.project)
    if row is None:
        console.print(f"[red]function {args.function!r} not found[/red]")
        sys.exit(1)
    min_match = None if args.all else args.min_match
    hits = _find(table, row, args.k, min_match, args.exclude_self_unit)
    _print_hits(f"{args.function} ({row['n_insns']} insns, "
                f"match {row['match_pct']:.2f})", hits)


def cmd_stats(args) -> None:
    table = _open(args)
    n = table.count_rows()
    console.print(f"{n} functions indexed (backend={args.backend})")


def cmd_eval(args) -> None:
    table = _open(args)
    pairs = json.load(open(args.pairs))
    ok = 0
    for case in pairs:
        row = _lookup(table, case["query"], case.get("project"))
        if row is None:
            console.print(f"[red]missing {case['query']}[/red]")
            continue
        hits = _find(table, row, args.k, None, False)
        names = [h["name"] for h in hits]
        want = case["expect"]
        rank = next((i + 1 for i, n in enumerate(names) if n in want), None)
        if rank is not None:
            ok += 1
            console.print(f"[green]HIT [/green] {case['query']}: "
                          f"{names[rank-1]} at rank {rank}")
        else:
            console.print(f"[red]MISS[/red] {case['query']}: wanted "
                          f"{want}, top: {names[:5]}")
    console.print(f"{ok}/{len(pairs)} recovered in top {args.k}")


def main() -> None:
    p = argparse.ArgumentParser(prog="dsearch")
    p.add_argument("--db", default=str(dbmod.DEFAULT_DB))
    p.add_argument("--backend", choices=["hashed", "local", "voyage"],
                   default=default_backend())
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest-dtk")
    pi.add_argument("root")
    pi.add_argument("--project", required=True)
    pi.add_argument("--version", default="GALE01")
    pi.add_argument("--report", help="decomp.dev report JSON path or URL")
    pi.add_argument("--min-insns", type=int, default=8)
    pi.add_argument("--full", action="store_true",
                    help="re-embed everything, ignoring stored vectors")
    pi.add_argument("--chunk", type=int, default=1024,
                    help="functions per embed+write chunk (DB write/resume unit)")
    pi.set_defaults(func=cmd_ingest_dtk)

    pf = sub.add_parser("find")
    pf.add_argument("function")
    pf.add_argument("--project")
    pf.add_argument("--min-match", type=float, default=99.5)
    pf.add_argument("--all", action="store_true",
                    help="no match%% filter (include unmatched functions)")
    pf.add_argument("-k", type=int, default=15)
    pf.add_argument("--exclude-self-unit", action="store_true")
    pf.set_defaults(func=cmd_find)

    ps = sub.add_parser("stats")
    ps.set_defaults(func=cmd_stats)

    pe = sub.add_parser("eval")
    pe.add_argument("--pairs", default=str(Path(__file__).resolve().parent.parent
                                           / "eval" / "known_pairs.json"))
    pe.add_argument("-k", type=int, default=20)
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
