"""decomp-twins CLI.

  ingest-dtk  ROOT --project NAME [--version GALE01] [--report PATH|URL]
  find        FN [--project P] [--min-match 99] [-k 15] [--all]
  stats
  eval        [--pairs eval/known_pairs.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           TextColumn, TimeElapsedColumn)
from rich.table import Table

from . import db as dbmod
from .embed import HASHED_DIM, default_backend, embed
from .ingest_dtk import (find_objdump, find_source_file, iter_units,
                         load_report, parse_object)
from .normalize import token_text

console = Console()


def cmd_ingest_dtk(args) -> None:
    root = Path(args.root).resolve()
    objdump = find_objdump(root)
    report = load_report(args.report) if args.report else {}
    backend = args.backend or default_backend()

    units = list(iter_units(root, args.version))
    records: list[dict] = []
    docs: list[str] = []

    with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        t = prog.add_task(f"disassembling {args.project}", total=len(units))
        for obj, unit in units:
            src = find_source_file(root, unit) or ""
            for fn in parse_object(objdump, obj, unit):
                if fn.size < args.min_insns:
                    continue
                pct, _ = report.get(fn.name, (-1.0, None))
                doc = token_text(fn)
                docs.append(doc)
                records.append({
                    "id": f"{args.project}:{unit}:{fn.name}",
                    "name": fn.name,
                    "project": args.project,
                    "unit": unit,
                    "src_path": src,
                    "n_insns": fn.size,
                    "match_pct": pct,
                    "backend": backend,
                    "tokens": doc,
                })
            prog.advance(t)

        t2 = prog.add_task(f"embedding ({backend})", total=1)
        vectors = embed(docs, backend)
        prog.advance(t2)

    for r, v in zip(records, vectors):
        r["vector"] = v

    dim = len(vectors[0]) if vectors else HASHED_DIM
    conn = dbmod.connect(args.db)
    table = dbmod.open_or_create(conn, dim)
    dbmod.replace_project(table, args.project)
    table.add(records)
    console.print(f"[green]ingested {len(records)} functions "
                  f"({args.project}, backend={backend})[/green]")


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


def cmd_find(args) -> None:
    conn = dbmod.connect(args.db)
    table = conn.open_table(dbmod.TABLE)
    row = _lookup(table, args.function, args.project)
    if row is None:
        console.print(f"[red]function {args.function!r} not found[/red]")
        sys.exit(1)
    min_match = None if args.all else args.min_match
    hits = _find(table, row, args.k, min_match, args.exclude_self_unit)
    _print_hits(f"{args.function} ({row['n_insns']} insns, "
                f"match {row['match_pct']:.2f})", hits)


def cmd_stats(args) -> None:
    conn = dbmod.connect(args.db)
    table = conn.open_table(dbmod.TABLE)
    n = table.count_rows()
    console.print(f"{n} functions indexed")


def cmd_eval(args) -> None:
    conn = dbmod.connect(args.db)
    table = conn.open_table(dbmod.TABLE)
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
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest-dtk")
    pi.add_argument("root")
    pi.add_argument("--project", required=True)
    pi.add_argument("--version", default="GALE01")
    pi.add_argument("--report", help="decomp.dev report JSON path or URL")
    pi.add_argument("--backend", choices=["hashed", "voyage"])
    pi.add_argument("--min-insns", type=int, default=8)
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
