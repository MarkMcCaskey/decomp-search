"""LanceDB storage layer."""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "index.lancedb"


def table_name(backend: str, kind: str = "functions") -> str:
    return f"{kind}_{backend}"


def connect(db_path: str | Path = DEFAULT_DB):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def has_table(db, name: str) -> bool:
    if hasattr(db, "list_tables"):
        res = db.list_tables()
        # newer lancedb returns a ListTablesResponse with .tables
        names = getattr(res, "tables", res)
    else:
        names = db.table_names()
    return name in list(names)


def schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("project", pa.string()),
        pa.field("unit", pa.string()),
        pa.field("src_path", pa.string()),
        pa.field("n_insns", pa.int32()),
        pa.field("match_pct", pa.float32()),
        pa.field("backend", pa.string()),
        pa.field("tokens", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


def open_or_create(db, dim: int, backend: str, kind: str = "functions"):
    name = table_name(backend, kind)
    if has_table(db, name):
        return db.open_table(name)
    return db.create_table(name, schema=schema(dim))


def fetch_project(table, project: str) -> list[dict]:
    n = table.count_rows(f"project = '{project}'")
    if n == 0:
        return []
    return (table.search().where(f"project = '{project}'", prefilter=True)
            .limit(n).to_list())


def delete_ids(table, ids: list[str], chunk: int = 500) -> None:
    for i in range(0, len(ids), chunk):
        quoted = ", ".join(f"'{x}'" for x in ids[i : i + chunk])
        table.delete(f"id IN ({quoted})")
