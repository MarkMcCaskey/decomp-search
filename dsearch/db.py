"""LanceDB storage layer."""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "index.lancedb"


def table_name(backend: str) -> str:
    return f"functions_{backend}"


def connect(db_path: str | Path = DEFAULT_DB):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


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


def open_or_create(db, dim: int, backend: str):
    name = table_name(backend)
    if name in db.table_names():
        return db.open_table(name)
    return db.create_table(name, schema=schema(dim))


def replace_project(table, project: str) -> None:
    table.delete(f"project = '{project}'")
