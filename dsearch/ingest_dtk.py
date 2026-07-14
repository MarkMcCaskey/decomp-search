"""Ingest adapter for dtk-based GameCube/Wii decomp projects.

Reads target objects from `<root>/build/<VERSION>/obj/**/*.o` via
powerpc-eabi-objdump, and match percentages from a decomp.dev progress
report (local JSON file or the `?mode=report` URL).
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Iterator

from .normalize import Function, Insn

_SYM = re.compile(r"^([0-9a-f]+) <(.+)>:")
_INSN = re.compile(
    r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} ){4}\s*([a-z0-9_.+-]+)\s*(.*?)\s*$"
)
_RELOC = re.compile(r"^\s*[0-9a-f]+:\s+(R_PPC_\S+)\s+(\S+)")


def find_objdump(project_root: Path) -> str:
    cand = project_root / "build" / "binutils" / "powerpc-eabi-objdump"
    if cand.exists():
        return str(cand)
    return "powerpc-eabi-objdump"  # hope it's on PATH


def load_report(source: str) -> dict[str, tuple[float, str]]:
    """-> {fn_name: (fuzzy_match_percent, unit_name)}"""
    if source.startswith("http"):
        with urllib.request.urlopen(source) as r:
            rep = json.load(r)
    else:
        rep = json.load(open(source))
    out: dict[str, tuple[float, str]] = {}
    for u in rep.get("units", []):
        for f in u.get("functions", []):
            out[f["name"]] = (float(f.get("fuzzy_match_percent", 0.0)),
                              u.get("name", "?"))
    return out


def parse_object(objdump: str, obj_path: Path, unit: str) -> Iterator[Function]:
    try:
        out = subprocess.run([objdump, "-dr", str(obj_path)],
                             capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return
    fn: Function | None = None
    for line in out.splitlines():
        m = _SYM.match(line)
        if m:
            if fn is not None and fn.insns:
                yield fn
            fn = Function(name=m.group(2), unit=unit)
            continue
        if fn is None:
            continue
        mr = _RELOC.match(line)
        if mr and fn.insns:
            fn.insns[-1].reloc = mr.group(1).removeprefix("R_PPC_")
            continue
        mi = _INSN.match(line)
        if mi:
            fn.insns.append(Insn(addr=int(mi.group(1), 16),
                                 mnemonic=mi.group(2),
                                 operands=mi.group(3)))
    if fn is not None and fn.insns:
        yield fn


def iter_units(project_root: Path, version: str) -> Iterator[tuple[Path, str]]:
    obj_root = project_root / "build" / version / "obj"
    if not obj_root.is_dir():
        raise FileNotFoundError(f"no object dir at {obj_root}")
    for obj in sorted(obj_root.rglob("*.o")):
        yield obj, str(obj.relative_to(obj_root))


def find_source_file(project_root: Path, unit: str) -> str | None:
    """Best-effort map from unit name (e.g. melee/mp/mplib.o) to a source path."""
    stem = unit.removesuffix(".o")
    for base in ("src", "source", ""):
        for ext in (".c", ".cpp"):
            p = project_root / base / (stem + ext)
            if p.exists():
                return str(p.relative_to(project_root))
    return None
