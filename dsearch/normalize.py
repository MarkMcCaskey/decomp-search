"""Normalize disassembled functions into token streams for embedding.

The token stream is designed to preserve the structural signals that matter
for compiler-similarity retrieval (instruction skeleton, operand shapes, branch
direction) while discarding the noise that varies between otherwise-similar functions (register
numbers, addresses, symbol names, literal values).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Insn:
    addr: int
    mnemonic: str
    operands: str
    reloc: str | None = None  # relocation type on this insn, if any


@dataclass
class Function:
    name: str
    unit: str  # translation unit (relative path of the object)
    insns: list[Insn] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.insns)


BRANCH_MNEMONICS = re.compile(
    r"^b(l|c|ne|eq|lt|gt|le|ge|so|ns|dnz|dz|ctr|lr)?[+-]?$|^b(ne|eq|lt|gt|le|ge)(lr|ctr)?[+-]?$"
)
_REG = re.compile(r"\b(r\d{1,2}|f\d{1,2}|cr\d|p\d{1,2}|qr\d)\b")
_HEXNUM = re.compile(r"-?0x[0-9a-fA-F]+|-?\b\d+\b")
_TARGET = re.compile(r"^(0x)?[0-9a-f]+$")


def _operand_shape(op: str) -> str:
    """Collapse one operand to a shape token: registers -> r/f/cr,
    immediates -> #, displacement forms -> #(r)."""
    op = op.strip()
    if not op:
        return ""
    m = re.fullmatch(r"(-?(?:0x)?[0-9a-fA-F]+)\((r\d{1,2}|sp|rtoc)\)", op)
    if m:
        return "#(r)"
    if re.fullmatch(r"r\d{1,2}|sp|rtoc", op):
        return "r"
    if re.fullmatch(r"f\d{1,2}", op):
        return "f"
    if re.fullmatch(r"cr\d", op):
        return "cr"
    if re.fullmatch(r"qr\d", op):
        return "q"
    if _TARGET.fullmatch(op):
        return "#"
    # symbol reference / reloc-ish operand
    return "@"


def insn_tokens(insn: Insn, fn_start: int) -> str:
    """One token per instruction: mnemonic plus operand shapes.

    Branches encode direction (fwd/back) instead of their target, which
    preserves loop geometry (e.g. `b(back)` = a backedge) — the property
    exact twin-scans key on.
    """
    mnem = insn.mnemonic
    ops = insn.operands

    if mnem == "b" or (mnem.startswith("b") and BRANCH_MNEMONICS.match(mnem)):
        if mnem in ("bl", "blr", "bctr", "bctrl", "blrl"):
            return mnem
        # find a hex target among operands
        parts = [p.strip() for p in ops.split(",")]
        direction = ""
        for p in parts:
            m = re.match(r"(?:0x)?([0-9a-f]+)\b", p)
            if m and _TARGET.fullmatch(p.split()[0]):
                tgt = int(m.group(1), 16)
                direction = "back" if tgt <= insn.addr else "fwd"
                break
        cond = ",".join(_operand_shape(p) for p in parts if _operand_shape(p) in ("cr",))
        return f"{mnem}({direction})" if direction else mnem

    if not ops:
        return mnem
    shapes = ",".join(s for s in (_operand_shape(p) for p in ops.split(",")) if s)
    tok = f"{mnem}({shapes})"
    if insn.reloc:
        tok += f"[{insn.reloc}]"
    return tok


def function_tokens(fn: Function) -> list[str]:
    if not fn.insns:
        return []
    start = fn.insns[0].addr
    return [insn_tokens(i, start) for i in fn.insns]


def token_text(fn: Function) -> str:
    """The document string handed to embedding backends."""
    toks = function_tokens(fn)
    return f"ppc {len(toks)}\n" + " ".join(toks)
