"""Incremental sync planning: diff desired records against stored rows.

The unit of change is the token text — it is exactly the embedding input,
so equal tokens means the stored vector is still valid. Metadata-only
changes (match %, src path) reuse the vector; only new or changed
functions are re-embedded. This also makes an interrupted ingest
resumable: rows already written are classified unchanged on the rerun.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SyncPlan:
    delete_ids: list[str] = field(default_factory=list)  # stale + changed
    rewrite: list[dict] = field(default_factory=list)  # meta changed, vector reused
    embed: list[dict] = field(default_factory=list)  # new or tokens changed
    unchanged: int = 0


def _meta_equal(prev: dict, rec: dict) -> bool:
    if abs(prev["match_pct"] - rec["match_pct"]) > 1e-3:
        return False
    return all(prev[f] == rec[f]
               for f in ("name", "unit", "src_path", "n_insns"))


def plan_sync(existing: list[dict], desired: list[dict],
              full: bool = False) -> SyncPlan:
    plan = SyncPlan()
    old = {r["id"]: r for r in existing}
    seen: set[str] = set()
    for rec in desired:
        if rec["id"] in seen:  # duplicate local symbol in one unit; keep first
            continue
        seen.add(rec["id"])
        prev = old.get(rec["id"])
        if full or prev is None or prev["tokens"] != rec["tokens"]:
            if prev is not None:
                plan.delete_ids.append(rec["id"])
            plan.embed.append(rec)
        elif not _meta_equal(prev, rec):
            plan.delete_ids.append(rec["id"])
            plan.rewrite.append({**rec, "vector": prev["vector"]})
        else:
            plan.unchanged += 1
    plan.delete_ids.extend(i for i in old if i not in seen)
    return plan
