"""Rank unmatched/low-% melee functions by how close their best matched
cross-unit neighbor is — the most 'solvable-by-donor' functions first."""

import lancedb

db = lancedb.connect("data/index.lancedb")
t = db.open_table("functions_local")
rows = (t.search().where("project = 'melee' AND match_pct >= 0 "
                         "AND match_pct < 95", prefilter=True)
        .limit(20000).to_list())
print(f"{len(rows)} melee fns below 95%", flush=True)

out = []
for r in rows:
    if r["n_insns"] < 16:
        continue
    hits = (t.search(r["vector"]).metric("cosine")
            .where("match_pct >= 99.5", prefilter=True)
            .limit(20).to_list())
    best = None
    for h in hits:
        if h["id"] == r["id"] or h["unit"] == r["unit"]:
            continue
        best = h
        break
    if best:
        out.append((1 - best["_distance"], r, best))

out.sort(key=lambda x: -x[0])
for sim, r, b in out[:25]:
    print(f"sim {sim:.3f}  {r['name']:<42} {r['match_pct']:6.2f}% "
          f"{r['n_insns']:4d}i <- {b['name']} ({b['match_pct']:.2f}%, "
          f"{b['n_insns']}i, {b['unit'].split('/')[-1]})", flush=True)
