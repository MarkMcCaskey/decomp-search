"""Throughput benchmark for embed_local on real corpus docs."""

import sys
import time

import lancedb

from dsearch import embed as E

db = lancedb.connect("data/index.lancedb")
t = db.open_table("functions_hashed")
arr = t.to_arrow()
tokens = [x.as_py() for x in arr["tokens"]]
names = [x.as_py() for x in arr["name"]]
projects = [x.as_py() for x in arr["project"]]

melee = [(n, d) for n, d, p in zip(names, tokens, projects) if p == "melee"]
sample = melee[:: max(1, len(melee) // 512)][:512]
docs = [d for _, d in sample]
print(f"sample: {len(docs)} docs from {len(melee)} melee fns", flush=True)

t0 = time.time()
vecs = E.embed_local(docs[:8])  # warm-up incl. model load
print(f"model load + warm-up: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
vecs = E.embed_local(docs)
dt = time.time() - t0
rate = len(docs) / dt
print(f"packed: {len(docs)} docs in {dt:.1f}s = {rate:.1f} docs/s "
      f"-> {len(melee)/rate/60:.1f} min for all melee", flush=True)

bad = sum(1 for v in vecs if not v or any(x != x for x in v))
norms = [sum(x * x for x in v) ** 0.5 for v in vecs[:20]]
print(f"NaN/empty: {bad}; norms[:5]: {[f'{n:.3f}' for n in norms[:5]]}",
      flush=True)

# quality spot-check: known twin pair should beat random pairs
byname = {n: d for n, d in melee}
pair = ["mpRightWallGetTop", "mpFloorGetRight"]
rand = [melee[100][0], melee[4000][0]]
qv = E.embed_local([byname[pair[0]], byname[pair[1]],
                    byname[rand[0]], byname[rand[1]]])
cos = lambda a, b: sum(x * y for x, y in zip(a, b))
print(f"cos({pair[0]}, {pair[1]}) = {cos(qv[0], qv[1]):.3f}", flush=True)
print(f"cos({pair[0]}, {rand[0]}) = {cos(qv[0], qv[2]):.3f}", flush=True)
print(f"cos({pair[0]}, {rand[1]}) = {cos(qv[0], qv[3]):.3f}", flush=True)

# old-style fixed-16 slicing for comparison (on a 128-doc subset)
sub = docs[:128]
t0 = time.time()
for i in range(0, len(sub), 16):
    E._local_model.encode_document(
        [d[:8000] for d in sub[i : i + 16]], batch_size=16,
        normalize_embeddings=True, show_progress_bar=False)
dt_old = time.time() - t0
print(f"old fixed-16: {len(sub)} docs in {dt_old:.1f}s = "
      f"{len(sub)/dt_old:.1f} docs/s "
      f"-> {len(melee)/(len(sub)/dt_old)/60:.1f} min for all melee", flush=True)
