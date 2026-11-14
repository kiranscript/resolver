"""Level 2: clean training set + merge/repo-grouped dev set + leakage report.

  .venv/bin/python scripts/build_level2.py
Inputs : data/train3424.jsonl, data/test806.jsonl
Outputs: data/train_clean.jsonl, data/dev300.jsonl, data/dev_sym.jsonl, results/level2.json

The dataset has no repository id, but every conflict label ends in the 40-hex SHA of the merged commit
(`>>>>>>> <sha>`), and many hunks carry a `package` line or a path label. Groups = connected components of
hunks that share a merge SHA or a package root. Whole groups are held out as dev.
"""
import collections, hashlib, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scoring import (score, summarize, trivial_resolve, swap_example, can_swap,  # noqa: E402
                     extract_code_block)

SEED, DEV_TARGET, MAX_DEV_GROUP, REGION_CAP, MERGE_CAP, JACCARD = 20261113, 300, 30, 3, 120, 0.6
load = lambda p: [json.loads(l) for l in open(ROOT / p)]
train, test = load("data/train3424.jsonl"), load("data/test806.jsonl")
h = lambda s: hashlib.sha1(s.encode()).hexdigest()
snippet = lambda q: q[q.index("```java\n") + 8:q.rindex("\n```")]
TLD = {"com", "org", "net", "io", "de", "fr", "nl", "edu", "gov", "me", "info", "eu", "co", "uk", "ru", "cn", "jp"}
GENERIC = {"java", "javax", "android", "junit", "static"}


def root_of(dotted):
    p = [x for x in dotted.strip(";").split(".") if x]
    if not p or p[0] in GENERIC:
        return None
    return ".".join(p[:3] if p[0] in TLD else p[:2]) if len(p) >= 2 else None


def features(r):
    s = snippet(r["question"])
    marks = [l for l in s.split("\n") if l.startswith(("<<<<<<<", ">>>>>>>"))]
    sha = next((m for l in marks for m in re.findall(r"\b[0-9a-f]{40}\b", l)), None)
    pk = re.findall(r"^package\s+([\w.]+)", s, re.M)
    roots = {root_of(p) for p in pk}
    for l in marks:  # path labels: HEAD:src/main/java/tools/jackson/core/X.java
        m = re.search(r":(?:.*/)?(?:java|src)/((?:[\w]+/)+)", l)
        if m:
            roots.add(root_of(m.group(1).strip("/").replace("/", ".")))
    return sha, {x for x in roots if x}


def region(q):
    out, inside = [], False
    for l in snippet(q).split("\n"):
        inside = inside or l.startswith("<<<<<<<")
        if inside:
            out.append(re.sub(r"\s+", " ", l.strip()))
        if l.startswith(">>>>>>>"):
            inside = False
    return "\n".join(out)


def shingles(text, k=8):
    t = re.findall(r"\w+|[^\w\s]", text)
    return {h(" ".join(t[i:i + k]))[:12] for i in range(max(1, len(t) - k + 1))}


for r in train + test:
    r["sha"], r["roots"] = features(r)
    r["region"] = region(r["question"])

rep = {"train_in": len(train)}
# ---- 1. duplicates ---------------------------------------------------------
seen, pool = set(), []
for r in train:
    if h(r["question"]) in seen:
        continue
    seen.add(h(r["question"])); pool.append(r)
rep["after_exact_dedup"] = len(pool)
cnt = collections.Counter(); capped = []
for r in pool:  # same conflict region repeated (licence headers, vendored code): keep at most REGION_CAP
    k = h(r["region"])
    cnt[k] += 1
    if cnt[k] <= REGION_CAP:
        capped.append(r)
rep["region_repeat_max"] = max(cnt.values())
rep["after_region_cap"] = len(capped); pool = capped

# ---- 2. train<->test leakage: exact merge SHA, package root, near-duplicate regions ---
test_sha = {r["sha"] for r in test if r["sha"]}
test_roots = set().union(*[r["roots"] for r in test])
test_sh = collections.defaultdict(set)
for i, r in enumerate(test):
    for s in shingles(r["region"]):
        test_sh[s].add(i)
def near_test(r):
    sh = shingles(r["region"]); hits = collections.Counter(i for s in sh for i in test_sh.get(s, ()))
    return any(c / (len(sh) + len(shingles(test[i]["region"])) - c) >= JACCARD for i, c in hits.items())
leak = {"sha": 0, "root": 0, "near_dup": 0}; kept = []
for r in pool:
    if r["sha"] in test_sha: leak["sha"] += 1; continue
    if r["roots"] & test_roots: leak["root"] += 1; continue
    if near_test(r): leak["near_dup"] += 1; continue
    kept.append(r)
rep["test_leakage_removed"] = leak; rep["after_test_guard"] = len(kept); pool = kept

# ---- 3. groups: union by merge SHA and package root -------------------------
parent = list(range(len(pool)))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b): parent[find(a)] = find(b)
by_key = {}
for i, r in enumerate(pool):
    for key in ([("sha", r["sha"])] if r["sha"] else []) + [("root", x) for x in r["roots"]]:
        if key in by_key: union(i, by_key[key])
        else: by_key[key] = i
groups = collections.defaultdict(list)
for i in range(len(pool)): groups[find(i)].append(i)
sizes = sorted((len(g) for g in groups.values()), reverse=True)
rep["groups"] = len(groups); rep["group_sizes_top10"] = sizes[:10]
rep["groups_le_30"] = sum(s <= MAX_DEV_GROUP for s in sizes)

# ---- 4. dev = whole groups (each <= MAX_DEV_GROUP hunks); train = rest, per-merge cap ----
order = sorted((g for g in groups.values() if len(g) <= MAX_DEV_GROUP), key=lambda g: h(f"{SEED}:{g[0]}"))
dev_idx = []
for g in order:
    if len(dev_idx) >= DEV_TARGET: break
    if len(dev_idx) + len(g) > DEV_TARGET + 15: continue
    dev_idx += g
dev_set = set(dev_idx)
dev = [pool[i] for i in sorted(dev_set)]
rest = [pool[i] for i in range(len(pool)) if i not in dev_set]
rng = random.Random(SEED); per = collections.defaultdict(list)
for r in rest: per[r["sha"] or id(r)].append(r)
train_clean, merge_capped = [], 0
for k, v in per.items():
    if len(v) > MERGE_CAP: merge_capped += len(v) - MERGE_CAP; v = rng.sample(v, MERGE_CAP)
    train_clean += v
train_clean.sort(key=lambda r: r["id"])
rep.update(merge_cap=MERGE_CAP, dropped_by_merge_cap=merge_capped, dev=len(dev), train_clean=len(train_clean))

# ---- 5. independent overlap checks (must all be 0) ---------------------------
sh = lambda rows: {r["sha"] for r in rows if r["sha"]}
rt = lambda rows: set().union(*[r["roots"] for r in rows]) if rows else set()
rep["overlap"] = {
    "dev_vs_train_sha": len(sh(dev) & sh(train_clean)), "dev_vs_train_root": len(rt(dev) & rt(train_clean)),
    "train_vs_test_sha": len(sh(train_clean) & test_sha), "dev_vs_test_sha": len(sh(dev) & test_sha),
    "train_vs_test_root": len(rt(train_clean) & test_roots), "dev_vs_test_root": len(rt(dev) & test_roots),
    "train_vs_test_exact_region": len({h(r["region"]) for r in train_clean} & {h(r["region"]) for r in test}),
}
rep["distinct_merge_sha"] = {"train_clean": len(sh(train_clean)), "dev": len(sh(dev)), "test": len(test_sha)}
rep["largest_merge_share_of_dev_pct"] = 100 * max(collections.Counter(r["sha"] for r in dev).values()) / len(dev)

# ---- 6. orientation-neutral dev ------------------------------------------------
dev_sym = [{k: r[k] for k in ("id", "question", "answer")} for r in dev]
n_conc = 0
for r in dev:
    sw = swap_example(r["question"], r["answer"])
    if sw:
        dev_sym.append({"id": f"{r['id']}s", "question": sw[0], "answer": sw[1]})
        n_conc += sw[1] != r["answer"]
rep["dev_sym"] = len(dev_sym); rep["dev_swapped_answer_reordered"] = n_conc

# ---- 7. representativeness -------------------------------------------------------
def triv(rows):
    return {s: round(summarize([score(trivial_resolve(r["question"], s), r["answer"]) for r in rows])["exact_pct"], 1)
            for s in ("left", "right", "both")}
def none_rate(rows):
    ok = lambda r: any(extract_code_block(trivial_resolve(r["question"], s)) == r["answer"].strip() for s in ("left", "right", "both", "both_reversed"))
    return round(100 * sum(not ok(r) for r in rows) / len(rows), 1)
def sidewins(rows, by):  # orientation per merge group, for groups >= 20 hunks
    g = collections.defaultdict(list)
    for r in rows: g[r[by]].append(r)
    out = []
    for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))[:6]:
        if len(v) >= 20 and k:
            out.append((len(v), round(triv(v)["left"]), round(triv(v)["right"])))
    return out
rep["trivial_exact_pct"] = {"train_clean": triv(train_clean), "dev": triv(dev), "dev_sym": triv(dev_sym), "test": triv(test)}
rep["pct_solved_by_no_trivial_strategy"] = {"train_clean": none_rate(train_clean), "dev": none_rate(dev), "test": none_rate(test)}
rep["largest_merges_(n,left%,right%)"] = {"train_clean": sidewins(train_clean, "sha"), "test": sidewins(test, "sha")}
has_path = lambda r: bool(re.search(r"^<<<<<<< HEAD:", r["question"], re.M))
rep["path_label_pct"] = {n: round(100 * sum(map(has_path, rows)) / len(rows), 1) for n, rows in (("train_clean", train_clean), ("test", test))}
rep["gate_min_clean"] = 2000; rep["gate_passed"] = len(train_clean) >= 2000

def dump(rows, p):
    with open(ROOT / p, "w") as f:
        for r in rows: f.write(json.dumps({k: r[k] for k in ("id", "question", "answer")}) + "\n")
dump(train_clean, "data/train_clean.jsonl"); dump(dev, "data/dev300.jsonl"); dump(dev_sym, "data/dev_sym.jsonl")
(ROOT / "results/level2.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
