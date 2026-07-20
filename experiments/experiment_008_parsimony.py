"""Experiment 008 — greedy redundancy elimination on V3 (parsimony study).

Method: iterative. Each round finds the highest |r| pair among REMAINING features
(threshold 0.85), evaluates dropping each member, and accepts the better drop iff it does
not cost more than 0.002 in either paired mean macro-F1 or paired mean accuracy.
Comparisons are PAIRED: identical grouped splits (seeds 0-4) for every candidate, so the
per-seed delta removes split variance and can resolve effects well below 0.002.
Pipeline, model, hyperparameters, population unchanged. Nothing else is redesigned.
"""
import json, os, time
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

SP = os.environ["SP"]
CLS = ["Withdrawn", "Fail", "Pass", "Distinction"]
THRESH_R = 0.85
TOL = 0.002
SEEDS = [0, 1, 2, 3, 4]
t00 = time.time()

df = pd.read_parquet(f"{SP}/exp006c_frame_probe.parquet")
meta = json.load(open(f"{SP}/exp006b_meta.json"))
CF0 = [c if c != "completion_ratio_cw" else "completion_ratio_avail" for c in meta["cls_features"]]
V3 = df[~(df["date_unregistration"] <= 0)].copy()
y = V3["target_multi"].values
g = V3["id_student"].values

splits = []
for s in SEEDS:
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(V3, y, groups=g))
    assert len(set(g[a]) & set(g[b])) == 0
    splits.append((a, b))

def xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, objective="multi:softmax",
                         num_class=4, eval_metric="mlogloss", n_jobs=-1)

CACHE = {}
def perseed(feats):
    """per-seed (acc, f1) arrays for a feature set — cached."""
    key = tuple(sorted(feats))
    if key in CACHE: return CACHE[key]
    X = V3[list(feats)]
    accs, f1s = [], []
    for a, b in splits:
        m = xgb(); m.fit(X.iloc[a], y[a]); p = m.predict(X.iloc[b])
        accs.append(accuracy_score(y[b], p)); f1s.append(f1_score(y[b], p, average="macro"))
    CACHE[key] = (np.array(accs), np.array(f1s))
    return CACHE[key]

def pairs_of(feats):
    c = V3[feats].astype(float).corr().abs()
    out = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            v = c.iloc[i, j]
            if not np.isnan(v) and v >= THRESH_R:
                out.append((feats[i], feats[j], float(v)))
    return sorted(out, key=lambda t: -t[2])

cur = list(CF0)
base_acc, base_f1 = perseed(cur)
log = {"start": {"n_features": len(cur), "acc": float(base_acc.mean()),
                 "f1": float(base_f1.mean()), "acc_std": float(base_acc.std()),
                 "f1_std": float(base_f1.std())},
       "rounds": [], "removed": [], "kept_pairs": []}
print(f"START {len(cur)} features: acc {base_acc.mean():.4f} f1 {base_f1.mean():.4f}")
print("initial pairs >=0.85:", [(a, b, round(r, 3)) for a, b, r in pairs_of(cur)], flush=True)

resolved = set()
rnd = 0
while True:
    ps = [p for p in pairs_of(cur) if (p[0], p[1]) not in resolved]
    if not ps: break
    fa, fb, r = ps[0]
    rnd += 1
    cand = {}
    for drop in (fa, fb):
        feats = [c for c in cur if c != drop]
        a_, f_ = perseed(feats)
        cand[drop] = {"acc": float(a_.mean()), "f1": float(f_.mean()),
                      "d_acc": float((a_ - base_acc).mean()), "d_f1": float((f_ - base_f1).mean()),
                      "d_f1_std": float((f_ - base_f1).std())}
        print(f"  [r{rnd}] pair({fa},{fb}) r={r:.3f} | drop {drop:24s} "
              f"dF1={cand[drop]['d_f1']:+.4f} dAcc={cand[drop]['d_acc']:+.4f}", flush=True)
    best = max(cand, key=lambda k: cand[k]["d_f1"])
    ok = (cand[best]["d_f1"] > -TOL) and (cand[best]["d_acc"] > -TOL)
    log["rounds"].append({"round": rnd, "pair": [fa, fb], "r": r,
                          "candidates": cand, "chosen": best, "accepted": bool(ok)})
    if ok:
        cur = [c for c in cur if c != best]
        base_acc, base_f1 = perseed(cur)     # re-baseline to the new set
        log["removed"].append({"feature": best, "pair_with": fb if best == fa else fa,
                               "r": r, "d_f1_at_removal": cand[best]["d_f1"],
                               "d_acc_at_removal": cand[best]["d_acc"]})
        print(f"  -> REMOVE {best} (now {len(cur)} features, "
              f"acc {base_acc.mean():.4f} f1 {base_f1.mean():.4f})", flush=True)
    else:
        resolved.add((fa, fb))
        log["kept_pairs"].append({"pair": [fa, fb], "r": r,
                                  "best_drop_cost_f1": cand[best]["d_f1"]})
        print(f"  -> KEEP BOTH (best drop costs {cand[best]['d_f1']:+.4f} F1)", flush=True)

# ---------- final validation of the reduced set ----------
start_acc, start_f1 = perseed(CF0)
fin_acc, fin_f1 = perseed(cur)
log["final"] = {"n_features": len(cur), "features": cur,
                "acc": float(fin_acc.mean()), "f1": float(fin_f1.mean()),
                "paired_d_acc_vs_start": float((fin_acc - start_acc).mean()),
                "paired_d_f1_vs_start": float((fin_f1 - start_f1).mean())}
cv = []
for name, feats in [("full_42", CF0), ("reduced", cur)]:
    X = V3[feats]
    fold = []
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups=g):
        assert len(set(g[a]) & set(g[b])) == 0
        m = xgb(); m.fit(X.iloc[a], y[a]); p = m.predict(X.iloc[b])
        fold.append((accuracy_score(y[b], p), f1_score(y[b], p, average="macro")))
    fa_, ff_ = np.array([x[0] for x in fold]), np.array([x[1] for x in fold])
    cv.append({"set": name, "n_features": len(feats), "acc": float(fa_.mean()),
               "acc_std": float(fa_.std()), "f1": float(ff_.mean()), "f1_std": float(ff_.std())})
    print(f"[SGKF5 {name:8s}] {len(feats):2d} feats  acc {fa_.mean():.4f}±{fa_.std():.4f}  "
          f"f1 {ff_.mean():.4f}±{ff_.std():.4f}", flush=True)
log["stratified_group_cv"] = cv
# per-class on the reduced set (headline split)
X = V3[cur]; a, b = splits[0]
m = xgb(); m.fit(X.iloc[a], y[a]); p = m.predict(X.iloc[b])
rep = classification_report(y[b], p, output_dict=True, zero_division=0, target_names=CLS)
log["final"]["per_class_f1_reduced"] = {c: float(rep[c]["f1-score"]) for c in CLS}
X = V3[CF0]; m = xgb(); m.fit(X.iloc[a], y[a]); p = m.predict(X.iloc[b])
rep0 = classification_report(y[b], p, output_dict=True, zero_division=0, target_names=CLS)
log["final"]["per_class_f1_full"] = {c: float(rep0[c]["f1-score"]) for c in CLS}
log["remaining_pairs_ge_0.85"] = [(a_, b_, round(r_, 3)) for a_, b_, r_ in pairs_of(cur)]

json.dump(log, open(f"{SP}/exp008_parsimony.json", "w"), indent=1)
print(f"\nremoved {len(log['removed'])}: {[r['feature'] for r in log['removed']]}")
print(f"final {len(cur)} features | TOTAL {time.time()-t00:.0f}s")
