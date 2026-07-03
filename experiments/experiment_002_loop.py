"""
Experiment 002 — greedy feature-group loop at cutoff 30.

Discipline: the official seed-42 grouped TEST is never touched. Every evaluation
is GroupKFold(3) inside the seed-42 TRAIN, with the tau (class-prior adjustment)
tuned on the same inner folds. Model fixed: baseline-v3 XGBoost hyperparameters.

Pre-registered acceptance rule (documented before running):
  accept the round's best candidate group iff
    (a) inner mean macro-F1 improves >= +0.002 over the current feature set, OR
    (b) macro-F1 within -0.001 AND Withdrawn recall improves >= +0.02.
  Stop when no candidate passes (plateau) or after 5 rounds.
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "experiments")
from feature_generation_002 import FEATURE_GROUPS
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score
from xgboost import XGBClassifier

SP = os.environ["SP"]; OUT = os.environ["OUT"]
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 42

man = json.load(open("data/processed/p3/c030/manifest.json"))
BASE_FEATS = man["active_features_v3"]
df = pd.read_parquet(f"{SP}/aug_c030.parquet")
y = df["target_multi"]; g = df["id_student"].values

tr, _te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
               .split(df, y, g))
dtr = df.iloc[tr]; ytr = y.iloc[tr]; gtr = g[tr]
folds = list(GroupKFold(n_splits=3).split(dtr, ytr, gtr))
print(f"[loop] train {dtr.shape}, folds ready", file=sys.stderr)

def xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                         objective="multi:softprob", num_class=4,
                         eval_metric="mlogloss", n_jobs=-1)

def evaluate(feats):
    scores = {t: {"f1": [], "acc": [], "wrec": []} for t in TAUS}
    for a, b in folds:
        Xa, ya = dtr.iloc[a][feats], ytr.iloc[a]
        Xb, yb = dtr.iloc[b][feats], ytr.iloc[b]
        m = xgb(); m.fit(Xa, ya)
        proba = m.predict_proba(Xb)
        pri = np.bincount(ya, minlength=4) / len(ya)
        for t in TAUS:
            pred = np.argmax(proba * (1.0 / pri) ** t, axis=1)
            scores[t]["f1"].append(f1_score(yb, pred, average="macro"))
            scores[t]["acc"].append(accuracy_score(yb, pred))
            scores[t]["wrec"].append(recall_score(yb, pred, labels=[0], average=None)[0])
    agg = {t: {k: float(np.mean(v)) for k, v in s.items()} for t, s in scores.items()}
    bt = max(TAUS, key=lambda t: agg[t]["f1"])
    return dict(tau=bt, f1=agg[bt]["f1"], acc=agg[bt]["acc"], wrec=agg[bt]["wrec"],
                tau_table={str(t): agg[t] for t in TAUS})

log = {"acceptance_rule": "dF1>=+0.002, or dF1>=-0.001 and dWrec>=+0.02; best group per round; stop on no-pass or 5 rounds",
       "base_features": BASE_FEATS, "rounds": []}

base_eval = evaluate(BASE_FEATS)
log["base_eval"] = base_eval
print(f"[loop] BASE(19) f1={base_eval['f1']:.4f} acc={base_eval['acc']:.4f} "
      f"wrec={base_eval['wrec']:.4f} tau={base_eval['tau']}", file=sys.stderr)

accepted, rejected = [], {}
current = list(BASE_FEATS); cur = base_eval
remaining = dict(FEATURE_GROUPS)

for rnd in range(1, 6):
    entries = {}
    for name, cols in remaining.items():
        r = evaluate(current + cols)
        d = r["f1"] - cur["f1"]; dw = r["wrec"] - cur["wrec"]
        entries[name] = dict(result=r, dF1=round(d, 4), dWrec=round(dw, 4))
        print(f"[loop] r{rnd} {name:18s} f1={r['f1']:.4f} (d{d:+.4f}) "
              f"wrec={r['wrec']:.4f} (d{dw:+.4f}) tau={r['tau']}", file=sys.stderr)
    def passes(e):
        return e["dF1"] >= 0.002 or (e["dF1"] >= -0.001 and e["dWrec"] >= 0.02)
    ok = {k: v for k, v in entries.items() if passes(v)}
    log["rounds"].append({"round": rnd, "evaluated": {k: {"dF1": v["dF1"], "dWrec": v["dWrec"],
                          "f1": v["result"]["f1"], "wrec": v["result"]["wrec"],
                          "tau": v["result"]["tau"]} for k, v in entries.items()},
                          "passed": list(ok)})
    if not ok:
        print(f"[loop] r{rnd}: no candidate passes — PLATEAU", file=sys.stderr)
        break
    best = max(ok, key=lambda k: ok[k]["dF1"])
    accepted.append(best)
    current = current + remaining.pop(best)
    cur = entries[best]["result"]
    log["rounds"][-1]["accepted"] = best
    log["rounds"][-1]["new_eval"] = cur
    print(f"[loop] r{rnd}: ACCEPT {best} -> f1={cur['f1']:.4f} wrec={cur['wrec']:.4f} "
          f"({len(current)} feats)", file=sys.stderr)

for name, e in (log["rounds"][-1]["evaluated"] if log["rounds"] else {}).items():
    if name not in accepted:
        rejected[name] = e
log["accepted_groups"] = accepted
log["rejected_last_round"] = rejected
log["final_features"] = current
log["final_inner_eval"] = cur
json.dump(log, open(OUT, "w"), indent=2)
print(f"[loop] DONE accepted={accepted} nfeat={len(current)} "
      f"f1={cur['f1']:.4f} (base {base_eval['f1']:.4f})", file=sys.stderr)
print("WROTE", OUT, file=sys.stderr)
