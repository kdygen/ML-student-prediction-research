"""Experiment 003 — feature contribution to each intervention index.

Permutation importance, model-agnostic, computed on the headline grouped test set:
for each feature, permute its test column (5 repeats, fixed rng), recompute the
XGB probabilities, and measure the drop in ROC-AUC of each index against its truth:
  WRI vs withdrawn, FRI vs fail, ASI vs atrisk.
The model is the official Baseline v4 XGB, untouched; only test inputs are permuted.
"""
import json, os, time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
SP = os.environ["SP"]
CUTS = [30, 90]
N_REP = 5
V4_NEW16 = {'rank_clicks','rank_wa','rank_active_days','mean_submit_lead','min_submit_lead',
 'late_submissions','submitted_count','first_submit_day','n_assess_types_submitted',
 'w1_clicks','w2_clicks','w3_clicks','w4_clicks','precourse_clicks','days_since_last','decay_clicks'}

out = {"method": "permutation importance on headline grouped test (5 repeats, rng seed 0); "
                 "metric = ROC-AUC drop of index vs its truth; official v4 XGB",
       "cutoffs": {}}

for c in CUTS:
    t0 = time.time()
    man = json.load(open(f"{REPO}/data/processed/p4/c{c:03d}/manifest.json"))
    feats = man["active_features_v4"]
    df = pd.read_parquet(f"{REPO}/data/processed/p4/c{c:03d}/mlDataV4.parquet")
    X, y, g = df[feats], df["target_multi"].values, df["id_student"].values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                  .split(X, y, groups=g))
    assert len(set(g[tr]) & set(g[te])) == 0
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, random_state=42, objective="multi:softmax",
                      num_class=4, eval_metric="mlogloss", n_jobs=-1)
    m.fit(X.iloc[tr], y[tr])
    Xte = X.iloc[te].reset_index(drop=True)
    y_te = y[te]
    truths = {"ASI": (y_te <= 1).astype(int),
              "WRI": (y_te == 0).astype(int),
              "FRI": (y_te == 1).astype(int)}

    def aucs(P):
        idx = {"WRI": P[:, 0], "FRI": P[:, 1], "ASI": P[:, 0] + P[:, 1]}
        return {k: roc_auc_score(truths[k], v) for k, v in idx.items()}

    base = aucs(m.predict_proba(Xte))
    rng = np.random.default_rng(0)
    imp = {k: {} for k in truths}
    for f in feats:
        drops = {k: [] for k in truths}
        for r in range(N_REP):
            Xp = Xte.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            a = aucs(m.predict_proba(Xp))
            for k in truths:
                drops[k].append(base[k] - a[k])
        for k in truths:
            imp[k][f] = {"mean_auc_drop": float(np.mean(drops[k])),
                         "std": float(np.std(drops[k]))}
    out["cutoffs"][f"c{c:03d}"] = {
        "base_auc": {k: float(v) for k, v in base.items()},
        "importance": imp}
    for k in truths:
        top = sorted(imp[k].items(), key=lambda kv: -kv[1]["mean_auc_drop"])[:8]
        tag = lambda f: "*" if f in V4_NEW16 else " "
        print(f"[c{c:03d}] {k}: " + ", ".join(f"{tag(f)}{f}={d['mean_auc_drop']:.4f}"
                                              for f, d in top), flush=True)
    print(f"[c{c:03d}] {time.time()-t0:.0f}s", flush=True)

json.dump(out, open(f"{SP}/exp003_perm.json", "w"), indent=1)
print("saved ->", f"{SP}/exp003_perm.json")
