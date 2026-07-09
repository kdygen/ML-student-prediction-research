"""Experiment 005 — additive demographic feature families on top of Baseline v4.

Arms (every arm = FULL official v4 35-feature set + additions; nothing removed):
  A0_v4            : reference (official v4)
  A1_all_demo      : + all 20 demographic columns
  A2_academic      : + studied_credits, num_of_prev_attempts
  A3_socioeconomic : + imd (decile+missing), region one-hots
  A4_personal      : + gender, age_band, disability

Official protocol per arm x cutoff, official XGB hyperparameters (identical across arms):
  headline GSS(seed 42) with per-class report + confusion;
  repeated grouped splits seeds 0-4 (acc, macro-F1, Withdrawn recall, at-risk AUC);
  GroupKFold(5) (acc, macro-F1).
RF (official params) additionally run for A0 and A1 (headline + repeats) as robustness.
Splits depend only on groups -> identical row indices across arms => paired comparisons.
No target-dependent feature selection anywhere: arms are fixed a priori.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score, recall_score)

REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
SP = os.environ["SP"]
CUTS = [14, 30, 60, 90, 140]
SEEDS = [0, 1, 2, 3, 4]
KEY = ["id_student", "code_module", "code_presentation"]

demo = pd.read_parquet(f"{SP}/exp005_demographics.parquet")
GROUPS = json.load(open(f"{SP}/exp005_groups.json"))
DEMO_ALL = GROUPS["academic"] + GROUPS["socioeconomic"] + GROUPS["personal"]
ARMS = {
    "A0_v4": [],
    "A1_all_demo": DEMO_ALL,
    "A2_academic": GROUPS["academic"],
    "A3_socioeconomic": GROUPS["socioeconomic"],
    "A4_personal": GROUPS["personal"],
}

def xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, objective="multi:softmax",
                         num_class=4, eval_metric="mlogloss", n_jobs=-1)
def rf():
    return RandomForestClassifier(random_state=42, class_weight="balanced",
                                  n_estimators=300, n_jobs=-1)

def eval_full(model, Xtr, ytr, Xte, yte, proba=True):
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    out = {"accuracy": float(accuracy_score(yte, p)),
           "macro_f1": float(f1_score(yte, p, average="macro")),
           "withdrawn_recall": float(recall_score(yte, p, labels=[0], average="macro",
                                                  zero_division=0))}
    if proba:
        P = model.predict_proba(Xte)
        out["atrisk_auc"] = float(roc_auc_score((yte <= 1).astype(int), P[:, 0] + P[:, 1]))
    return out, p, model

results = {"meta": {
    "arms": {k: v for k, v in ARMS.items()},
    "protocol": "official v4: GSS(seed42) headline + seeds 0-4 repeats + GroupKFold(5); "
                "official XGB hyperparameters identical across arms; RF for A0/A1",
    "note": "splits depend only on groups -> identical across arms (paired)",
}, "cutoffs": {}}

t00 = time.time()
for c in CUTS:
    t0 = time.time()
    man = json.load(open(f"{REPO}/data/processed/p4/c{c:03d}/manifest.json"))
    v4f = man["active_features_v4"]
    df = pd.read_parquet(f"{REPO}/data/processed/p4/c{c:03d}/mlDataV4.parquet")
    df = df.merge(demo, on=KEY, how="left", validate="one_to_one")
    assert df[DEMO_ALL].isna().sum().sum() == 0
    y, g = df["target_multi"].values, df["id_student"].values

    tr42, te42 = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                      .split(df, y, groups=g))
    assert len(set(g[tr42]) & set(g[te42])) == 0
    seed_splits = {}
    for s in SEEDS:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
                    .split(df, y, groups=g))
        assert len(set(g[a]) & set(g[b])) == 0
        seed_splits[s] = (a, b)
    gkf_splits = list(GroupKFold(n_splits=5).split(df, y, groups=g))
    for a, b in gkf_splits:
        assert len(set(g[a]) & set(g[b])) == 0

    cres = {}
    for arm, extra in ARMS.items():
        F = v4f + extra
        X = df[F]
        # headline (XGB) with per-class + confusion
        hl, p, mdl = eval_full(xgb(), X.iloc[tr42], y[tr42], X.iloc[te42], y[te42])
        rep = classification_report(y[te42], p, output_dict=True, zero_division=0,
                                    target_names=["Withdrawn", "Fail", "Pass", "Distinction"])
        hl["per_class"] = {k: {kk: float(vv) for kk, vv in v.items()}
                           for k, v in rep.items() if isinstance(v, dict)}
        hl["confusion"] = confusion_matrix(y[te42], p).tolist()
        if extra:  # importance rank of added features (gain)
            imp = pd.Series(mdl.feature_importances_, index=F).sort_values(ascending=False)
            hl["added_importance_ranks"] = {f: int(list(imp.index).index(f) + 1)
                                            for f in extra}
            hl["top10_features"] = {f: float(v) for f, v in imp.head(10).items()}
        arm_res = {"xgb_headline": hl, "xgb_repeats": [], "xgb_gkf5": []}
        for s in SEEDS:
            a, b = seed_splits[s]
            r, _, _ = eval_full(xgb(), X.iloc[a], y[a], X.iloc[b], y[b])
            r["seed"] = s
            arm_res["xgb_repeats"].append(r)
        for i, (a, b) in enumerate(gkf_splits):
            r, _, _ = eval_full(xgb(), X.iloc[a], y[a], X.iloc[b], y[b], proba=False)
            r["fold"] = i
            arm_res["xgb_gkf5"].append(r)
        if arm in ("A0_v4", "A1_all_demo"):
            rhl, _, _ = eval_full(rf(), X.iloc[tr42], y[tr42], X.iloc[te42], y[te42])
            arm_res["rf_headline"] = rhl
            arm_res["rf_repeats"] = []
            for s in SEEDS:
                a, b = seed_splits[s]
                r, _, _ = eval_full(rf(), X.iloc[a], y[a], X.iloc[b], y[b])
                r["seed"] = s
                arm_res["rf_repeats"].append(r)
        cres[arm] = arm_res
        f1m = np.mean([r["macro_f1"] for r in arm_res["xgb_repeats"]])
        print(f"[c{c:03d}] {arm:17s} hl acc={hl['accuracy']:.4f} F1={hl['macro_f1']:.4f} "
              f"| repeats F1={f1m:.4f}", flush=True)
    results["cutoffs"][f"c{c:03d}"] = cres
    print(f"[c{c:03d}] done {time.time()-t0:.0f}s", flush=True)

json.dump(results, open(f"{SP}/exp005_raw.json", "w"), indent=1)
print(f"TOTAL {time.time()-t00:.0f}s")

# quick sanity: A0 headline must equal official
off = json.load(open(f"{REPO}/reports/baseline_v4_results.json"))
for c in CUTS:
    o = off[str(c)]["v4"]["models"]["xgboost"]
    a = results["cutoffs"][f"c{c:03d}"]["A0_v4"]["xgb_headline"]
    d = abs(a["accuracy"] - o["accuracy"]) + abs(a["macro_f1"] - o["macro_f1"])
    print(f"sanity c{c:03d}: A0 vs official diff = {d:.2e}")
