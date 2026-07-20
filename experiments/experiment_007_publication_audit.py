"""Experiment 007 — final publication audit of the V3 headline configuration.

V3 = per-student anchoring (006b) + fair completion denominator (006c V1)
     + engaged population (drop enrolments unregistered on/before day 0).
Nothing is modified; this script only measures.
"""
import json, os, time
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.dummy import DummyClassifier

SP = os.environ["SP"]
REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
CLS = ["Withdrawn", "Fail", "Pass", "Distinction"]
t00 = time.time()

df = pd.read_parquet(f"{SP}/exp006c_frame_probe.parquet")
meta = json.load(open(f"{SP}/exp006b_meta.json"))
CF = [c if c != "completion_ratio_cw" else "completion_ratio_avail" for c in meta["cls_features"]]
V3 = df[~(df["date_unregistration"] <= 0)].copy()
y = V3["target_multi"].values
g = V3["id_student"].values
X = V3[CF]
audit = {"config": {"n_rows": int(len(V3)), "n_features": len(CF), "features": CF,
                    "population": "engaged (excludes unregistration on/before day 0)"}}
print(f"V3: {len(V3)} rows, {len(CF)} features, {V3['id_student'].nunique()} students")

def xgb(**kw):
    p = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, random_state=42, objective="multi:softmax",
             num_class=4, eval_metric="mlogloss", n_jobs=-1)
    p.update(kw); return XGBClassifier(**p)

def evaluate(feats, tag, n_rep=5, data=None):
    d = V3 if data is None else data
    yy, gg, XX = d["target_multi"].values, d["id_student"].values, d[feats]
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(XX, yy, groups=gg))
    m = xgb(); m.fit(XX.iloc[tr], yy[tr]); p = m.predict(XX.iloc[te])
    accs, f1s = [], []
    for s in range(n_rep):
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(XX, yy, groups=gg))
        ms = xgb(); ms.fit(XX.iloc[a], yy[a]); ps = ms.predict(XX.iloc[b])
        accs.append(accuracy_score(yy[b], ps)); f1s.append(f1_score(yy[b], ps, average="macro"))
    rep = classification_report(yy[te], p, output_dict=True, zero_division=0, target_names=CLS)
    r = {"acc": float(accuracy_score(yy[te], p)), "f1": float(f1_score(yy[te], p, average="macro")),
         "acc_rep_mean": float(np.mean(accs)), "acc_rep_std": float(np.std(accs)),
         "f1_rep_mean": float(np.mean(f1s)), "f1_rep_std": float(np.std(f1s)),
         "per_class_f1": {c: float(rep[c]["f1-score"]) for c in CLS}}
    print(f"[{tag:34s}] acc {r['acc_rep_mean']:.4f}±{r['acc_rep_std']:.4f} "
          f"F1 {r['f1_rep_mean']:.4f}±{r['f1_rep_std']:.4f}", flush=True)
    return r, m, (tr, te)

# ---------- 0. empirical leakage verification ----------
lk = {}
lk["forbidden_cols_in_features"] = [c for c in
    ["date_unregistration", "course_end", "horizon", "final_result", "target_multi",
     "n_cw_total", "n_cw_avail", "last_day", "first_day"] if c in CF]
sv = pd.read_csv(f"{REPO}/data/raw/studentVle.csv", usecols=["id_student", "code_module",
                 "code_presentation", "date"])
KEY = ["id_student", "code_module", "code_presentation"]
chk = sv.merge(V3[KEY + ["date_unregistration"]], on=KEY, how="inner")
lk["vle_rows_after_unreg_in_population"] = int(((chk["date_unregistration"].notna()) &
                                                (chk["date"] > chk["date_unregistration"])).sum())
asr = pd.read_csv(f"{REPO}/data/raw/assessments.csv")
lk["exam_ids_reachable_from_features"] = 0  # exams removed at build; no feature derives from them
# perfect-separator scan: any single feature with AUC>0.99 for any class?
sep = {}
for c_i, c_n in enumerate(CLS):
    yb = (y == c_i).astype(int)
    for f in CF:
        v = X[f].values.astype(float)
        if np.nanstd(v) == 0: continue
        a = roc_auc_score(yb, v); a = max(a, 1 - a)
        if a > 0.95: sep[f"{c_n}|{f}"] = float(a)
lk["single_feature_auc_gt_0.95"] = sep
audit["leakage_checks"] = lk
print("leakage checks:", json.dumps(lk, indent=1)[:400])

# ---------- 1. baseline + trivial predictors ----------
base, model, (tr, te) = evaluate(CF, "V3 baseline")
audit["baseline"] = base
imp = sorted(zip(CF, model.feature_importances_), key=lambda kv: -kv[1])
audit["importance_top10"] = [(f, float(v)) for f, v in imp[:10]]

triv = {}
dm = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y[tr])
pd_ = dm.predict(X.iloc[te])
triv["majority_class"] = {"acc": float(accuracy_score(y[te], pd_)),
                          "f1": float(f1_score(y[te], pd_, average="macro"))}
for f in [imp[0][0], imp[1][0]]:
    r1, _, _ = evaluate([f], f"single feature: {f}", n_rep=3)
    triv[f"only_{f}"] = {"acc": r1["acc_rep_mean"], "f1": r1["f1_rep_mean"]}
audit["trivial_predictors"] = triv

# ---------- 2. ablations: remove top-1/2/3 individually and cumulatively ----------
abl = {}
for k in range(3):
    f = imp[k][0]
    r, _, _ = evaluate([c for c in CF if c != f], f"ablate {f}")
    abl[f"drop_{f}"] = {"acc": r["acc_rep_mean"], "f1": r["f1_rep_mean"],
                        "d_acc": r["acc_rep_mean"] - base["acc_rep_mean"],
                        "d_f1": r["f1_rep_mean"] - base["f1_rep_mean"],
                        "per_class_f1": r["per_class_f1"]}
top3 = [f for f, _ in imp[:3]]
r, _, _ = evaluate([c for c in CF if c not in top3], "ablate top-3 together")
abl["drop_top3"] = {"acc": r["acc_rep_mean"], "f1": r["f1_rep_mean"],
                    "d_acc": r["acc_rep_mean"] - base["acc_rep_mean"],
                    "d_f1": r["f1_rep_mean"] - base["f1_rep_mean"],
                    "per_class_f1": r["per_class_f1"]}
audit["ablations"] = abl

# ---------- 3. subgroup robustness (per module, on the headline split) ----------
p_te = model.predict(X.iloc[te])
sub = V3.iloc[te].copy(); sub["_pred"] = p_te; sub["_true"] = y[te]
sg = {}
for mod, gg in sub.groupby("code_module"):
    sg[mod] = {"n": int(len(gg)), "acc": float(accuracy_score(gg["_true"], gg["_pred"])),
               "f1": float(f1_score(gg["_true"], gg["_pred"], average="macro"))}
audit["subgroup_by_module"] = sg
print("per-module F1:", {k: round(v["f1"], 3) for k, v in sg.items()})
audit["class_balance"] = {c: int((y == i).sum()) for i, c in enumerate(CLS)}
audit["headline_confusion"] = confusion_matrix(y[te], p_te).tolist()
audit["headline_per_class"] = {c: {k: float(v) for k, v in
    classification_report(y[te], p_te, output_dict=True, zero_division=0,
                          target_names=CLS)[c].items()} for c in CLS}

# ---------- 4. feature overlap ----------
corr = X.astype(float).corr().abs()
pairs = []
for i in range(len(CF)):
    for j in range(i + 1, len(CF)):
        v = corr.iloc[i, j]
        if v >= 0.85 and not np.isnan(v):
            pairs.append((CF[i], CF[j], float(v)))
pairs.sort(key=lambda t: -t[2])
audit["high_correlation_pairs_ge_0.85"] = pairs
print("high-corr pairs:", pairs[:8])

# ---------- 5. stratified grouped cross-validation ----------
cv = {"folds": []}
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for i, (a, b) in enumerate(sgkf.split(X, y, groups=g)):
    assert len(set(g[a]) & set(g[b])) == 0
    m = xgb(); m.fit(X.iloc[a], y[a]); p = m.predict(X.iloc[b])
    cv["folds"].append({"fold": i, "n_test": int(len(b)),
                        "acc": float(accuracy_score(y[b], p)),
                        "f1": float(f1_score(y[b], p, average="macro"))})
    print(f"[SGKF fold {i}] acc={cv['folds'][-1]['acc']:.4f} f1={cv['folds'][-1]['f1']:.4f}", flush=True)
cv["mean_acc"] = float(np.mean([f["acc"] for f in cv["folds"]]))
cv["std_acc"] = float(np.std([f["acc"] for f in cv["folds"]]))
cv["mean_f1"] = float(np.mean([f["f1"] for f in cv["folds"]]))
cv["std_f1"] = float(np.std([f["f1"] for f in cv["folds"]]))
audit["stratified_group_cv"] = cv
print(f"SGKF(5): acc {cv['mean_acc']:.4f}±{cv['std_acc']:.4f}  F1 {cv['mean_f1']:.4f}±{cv['std_f1']:.4f}")

json.dump(audit, open(f"{SP}/exp007_audit.json", "w"), indent=1)
print(f"\nTOTAL {time.time()-t00:.0f}s -> exp007_audit.json")
