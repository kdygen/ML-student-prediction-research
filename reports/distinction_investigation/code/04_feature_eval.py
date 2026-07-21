"""Distinction investigation 04 — do the new feature groups move the P-vs-D ceiling?

Design: the threshold-free information metric is the balanced P-vs-D binary AUC/PR-AUC on
the P+D subset (same grouped folds as M5). Each hypothesis group is added to the 33-feature
base separately, then the helpful ones combined. Finally the best combined set is run through
the full 4-class pipeline with M1 weighting (the new operating point) on the SAME folds.
Also: effect sizes of every new feature (D vs P) for the log.
"""
import json, os, time
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                             roc_auc_score, average_precision_score)

SP = os.environ["SP"]
REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
OUT = REPO + "/reports/distinction_investigation"
CLS = ["Withdrawn", "Fail", "Pass", "Distinction"]
KEY = ["id_student", "code_module", "code_presentation"]

df = pd.read_parquet(SP + "/exp006c_frame_probe.parquet").copy()
df.loc[df["first_submit_day"] >= (df["course_end"] + 29), "first_submit_day"] = 999
newf = pd.read_parquet(SP + "/dx_new_features.parquet")
groups = json.load(open(SP + "/dx_feature_groups.json"))
df = df.merge(newf, on=KEY, how="left", validate="one_to_one")
F36 = json.load(open(REPO + "/reports/experiment_008_parsimony/parsimony_metrics.json"))["final"]["features"]
F33 = [f for f in F36 if f not in ["rank_wa", "score_slope_cw", "score_std_cw"]]
V3 = df[~(df["date_unregistration"] <= 0)].reset_index(drop=True)
for c in [c for g in groups.values() for c in g]:
    V3[c] = pd.to_numeric(V3[c], errors="coerce").fillna(0)
y = V3["target_multi"].values; g_ = V3["id_student"].values

# ---------- effect sizes of new features (D vs P) ----------
eff = []
for f in [c for gg in groups.values() for c in gg]:
    a = V3.loc[y == 3, f].astype(float); b = V3.loc[y == 2, f].astype(float)
    s = np.sqrt((a.std()**2 + b.std()**2) / 2)
    eff.append({"feature": f, "cohens_d_D_minus_P": float((a.mean() - b.mean()) / s) if s > 0 else 0.0})
eff.sort(key=lambda r: -abs(r["cohens_d_D_minus_P"]))
print("new-feature effect sizes (D vs P), top 12:")
for r in eff[:12]: print(f"  {r['feature']:26s} d={r['cohens_d_D_minus_P']:+.3f}")

# ---------- P-vs-D binary AUC per feature set ----------
pdm = np.isin(y, [2, 3])
Xall_idx = V3[pdm].reset_index(drop=True)
yb = (y[pdm] == 3).astype(int); gb = g_[pdm]
bfolds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(Xall_idx, yb, groups=gb))

def pvd_auc(feats, tag):
    Xp = Xall_idx[feats]
    aucs, prs = [], []
    for a, b in bfolds:
        spw = (yb[a] == 0).sum() / (yb[a] == 1).sum()
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, eval_metric="logloss",
                          scale_pos_weight=spw, n_jobs=-1)
        m.fit(Xp.iloc[a], yb[a]); p = m.predict_proba(Xp.iloc[b])[:, 1]
        aucs.append(roc_auc_score(yb[b], p)); prs.append(average_precision_score(yb[b], p))
    r = {"set": tag, "n_features": len(feats), "auc": float(np.mean(aucs)),
         "auc_std": float(np.std(aucs)), "pr_auc": float(np.mean(prs))}
    print(f"[PvD {tag:22s}] {len(feats):2d}+ feats AUC {r['auc']:.4f}±{r['auc_std']:.4f} PR {r['pr_auc']:.4f}", flush=True)
    return r

runs = [pvd_auc(F33, "base_33")]
for gname, cols in groups.items():
    runs.append(pvd_auc(F33 + cols, f"+{gname}"))
runs.append(pvd_auc(F33 + [c for g2 in groups.values() for c in g2], "+ALL_new"))
base_auc = runs[0]["auc"]
helpful = [r["set"][1:] for r in runs[1:-1] if r["auc"] > base_auc + 0.002]
print("groups beating base by >0.002 AUC:", helpful if helpful else "NONE")
combo_cols = [c for name in helpful for c in groups[name]]
if helpful and len(helpful) > 1:
    runs.append(pvd_auc(F33 + combo_cols, "+helpful_combo"))

# ---------- full 4-class with M1 weighting: base vs best feature set ----------
best_feats = F33 + (combo_cols if helpful else [c for g2 in groups.values() for c in g2])
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(V3[F33], y, groups=g_))
cw = {c: len(y) / (4 * (y == c).sum()) for c in range(4)}
def multi(feats, tag):
    X = V3[feats]
    oof = np.full(len(y), -1)
    for tr, te in folds:
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, objective="multi:softmax",
                          num_class=4, eval_metric="mlogloss", n_jobs=-1)
        m.fit(X.iloc[tr], y[tr], sample_weight=np.vectorize(cw.get)(y[tr]))
        oof[te] = m.predict(X.iloc[te])
    A = [accuracy_score(y[te], oof[te]) for _, te in folds]
    F = [f1_score(y[te], oof[te], average="macro") for _, te in folds]
    rep = classification_report(y, oof, target_names=CLS, output_dict=True, zero_division=0)
    r = {"set": tag, "n_features": len(feats), "accuracy": float(np.mean(A)),
         "macro_f1": float(np.mean(F)), "macro_f1_std": float(np.std(F)),
         "D_precision": float(rep["Distinction"]["precision"]),
         "D_recall": float(rep["Distinction"]["recall"]),
         "D_f1": float(rep["Distinction"]["f1-score"]),
         "P_f1": float(rep["Pass"]["f1-score"]), "F_f1": float(rep["Fail"]["f1-score"]),
         "W_f1": float(rep["Withdrawn"]["f1-score"])}
    print(f"[4cls {tag:22s}] acc {r['accuracy']:.4f} F1 {r['macro_f1']:.4f}±{r['macro_f1_std']:.4f} "
          f"| D {r['D_f1']:.3f} (P{r['D_precision']:.2f}/R{r['D_recall']:.2f}) "
          f"| W {r['W_f1']:.3f} F {r['F_f1']:.3f} P {r['P_f1']:.3f}", flush=True)
    return r

m_base = multi(F33, "weighted_base_33")
m_best = multi(best_feats, "weighted_best_feats")

json.dump({"new_feature_effect_sizes": eff, "pvd_auc_runs": runs,
           "multiclass_weighted": [m_base, m_best],
           "best_feature_set": best_feats},
          open(OUT + "/iteration2_features.json", "w"), indent=1)
print("\nsaved -> iteration2_features.json")
