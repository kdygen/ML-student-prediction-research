"""Distinction investigation 02 — model/operating-point experiments (Iteration 1).

Hypothesis I1: Distinction F1 is suppressed by the argmax operating point under 4.1:1
imbalance; the score-free ranking already carries signal (P-vs-D AUC 0.746). Fixing the
operating point should recover F1 without new information.

Arms (same folds as baseline, all tuning INSIDE training folds only):
  M1 sample-weighted XGB (inverse class frequency)
  M2 D-threshold rescue on baseline probas: predict D if p(D) > tau, tau tuned on inner
     GroupKFold(3) OOF within each training fold (maximise D F1 s.t. macro-F1 not worse
     than argmax - 0.005 on inner OOF)
  M3 hierarchical: keep baseline W/F; re-decide P-vs-D with a dedicated balanced binary
     classifier (threshold tuned on inner OOF for D F1)
  M4 ordinal: XGB regressor on ordered target, cut points tuned on inner OOF (macro-F1)
  M5 P-vs-D information ceiling: dedicated binary on P+D subset only (balanced), grouped
     folds -> AUC / PR-AUC / best-inner-threshold F1. This is the feature-set ceiling.
"""
import json, os, time
import numpy as np, pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             roc_auc_score, average_precision_score)

SP = os.environ["SP"]
REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
OUT = REPO + "/reports/distinction_investigation"
CLS = ["Withdrawn", "Fail", "Pass", "Distinction"]

df = pd.read_parquet(SP + "/exp006c_frame_probe.parquet").copy()
df.loc[df["first_submit_day"] >= (df["course_end"] + 29), "first_submit_day"] = 999
F36 = json.load(open(REPO + "/reports/experiment_008_parsimony/parsimony_metrics.json"))["final"]["features"]
F33 = [f for f in F36 if f not in ["rank_wa", "score_slope_cw", "score_std_cw"]]
V3 = df[~(df["date_unregistration"] <= 0)].reset_index(drop=True)
X, y, g = V3[F33], V3["target_multi"].values, V3["id_student"].values
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups=g))

def xgb_c(**kw):
    p = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, random_state=42, objective="multi:softmax",
             num_class=4, eval_metric="mlogloss", n_jobs=-1)
    p.update(kw); return XGBClassifier(**p)

def summarize(name, oof_pred, extra=None):
    A = [accuracy_score(y[te], oof_pred[te]) for _, te in folds]
    F = [f1_score(y[te], oof_pred[te], average="macro") for _, te in folds]
    rep = classification_report(y, oof_pred, target_names=CLS, output_dict=True, zero_division=0)
    r = {"arm": name, "accuracy": float(np.mean(A)), "accuracy_std": float(np.std(A)),
         "macro_f1": float(np.mean(F)), "macro_f1_std": float(np.std(F)),
         "weighted_f1": float(rep["weighted avg"]["f1-score"]),
         "D_precision": float(rep["Distinction"]["precision"]),
         "D_recall": float(rep["Distinction"]["recall"]),
         "D_f1": float(rep["Distinction"]["f1-score"]),
         "P_f1": float(rep["Pass"]["f1-score"]), "F_f1": float(rep["Fail"]["f1-score"]),
         "W_f1": float(rep["Withdrawn"]["f1-score"])}
    if extra: r.update(extra)
    print(f"[{name:24s}] acc {r['accuracy']:.4f} F1 {r['macro_f1']:.4f}±{r['macro_f1_std']:.4f} "
          f"| D P/R/F1 {r['D_precision']:.3f}/{r['D_recall']:.3f}/{r['D_f1']:.3f} "
          f"| W {r['W_f1']:.3f} F {r['F_f1']:.3f} P {r['P_f1']:.3f}", flush=True)
    return r

results = []
# ---------- M4: ordinal (regressor + tuned cuts) ----------
t0 = time.time()
oof = np.full(len(y), -1)
for tr, te in folds:
    Xtr, ytr, gtr = X.iloc[tr], y[tr], g[tr]
    inner = np.full(len(tr), np.nan)
    for a, b in GroupKFold(3).split(Xtr, ytr, groups=gtr):
        mi = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, n_jobs=-1)
        mi.fit(Xtr.iloc[a], ytr[a]); inner[b] = mi.predict(Xtr.iloc[b])
    best_cuts, best = None, -1
    for c1 in np.arange(0.3, 1.15, 0.1):
        for c2 in np.arange(c1 + 0.25, 2.15, 0.1):
            for c3 in np.arange(c2 + 0.25, 3.0, 0.1):
                pred = np.digitize(inner, [c1, c2, c3])
                f = f1_score(ytr, pred, average="macro")
                if f > best: best, best_cuts = f, (c1, c2, c3)
    m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr)
    oof[te] = np.digitize(m.predict(X.iloc[te]), list(best_cuts))
results.append(summarize("M4_ordinal_regression", oof, {"runtime_s": round(time.time()-t0)}))

# ---------- M5: P-vs-D dedicated binary (information ceiling of 33 features) ----------
t0 = time.time()
pdm = np.isin(y, [2, 3])
Xp, yp, gp = X[pdm].reset_index(drop=True), (y[pdm] == 3).astype(int), g[pdm]
aucs, praucs, f1s = [], [], []
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(Xp, yp, groups=gp):
    spw = (yp[a] == 0).sum() / (yp[a] == 1).sum()
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, random_state=42, eval_metric="logloss",
                      scale_pos_weight=spw, n_jobs=-1)
    m.fit(Xp.iloc[a], yp[a]); pr = m.predict_proba(Xp.iloc[b])[:, 1]
    aucs.append(roc_auc_score(yp[b], pr)); praucs.append(average_precision_score(yp[b], pr))
    # inner-tuned threshold
    inner = np.full(len(a), np.nan)
    for a2, b2 in GroupKFold(3).split(Xp.iloc[a], yp[a], groups=gp[a]):
        mi = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, random_state=42, eval_metric="logloss",
                           scale_pos_weight=spw, n_jobs=-1)
        mi.fit(Xp.iloc[a].iloc[a2], yp[a][a2]); inner[b2] = mi.predict_proba(Xp.iloc[a].iloc[b2])[:, 1]
    best_tau, best = 0.5, -1
    for tau in np.arange(0.2, 0.81, 0.02):
        f = f1_score(yp[a], (inner > tau).astype(int), zero_division=0)
        if f > best: best, best_tau = f, tau
    f1s.append(f1_score(yp[b], (pr > best_tau).astype(int), zero_division=0))
ceil = {"arm": "M5_PvsD_binary_ceiling", "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "pr_auc": float(np.mean(praucs)), "pr_auc_std": float(np.std(praucs)),
        "D_f1_at_inner_tuned_threshold": float(np.mean(f1s)), "D_f1_std": float(np.std(f1s)),
        "runtime_s": round(time.time()-t0)}
print(f"[M5_PvsD_binary_ceiling   ] AUC {ceil['auc']:.4f}±{ceil['auc_std']:.4f} "
      f"PR-AUC {ceil['pr_auc']:.4f} | D-F1@tuned-tau {ceil['D_f1_at_inner_tuned_threshold']:.3f}±{ceil['D_f1_std']:.3f}")
results.append(ceil)

json.dump(results, open(OUT + "/iteration1_models.json", "w"), indent=1)
pd.DataFrame(results).to_csv(OUT + "/experiment_results.csv", index=False)
print("\nsaved -> iteration1_models.json / experiment_results.csv")
