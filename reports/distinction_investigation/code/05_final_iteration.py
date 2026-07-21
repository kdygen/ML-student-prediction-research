"""Distinction investigation 05 — final iteration.

I3 hypotheses:
  (a) module identity: per-module Distinction F1 varies 0.00-0.40, so module dummies may let
      the model calibrate class priors per course (static metadata, leakage-free).
  (b) alternative GBDT: sklearn HistGradientBoosting (in the pinned env) as a check that the
      ceiling is not an XGBoost quirk. LightGBM/CatBoost skipped: not in the pinned
      environment and installing them would break reproducibility guarantees.
  (c) best stack: weighted 4-class on 33+H2+module dummies, plus inner-tuned D-threshold.
Also produces the final artifacts: P(D) distribution figure + per-module D-F1 figure.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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
H2 = groups["H2_depth"]
V3 = df[~(df["date_unregistration"] <= 0)].reset_index(drop=True)
for c in [c for g in groups.values() for c in g]:
    V3[c] = pd.to_numeric(V3[c], errors="coerce").fillna(0)
mod = pd.get_dummies(V3["code_module"], prefix="mod", drop_first=True).astype(float)
V3 = pd.concat([V3, mod], axis=1)
MODS = list(mod.columns)
y = V3["target_multi"].values; g_ = V3["id_student"].values

pdm = np.isin(y, [2, 3])
Xi = V3[pdm].reset_index(drop=True); yb = (y[pdm] == 3).astype(int); gb = g_[pdm]
bfolds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(Xi, yb, groups=gb))

def pvd(feats, tag, model="xgb"):
    Xp = Xi[feats]; aucs, prs = [], []
    for a, b in bfolds:
        spw = (yb[a] == 0).sum() / (yb[a] == 1).sum()
        if model == "xgb":
            m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, eval_metric="logloss",
                              scale_pos_weight=spw, n_jobs=-1)
            m.fit(Xp.iloc[a], yb[a]); p = m.predict_proba(Xp.iloc[b])[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                               random_state=42,
                                               class_weight={0: 1.0, 1: float(spw)})
            m.fit(Xp.iloc[a], yb[a]); p = m.predict_proba(Xp.iloc[b])[:, 1]
        aucs.append(roc_auc_score(yb[b], p)); prs.append(average_precision_score(yb[b], p))
    r = {"set": tag, "model": model, "n_features": len(feats), "auc": float(np.mean(aucs)),
         "auc_std": float(np.std(aucs)), "pr_auc": float(np.mean(prs))}
    print(f"[PvD {tag:26s} {model:6s}] AUC {r['auc']:.4f}±{r['auc_std']:.4f} PR {r['pr_auc']:.4f}", flush=True)
    return r

runs = []
runs.append(pvd(F33 + H2, "33+H2"))
runs.append(pvd(F33 + H2 + MODS, "33+H2+module"))
runs.append(pvd(F33 + H2, "33+H2", model="histgb"))
runs.append(pvd(F33 + H2 + MODS, "33+H2+module", model="histgb"))

# ---------- final stack: weighted multiclass + best feats + inner-tuned D threshold ----------
BEST = F33 + H2 + MODS
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(V3[F33], y, groups=g_))
cw = {c: len(y) / (4 * (y == c).sum()) for c in range(4)}
def xgb4():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, objective="multi:softmax",
                         num_class=4, eval_metric="mlogloss", n_jobs=-1)
X = V3[BEST]
oof = np.full(len(y), -1); oof_p = np.full((len(y), 4), np.nan)
taus = []
for tr, te in folds:
    assert len(set(g_[tr]) & set(g_[te])) == 0
    Xtr, ytr, gtr = X.iloc[tr], y[tr], g_[tr]
    inner = np.full((len(tr), 4), np.nan)
    for a, b in GroupKFold(3).split(Xtr, ytr, groups=gtr):
        mi = xgb4(); mi.fit(Xtr.iloc[a], ytr[a], sample_weight=np.vectorize(cw.get)(ytr[a]))
        inner[b] = mi.predict_proba(Xtr.iloc[b])
    ip = inner.argmax(1)
    base_macro = f1_score(ytr, ip, average="macro")
    best_tau, best_df1 = None, f1_score(ytr == 3, ip == 3, zero_division=0)
    for tau in np.arange(0.20, 0.61, 0.02):
        cand = ip.copy(); cand[inner[:, 3] > tau] = 3
        d = f1_score(ytr == 3, cand == 3, zero_division=0)
        if d > best_df1 and f1_score(ytr, cand, average="macro") >= base_macro - 0.005:
            best_tau, best_df1 = tau, d
    taus.append(best_tau)
    m = xgb4(); m.fit(Xtr, ytr, sample_weight=np.vectorize(cw.get)(ytr))
    P = m.predict_proba(X.iloc[te]); pred = P.argmax(1)
    if best_tau is not None: pred[P[:, 3] > best_tau] = 3
    oof[te] = pred; oof_p[te] = P
A = [accuracy_score(y[te], oof[te]) for _, te in folds]
F = [f1_score(y[te], oof[te], average="macro") for _, te in folds]
Df = [f1_score(y[te] == 3, oof[te] == 3, zero_division=0) for _, te in folds]
rep = classification_report(y, oof, target_names=CLS, output_dict=True, zero_division=0)
final = {"arm": "FINAL_weighted+H2+module+tuned_tau", "taus": taus,
         "accuracy": float(np.mean(A)), "macro_f1": float(np.mean(F)),
         "macro_f1_std": float(np.std(F)), "D_f1_per_fold": [float(x) for x in Df],
         "D_f1_mean": float(np.mean(Df)), "D_f1_std": float(np.std(Df)),
         "D_precision": float(rep["Distinction"]["precision"]),
         "D_recall": float(rep["Distinction"]["recall"]),
         "D_f1": float(rep["Distinction"]["f1-score"]),
         "P_f1": float(rep["Pass"]["f1-score"]), "F_f1": float(rep["Fail"]["f1-score"]),
         "W_f1": float(rep["Withdrawn"]["f1-score"])}
print(f"\n[FINAL stack] acc {final['accuracy']:.4f} F1 {final['macro_f1']:.4f}±{final['macro_f1_std']:.4f} "
      f"| D {final['D_f1']:.3f} (P{final['D_precision']:.2f}/R{final['D_recall']:.2f}) "
      f"per-fold {['%.3f'%x for x in Df]} | W {final['W_f1']:.3f} F {final['F_f1']:.3f} P {final['P_f1']:.3f}")

# ---------- figures ----------
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
pD = oof_p[:, 3]
ax[0].hist(pD[y == 2], bins=40, alpha=0.6, density=True, label="true Pass")
ax[0].hist(pD[y == 3], bins=40, alpha=0.6, density=True, label="true Distinction")
ax[0].set_xlabel("model P(Distinction)"); ax[0].set_title("Score-free P(D): heavy overlap")
ax[0].legend(); ax[0].grid(alpha=0.3)
V3d = V3.copy(); V3d["_pred"] = oof
mods = sorted(V3d["code_module"].unique())
df1 = [f1_score((V3d[V3d.code_module == m]["target_multi"] == 3),
                (V3d[V3d.code_module == m]["_pred"] == 3), zero_division=0) for m in mods]
dsh = [float((V3d[V3d.code_module == m]["target_multi"] == 3).mean()) for m in mods]
ax[1].bar(mods, df1, color="tab:blue", label="Distinction F1 (final stack)")
ax[1].plot(mods, dsh, "ro-", label="Distinction share")
ax[1].set_title("Per-module Distinction F1"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT + "/figures/final_distributions.png", dpi=150)

json.dump({"pvd_runs": runs, "final_stack": final},
          open(OUT + "/iteration3_final.json", "w"), indent=1)
print("saved -> iteration3_final.json + figures/final_distributions.png")
