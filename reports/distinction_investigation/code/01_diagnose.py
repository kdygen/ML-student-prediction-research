"""Distinction investigation 01 — reproduce the score-free baseline and diagnose
the Pass-vs-Distinction confusion. Protocol: engaged population (29,496), 33 features
(36 minus rank_wa/score_slope_cw/score_std_cw), StratifiedGroupKFold(5, shuffle, rs=42)
grouped by id_student — identical to Experiment 009 Arm B.
"""
import json, os, hashlib
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score, average_precision_score)

SP = os.environ["SP"]
REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
OUT = REPO + "/reports/distinction_investigation"
CLS = ["Withdrawn", "Fail", "Pass", "Distinction"]

df = pd.read_parquet(SP + "/exp006c_frame_probe.parquet").copy()
sent = df["first_submit_day"] >= (df["course_end"] + 29)
df.loc[sent, "first_submit_day"] = 999
F36 = json.load(open(REPO + "/reports/experiment_008_parsimony/parsimony_metrics.json"))["final"]["features"]
SCORE = ["rank_wa", "score_slope_cw", "score_std_cw"]
F33 = [f for f in F36 if f not in SCORE]
V3 = df[~(df["date_unregistration"] <= 0)].reset_index(drop=True)
X, y, g = V3[F33], V3["target_multi"].values, V3["id_student"].values

def xgb(**kw):
    p = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, random_state=42, objective="multi:softmax",
             num_class=4, eval_metric="mlogloss", n_jobs=-1)
    p.update(kw); return XGBClassifier(**p)

# ---------- baseline reproduction with OOF probabilities ----------
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups=g))
fold_hash = hashlib.sha256(b"".join(np.sort(te).tobytes() for _, te in folds)).hexdigest()[:16]
oof_pred = np.full(len(y), -1); oof_proba = np.full((len(y), 4), np.nan)
A, F = [], []
for tr, te in folds:
    assert len(set(g[tr]) & set(g[te])) == 0
    m = xgb(); m.fit(X.iloc[tr], y[tr])
    oof_pred[te] = m.predict(X.iloc[te]); oof_proba[te] = m.predict_proba(X.iloc[te])
    A.append(accuracy_score(y[te], oof_pred[te])); F.append(f1_score(y[te], oof_pred[te], average="macro"))
rep = classification_report(y, oof_pred, target_names=CLS, output_dict=True, zero_division=0)
base = {"fold_hash": fold_hash, "accuracy": float(np.mean(A)), "accuracy_std": float(np.std(A)),
        "macro_f1": float(np.mean(F)), "macro_f1_std": float(np.std(F)),
        "per_class_f1": {c: float(rep[c]["f1-score"]) for c in CLS}}
print(f"[baseline B reproduced] acc {base['accuracy']:.4f}±{base['accuracy_std']:.4f} "
      f"F1 {base['macro_f1']:.4f}±{base['macro_f1_std']:.4f}")
print(f"  per-class F1: {base['per_class_f1']}")
prev = json.load(open(REPO + "/reports/experiment_009_feature_access_results.json"))["engaged_29496"]["B_no_scores_33"]
print(f"  matches Exp009 B: dAcc={base['accuracy']-prev['accuracy']:+.6f} dF1={base['macro_f1']-prev['macro_f1']:+.6f}")
np.save(SP + "/dx_oof_proba.npy", oof_proba); np.save(SP + "/dx_oof_pred.npy", oof_pred)

dx = {"baseline": base}

# ---------- P vs D diagnosis ----------
pd_mask = np.isin(y, [2, 3])
yp = y[pd_mask]; pp = oof_pred[pd_mask]; pr = oof_proba[pd_mask]
dD = V3[pd_mask]
dx["population"] = {"pass": int((y == 2).sum()), "distinction": int((y == 3).sum()),
                    "imbalance_P_to_D": float((y == 2).sum() / (y == 3).sum())}
# where do true Distinctions go?
dgo = pd.Series(oof_pred[y == 3]).map(dict(enumerate(CLS))).value_counts(normalize=True)
dx["true_distinction_predicted_as"] = {k: float(v) for k, v in dgo.items()}
# P(D) probability separation
pD_dist = oof_proba[y == 3, 3]; pD_pass = oof_proba[y == 2, 3]
dx["prob_D"] = {"true_D_mean": float(pD_dist.mean()), "true_D_median": float(np.median(pD_dist)),
                "true_P_mean": float(pD_pass.mean()),
                "true_D_share_above_0.5": float((pD_dist > 0.5).mean()),
                "auc_P_vs_D_from_multiclass_pD": float(roc_auc_score((yp == 3).astype(int), pr[:, 3])),
                "prauc_P_vs_D": float(average_precision_score((yp == 3).astype(int), pr[:, 3]))}
# effect sizes: Pass vs Distinction on all 33 features (Cohen's d)
eff = []
for f in F33:
    a = V3.loc[y == 3, f].astype(float); b = V3.loc[y == 2, f].astype(float)
    s = np.sqrt((a.std()**2 + b.std()**2) / 2)
    eff.append((f, float((a.mean() - b.mean()) / s) if s > 0 else 0.0,
                float(a.mean()), float(b.mean())))
eff.sort(key=lambda t: -abs(t[1]))
dx["effect_sizes_D_minus_P"] = [{"feature": f, "cohens_d": d, "mean_D": mD, "mean_P": mP}
                                for f, d, mD, mP in eff]
print("\ntop |d| features D vs P:")
for f, d, mD, mP in eff[:10]: print(f"  {f:28s} d={d:+.3f}  D={mD:.2f} P={mP:.2f}")

# misclassified vs correctly classified Distinction
corr = (y == 3) & (oof_pred == 3); miss = (y == 3) & (oof_pred == 2)
dx["distinction_split"] = {"n_correct": int(corr.sum()), "n_missed_as_pass": int(miss.sum())}
mm = []
for f in F33:
    a = V3.loc[corr, f].astype(float); b = V3.loc[miss, f].astype(float); c = V3.loc[y == 2, f].astype(float)
    s = np.sqrt((a.std()**2 + b.std()**2) / 2)
    d_cm = float((a.mean() - b.mean()) / s) if s > 0 else 0.0
    s2 = np.sqrt((b.std()**2 + c.std()**2) / 2)
    d_mp = float((b.mean() - c.mean()) / s2) if s2 > 0 else 0.0
    mm.append((f, d_cm, d_mp))
mm.sort(key=lambda t: -abs(t[1]))
dx["correct_vs_missed_D"] = [{"feature": f, "d_correct_vs_missed": a, "d_missed_vs_pass": b}
                             for f, a, b in mm]
big_missed_vs_pass = [abs(b) for _, _, b in mm]
dx["missed_D_indistinguishable_from_P"] = {"max_abs_d_missed_vs_pass": float(max(big_missed_vs_pass)),
                                           "mean_abs_d": float(np.mean(big_missed_vs_pass))}
print(f"\nmissed-D vs Pass: max |d| = {max(big_missed_vs_pass):.3f} (near 0 => truly Pass-like)")
print("top features separating CORRECT-D from MISSED-D:")
for f, a, b in mm[:8]: print(f"  {f:28s} d(corr,miss)={a:+.3f}  d(miss,Pass)={b:+.3f}")

# per-module / per-presentation Distinction F1 + subgroups
sub = {}
V3d = V3.copy(); V3d["_pred"] = oof_pred; V3d["_true"] = y
for col in ["code_module", "code_presentation"]:
    r = {}
    for k, gg in V3d.groupby(col):
        yt, yp_ = gg["_true"].values, gg["_pred"].values
        nD = int((yt == 3).sum())
        r[str(k)] = {"n_D": nD, "D_share": float(nD / len(gg)),
                     "D_f1": float(f1_score(yt == 3, yp_ == 3, zero_division=0))}
    sub[col] = r
demo = pd.read_csv(REPO + "/data/raw/studentInfo.csv")
V3d = V3d.merge(demo[["id_student", "code_module", "code_presentation", "gender", "age_band",
                      "imd_band", "disability", "num_of_prev_attempts"]],
                on=["id_student", "code_module", "code_presentation"], how="left")
for col in ["gender", "age_band", "disability"]:
    r = {}
    for k, gg in V3d.groupby(col):
        yt, yp_ = gg["_true"].values, gg["_pred"].values
        r[str(k)] = {"n_D": int((yt == 3).sum()),
                     "D_f1": float(f1_score(yt == 3, yp_ == 3, zero_division=0))}
    sub[col] = r
dx["subgroup_D_f1"] = sub
print("\nper-module Distinction F1:", {k: round(v["D_f1"], 3) for k, v in sub["code_module"].items()})

json.dump(dx, open(OUT + "/phase1_diagnosis.json", "w"), indent=1)
print("\nsaved -> reports/distinction_investigation/phase1_diagnosis.json")
