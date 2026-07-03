"""
Experiment 002 — final confirmation of the accepted feature set at all cutoffs.

Arms (identical XGB hyperparameters everywhere):
  A) baseline_v3   : official 19 features, tau=0            (Baseline v3 reference)
  B) exp001_winner : official 19 features, tau=0.75         (Experiment 001 winner)
  C) exp002        : official 19 + accepted groups, tau re-tuned per cutoff by
                     GroupKFold(3) inside the seed-42 train (never on test)

Per cutoff: one evaluation on the held-out seed-42 test, then repeated grouped
splits (seeds 0-4) with everything frozen; all arms share identical splits for
paired statistics. Also records the tau=0 accuracy operating point of arm C
(70%-accuracy stretch check), per-class metrics, confusions, importance.
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "experiments")
from feature_generation_002 import FEATURE_GROUPS
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (f1_score, accuracy_score, recall_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier

SP = os.environ["SP"]; OUT = os.environ["OUT"]
LOOP = json.load(open(os.environ["LOOP_JSON"]))
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 42
CUTOFFS = [14, 30, 60, 90, 140]

BASE = LOOP["base_features"]
ACCEPTED = LOOP["accepted_groups"]
EXTRA = [c for gname in ACCEPTED for c in FEATURE_GROUPS[gname]]
AUG = BASE + EXTRA
print(f"[final] accepted groups: {ACCEPTED} (+{len(EXTRA)} cols)", file=sys.stderr)

def xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                         objective="multi:softprob", num_class=4,
                         eval_metric="mlogloss", n_jobs=-1)

def run(Xtr, ytr, Xte, yte, tau):
    m = xgb(); m.fit(Xtr, ytr)
    pri = np.bincount(ytr, minlength=4) / len(ytr)
    pred = np.argmax(m.predict_proba(Xte) * (1.0 / pri) ** tau, axis=1)
    return m, pred

def metrics(yte, pred):
    rep = classification_report(yte, pred, output_dict=True, zero_division=0,
                                target_names=["Withdrawn", "Fail", "Pass", "Distinction"])
    return dict(accuracy=float(accuracy_score(yte, pred)),
                macro_f1=float(f1_score(yte, pred, average="macro")),
                withdrawn_recall=float(recall_score(yte, pred, labels=[0], average=None)[0]),
                per_class={k: {kk: float(vv) for kk, vv in v.items()}
                           for k, v in rep.items() if isinstance(v, dict)},
                confusion=confusion_matrix(yte, pred).tolist())

def tune_tau(df, feats, tr_idx, ytr_all, gtr):
    folds = GroupKFold(n_splits=3).split(df.iloc[tr_idx], ytr_all, gtr)
    f1s = {t: [] for t in TAUS}
    for a, b in folds:
        Xa = df.iloc[tr_idx].iloc[a][feats]; ya = ytr_all.iloc[a]
        Xb = df.iloc[tr_idx].iloc[b][feats]; yb = ytr_all.iloc[b]
        m = xgb(); m.fit(Xa, ya)
        proba = m.predict_proba(Xb)
        pri = np.bincount(ya, minlength=4) / len(ya)
        for t in TAUS:
            f1s[t].append(f1_score(yb, np.argmax(proba * (1.0/pri) ** t, axis=1),
                                   average="macro"))
    return max(TAUS, key=lambda t: float(np.mean(f1s[t])))

results = {"accepted_groups": ACCEPTED, "extra_features": EXTRA, "cutoffs": {}}

for C in CUTOFFS:
    df = pd.read_parquet(f"{SP}/aug_c{C:03d}.parquet")
    y = df["target_multi"]; g = df["id_student"].values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                  .split(df, y, g))
    ytr, yte = y.iloc[tr], y.iloc[te]

    tauC = tune_tau(df, AUG, tr, ytr, g[tr]) if C != 30 else LOOP["final_inner_eval"]["tau"]
    R = {"tau_exp002": tauC, "n": int(len(df))}

    arms = {"baseline_v3": (BASE, 0.0), "exp001_winner": (BASE, 0.75),
            "exp002": (AUG, tauC), "exp002_tau0": (AUG, 0.0)}
    # held-out official test
    for name, (feats, tau) in arms.items():
        m, pred = run(df.iloc[tr][feats], ytr, df.iloc[te][feats], yte, tau)
        R[f"test_{name}"] = metrics(yte, pred)
        if name == "exp002" and C in (30, 140):
            R["importance"] = sorted(({"feature": f, "importance": float(v)}
                                      for f, v in zip(feats, m.feature_importances_)),
                                     key=lambda d: -d["importance"])[:15]
    # repeats, shared splits
    reps = {k: [] for k in arms}
    for seed in range(5):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    .split(df, y, g))
        for name, (feats, tau) in arms.items():
            _, pred = run(df.iloc[a][feats], y.iloc[a], df.iloc[b][feats], y.iloc[b], tau)
            mm = metrics(y.iloc[b], pred)
            reps[name].append({k: mm[k] for k in ("accuracy", "macro_f1", "withdrawn_recall")})
    def agg(rs, key):
        v = [r[key] for r in rs]
        return dict(mean=float(np.mean(v)), std=float(np.std(v)), all=[round(x, 4) for x in v])
    R["repeats"] = {k: {m: agg(v, m) for m in ("accuracy", "macro_f1", "withdrawn_recall")}
                    for k, v in reps.items()}
    for ref in ("exp001_winner", "baseline_v3"):
        d = [reps["exp002"][i]["macro_f1"] - reps[ref][i]["macro_f1"] for i in range(5)]
        R[f"paired_dF1_vs_{ref}"] = dict(per_seed=[round(x, 4) for x in d],
                                          mean=float(np.mean(d)), std=float(np.std(d)),
                                          all_positive=bool(all(x > 0 for x in d)))
    results["cutoffs"][str(C)] = R
    print(f"[final] c{C}: tau={tauC} | test F1 "
          f"v3={R['test_baseline_v3']['macro_f1']:.4f} "
          f"e1={R['test_exp001_winner']['macro_f1']:.4f} "
          f"e2={R['test_exp002']['macro_f1']:.4f} | rep e2 "
          f"{R['repeats']['exp002']['macro_f1']['mean']:.4f}±{R['repeats']['exp002']['macro_f1']['std']:.4f} "
          f"(e1 {R['repeats']['exp001_winner']['macro_f1']['mean']:.4f}) | "
          f"Wrec e2={R['repeats']['exp002']['withdrawn_recall']['mean']:.4f} | "
          f"acc@tau0 e2={R['repeats']['exp002_tau0']['accuracy']['mean']:.4f}", file=sys.stderr)

json.dump(results, open(OUT, "w"), indent=2)
print("WROTE", OUT, file=sys.stderr)
