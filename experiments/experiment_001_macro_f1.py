"""
Experiment 001 — macro-F1 optimization under the official v3 protocol.

Input: canonical cache data/processed/p3/c*/mlDataV3.parquet (never modified).
Protocol per cutoff:
  - official split = GroupShuffleSplit(group=id_student, test_size=0.2, seed=42);
    the test side is NEVER used for any tuning decision.
  - all tuning = GroupKFold(3) inside the seed-42 TRAIN, including the
    class-prior adjustment exponent tau (threshold tuning for macro-F1):
       p_adj[:, c] = proba[:, c] * (1/prior_c)^tau,  argmax over classes
    tau=0 reproduces plain argmax. Priors come from the fitted training data.
  - SMOTE variants resample INSIDE training folds only.
  - screening of all candidate configs happens at cutoff 30; the top-3 shortlist
    is re-selected per cutoff by inner CV; the per-cutoff winner is refit on the
    full seed-42 train and evaluated once on the held-out test, then re-evaluated
    on repeated grouped splits (seeds 0-4) with hyperparameters/tau FROZEN.
  - paired baseline comparison: baseline-v3 XGBoost (unweighted) and RF
    (balanced) evaluated on the identical repeat splits.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, recall_score)
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

CACHE = "data/processed/p3"
OUT = os.environ["OUT"]
CUTOFFS = [14, 30, 60, 90, 140]
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 42

def load(C):
    df = pd.read_parquet(f"{CACHE}/c{C:03d}/mlDataV3.parquet")
    man = json.load(open(f"{CACHE}/c{C:03d}/manifest.json"))
    feats = man["active_features_v3"]
    return df, df[feats], df["target_multi"], df["id_student"].values

# ---------------- candidate configs ----------------
def xgb(**kw):
    p = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, random_state=SEED, objective="multi:softprob",
             num_class=4, eval_metric="mlogloss", n_jobs=-1)
    p.update(kw); return XGBClassifier(**p)

def rf(**kw):
    p = dict(random_state=SEED, class_weight="balanced", n_estimators=300, n_jobs=-1)
    p.update(kw); return RandomForestClassifier(**p)

CONFIGS = {
    # references (baseline v3 models)
    "xgb_base":        dict(mk=lambda: xgb(), w=False, sm=None),
    "rf_bal":          dict(mk=lambda: rf(), w=False, sm=None),
    "logreg_bal":      dict(mk=lambda: SkPipeline([("sc", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=5000, random_state=SEED,
                                                       class_weight="balanced"))]),
                            w=False, sm=None),
    # class-weighted XGBoost family
    "xgb_bal":         dict(mk=lambda: xgb(), w=True, sm=None),
    "xgb_bal_d4_lr01": dict(mk=lambda: xgb(max_depth=4, learning_rate=0.1,
                                            n_estimators=600), w=True, sm=None),
    "xgb_bal_d8":      dict(mk=lambda: xgb(max_depth=8), w=True, sm=None),
    "xgb_bal_mcw10":   dict(mk=lambda: xgb(min_child_weight=10), w=True, sm=None),
    "xgb_bal_sub06":   dict(mk=lambda: xgb(subsample=0.6, colsample_bytree=0.6),
                            w=True, sm=None),
    # RF variants
    "rf_balsub_600":   dict(mk=lambda: rf(class_weight="balanced_subsample",
                                           n_estimators=600), w=False, sm=None),
    "rf_bal_leaf5":    dict(mk=lambda: rf(min_samples_leaf=5, n_estimators=600),
                            w=False, sm=None),
    # resampling inside folds only
    "xgb_smote":       dict(mk=lambda: xgb(), w=False, sm="smote"),
    "rf_smote":        dict(mk=lambda: RandomForestClassifier(random_state=SEED,
                                        n_estimators=300, n_jobs=-1), w=False, sm="smote"),
    "xgb_smotetomek":  dict(mk=lambda: xgb(), w=False, sm="smotetomek"),
}

def fit_predict_proba(cfg, Xtr, ytr, Xev):
    Xf, yf = Xtr, ytr
    if cfg["sm"] == "smote":
        Xf, yf = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    elif cfg["sm"] == "smotetomek":
        Xf, yf = SMOTETomek(random_state=SEED).fit_resample(Xtr, ytr)
    m = cfg["mk"]()
    if cfg["w"]:
        m.fit(Xf, yf, sample_weight=compute_sample_weight("balanced", yf))
    else:
        m.fit(Xf, yf)
    return m, m.predict_proba(Xev), yf

def tau_predict(proba, priors, tau):
    adj = proba * (1.0 / priors) ** tau
    return np.argmax(adj, axis=1)

def priors_of(y):
    p = np.bincount(y, minlength=4).astype(float)
    return p / p.sum()

def inner_cv(cfg, Xtr, ytr, gtr, n_splits=3):
    """GroupKFold CV inside the training side; returns per-tau mean scores."""
    gkf = GroupKFold(n_splits=n_splits)
    f1s = {t: [] for t in TAUS}; accs = {t: [] for t in TAUS}; wrec = {t: [] for t in TAUS}
    for tr, va in gkf.split(Xtr, ytr, gtr):
        Xa, ya = Xtr.iloc[tr], ytr.iloc[tr]
        Xb, yb = Xtr.iloc[va], ytr.iloc[va]
        _, proba, yfit = fit_predict_proba(cfg, Xa, ya, Xb)
        pri = priors_of(yfit)
        for t in TAUS:
            pred = tau_predict(proba, pri, t)
            f1s[t].append(f1_score(yb, pred, average="macro"))
            accs[t].append(accuracy_score(yb, pred))
            wrec[t].append(recall_score(yb, pred, labels=[0], average=None)[0])
    out = {}
    for t in TAUS:
        out[t] = dict(macro_f1=float(np.mean(f1s[t])), acc=float(np.mean(accs[t])),
                      withdrawn_recall=float(np.mean(wrec[t])))
    best_t = max(TAUS, key=lambda t: out[t]["macro_f1"])
    return best_t, out

def evaluate(cfg, tau, Xtr, ytr, Xte, yte):
    m, proba, yfit = fit_predict_proba(cfg, Xtr, ytr, Xte)
    pred = tau_predict(proba, priors_of(yfit), tau)
    rep = classification_report(yte, pred, output_dict=True, zero_division=0,
                                target_names=["Withdrawn", "Fail", "Pass", "Distinction"])
    return m, dict(accuracy=float(accuracy_score(yte, pred)),
                   macro_f1=float(f1_score(yte, pred, average="macro")),
                   withdrawn_recall=float(recall_score(yte, pred, labels=[0], average=None)[0]),
                   per_class={k: {kk: float(vv) for kk, vv in v.items()}
                              for k, v in rep.items() if isinstance(v, dict)},
                   confusion=confusion_matrix(yte, pred).tolist())

results = {"protocol": {
    "split": "GroupShuffleSplit(group=id_student, test_size=0.2, seed=42); test never used for tuning",
    "tuning": "GroupKFold(3) inside seed-42 train; tau (class-prior adjustment) tuned there too",
    "repeats": "GroupShuffleSplit seeds 0-4, config+tau frozen",
    "taus": TAUS}, "screening_c30": {}, "cutoffs": {}}

# ---------------- 1) screening at cutoff 30 ----------------
df, X, y, g = load(30)
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, g))
Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], g[tr]
print("[screen] cutoff 30 train:", Xtr.shape, file=sys.stderr)
screen = {}
for name, cfg in CONFIGS.items():
    best_t, table = inner_cv(cfg, Xtr, ytr, gtr)
    screen[name] = dict(best_tau=best_t, at_best=table[best_t], tau_table=table)
    print(f"[screen] {name:18s} tau*={best_t:4} innerF1={table[best_t]['macro_f1']:.4f} "
          f"acc={table[best_t]['acc']:.4f} Wrec={table[best_t]['withdrawn_recall']:.4f}",
          file=sys.stderr)
results["screening_c30"] = screen
shortlist = sorted((k for k in CONFIGS), key=lambda k: -screen[k]["at_best"]["macro_f1"])[:3]
print("[screen] shortlist:", shortlist, file=sys.stderr)
results["shortlist"] = shortlist

# ---------------- 2) per-cutoff selection + final + repeats ----------------
for C in CUTOFFS:
    df, X, y, g = load(C)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, g))
    Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], g[tr]
    Xte, yte = X.iloc[te], y.iloc[te]
    R = {"n": int(len(X)), "target_dist": {int(k): int(v) for k, v in y.value_counts().sort_index().items()}}

    # per-cutoff winner selection on inner CV (shortlist only; c30 reuses screening)
    sel = {}
    for name in shortlist:
        if C == 30:
            sel[name] = dict(best_tau=screen[name]["best_tau"],
                             macro_f1=screen[name]["at_best"]["macro_f1"])
        else:
            bt, tab = inner_cv(CONFIGS[name], Xtr, ytr, gtr)
            sel[name] = dict(best_tau=bt, macro_f1=tab[bt]["macro_f1"])
    winner = max(sel, key=lambda k: sel[k]["macro_f1"])
    wtau = sel[winner]["best_tau"]
    R["selection"] = sel; R["winner"] = winner; R["winner_tau"] = wtau

    # final on official held-out test
    wm, R["winner_test"] = evaluate(CONFIGS[winner], wtau, Xtr, ytr, Xte, yte)
    _, R["baseline_xgb_test"] = evaluate(CONFIGS["xgb_base"], 0.0, Xtr, ytr, Xte, yte)
    _, R["baseline_rf_test"] = evaluate(CONFIGS["rf_bal"], 0.0, Xtr, ytr, Xte, yte)

    # feature importance of the winner (gain-based for XGB, impurity for RF)
    try:
        imp = getattr(wm, "feature_importances_", None)
        if imp is None and hasattr(wm, "named_steps"):
            imp = None
        if imp is not None:
            R["winner_importance"] = sorted(
                ({"feature": f, "importance": float(v)} for f, v in zip(X.columns, imp)),
                key=lambda d: -d["importance"])
    except Exception as e:
        R["winner_importance_error"] = str(e)

    # repeats seeds 0-4, everything frozen; paired baseline on identical splits
    reps = {"winner": [], "baseline_xgb": [], "baseline_rf": []}
    for seed in range(5):
        t0, t1 = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, g))
        Xa, ya = X.iloc[t0], y.iloc[t0]; Xb, yb = X.iloc[t1], y.iloc[t1]
        _, rw = evaluate(CONFIGS[winner], wtau, Xa, ya, Xb, yb)
        _, rx = evaluate(CONFIGS["xgb_base"], 0.0, Xa, ya, Xb, yb)
        _, rr = evaluate(CONFIGS["rf_bal"], 0.0, Xa, ya, Xb, yb)
        for k, r in [("winner", rw), ("baseline_xgb", rx), ("baseline_rf", rr)]:
            reps[k].append({kk: r[kk] for kk in ("accuracy", "macro_f1", "withdrawn_recall")})
    def agg(rs, key):
        v = [r[key] for r in rs]; return dict(mean=float(np.mean(v)), std=float(np.std(v)),
                                              all=[round(x, 4) for x in v])
    R["repeats"] = {k: {m: agg(v, m) for m in ("accuracy", "macro_f1", "withdrawn_recall")}
                    for k, v in reps.items()}
    d = [reps["winner"][i]["macro_f1"] - reps["baseline_xgb"][i]["macro_f1"] for i in range(5)]
    R["paired_delta_f1_vs_xgb"] = dict(per_seed=[round(x, 4) for x in d],
                                       mean=float(np.mean(d)), std=float(np.std(d)),
                                       all_positive=bool(all(x > 0 for x in d)),
                                       sign_test_p_one_sided=float(0.5 ** 5) if all(x > 0 for x in d) else None)
    d2 = [reps["winner"][i]["macro_f1"] - reps["baseline_rf"][i]["macro_f1"] for i in range(5)]
    R["paired_delta_f1_vs_rf"] = dict(per_seed=[round(x, 4) for x in d2],
                                      mean=float(np.mean(d2)), std=float(np.std(d2)),
                                      all_positive=bool(all(x > 0 for x in d2)))
    results["cutoffs"][str(C)] = R
    print(f"[final] c{C}: winner={winner} tau={wtau} "
          f"testF1={R['winner_test']['macro_f1']:.4f} (xgb_base {R['baseline_xgb_test']['macro_f1']:.4f}) "
          f"repF1={R['repeats']['winner']['macro_f1']['mean']:.4f}±{R['repeats']['winner']['macro_f1']['std']:.4f} "
          f"Wrec={R['repeats']['winner']['withdrawn_recall']['mean']:.4f}", file=sys.stderr)

json.dump(results, open(OUT, "w"), indent=2)
print("WROTE", OUT, file=sys.stderr)
