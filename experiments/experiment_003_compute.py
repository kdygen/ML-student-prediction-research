"""Experiment 003 — Early Intervention Index: compute candidate indices + ranking metrics.

Evaluation-layer only: official Baseline v4 models/hyperparameters/features, p4 cache,
grouped splits. Nothing cached or official is modified.

Discipline:
- Headline split: GroupShuffleSplit(test_size=0.2, random_state=42), groups=id_student.
- Robustness: repeated grouped splits, seeds 0-4.
- Any tunable index choice (weight w in w*P(W)+(1-w)*P(F)) is selected on inner
  GroupKFold(3) out-of-fold predictions within the TRAIN side only, then evaluated once
  on test. Isotonic calibration is fitted on inner OOF scores only.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
SP = os.environ["SP"]
CUTS = [14, 30, 60, 90, 140]
SEEDS = [0, 1, 2, 3, 4]
KS = [0.05, 0.10, 0.20]
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def official_models():
    # verbatim from the Baseline v4 notebook cell (n_jobs only parallelizes RF; results identical)
    return {
        "logreg": SkPipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=5000, random_state=42, class_weight="balanced")),
        ]),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "random_forest": RandomForestClassifier(
            random_state=42, class_weight="balanced", n_estimators=300, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, objective="multi:softmax",
            num_class=4, eval_metric="mlogloss", n_jobs=-1),
    }

def topk_metrics(score, truth, ks=KS):
    n = len(score)
    order = np.argsort(-score, kind="stable")
    base = truth.mean()
    out = {}
    for k in ks:
        m = max(1, int(round(k * n)))
        top = truth[order[:m]]
        p_at = float(top.mean())
        out[f"precision_at_{int(k*100)}"] = p_at
        out[f"recall_at_{int(k*100)}"] = float(top.sum() / max(1, truth.sum()))
        out[f"lift_at_{int(k*100)}"] = float(p_at / base) if base > 0 else None
        out[f"hits_at_{int(k*100)}"] = int(top.sum())
        out[f"n_at_{int(k*100)}"] = int(m)
    return out

def ece(prob, truth, n_bins=10):
    prob = np.clip(prob, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    e, n = 0.0, len(prob)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(prob[m].mean() - truth[m].mean())
    return float(e)

def rank_metrics(score, truth, prob_scale=False):
    d = {
        "roc_auc": float(roc_auc_score(truth, score)),
        "avg_precision": float(average_precision_score(truth, score)),
        "prevalence": float(truth.mean()),
    }
    d.update(topk_metrics(score, truth))
    if prob_scale:
        d["brier"] = float(brier_score_loss(truth, np.clip(score, 0, 1)))
        d["ece_10bin"] = ece(score, truth)
    return d

def entropy4(P):
    Pc = np.clip(P, 1e-12, 1)
    return -(Pc * np.log(Pc)).sum(axis=1) / np.log(4)

def indices_from_probs(P):
    """Fixed-formula candidate indices from a (n,4) prob matrix.
    Classes: 0=Withdrawn 1=Fail 2=Pass 3=Distinction."""
    H = entropy4(P)
    asi = P[:, 0] + P[:, 1]
    return {
        "WRI":      P[:, 0],                       # withdrawal risk
        "FRI":      P[:, 1],                       # failure risk
        "ASI":      asi,                           # academic support index = P(adverse)
        "SEV2":     2 * P[:, 0] + P[:, 1],         # severity-weighted (withdrawal x2)
        "ENT_ASI":  asi * (1 - H),                 # confidence(entropy)-adjusted ASI
    }

TRUTHS = {"atrisk": lambda y: (y <= 1).astype(int),
          "withdrawn": lambda y: (y == 0).astype(int),
          "fail": lambda y: (y == 1).astype(int)}
PROB_SCALE = {"WRI", "FRI", "ASI", "ISO_ASI"}   # indices on a probability scale

results = {"meta": {
    "experiment": "003_intervention_index",
    "models": "official Baseline v4 (verbatim hyperparameters)",
    "features": "official v4 active set (35), p4 cache",
    "split": "GroupShuffleSplit(group=id_student, test_size=0.2), headline seed 42, repeats 0-4",
    "truth_definitions": {"atrisk": "final_result in {Withdrawn, Fail}",
                          "withdrawn": "final_result == Withdrawn",
                          "fail": "final_result == Fail"},
    "class_order": ["Withdrawn", "Fail", "Pass", "Distinction"],
    "ks": KS, "w_grid": W_GRID,
}, "cutoffs": {}}

t00 = time.time()
for c in CUTS:
    t0 = time.time()
    man = json.load(open(f"{REPO}/data/processed/p4/c{c:03d}/manifest.json"))
    feats = man["active_features_v4"]
    df = pd.read_parquet(f"{REPO}/data/processed/p4/c{c:03d}/mlDataV4.parquet")
    X, y, g = df[feats], df["target_multi"].values, df["id_student"].values
    course = (df["code_module"].astype(str) + "_" + df["code_presentation"].astype(str)).values

    cres = {"n_rows": int(len(df)), "headline": {}, "repeats": {}}

    # ---------- headline split (seed 42) ----------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups=g))
    assert len(set(g[tr]) & set(g[te])) == 0
    y_te, course_te = y[te], course[te]
    truths = {k: f(y_te) for k, f in TRUTHS.items()}

    probs = {}
    models = official_models()
    for name, m in models.items():
        m.fit(X.iloc[tr], y[tr])
        P = m.predict_proba(X.iloc[te])
        assert P.shape[1] == 4 and np.allclose(P.sum(axis=1), 1, atol=1e-5)
        probs[name] = P

    # inner OOF (train-only) for XGB: weight selection + isotonic calibration
    gkf = GroupKFold(n_splits=3)
    Xtr, ytr, gtr = X.iloc[tr], y[tr], g[tr]
    oof = np.full((len(tr), 4), np.nan)
    for itr, iva in gkf.split(Xtr, ytr, groups=gtr):
        mm = official_models()["xgboost"]
        mm.fit(Xtr.iloc[itr], ytr[itr])
        oof[iva] = mm.predict_proba(Xtr.iloc[iva])
    assert not np.isnan(oof).any()
    oof_truth = {k: f(ytr) for k, f in TRUTHS.items()}

    # select w per truth on inner OOF (max average precision)
    w_sel = {}
    for tname in TRUTHS:
        aps = [average_precision_score(oof_truth[tname], w * oof[:, 0] + (1 - w) * oof[:, 1])
               for w in W_GRID]
        w_sel[tname] = {"w": float(W_GRID[int(np.argmax(aps))]),
                        "inner_ap_curve": [float(a) for a in aps]}

    # isotonic calibration of ASI (fit on inner OOF only)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof[:, 0] + oof[:, 1], oof_truth["atrisk"])

    # ---------- assemble candidate indices on the headline test ----------
    per_model = {}
    for name, P in probs.items():
        per_model[name] = indices_from_probs(P)
    # XGB-only extras
    Px = probs["xgboost"]
    xgb_idx = per_model["xgboost"]
    xgb_idx["WSEL_atrisk"] = w_sel["atrisk"]["w"] * Px[:, 0] + (1 - w_sel["atrisk"]["w"]) * Px[:, 1]
    xgb_idx["ISO_ASI"] = iso.predict(Px[:, 0] + Px[:, 1])
    # course-normalized percentile of ASI (within module_presentation, on test)
    s = pd.Series(Px[:, 0] + Px[:, 1])
    xgb_idx["ASI_COURSE_PCT"] = s.groupby(pd.Series(course_te)).rank(pct=True).values
    # ensemble across the 4 official models
    asi_stack = np.stack([per_model[m]["ASI"] for m in probs], axis=0)
    xgb_idx["ENS_ASI"] = asi_stack.mean(axis=0)
    xgb_idx["ENS_ASI_MINUS_STD"] = asi_stack.mean(axis=0) - asi_stack.std(axis=0)

    hl = {"w_selected": w_sel, "models": {}}
    for name in per_model:
        hl["models"][name] = {}
        for iname, score in per_model[name].items():
            hl["models"][name][iname] = {
                t: rank_metrics(score, truths[t], prob_scale=(iname in PROB_SCALE))
                for t in truths}
    cres["headline"] = hl

    # arrays for figures (XGB + ensemble indices, truths, probs)
    np.savez_compressed(
        f"{SP}/exp003_arrays_c{c:03d}.npz",
        y_test=y_te, course_test=course_te, P_xgb=Px,
        P_rf=probs["random_forest"], P_lr=probs["logreg"], P_dt=probs["decision_tree"],
        **{f"idx_{k}": v for k, v in xgb_idx.items()})

    # ---------- repeats: seeds 0-4, fixed-formula indices, all models ----------
    for seed in SEEDS:
        gss_s = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        trs, tes = next(gss_s.split(X, y, groups=g))
        assert len(set(g[trs]) & set(g[tes])) == 0
        y_s = y[tes]
        truths_s = {k: f(y_s) for k, f in TRUTHS.items()}
        rep = {}
        probs_s = {}
        for name, m in official_models().items():
            m.fit(X.iloc[trs], y[trs])
            probs_s[name] = m.predict_proba(X.iloc[tes])
        for name, P in probs_s.items():
            rep[name] = {}
            for iname, score in indices_from_probs(P).items():
                rep[name][iname] = {
                    t: {"roc_auc": float(roc_auc_score(truths_s[t], score)),
                        "avg_precision": float(average_precision_score(truths_s[t], score)),
                        **{k2: v2 for k2, v2 in topk_metrics(score, truths_s[t]).items()
                           if k2.startswith(("precision_at", "recall_at"))}}
                    for t in truths_s}
        # ensemble ASI in repeats too
        asi_stack_s = np.stack([indices_from_probs(P)["ASI"] for P in probs_s.values()], axis=0)
        rep["ensemble"] = {"ENS_ASI": {
            t: {"roc_auc": float(roc_auc_score(truths_s[t], asi_stack_s.mean(axis=0))),
                "avg_precision": float(average_precision_score(truths_s[t], asi_stack_s.mean(axis=0))),
                **{k2: v2 for k2, v2 in topk_metrics(asi_stack_s.mean(axis=0), truths_s[t]).items()
                   if k2.startswith(("precision_at", "recall_at"))}}
            for t in truths_s}}
        cres["repeats"][f"seed{seed}"] = rep

    results["cutoffs"][f"c{c:03d}"] = cres
    print(f"[c{c:03d}] done in {time.time()-t0:.0f}s "
          f"(XGB ASI atrisk AUC={hl['models']['xgboost']['ASI']['atrisk']['roc_auc']:.4f} "
          f"AP={hl['models']['xgboost']['ASI']['atrisk']['avg_precision']:.4f})", flush=True)

json.dump(results, open(f"{SP}/exp003_results_raw.json", "w"), indent=1)
print(f"TOTAL {time.time()-t00:.0f}s -> {SP}/exp003_results_raw.json")
