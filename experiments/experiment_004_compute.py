"""Experiment 004 — progressive prediction analysis on a fixed student panel.

Evaluation/decision layer only: official Baseline v4 XGB (verbatim hyperparameters),
p4 cache, no feature/model/cache changes.

Fixed student panel: one global grouped 80/20 split of the UNION of student IDs across
all cutoffs (rng seed 42). Every test student is out-of-sample at every cutoff, so
individual risk trajectories can be tracked across cutoffs without leakage. Grouped by
construction (the split IS at student level).

Per cutoff: fit official XGB on train-panel rows, predict test-panel rows; isotonic
calibration of ASI fitted on inner GroupKFold(3) out-of-fold train predictions only.
"""
import json, os, time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss)
from sklearn.isotonic import IsotonicRegression

REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
SP = os.environ["SP"]
CUTS = [14, 30, 60, 90, 140]
KS = [0.05, 0.10, 0.20]

def xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, objective="multi:softmax",
                         num_class=4, eval_metric="mlogloss", n_jobs=-1)

def ece(prob, truth, n_bins=10):
    prob = np.clip(prob, 0, 1)
    idx = np.clip(np.digitize(prob, np.linspace(0, 1, n_bins + 1)) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += (m.sum() / len(prob)) * abs(prob[m].mean() - truth[m].mean())
    return float(e)

def topk(score, truth, k):
    n = len(score)
    m = max(1, int(round(k * n)))
    order = np.argsort(-score, kind="stable")
    top = truth[order[:m]]
    return {"precision": float(top.mean()), "recall": float(top.sum() / max(1, truth.sum())),
            "lift": float(top.mean() / truth.mean()), "hits": int(top.sum()), "n": int(m)}

# ---------- fixed student panel ----------
frames = {c: pd.read_parquet(f"{REPO}/data/processed/p4/c{c:03d}/mlDataV4.parquet")
          for c in CUTS}
feats = {c: json.load(open(f"{REPO}/data/processed/p4/c{c:03d}/manifest.json"))["active_features_v4"]
         for c in CUTS}
all_students = np.array(sorted(set().union(*[set(frames[c]["id_student"]) for c in CUTS])))
rng = np.random.default_rng(42)
perm = rng.permutation(len(all_students))
n_test = int(round(0.2 * len(all_students)))
test_students = set(all_students[perm[:n_test]])
print(f"panel: {len(all_students)} students total, {len(test_students)} test")

meta = {"panel": {"n_students": int(len(all_students)), "n_test_students": int(len(test_students)),
                  "rule": "global grouped 80/20 split of the union of id_student (rng seed 42); "
                          "identical membership at every cutoff"}}
results = {"meta": meta, "cutoffs": {}}
records = []

for c in CUTS:
    t0 = time.time()
    df = frames[c]
    F = feats[c]
    is_te = df["id_student"].isin(test_students).values
    tr, te = np.where(~is_te)[0], np.where(is_te)[0]
    assert len(set(df["id_student"].values[tr]) & set(df["id_student"].values[te])) == 0
    X, y = df[F], df["target_multi"].values

    m = xgb()
    m.fit(X.iloc[tr], y[tr])
    P = m.predict_proba(X.iloc[te])
    assert np.allclose(P.sum(axis=1), 1, atol=1e-5)
    y_te = y[te]
    pred = P.argmax(axis=1)

    # inner OOF isotonic (train panel only, grouped by student)
    gkf = GroupKFold(n_splits=3)
    Xtr, ytr, gtr = X.iloc[tr], y[tr], df["id_student"].values[tr]
    oof = np.full(len(tr), np.nan)
    for itr, iva in gkf.split(Xtr, ytr, groups=gtr):
        mm = xgb()
        mm.fit(Xtr.iloc[itr], ytr[itr])
        Pv = mm.predict_proba(Xtr.iloc[iva])
        oof[iva] = Pv[:, 0] + Pv[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, (ytr <= 1).astype(int))

    asi = P[:, 0] + P[:, 1]
    wri = P[:, 0]
    iso_asi = iso.predict(asi)
    Pc = np.clip(P, 1e-12, 1)
    ent = -(Pc * np.log(Pc)).sum(axis=1) / np.log(4)
    atrisk = (y_te <= 1).astype(int)
    withdrawn = (y_te == 0).astype(int)

    # bands within this cutoff's test ranking: red = top 5%, amber = 5-20%, green = rest
    pct = pd.Series(asi).rank(pct=True, method="first").values
    band = np.where(pct >= 0.95, "red", np.where(pct >= 0.80, "amber", "green"))

    # accuracy by entropy quartile (confidence evidence)
    q = pd.qcut(ent, 4, labels=["q1_lowH", "q2", "q3", "q4_highH"])
    acc_by_ent = {str(k): float((pred[q == k] == y_te[q == k]).mean()) for k in q.categories}
    # band boundary values on the calibrated scale
    red_line = float(np.quantile(asi, 0.95)); amber_line = float(np.quantile(asi, 0.80))

    cres = {
        "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "prevalence_atrisk": float(atrisk.mean()),
        "prevalence_withdrawn": float(withdrawn.mean()),
        "ASI": {"roc_auc": float(roc_auc_score(atrisk, asi)),
                "avg_precision": float(average_precision_score(atrisk, asi)),
                "brier": float(brier_score_loss(atrisk, asi)), "ece": ece(asi, atrisk),
                **{f"top{int(k*100)}": topk(asi, atrisk, k) for k in KS}},
        "ISO_ASI": {"brier": float(brier_score_loss(atrisk, iso_asi)),
                    "ece": ece(iso_asi, atrisk)},
        "WRI": {"roc_auc": float(roc_auc_score(withdrawn, wri)),
                "avg_precision": float(average_precision_score(withdrawn, wri)),
                **{f"top{int(k*100)}": topk(wri, withdrawn, k) for k in KS}},
        "uncertainty": {"mean_entropy": float(ent.mean()),
                        "share_confident": float((ent < 0.5).mean()),
                        "accuracy_by_entropy_quartile": acc_by_ent},
        "band_lines_raw_ASI": {"red_top5": red_line, "amber_top20": amber_line},
        "band_lines_calibrated": {"red_top5": float(iso.predict([red_line])[0]),
                                  "amber_top20": float(iso.predict([amber_line])[0])},
        "band_outcome_rates": {},
    }
    for b in ["red", "amber", "green"]:
        mk = band == b
        cres["band_outcome_rates"][b] = {
            "n": int(mk.sum()),
            "P_adverse": float(atrisk[mk].mean()),
            "P_withdrawn": float(withdrawn[mk].mean()),
            "P_fail": float((y_te[mk] == 1).mean()),
        }
    results["cutoffs"][f"c{c:03d}"] = cres

    rec = df.iloc[te][["id_student", "code_module", "code_presentation", "final_result",
                       "date_registration", "date_unregistration"]].copy()
    rec["cutoff"] = c
    rec["ASI"] = asi; rec["ISO_ASI"] = iso_asi; rec["WRI"] = wri
    rec["entropy"] = ent; rec["band"] = band; rec["pred"] = pred
    records.append(rec)
    print(f"[c{c:03d}] acc={cres['accuracy']:.4f} F1={cres['macro_f1']:.4f} "
          f"ASI AUC={cres['ASI']['roc_auc']:.4f} ECE={cres['ASI']['ece']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

long = pd.concat(records, ignore_index=True)
long.to_parquet(f"{SP}/exp004_panel_long.parquet")
json.dump(results, open(f"{SP}/exp004_progressive.json", "w"), indent=1)

# sanity: compare panel headline vs official GSS-42 baseline (should sit within split noise)
off = json.load(open(f"{REPO}/reports/baseline_v4_results.json"))
print("\nsanity vs official GSS-42 (XGB acc/F1):")
for c in CUTS:
    o = off[str(c)]["v4"]["models"]["xgboost"]
    print(f"  c{c:03d}: panel {results['cutoffs'][f'c{c:03d}']['accuracy']:.4f}/"
          f"{results['cutoffs'][f'c{c:03d}']['macro_f1']:.4f} vs official "
          f"{o['accuracy']:.4f}/{o['macro_f1']:.4f}")
print("done")
