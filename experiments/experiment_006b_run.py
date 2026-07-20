"""Experiment 006 — full-course leakage-free prediction: multiclass, binary, regression.

Protocol: grouped by id_student everywhere (GroupShuffleSplit 80/20 seed 42 headline;
repeated grouped splits seeds 0-4; GroupKFold(5) for multiclass XGB/RF robustness).
Classification models = official v4 models verbatim. Regressors mirror them.
"""
import json, os, time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve,
                             mean_absolute_error, mean_squared_error, r2_score)

SP = os.environ["SP"]
SEEDS = [0, 1, 2, 3, 4]
CLASSES = ["Withdrawn", "Fail", "Pass", "Distinction"]

df = pd.read_parquet(f"{SP}/exp006b_frame.parquet")
meta = json.load(open(f"{SP}/exp006b_meta.json"))
CF, RF_ = meta["cls_features"], meta["reg_features"]
y = df["target_multi"].values
g = df["id_student"].values
X = df[CF]

def cls_models():
    return {
        "logreg": SkPipeline([("scaler", StandardScaler()),
                              ("logreg", LogisticRegression(max_iter=5000, random_state=42,
                                                            class_weight="balanced"))]),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "random_forest": RandomForestClassifier(random_state=42, class_weight="balanced",
                                                n_estimators=300, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 objective="multi:softmax", num_class=4,
                                 eval_metric="mlogloss", n_jobs=-1),
    }

def reg_models():
    return {
        "linear": SkPipeline([("scaler", StandardScaler()), ("lin", LinearRegression())]),
        "rf_reg": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "xgb_reg": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=42,
                                n_jobs=-1),
    }

results = {"meta": {k: meta[k] for k in ["n_rows", "n_cls_features", "n_reg_features",
                                         "regression_target", "n_regression_rows", "audit"]},
           "multiclass": {}, "binary": {}, "regression": {}}
arrays = {}

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=g))
assert len(set(g[tr]) & set(g[te])) == 0
seed_splits = {}
for s in SEEDS:
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s).split(X, y, groups=g))
    assert len(set(g[a]) & set(g[b])) == 0
    seed_splits[s] = (a, b)

# ================= 1. MULTICLASS =================
t0 = time.time()
for name, m in cls_models().items():
    m.fit(X.iloc[tr], y[tr]); p = m.predict(X.iloc[te])
    rep = classification_report(y[te], p, output_dict=True, zero_division=0, target_names=CLASSES)
    r = {"accuracy": float(accuracy_score(y[te], p)),
         "macro_precision": float(precision_score(y[te], p, average="macro", zero_division=0)),
         "macro_recall": float(recall_score(y[te], p, average="macro", zero_division=0)),
         "macro_f1": float(f1_score(y[te], p, average="macro")),
         "per_class": {k: {kk: float(vv) for kk, vv in v.items()}
                       for k, v in rep.items() if isinstance(v, dict)},
         "confusion": confusion_matrix(y[te], p).tolist(), "repeats": []}
    for s in SEEDS:
        a, b = seed_splits[s]
        ms = cls_models()[name]; ms.fit(X.iloc[a], y[a]); ps = ms.predict(X.iloc[b])
        r["repeats"].append({"seed": s, "accuracy": float(accuracy_score(y[b], ps)),
                             "macro_f1": float(f1_score(y[b], ps, average="macro"))})
    if name in ("random_forest", "xgboost"):
        r["gkf5"] = []
        for i, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g)):
            mk = cls_models()[name]; mk.fit(X.iloc[a], y[a]); pk = mk.predict(X.iloc[b])
            r["gkf5"].append({"fold": i, "accuracy": float(accuracy_score(y[b], pk)),
                              "macro_f1": float(f1_score(y[b], pk, average="macro"))})
        imp = (m.feature_importances_ if name == "random_forest"
               else m.feature_importances_)
        r["feature_importance"] = {f: float(v) for f, v in
                                   sorted(zip(CF, imp), key=lambda kv: -kv[1])}
    results["multiclass"][name] = r
    print(f"[multi] {name:15s} acc={r['accuracy']:.4f} F1={r['macro_f1']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
arrays["multi_confusion_xgb"] = np.array(results["multiclass"]["xgboost"]["confusion"])

# ================= 2. BINARY =================
TASKS = {
    "withdrawn_vs_rest": (y == 0).astype(int),
    "fail_vs_rest": (y == 1).astype(int),
    "pass_vs_rest": (y == 2).astype(int),
    "distinction_vs_rest": (y == 3).astype(int),
    "atrisk_WF_vs_PD": (y <= 1).astype(int),
}
# note: Completed-vs-Withdrawn is the label complement of withdrawn_vs_rest — identical task.
for tname, yb in TASKS.items():
    results["binary"][tname] = {"prevalence": float(yb[te].mean()), "models": {}}
    for name, m in cls_models().items():
        if name == "xgboost":
            m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              eval_metric="logloss", n_jobs=-1)
        m.fit(X.iloc[tr], yb[tr])
        p = m.predict(X.iloc[te]); pr = m.predict_proba(X.iloc[te])[:, 1]
        r = {"accuracy": float(accuracy_score(yb[te], p)),
             "precision": float(precision_score(yb[te], p, zero_division=0)),
             "recall": float(recall_score(yb[te], p, zero_division=0)),
             "f1": float(f1_score(yb[te], p, zero_division=0)),
             "roc_auc": float(roc_auc_score(yb[te], pr)),
             "pr_auc": float(average_precision_score(yb[te], pr)),
             "confusion": confusion_matrix(yb[te], p).tolist()}
        results["binary"][tname]["models"][name] = r
        if name == "xgboost":
            fpr, tpr, _ = roc_curve(yb[te], pr)
            prec, rec, _ = precision_recall_curve(yb[te], pr)
            arrays[f"roc_{tname}"] = np.stack([fpr, tpr])
            arrays[f"pr_{tname}"] = np.stack([rec, prec])
            reps = []
            for s in SEEDS:
                a, b = seed_splits[s]
                ms = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   eval_metric="logloss", n_jobs=-1)
                ms.fit(X.iloc[a], yb[a]); prs = ms.predict_proba(X.iloc[b])[:, 1]
                reps.append({"seed": s, "roc_auc": float(roc_auc_score(yb[b], prs)),
                             "pr_auc": float(average_precision_score(yb[b], prs))})
            results["binary"][tname]["xgb_repeats"] = reps
    x = results["binary"][tname]["models"]["xgboost"]
    print(f"[bin] {tname:22s} XGB acc={x['accuracy']:.4f} F1={x['f1']:.4f} "
          f"AUC={x['roc_auc']:.4f} PR-AUC={x['pr_auc']:.4f}", flush=True)

# ================= 3. REGRESSION =================
rmask = df["final_cw_score"].notna().values
Xr, yr, gr = df.loc[rmask, RF_], df.loc[rmask, "final_cw_score"].values, g[rmask]
tr_r, te_r = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                  .split(Xr, yr, groups=gr))
assert len(set(gr[tr_r]) & set(gr[te_r])) == 0
results["regression"]["n_rows"] = int(rmask.sum())
results["regression"]["target_stats"] = {"mean": float(yr.mean()), "std": float(yr.std())}
results["regression"]["models"] = {}
for name, m in reg_models().items():
    m.fit(Xr.iloc[tr_r], yr[tr_r]); p = m.predict(Xr.iloc[te_r])
    r = {"mae": float(mean_absolute_error(yr[te_r], p)),
         "rmse": float(np.sqrt(mean_squared_error(yr[te_r], p))),
         "r2": float(r2_score(yr[te_r], p)), "repeats": []}
    for s in SEEDS:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
                    .split(Xr, yr, groups=gr))
        ms = reg_models()[name]; ms.fit(Xr.iloc[a], yr[a]); ps = ms.predict(Xr.iloc[b])
        r["repeats"].append({"seed": s, "mae": float(mean_absolute_error(yr[b], ps)),
                             "rmse": float(np.sqrt(mean_squared_error(yr[b], ps))),
                             "r2": float(r2_score(yr[b], ps))})
    if name in ("rf_reg", "xgb_reg"):
        r["feature_importance"] = {f: float(v) for f, v in
                                   sorted(zip(RF_, m.feature_importances_), key=lambda kv: -kv[1])}
    results["regression"]["models"][name] = r
    print(f"[reg] {name:10s} MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} R2={r['r2']:.4f}", flush=True)
    if name == "xgb_reg":
        arrays["reg_actual"] = yr[te_r]; arrays["reg_pred"] = p

json.dump(results, open(f"{SP}/exp006b_results.json", "w"), indent=1)
np.savez_compressed(f"{SP}/exp006b_arrays.npz", **arrays)
print("saved results + arrays")
