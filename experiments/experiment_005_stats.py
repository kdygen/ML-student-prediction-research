"""Experiment 005 — paired significance tests vs A0_v4.

Splits are identical across arms (same groups, same seeds/folds), so all comparisons are
paired. Per arm we report, for macro-F1 / accuracy / Withdrawn recall / at-risk AUC:
  - per-cutoff seed-paired deltas (n=5) and GKF fold-paired deltas (n=5, F1/acc only);
  - pooled across cutoffs (n=25 seed pairs; n=25 fold pairs);
  - paired t-test, Wilcoxon signed-rank, and exact sign counts.
n is small per cutoff; the pooled tests and sign consistency carry the inference weight.
"""
import json, os, sys
import numpy as np
from scipy import stats

SP = os.environ["SP"]
raw = json.load(open(f"{SP}/exp005_raw.json"))
CUTS = sorted(raw["cutoffs"].keys())
ARMS = [a for a in raw["cutoffs"][CUTS[0]] if a != "A0_v4"]

def paired(arm, metric, kind):
    """list of (cutoff, deltas array) using seed- or fold-paired obs"""
    out = []
    for c in CUTS:
        a0 = raw["cutoffs"][c]["A0_v4"][kind]
        a1 = raw["cutoffs"][c][arm][kind]
        if metric not in a0[0]:
            continue
        d = np.array([x[metric] for x in a1]) - np.array([x[metric] for x in a0])
        out.append((c, d))
    return out

def tests(d):
    d = np.asarray(d)
    r = {"n": int(len(d)), "mean_delta": float(d.mean()), "std": float(d.std(ddof=1)),
         "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum())}
    if len(d) >= 2 and d.std(ddof=1) > 0:
        t, pt = stats.ttest_rel(d, np.zeros_like(d))  # equivalent to 1-sample t on deltas
        r["t_p"] = float(pt)
        try:
            w, pw = stats.wilcoxon(d)
            r["wilcoxon_p"] = float(pw)
        except ValueError:
            r["wilcoxon_p"] = None
    return r

report = {}
for arm in ARMS:
    ar = {}
    for metric in ["macro_f1", "accuracy", "withdrawn_recall", "atrisk_auc"]:
        m = {}
        for kind, label in [("xgb_repeats", "seeds"), ("xgb_gkf5", "gkf")]:
            per = paired(arm, metric, kind)
            if not per:
                continue
            m[label] = {
                "per_cutoff": {c: tests(d) for c, d in per},
                "pooled": tests(np.concatenate([d for _, d in per])),
            }
        ar[metric] = m
    report[arm] = ar

json.dump(report, open(f"{SP}/exp005_stats.json", "w"), indent=1)

# console summary
for arm in ARMS:
    print(f"\n===== {arm} vs A0_v4 =====")
    for metric in ["macro_f1", "accuracy", "withdrawn_recall", "atrisk_auc"]:
        for label in ["seeds", "gkf"]:
            if label not in report[arm][metric]:
                continue
            p = report[arm][metric][label]["pooled"]
            tp = p.get("t_p"); wp = p.get("wilcoxon_p")
            print(f"{metric:17s} [{label:5s}] pooled Δ={p['mean_delta']:+.4f}±{p['std']:.4f} "
                  f"({p['n_positive']}/{p['n']} pos) t_p={tp if tp is None else round(tp,4)} "
                  f"w_p={wp if wp is None else round(wp,4)}")
