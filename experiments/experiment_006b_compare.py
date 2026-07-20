"""006 vs 006b comparison: tables to stdout, figure to the 006b results folder."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
SP = os.environ["SP"]
OUT = f"{REPO}/reports/experiment_006b_per_student_anchor"
os.makedirs(f"{OUT}/figures", exist_ok=True)
A = json.load(open(f"{SP}/exp006_results.json"))    # original (global anchor)
B = json.load(open(f"{SP}/exp006b_results.json"))   # per-student anchor
CLASSES = ["Withdrawn", "Fail", "Pass", "Distinction"]

print("=== multiclass (headline seed 42): acc / macro-F1  [006 -> 006b] ===")
for m in A["multiclass"]:
    a, b = A["multiclass"][m], B["multiclass"][m]
    print(f"{m:15s} acc {a['accuracy']:.4f} -> {b['accuracy']:.4f} ({b['accuracy']-a['accuracy']:+.4f})"
          f"   F1 {a['macro_f1']:.4f} -> {b['macro_f1']:.4f} ({b['macro_f1']-a['macro_f1']:+.4f})")

print("\n=== XGB repeats mean±std ===")
import statistics as st
for tag, R in [("006 ", A), ("006b", B)]:
    r = R["multiclass"]["xgboost"]["repeats"]
    print(f"{tag}: acc {st.mean([x['accuracy'] for x in r]):.4f}±{st.pstdev([x['accuracy'] for x in r]):.4f}"
          f"  F1 {st.mean([x['macro_f1'] for x in r]):.4f}±{st.pstdev([x['macro_f1'] for x in r]):.4f}")

print("\n=== XGB per-class P/R/F1 [006 -> 006b] ===")
for c in CLASSES:
    a = A["multiclass"]["xgboost"]["per_class"][c]
    b = B["multiclass"]["xgboost"]["per_class"][c]
    print(f"{c:12s} P {a['precision']:.3f}->{b['precision']:.3f}  R {a['recall']:.3f}->{b['recall']:.3f}"
          f"  F1 {a['f1-score']:.3f}->{b['f1-score']:.3f} ({b['f1-score']-a['f1-score']:+.3f})")

print("\n=== binary XGB: ROC-AUC / PR-AUC [006 -> 006b] ===")
for t in A["binary"]:
    a, b = A["binary"][t]["models"]["xgboost"], B["binary"][t]["models"]["xgboost"]
    print(f"{t:22s} AUC {a['roc_auc']:.4f} -> {b['roc_auc']:.4f} ({b['roc_auc']-a['roc_auc']:+.4f})"
          f"   PR {a['pr_auc']:.4f} -> {b['pr_auc']:.4f} ({b['pr_auc']-a['pr_auc']:+.4f})")

print("\n=== regression [006 -> 006b] ===")
for m in A["regression"]["models"]:
    a, b = A["regression"]["models"][m], B["regression"]["models"][m]
    print(f"{m:8s} MAE {a['mae']:.3f}->{b['mae']:.3f}  RMSE {a['rmse']:.3f}->{b['rmse']:.3f}"
          f"  R2 {a['r2']:.4f}->{b['r2']:.4f} ({b['r2']-a['r2']:+.4f})")

print("\n=== XGB multiclass importance: top-12 side by side ===")
ia = list(A["multiclass"]["xgboost"]["feature_importance"].items())[:12]
ib = list(B["multiclass"]["xgboost"]["feature_importance"].items())[:12]
for i in range(12):
    print(f"{i+1:2d}. {ia[i][0]:26s}{ia[i][1]:.4f}   | {ib[i][0]:26s}{ib[i][1]:.4f}")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
x = np.arange(4); w = 0.38
a_f1 = [A["multiclass"]["xgboost"]["per_class"][c]["f1-score"] for c in CLASSES]
b_f1 = [B["multiclass"]["xgboost"]["per_class"][c]["f1-score"] for c in CLASSES]
axes[0].bar(x - w/2, a_f1, w, label="006 global anchor", color="tab:blue")
axes[0].bar(x + w/2, b_f1, w, label="006b per-student anchor", color="tab:orange")
axes[0].set_xticks(x, CLASSES, fontsize=8); axes[0].set_ylabel("F1")
axes[0].set_title("Per-class F1 (XGB)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")

tasks = list(A["binary"].keys())
x = np.arange(len(tasks))
a_auc = [A["binary"][t]["models"]["xgboost"]["roc_auc"] for t in tasks]
b_auc = [B["binary"][t]["models"]["xgboost"]["roc_auc"] for t in tasks]
axes[1].bar(x - w/2, a_auc, w, label="006", color="tab:blue")
axes[1].bar(x + w/2, b_auc, w, label="006b", color="tab:orange")
axes[1].set_xticks(x, [t.replace("_vs_", "\nvs ") for t in tasks], fontsize=7)
axes[1].set_ylim(0.5, 1.0); axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("Binary ROC-AUC (XGB)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")

ib_full = B["multiclass"]["xgboost"]["feature_importance"]
top = list(ib_full.items())[:15][::-1]
ia_full = A["multiclass"]["xgboost"]["feature_importance"]
names = [k for k, _ in top]
axes[2].barh(names, [v for _, v in top], color="tab:orange", label="006b")
axes[2].barh(names, [ia_full.get(k, 0) for k in names], height=0.4, color="tab:blue",
             alpha=0.85, label="006 (same features)")
axes[2].tick_params(labelsize=7); axes[2].legend(fontsize=8)
axes[2].set_title("006b top-15 importance (XGB) vs 006"); axes[2].grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(f"{OUT}/figures/fig1_comparison.png", dpi=160); plt.close(fig)
print("\nfigure written ->", f"{OUT}/figures/fig1_comparison.png")
