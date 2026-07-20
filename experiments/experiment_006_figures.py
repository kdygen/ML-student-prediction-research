"""Experiment 006 — figures. Writes PNGs to reports/experiment_006_full_course/figures/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
SP = os.environ["SP"]
OUT = f"{REPO}/reports/experiment_006_full_course/figures"
os.makedirs(OUT, exist_ok=True)
res = json.load(open(f"{SP}/exp006_results.json"))
arr = np.load(f"{SP}/exp006_arrays.npz")
CLASSES = ["Withdrawn", "Fail", "Pass", "Distinction"]

# fig 1: multiclass confusion matrices (all 4 models)
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, (name, r) in zip(axes, res["multiclass"].items()):
    cm = np.array(r["confusion"])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASSES, rotation=45, fontsize=7)
    ax.set_yticks(range(4), CLASSES, fontsize=7)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_title(f"{name}\nacc={r['accuracy']:.3f} F1={r['macro_f1']:.3f}", fontsize=9)
fig.suptitle("Multiclass confusion matrices — full-course leakage-free (grouped test, seed 42)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_confusion_multiclass.png", dpi=160); plt.close(fig)

# fig 2: ROC curves (XGB) for all binary tasks
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for tname, d in res["binary"].items():
    fpr, tpr = arr[f"roc_{tname}"]
    axes[0].plot(fpr, tpr, lw=1.6,
                 label=f"{tname} (AUC={d['models']['xgboost']['roc_auc']:.3f})")
    rec, prec = arr[f"pr_{tname}"]
    axes[1].plot(rec, prec, lw=1.6,
                 label=f"{tname} (PR-AUC={d['models']['xgboost']['pr_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC — XGB, binary tasks")
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("PR — XGB, binary tasks")
for ax in axes: ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_roc_pr_binary.png", dpi=160); plt.close(fig)

# fig 3: regression scatter + residuals (XGB)
ya, yp = arr["reg_actual"], arr["reg_pred"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
axes[0].scatter(ya, yp, s=4, alpha=0.25)
axes[0].plot([0, 100], [0, 100], "r--", lw=1)
r2 = res["regression"]["models"]["xgb_reg"]["r2"]
axes[0].set_xlabel("actual final coursework score"); axes[0].set_ylabel("predicted")
axes[0].set_title(f"Predicted vs actual (XGB) — R²={r2:.3f} (behavioral features only)")
resid = yp - ya
axes[1].scatter(yp, resid, s=4, alpha=0.25)
axes[1].axhline(0, color="r", ls="--", lw=1)
axes[1].set_xlabel("predicted"); axes[1].set_ylabel("residual (pred − actual)")
axes[1].set_title(f"Residuals — MAE={res['regression']['models']['xgb_reg']['mae']:.2f}")
for ax in axes: ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_regression.png", dpi=160); plt.close(fig)

# fig 4: top-20 feature importances (RF + XGB, multiclass)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5))
for ax, name in zip(axes, ["random_forest", "xgboost"]):
    imp = res["multiclass"][name]["feature_importance"]
    top = list(imp.items())[:20][::-1]
    ax.barh([k for k, _ in top], [v for _, v in top], color="tab:blue")
    ax.set_title(f"{name} — top 20 (multiclass)", fontsize=10)
    ax.tick_params(labelsize=7.5); ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_importance_multiclass.png", dpi=160); plt.close(fig)

# fig 5: full-course vs early-prediction comparison (XGB acc/F1 across cutoffs + full)
off = json.load(open(f"{REPO}/reports/baseline_v4_results.json"))
cuts = [14, 30, 60, 90, 140]
acc = [off[str(c)]["v4"]["models"]["xgboost"]["accuracy"] for c in cuts]
f1 = [off[str(c)]["v4"]["models"]["xgboost"]["macro_f1"] for c in cuts]
fa, ff = res["multiclass"]["xgboost"]["accuracy"], res["multiclass"]["xgboost"]["macro_f1"]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(cuts, acc, "o-", label="accuracy (Baseline v4, early cutoffs)")
ax.plot(cuts, f1, "s-", label="macro-F1 (Baseline v4)")
ax.axhline(fa, color="tab:blue", ls="--", lw=1.2,
           label=f"full-course accuracy = {fa:.3f}")
ax.axhline(ff, color="tab:orange", ls="--", lw=1.2,
           label=f"full-course macro-F1 = {ff:.3f}")
ax.set_xlabel("cutoff (course day)"); ax.set_title("Early prediction vs full-course ceiling (XGB)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_vs_early_prediction.png", dpi=160); plt.close(fig)

print("figures written:")
for f in sorted(os.listdir(OUT)): print(" -", f)
