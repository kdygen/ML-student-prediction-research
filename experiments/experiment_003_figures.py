"""Experiment 003 — comparison figures. Reads exp003 arrays/JSONs, writes PNGs to
reports/figures/experiment_003/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
SP = os.environ["SP"]
FIG = f"{REPO}/reports/figures/experiment_003"
os.makedirs(FIG, exist_ok=True)
CUTS = [14, 30, 60, 90, 140]
IDX_MAIN = ["WRI", "FRI", "ASI", "SEV2", "ENT_ASI", "WSEL_atrisk", "ISO_ASI",
            "ASI_COURSE_PCT", "ENS_ASI", "ENS_ASI_MINUS_STD"]

res = json.load(open(f"{SP}/exp003_results_raw.json"))
perm = json.load(open(f"{SP}/exp003_perm.json"))
arrays = {c: np.load(f"{SP}/exp003_arrays_c{c:03d}.npz") for c in CUTS}

def truth_of(y, t):
    return {"atrisk": (y <= 1), "withdrawn": (y == 0), "fail": (y == 1)}[t].astype(int)

# ---------------- fig 1: heatmap of ROC-AUC and Recall@10 across indices/cutoffs ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, metric, title in [(axes[0], "roc_auc", "ROC-AUC (truth = at-risk)"),
                          (axes[1], "recall_at_10", "Recall@10% (truth = at-risk)")]:
    M = np.zeros((len(IDX_MAIN), len(CUTS)))
    for j, c in enumerate(CUTS):
        hl = res["cutoffs"][f"c{c:03d}"]["headline"]["models"]["xgboost"]
        for i, idx in enumerate(IDX_MAIN):
            M[i, j] = hl[idx]["atrisk"][metric]
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(CUTS)), [f"c{c}" for c in CUTS])
    ax.set_yticks(range(len(IDX_MAIN)), IDX_MAIN)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] < M.max() - 0.03 else "black", fontsize=7.5)
    ax.set_title(title + " — XGB indices, grouped test (seed 42)")
    fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_index_heatmap.png", dpi=160)
plt.close(fig)

# ---------------- fig 2: precision@K / recall@K curves at c30 and c90 -----------------
SHOW = ["ASI", "WRI", "FRI", "ENS_ASI", "ASI_COURSE_PCT", "ENT_ASI"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for col, c in enumerate([30, 90]):
    a = arrays[c]
    y = a["y_test"]
    tt = truth_of(y, "atrisk")
    n = len(y)
    ks = np.arange(1, 31) / 100.0
    for iname in SHOW:
        s = a[f"idx_{iname}"]
        order = np.argsort(-s, kind="stable")
        cums = np.cumsum(tt[order])
        prec, rec = [], []
        for k in ks:
            m = max(1, int(round(k * n)))
            prec.append(cums[m - 1] / m)
            rec.append(cums[m - 1] / tt.sum())
        axes[0, col].plot(ks * 100, prec, label=iname, lw=1.6)
        axes[1, col].plot(ks * 100, rec, label=iname, lw=1.6)
    axes[0, col].axhline(tt.mean(), color="gray", ls="--", lw=1, label="prevalence")
    axes[0, col].set_title(f"Precision@K — cutoff {c} (truth = at-risk)")
    axes[1, col].set_title(f"Recall@K — cutoff {c}")
    for r in (0, 1):
        axes[r, col].set_xlabel("K (% of ranked students)")
        axes[r, col].grid(alpha=0.3)
    for kv in (5, 10, 20):
        axes[0, col].axvline(kv, color="k", ls=":", lw=0.7)
        axes[1, col].axvline(kv, color="k", ls=":", lw=0.7)
axes[0, 0].set_ylabel("precision")
axes[1, 0].set_ylabel("recall")
axes[0, 0].legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_topk_curves.png", dpi=160)
plt.close(fig)

# ---------------- fig 3: reliability (calibration) — raw ASI vs isotonic --------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, c in zip(axes, [30, 90]):
    a = arrays[c]
    tt = truth_of(a["y_test"], "atrisk")
    for iname, lab, colr in [("ASI", "raw ASI", "tab:red"),
                             ("ISO_ASI", "isotonic ASI (train-only fit)", "tab:blue")]:
        p = np.clip(a[f"idx_{iname}"], 0, 1)
        bins = np.linspace(0, 1, 11)
        bi = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs, ys, ws = [], [], []
        for b in range(10):
            m = bi == b
            if m.sum() >= 20:
                xs.append(p[m].mean()); ys.append(tt[m].mean()); ws.append(m.sum())
        ax.plot(xs, ys, "o-", color=colr, label=lab, ms=4)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    hl = res["cutoffs"][f"c{c:03d}"]["headline"]["models"]["xgboost"]
    ax.set_title(f"Reliability, cutoff {c} — ECE raw={hl['ASI']['atrisk']['ece_10bin']:.3f}, "
                 f"iso={hl['ISO_ASI']['atrisk']['ece_10bin']:.3f}")
    ax.set_xlabel("predicted P(at-risk)"); ax.set_ylabel("observed at-risk rate")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_reliability.png", dpi=160)
plt.close(fig)

# ---------------- fig 4: lift curves ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, c in zip(axes, [30, 140]):
    a = arrays[c]
    tt = truth_of(a["y_test"], "atrisk")
    n = len(tt)
    ks = np.arange(1, 51) / 100.0
    for iname in ["ASI", "ENS_ASI", "WRI", "FRI"]:
        s = a[f"idx_{iname}"]
        order = np.argsort(-s, kind="stable")
        cums = np.cumsum(tt[order])
        lift = [(cums[max(1, int(round(k * n))) - 1] / max(1, int(round(k * n)))) / tt.mean()
                for k in ks]
        ax.plot(ks * 100, lift, label=iname, lw=1.6)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_title(f"Lift — cutoff {c} (truth = at-risk)")
    ax.set_xlabel("K (% of ranked students)"); ax.set_ylabel("lift over random")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_lift.png", dpi=160)
plt.close(fig)

# ---------------- fig 5: feature contribution (permutation importance) ----------------
NEW16 = {'rank_clicks','rank_wa','rank_active_days','mean_submit_lead','min_submit_lead',
 'late_submissions','submitted_count','first_submit_day','n_assess_types_submitted',
 'w1_clicks','w2_clicks','w3_clicks','w4_clicks','precourse_clicks','days_since_last','decay_clicks'}
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for r, c in enumerate([30, 90]):
    for j, idx in enumerate(["ASI", "WRI", "FRI"]):
        imp = perm["cutoffs"][f"c{c:03d}"]["importance"][idx]
        top = sorted(imp.items(), key=lambda kv: -kv[1]["mean_auc_drop"])[:12][::-1]
        names = [f for f, _ in top]
        vals = [d["mean_auc_drop"] for _, d in top]
        errs = [d["std"] for _, d in top]
        colors = ["tab:orange" if f in NEW16 else "tab:blue" for f in names]
        ax = axes[r, j]
        ax.barh(names, vals, xerr=errs, color=colors)
        ax.set_title(f"{idx} — cutoff {c} (AUC drop)", fontsize=10)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.3, axis="x")
fig.suptitle("Permutation importance toward each index (orange = promoted v4 feature)", y=1.0)
fig.tight_layout()
fig.savefig(f"{FIG}/fig5_feature_contribution.png", dpi=160)
plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print(" -", f)
