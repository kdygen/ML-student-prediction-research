"""Experiment 004 — timeline visualizations, trajectory flows, policy comparison,
decision-flow diagram. Writes PNGs to reports/figures/experiment_004/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
SP = os.environ["SP"]
FIG = f"{REPO}/reports/figures/experiment_004"
os.makedirs(FIG, exist_ok=True)
CUTS = [14, 30, 60, 90, 140]
CK = [f"c{c:03d}" for c in CUTS]

prog = json.load(open(f"{SP}/exp004_progressive.json"))["cutoffs"]
traj = json.load(open(f"{SP}/exp004_trajectory.json"))
pol = json.load(open(f"{SP}/exp004_policies.json"))

# ---------- fig 1: progressive prediction quality timeline ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axes[0]
ax.plot(CUTS, [prog[c]["accuracy"] for c in CK], "o-", label="4-class accuracy")
ax.plot(CUTS, [prog[c]["macro_f1"] for c in CK], "s-", label="macro-F1")
ax.plot(CUTS, [prog[c]["ASI"]["roc_auc"] for c in CK], "^-", label="ASI AUC (at-risk)")
ax.plot(CUTS, [prog[c]["WRI"]["roc_auc"] for c in CK], "v-", label="WRI AUC (withdrawn)")
ax.set_title("Prediction quality vs cutoff"); ax.set_xlabel("cutoff (course day)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(CUTS, [prog[c]["ASI"]["ece"] for c in CK], "o-", label="ASI ECE (raw)")
ax.plot(CUTS, [prog[c]["ISO_ASI"]["ece"] for c in CK], "s-", label="ASI ECE (isotonic)")
ax.plot(CUTS, [prog[c]["ASI"]["brier"] for c in CK], "^-", label="ASI Brier")
ax.plot(CUTS, [prog[c]["uncertainty"]["mean_entropy"] for c in CK], "v-",
        label="mean posterior entropy")
ax.set_title("Calibration & uncertainty vs cutoff"); ax.set_xlabel("cutoff (course day)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[2]
for k, mk in [("top5", "o-"), ("top10", "s-"), ("top20", "^-")]:
    ax.plot(CUTS, [prog[c]["ASI"][k]["precision"] for c in CK], mk, label=f"precision@{k[3:]}%")
for k, mk in [("top5", "o--"), ("top10", "s--"), ("top20", "^--")]:
    ax.plot(CUTS, [prog[c]["ASI"][k]["recall"] for c in CK], mk, alpha=0.6,
            label=f"recall@{k[3:]}%")
ax.set_title("Top-K performance vs cutoff (ASI, at-risk)"); ax.set_xlabel("cutoff (course day)")
ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_progressive_timeline.png", dpi=160); plt.close(fig)

# ---------- fig 2: the timing trade-off (quality up, reachability down) ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
ax.plot(CUTS, [prog[c]["ASI"]["roc_auc"] for c in CK], "o-", color="tab:blue",
        label="ranking quality (ASI AUC)")
ax.plot(CUTS, [pol["reachability"][c]["adverse_reachable_share"] for c in CK], "s-",
        color="tab:green", label="at-risk still reachable")
ax.plot(CUTS, [pol["reachability"][c]["withdrawn_reachable_share"] for c in CK], "v-",
        color="tab:red", label="withdrawn still reachable")
ax.set_title("The timing trade-off: quality rises, opportunity shrinks")
ax.set_xlabel("cutoff (course day)"); ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
ao = [pol["one_shot"][f"top20_at_{c}"] for c in CK]
ax.plot(CUTS, [d["recall_of_reference_adverse"] for d in ao], "o-",
        label="adverse reached (recall, top-20% one-shot)")
ax.plot(CUTS, [d["recall_of_reference_withdrawn"] for d in ao], "v-",
        label="withdrawn reached (recall)")
ax.plot(CUTS, [d["precision"] for d in ao], "s-", label="precision of flags")
ax.set_title("One-shot intervention value by cutoff (budget = 20%)")
ax.set_xlabel("cutoff of the single intervention (course day)")
ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_timing_tradeoff.png", dpi=160); plt.close(fig)

# ---------- fig 3: trajectories ----------
fig = plt.figure(figsize=(14, 4.8))
ax = fig.add_subplot(1, 3, 1)
cats = sorted(traj["trajectory_categories"].items(), key=lambda kv: -kv[1]["P_adverse"])
names = [k for k, _ in cats]
padv = [v["P_adverse"] for _, v in cats]
ns = [v["n"] for _, v in cats]
bars = ax.barh(names, padv, color="tab:red", alpha=0.75)
for b, n in zip(bars, ns):
    ax.text(b.get_width() + 0.01, b.get_y() + b.get_height()/2, f"n={n}", va="center", fontsize=8)
ax.set_xlim(0, 1.15); ax.set_xlabel("P(adverse outcome)"); ax.grid(alpha=0.3, axis="x")
ax.set_title("Outcome rate by risk trajectory\n(test panel, cases present at c14)", fontsize=10)

ax = fig.add_subplot(1, 3, 2)
M = traj["band_transitions"]["c030->c060"]
bands = ["red", "amber", "green"]
mat = np.array([[M[r].get(cc, 0.0) for cc in bands] for r in bands])
im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1)
ax.set_xticks(range(3), [f"{b} @c60" for b in bands])
ax.set_yticks(range(3), [f"{b} @c30" for b in bands])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                color="white" if mat[i, j] > 0.5 else "black")
ax.set_title("Band transition matrix c30→c60\n(row-normalized, survivors)", fontsize=10)

ax = fig.add_subplot(1, 3, 3)
ex30 = traj["band_transitions"]["c030->c060_exits"]
ex14 = traj["band_transitions"]["c014->c030_exits"]
x = np.arange(3); w = 0.35
ax.bar(x - w/2, [ex14[b]["exit_rate"] for b in bands], w, label="c14→c30", color="tab:orange")
ax.bar(x + w/2, [ex30[b]["exit_rate"] for b in bands], w, label="c30→c60", color="tab:red")
ax.set_xticks(x, bands); ax.set_ylabel("share unregistering before next cutoff")
ax.set_title("Imminent-withdrawal rate by band", fontsize=10)
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_trajectories.png", dpi=160); plt.close(fig)

# ---------- fig 4: policy comparison ----------
pols = ["oneshot_c30_20pct", "oneshot_c90_20pct", "oneshot_c140_20pct",
        "c30_10_c90_10", "c30_10_c60_5_c90_5", "c14_5_c30_5_c60_5_c90_5"]
labels = ["one-shot\nc30 (20%)", "one-shot\nc90 (20%)", "one-shot\nc140 (20%)",
          "staged\nc30+c90", "staged\nc30+c60+c90", "staged\nc14..c90 (4x5%)"]
fig, ax = plt.subplots(figsize=(11.5, 4.8))
x = np.arange(len(pols)); w = 0.27
ax.bar(x - w, [pol["staged"][p]["precision"] if p in pol["staged"] else None for p in pols],
       w, label="precision of flags", color="tab:blue")
ax.bar(x, [pol["staged"][p]["recall_of_reference_adverse"] for p in pols],
       w, label="adverse reached (recall)", color="tab:green")
ax.bar(x + w, [pol["staged"][p]["recall_of_reference_withdrawn"] for p in pols],
       w, label="withdrawn reached (recall)", color="tab:red")
for i, p in enumerate(pols):
    ld = pol["staged"][p]["median_lead_days_withdrawn"]
    ax.text(i, 0.03, f"lead {ld:.0f}d", ha="center", fontsize=8, color="white",
            fontweight="bold")
ax.set_xticks(x, labels, fontsize=8.5)
ax.set_title("Intervention policies at equal total budget (20% of cohort) — test panel")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9); ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_policies.png", dpi=160); plt.close(fig)

# ---------- fig 5: decision-flow diagram ----------
fig, ax = plt.subplots(figsize=(12.5, 8.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, text, fc, fontsize=9, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec="black", lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, color=tc)

def arrow(x1, y1, x2, y2, text="", dx=0.12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.2, color="black"))
    if text:
        ax.text((x1 + x2)/2 + dx, (y1 + y2)/2, text, fontsize=8, ha="left")

box(2.6, 9.0, 4.8, 0.8, "Each checkpoint (day 14, 30, 60, 90, 140):\nscore all enrolled students — calibrated ASI (official v4 XGB)", "#dbe9ff")
box(2.6, 7.6, 4.8, 0.75, "Rank students; compare with previous checkpoint\n(band + risk velocity ΔASI)", "#dbe9ff")
arrow(5.0, 9.0, 5.0, 8.42)

box(0.3, 5.6, 2.5, 1.0, "RED band — top 5%\n(calibrated P ≥ 0.92 at c30)\nP(adverse) = 0.98–1.00", "#ffb3b3", 8)
box(3.1, 5.6, 2.3, 1.0, "RAPID RISER\nΔ calibrated ASI ≥ +0.15\nP(adverse) = 0.68–0.80", "#ffd6a5", 8)
box(5.7, 5.6, 2.0, 1.0, "AMBER band — 5–20%\nP(adverse) = 0.71–0.95", "#fff3b0", 8)
box(8.0, 5.6, 1.8, 1.0, "GREEN band — rest\nP(adverse) = 0.21–0.38", "#c8e6c9", 8)
arrow(3.6, 7.6, 1.6, 6.68); arrow(4.6, 7.6, 4.2, 6.68)
arrow(5.6, 7.6, 6.6, 6.68); arrow(6.6, 7.6, 8.8, 6.68)

box(0.3, 3.6, 2.5, 1.0, "IMMEDIATE INTERVENTION\npersonal outreach now;\nred sticks (71% stay red)\n& 18% unregister by next cp", "#ff8080", 8)
box(3.1, 3.6, 2.3, 1.0, "IMMEDIATE INTERVENTION\nescalating trajectory:\nP(adverse) = 0.99", "#ff9f66", 8)
box(5.7, 3.6, 2.0, 1.0, "MONITOR CLOSELY\nlight-touch contact;\nrecheck next checkpoint", "#ffe680", 8)
box(8.0, 3.6, 1.8, 1.0, "LOW RISK\nno action;\nre-score next checkpoint", "#a5d6a7", 8)
arrow(1.6, 5.6, 1.6, 4.72); arrow(4.2, 5.6, 4.2, 4.72)
arrow(6.6, 5.6, 6.6, 4.72); arrow(8.8, 5.6, 8.8, 4.72)

box(5.7, 2.0, 2.0, 0.85, "CONTINUE MONITORING\nif amber 2+ checkpoints\nor high entropy (top quartile):\nadvisor verifies manually", "#fff3b0", 7.5)
arrow(6.6, 3.6, 6.6, 2.97)

box(0.3, 0.3, 9.5, 1.05,
    "Capacity rule: flag lists sized to advising capacity (top-K), not fixed thresholds. Timing rule: first full pass at day 30\n"
    "(earliest reliable point: red-band precision 0.98, ECE 0.02); stage remaining capacity at day 60/90 for late-emerging risk\n"
    "(staged c30+c90 doubles withdrawn reach vs waiting to c140 at equal precision). Median lead time: 58 days before unregistration.",
    "#e8e8e8", 8.5)
ax.set_title("Experiment 004 — Early-intervention decision framework (all rates measured on the out-of-sample test panel)",
             fontsize=11)
fig.savefig(f"{FIG}/fig5_decision_flow.png", dpi=160, bbox_inches="tight"); plt.close(fig)

print("figures written:")
for f in sorted(os.listdir(FIG)):
    print(" -", f)
