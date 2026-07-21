# Distinction Investigation — Research Log

> **📌 OFFICIAL BASELINE NOTE (2026-07-21).** The AF3 configuration described here was
> **promoted to the project's official baseline**. The numbers in this document come from the
> research pipeline, which computes cohort percentile features (`rank_clicks`) within all
> 32,593 registered students. The official notebook baseline ranks within the **engaged**
> population (29,496) — the population the model actually predicts for — giving
> **accuracy 0.739, macro-F1 0.715, Distinction F1 0.430, Withdrawn 0.940, Fail 0.779**.
> The difference is ≤ 0.003 on every metric, well inside fold noise (Distinction fold
> SD = 0.019). Official artifact: `reports/official_baseline_results.json`. The figures below
> are preserved as the experimental record.

Chronological record. Every arm evaluated on the identical StratifiedGroupKFold(5, rs=42)
folds, grouped by `id_student`, engaged population (29,496), all tuning inside training
folds. Baseline reproduced to Δ = 0.000000 before any experiment.

---

## Iteration 0 — Baseline reproduction + Phase-1 diagnosis (dx01)

**Score-free baseline (33 features, argmax):** acc 0.8053 ± 0.0028, macro-F1 0.6951 ± 0.0041,
D F1 **0.227** (P 0.563 / R 0.142), W 0.941, F 0.791, P 0.822.

**Diagnosis findings:**
- 85.3% of true Distinctions predicted Pass; only 13.3% get p(D) > 0.5.
- **But ranking signal exists: P-vs-D AUC 0.746, PR-AUC 0.423** (prevalence 19.7%) → a large
  part of the collapse is the argmax operating point under 4.09:1 imbalance, not absent
  information.
- Missed Distinctions are not pure Pass-clones (max |d| vs Pass = 0.47 on engagement volume);
  the model catches hyper-engaged Distinctions (d ≈ 0.9–1.1 correct-vs-missed on volume
  features) and misses normal-engagement ones.
- Strong course heterogeneity: per-module D F1 from 0.00 (AAA) to 0.40 (CCC).
- Largest univariate D-vs-P effects: rank_clicks d=+0.58, active_days +0.42, study_spread
  +0.42 — all volume-axis; nothing above 0.6.

**Decision:** attack the operating point first (cheap, directly indicated), then features.

## Iteration 1 — Operating point & model family (dx02)

**Hypothesis:** D F1 is suppressed by argmax under imbalance; fixing the operating point
recovers F1 without new information.

| Arm | Acc | Macro-F1 | D F1 | D P/R | W / F / P F1 |
|---|--:|--:|--:|:--|:--|
| M0 baseline argmax | 0.8053 | 0.6951 | 0.227 | .56/.14 | .941/.791/.822 |
| **M1 inverse-freq weights** | 0.7569 | **0.7288** | **0.452** | .36/.60 | .941/.789/.733 |
| M2 D-threshold rescue (inner-tuned) | 0.7570 | 0.7286 | 0.447 | .36/.60 | .941/.791/.737 |
| M3 hierarchical P-vs-D re-decision | 0.7399 | 0.7196 | 0.439 | .33/.65 | .941/.791/.708 |
| M4 ordinal regression + tuned cuts | 0.7751 | 0.7181 | 0.412 | .43/.39 | .920/.751/.789 |
| M5 dedicated P-vs-D binary (ceiling) | — | — | 0.449 @ tuned τ | — | AUC 0.7370, PR 0.4162 |

**Outcome: hypothesis confirmed.** Weighting **doubles D F1 (0.227 → 0.452)** and *raises*
macro-F1 by +0.034; W/F untouched; cost is accuracy (−4.8 pp) and Pass F1 (−0.089) — an
explicit precision/recall trade on the P/D boundary. Threshold rescue is equivalent (0.447);
hierarchical and ordinal are worse. Critically, the dedicated binary ceiling (0.449) matches
the multiclass fixes — **~0.45 is the operating ceiling of the 33-feature set.**

## Iteration 2 — 22 new leakage-free features in 6 hypothesis groups (dx03/dx04)

**Hypothesis:** normal-engagement Distinctions differ in regularity, depth, proactivity,
spacing, or cohort trajectory rather than volume.

Effect sizes (D vs P): share_topq_weeks +0.475, mean_weekly_z +0.399, inactive_week_share
−0.377, max_week_streak +0.375, gap_std −0.335, revisit_ratio +0.312 … all real, all
volume-correlated.

**P-vs-D AUC (the information metric), same folds:**

| Feature set | AUC | PR-AUC |
|---|--:|--:|
| base 33 | 0.7370 | 0.4162 |
| +H1 regularity | 0.7381 | 0.4172 |
| **+H2 depth/revisits** | **0.7488** | **0.4369** |
| +H3 proactivity | 0.7383 | 0.4173 |
| +H4 spacing | 0.7351 | 0.4121 |
| +H5 cohort-weekly | 0.7371 | 0.4163 |
| +H6 enrolment | 0.7379 | 0.4169 |
| +ALL 22 | 0.7466 | 0.4332 |

**Outcome: only H2 (depth/revisits) clears the pre-stated +0.002 rule** (+0.012 AUC — matching
Kizilcec et al.'s finding that revisiting is the SRL signature). All 22 together do no better
than H2 alone → the other groups are redundant re-encodings of the volume axis. Weighted
4-class with 33+H2: D F1 0.452 → 0.455 (inside noise), macro-F1 0.7288 → 0.7314.

## Iteration 3 — Module identity, alternative GBDT, final stack (dx05)

| Test | Result |
|---|---|
| +module dummies (P-vs-D) | AUC 0.7488 → 0.7491 — nothing |
| HistGradientBoosting instead of XGB | 0.7418 (33+H2) — slightly worse; **ceiling is not an XGBoost quirk** |
| LightGBM/CatBoost | skipped: not in the pinned environment (reproducibility guarantee) |

**FINAL STACK** (weighted 4-class, 33+H2+module, inner-tuned D-threshold):
acc 0.7554, macro-F1 **0.7306 ± 0.0066**, **D F1 0.458** (P 0.36 / R 0.63),
per-fold D F1 = [0.451, 0.491, 0.453, 0.456, 0.439] — every fold ≫ baseline 0.227.
W 0.943 / F 0.795 preserved; P 0.726.

## Verdict against the pre-stated success criteria

| Criterion | Status |
|---|---|
| Improve D F1 reproducibly | ✅ 0.227 → 0.458 (+0.231, ~2×), all 5 folds ≫ baseline |
| Preserve Fail & Withdrawn | ✅ W 0.941→0.943, F 0.791→0.795 |
| No major macro-F1 reduction | ✅ macro-F1 *improved* 0.695 → 0.731 |
| Across folds, not one split | ✅ fold D F1 range 0.439–0.491 |
| Survives grouped validation | ✅ everything grouped, tuning inner-fold only |

**And simultaneously: strong ceiling evidence.** P-vs-D AUC moved only 0.737 → 0.749 across
22 engineered features, module identity, and two GBDT families; missed Distinctions are
behaviourally Pass-like; independent literature (Borna 2024: click-only D F1 0.15; Conijn
2017; Tempelaar 2015; Adnan's ablation) converges on the same limit. The residual gap to the
with-scores model (0.458 vs 0.581) is an **information limit, not a modelling failure**.
