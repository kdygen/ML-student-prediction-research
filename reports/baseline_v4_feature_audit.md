# Baseline v4 — Feature Promotion Audit

**Date:** 2026-07-02
**Scope:** the 16 Experiment-002 accepted features, audited for promotion into Baseline v4.
**Standard:** scientific promotion, not a copy — every feature independently recomputed,
traced, leakage-proven, redundancy-checked, and ablation-confirmed before entering the
official pipeline.
**Environment:** pinned (Python 3.12.6, sklearn 1.6.1, pandas 2.2.2, numpy 2.0.2, xgboost 3.3.0).

---

## 1. Independent recomputation (implementation correctness)

Each feature was recomputed from raw OULAD tables via a **deliberately different code path**
from the Experiment-002 generator, then compared value-by-value on the full population:

| Aspect | Exp-002 generator | Independent verifier |
|---|---|---|
| window sums | per-day aggregate (`daily`) then window filter | boolean masks over **raw click rows** |
| decay / recency | on daily sums | on raw rows |
| submissions | `studentAssessment.merge(assessments)` | **reversed** merge direction; numpy lead/late arithmetic |
| first submission | `groupby.min` | `sort_values` + `drop_duplicates` |
| cohort ranks | pandas `.rank(pct=True)` | `scipy.stats.rankdata(method="average")/n` per group |

**Result: all 16 features match at both audited cutoffs (30 and 90) with worst-case
`max_abs_diff = 2.27e-13`** (float summation-order noise; integer/count features exact).
The notebook v4 cell was additionally cross-checked against the Experiment-002 frames:
max deviation 2.27e-13 across all 35 active features.

## 2. Dependency trace and leakage proof (per feature)

All temporal guards are hard `assert`s that execute on every run (verifier **and** the v4
notebook cell). Measured on this audit: max `studentVle.date` used = 30.0/90.0 ≤ cutoff; max
`date_submitted` used = 30.0/90.0 ≤ cutoff; duplicate keys = 0.

| Feature | Raw dependencies | Filter / proof |
|---|---|---|
| `w1..w4_clicks` | `studentVle.date, sum_click` | clicks in [C−6,C], [C−13,C−7], [C−20,C−14], [C−27,C−21]; date = day click happened, all ≤ C |
| `precourse_clicks` | same | `date < 0` (before course start, trivially ≤ C) |
| `days_since_last` | `studentVle.date` | `C − max(date ≤ C)`; sentinel C+30 when no activity |
| `decay_clicks` | `studentVle.date, sum_click` | `Σ clicks·exp(−(C−date)/7)` over `date ≤ C` |
| `mean/min_submit_lead` | `studentAssessment.date_submitted`, `assessments.date` (deadline) | submissions with `date_submitted ≤ C`; deadline is published schedule (known in advance) |
| `late_submissions` | same | count of `date_submitted > deadline` among submissions ≤ C — lateness is observed the day it happens |
| `submitted_count`, `first_submit_day`, `n_assess_types_submitted` | `studentAssessment.date_submitted`, `assessments.assessment_type` | all over `date_submitted ≤ C`; type is static metadata |
| `rank_clicks/wa/active_days` | official ≤C features + `code_module`, `code_presentation` | percentile rank within the (module, presentation) cohort of ≤C **behaviour** — peers' labels never used; a deployed system has exactly this information on day C |

**No feature reads `final_result`, unregistration dates, post-cutoff clicks, or
post-cutoff submissions. Verdict: all 16 leakage-free.**

## 3. Redundancy analysis (c30, Pearson vs all 34 other active features)

- **No new feature reaches |r| ≥ 0.9 against any official v3 feature** (max: `rank_active_days`
  ~ `study_spread`, r=0.878; `days_since_last` ~ `has_vle_activity`, r=0.831 — informative but
  distinct).
- Exactly **one within-new pair flagged**: `w1_clicks ~ decay_clicks` (r=0.948). Both were
  subjected to leave-one-feature-out checks (§4): removing `decay_clicks` costs −0.0038
  inner F1 at c30, removing `w1_clicks` costs −0.0010 — **both carry independent signal;
  both kept** (at c90 each is a ±0.001 wash, documented).

## 4. Leave-one-group-out ablations (consistent-benefit test)

Train-only discipline: GroupKFold(3) inside the seed-42 grouped train; model fixed
(baseline XGBoost); macro-F1 at the τ=0.5 operating point. **Pre-registered rule:** promote a
group iff removing it lowers c30 inner F1 by ≥ 0.001 AND does not raise c90 inner F1 by
> 0.002.

| Group removed | ΔF1 at c30 | ΔF1 at c90 | Verdict |
|---|--:|--:|---|
| module_norm        | **−0.0088** | **−0.0088** | PROMOTE |
| submission_timing  | **−0.0096** | **−0.0047** | PROMOTE |
| recent_windows     | **−0.0086** | **−0.0064** | PROMOTE |
| recency_decay      | **−0.0023** | **−0.0017** | PROMOTE |

Every group contributes at **both** the early and late regime — nothing rides for free.
Full frame reference: 0.4424 (c30) / 0.5000 (c90) inner F1.

## 5. Promotion decision

**Promoted: all 16 features (4 groups).** Removed: none — no feature failed verification,
validity, redundancy, or consistent-benefit checks. (The two redundancy-flagged features
survived explicit LOO tests; their retention is evidence-based, not default.)

| Group | Features (16) |
|---|---|
| module_norm | rank_clicks, rank_wa, rank_active_days |
| submission_timing | mean_submit_lead, min_submit_lead, late_submissions, submitted_count, first_submit_day, n_assess_types_submitted |
| recent_windows | w1_clicks, w2_clicks, w3_clicks, w4_clicks, precourse_clicks |
| recency_decay | days_since_last, decay_clicks |

Baseline v4 active feature set = official v3 19 + these 16 = **35 features**. Population,
protocol, models, and hyperparameters are unchanged from v3. Official results and validity
re-verification: `reports/baseline_v4.md`.
