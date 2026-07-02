# Baseline v2 — OULAD Early Prediction (leakage-corrected)

**Date:** 2026-07-02
**Notebook:** `notebook/OULAD_early_prediction_v1 (1).ipynb` (single notebook; v2 section added)
**Builds on:** Baseline v1 (`reports/baseline_v1.md`, immutable)
**Status:** Official leakage-corrected baseline. Same pipeline as v1 with three
assessment-derived features recomputed to remove confirmed future-information leakage.

---

## 1. What changed vs v1 (one-line summary)

Three active features (`weighted_average`, `clicks_per_assessment`, `recovery_slope`) were
filtered in v1 by the assessment **due date** (`assessments.date <= CUTOFF`). v2 filters them
by **submission date** (`date_submitted <= CUTOFF`) — the day a score first exists — which is
the only information truly available at the cutoff. **Nothing else changed:** same rows, same
17-feature set, same 80/20 split (seed 42), same models/hyperparameters, same environment. Full
audit and evidence: `reports/leakage_audit.md`.

## 2. Environment (pinned — identical to v1)

Python 3.12.6 · scikit-learn **1.6.1** · pandas 2.2.2 · numpy 2.0.2 · xgboost 3.3.0 ·
imbalanced-learn 0.13.0 · this run on macOS/arm64. RandomForest metrics are
sklearn-version- and platform-sensitive (~±0.4% cross-platform); see v1 report §7. All v1↔v2
comparisons below are within this single pinned environment, so deltas are attributable to the
feature change alone.

## 3. The change in detail

Implemented as an additive notebook section ("Baseline v2 — Leakage-Free Assessment Features")
inserted **before** the train/test split; the original v1 feature cells are preserved as
history and then overwritten in `mlData`. The notebook remains a **single notebook that works
for every cutoff by changing only `CUTOFF`**.

| Feature | v1 filter | v2 filter | Everything else |
|---|---|---|---|
| `weighted_average` | `assessment_type!=Exam` & `date (due) <= CUTOFF` | `assessment_type!=Exam` & `date_submitted <= CUTOFF` | unchanged (same weighted mean, `fillna(0)`) |
| `clicks_per_assessment` | `assessment_count` from due-date filter | `assessment_count` from `date_submitted` filter | unchanged (`sum_click/(count+1)`) |
| `recovery_slope` | scores filtered by due date | scores filtered by `date_submitted` | unchanged (mean consecutive Δscore) |

**Assumption:** OULAD has no grading date, so `date_submitted` is the proxy for "score is
available" (conservative — a score can't be known before submission).

**Why this both removes leakage and recovers signal:** the due-date filter (a) *leaked* work
submitted after the cutoff and (b) *discarded* work submitted before the cutoff but due later.
v2 fixes both. At cutoff 30, v1 leaked 473 late scores (2.1% of coursework rows) and discarded
4,962 early submissions; v2 restores the latter as legitimate early signal.

## 4. Dataset shape per cutoff (identical to v1)

Membership and split are unchanged, so v1↔v2 is a clean feature-by-feature comparison.

| Cutoff | mlData | X | Train | Test | W(0) | F(1) | P(2) | D(3) |
|-------:|:-------|:--|-----:|-----:|-----:|-----:|------:|-----:|
| 14  | (1188, 78)  | (1188, 17)  | 950   | 238  | 148  | 316  | 558   | 166  |
| 30  | (19300, 78) | (19300, 17) | 15440 | 3860 | 3900 | 4406 | 8906  | 2088 |
| 60  | (23411, 78) | (23411, 17) | 18728 | 4683 | 4400 | 5163 | 11220 | 2628 |
| 90  | (23452, 78) | (23452, 17) | 18761 | 4691 | 4414 | 5180 | 11230 | 2628 |
| 140 | (23478, 78) | (23478, 17) | 18782 | 4696 | 4423 | 5192 | 11235 | 2628 |

Active feature list is identical to v1 (13 `base_features` + 4 `highest_education_*`).

## 5. Baseline v2 results — multiclass (accuracy / macro-F1)

**Bold** = best per column-group per row.

| Cutoff | LogReg acc | LogReg F1 | Tree acc | Tree F1 | RF acc | RF F1 | XGB acc | XGB F1 |
|-------:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 14  | 0.3403 | 0.3135 | **0.5210** | 0.3350 | 0.4832 | 0.3003 | 0.4874 | **0.3646** |
| 30  | 0.3904 | 0.3857 | 0.5044 | 0.3167 | 0.5122 | 0.3731 | **0.5189** | **0.3948** |
| 60  | 0.4281 | 0.4206 | 0.5253 | 0.3645 | 0.5447 | 0.4161 | **0.5499** | **0.4425** |
| 90  | 0.4498 | 0.4394 | 0.5543 | 0.4285 | 0.5670 | 0.4458 | **0.5760** | **0.4735** |
| 140 | 0.4977 | 0.4858 | 0.5730 | 0.4859 | 0.6078 | 0.5143 | **0.6180** | **0.5399** |

### Binary Random Forest (paper-style pairs) — accuracy

| Cutoff | Pass vs Fail | Distinction vs Fail | Distinction vs Pass | Withdrawn vs Pass |
|-------:|-----:|-----:|-----:|-----:|
| 14  | 0.6229 | 0.7216 | 0.7793 | 0.7746 |
| 30  | 0.7195 | 0.8214 | 0.8099 | 0.7506 |
| 60  | 0.7489 | 0.8576 | 0.8256 | 0.7891 |
| 90  | 0.7626 | 0.8828 | 0.8344 | 0.8277 |
| 140 | 0.8022 | 0.9239 | 0.8345 | 0.8512 |

## 6. v1 → v2 comparison (metric-by-metric)

Multiclass Δ = v2 − v1 (accuracy / macro-F1):

| Cutoff | LogReg | Tree | RF | XGB |
|-------:|:------|:------|:------|:------|
| 14  | +0.0252 / +0.0185 | +0.0168 / +0.0577 | +0.0168 / +0.0035 | **+0.0504 / +0.0412** |
| 30  | +0.0018 / +0.0006 | −0.0039 / −0.0216 | +0.0060 / +0.0112 | +0.0047 / +0.0112 |
| 60  | +0.0009 / +0.0018 | −0.0021 / −0.0164 | −0.0023 / +0.0033 | −0.0011 / +0.0004 |
| 90  | −0.0009 / +0.0006 | +0.0034 / −0.0150 | −0.0004 / −0.0018 | −0.0021 / −0.0020 |
| 140 | −0.0026 / −0.0010 | −0.0040 / −0.0055 | −0.0028 / +0.0008 | −0.0040 / −0.0007 |

**Which features changed and why:** only the three assessment features
(`weighted_average`, `clicks_per_assessment`, `recovery_slope`), because they were the only
active features using the leaky due-date filter (see audit). The 14 SAFE features
(9 VLE/behavioural, `assessment_focus`, 4 education dummies) are byte-identical to v1.

**Performance direction:**
- **Earliest cutoffs (14, 30) — improved.** Biggest gains at cutoff 14 (XGB +0.050 acc /
  +0.041 F1; Decision Tree +0.058 F1). Here v1 was information-starved (almost nothing is *due*
  by day 14), and v2 recovers many legitimately-available early submissions → more valid *and*
  more predictive.
- **Later cutoffs (60, 90, 140) — neutral.** Deltas are within ±0.004 accuracy and ±0.02
  macro-F1, mixed sign — by these cutoffs almost all due work is already submitted, so the
  leak/discard largely cancel.
- **Binary tasks at cutoff 14 dropped** (Pass-vs-Fail −0.034, Distinction-vs-Fail −0.051):
  those score-driven pairs were partly *inflated* by leaked future scores at the earliest
  cutoff; removing the leak makes them more honest. Later binary cutoffs are near-neutral.

**Net:** removing the confirmed leakage did **not** cost performance — it was neutral overall
and clearly positive at the early cutoffs that matter most for *early* prediction. This is the
expected signature of a leak that was simultaneously masking legitimately-available signal.

## 7. Is v2 a more valid early-prediction benchmark?

**Yes.** Every active feature now uses only information available on/before the cutoff
(scores gated by `date_submitted`, not by deadline). The three confirmed leaks are removed with
no loss of predictive value, and the fix restores real early signal v1 discarded. v2 is
therefore a strictly more valid early-prediction benchmark than v1, and is the recommended
reference going forward.

**Residual (documented, deferred to v3):** the *sample membership* itself is still defined by
having **submitted** weighted coursework due ≤ cutoff (survivorship: never-submitters excluded;
due-date seed). This is a pipeline-level leak, not a feature leak; fixing it changes the sample
and would break the clean v1↔v2 comparison, so it is deferred. See `leakage_audit.md` §5. Until
then, v2 metrics describe "outcome prediction among students with early submitted coursework,"
using leakage-free features.

## 8. Observations (unchanged qualitative story)

- Monotonic improvement with later cutoffs holds for every model.
- **XGBoost is the strongest model** — best macro-F1 at every cutoff and best accuracy at
  30/60/90/140 (Decision Tree tops accuracy only at the data-starved cutoff 14).
- Class imbalance (~48% Pass) unchanged; RandomForest remains majority-leaning under sklearn
  1.6.1 (lower macro-F1 than XGBoost).

## 9. Artifacts

- Raw per-cutoff metrics: `reports/baseline_v2_results.json`
- Leakage audit (all 17 features + evidence): `reports/leakage_audit.md`
- Immutable prior baseline: `reports/baseline_v1.md`, `reports/baseline_v1_results.json`
