# Experiment 007 — Final Publication Audit of the V3 Configuration

> **📌 These figures describe the WITH-SCORES comparison arm, not the official baseline.**
> As of 2026-07-21 the official baseline is the **assessment-free** pipeline: accuracy 0.739,
> macro-F1 0.715, per-class F1 W 0.940 / F 0.779 / P 0.709 / D 0.430
> (`reports/official_baseline_results.json`). The numbers below remain the correct record
> for the with-scores configuration and for comparison against papers that use assessment
> scores.


**Date:** 2026-07-20
**Subject:** V3 = per-student timeline anchoring (006b) + fair completion denominator (006c V1)
+ engaged population (excludes enrolments unregistered on/before day 0).
**n = 29,496 enrolments, 26,358 students, 42 features.**
**Mandate:** verify soundness; change nothing unless a genuine issue is found.
**Outcome: no methodological change was required.** One reporting correction and one
optional simplification are recommended; both are documented below.
**Artifacts:** `reports/experiment_007_publication_audit/audit_metrics.json`,
`experiments/experiment_007_publication_audit.py`. Environment pinned and previously verified
to reproduce Baseline v4 bit-exactly.

---

## 1. Leakage audit

### 1.1 Structural checks (all passed)

| Check | Result |
|---|---|
| Forbidden columns present as features (`date_unregistration`, `course_end`, `horizon`, `final_result`, `target_multi`, `n_cw_total/avail`, `last_day`) | **none** (empty set) |
| Exam-derived information reachable from any feature | **none** — exams removed at build (4,959 rows) before feature computation |
| Any single feature separating any class with AUC > 0.95 | **none found** |
| Max \|correlation\| between any feature and any single class | **0.559** (`active_weeks`) — no target encoding |

### 1.2 Direct verification that censoring was applied (the decisive test)

A structural check is insufficient — features could have been computed from raw data by
mistake. I recomputed `active_days` from raw `studentVle` both **with** and **without** the
withdrawal censor and compared against the stored values, restricted to the 7,067 withdrawn
students in V3:

| | stored == censored recompute | stored == uncensored recompute |
|---|--:|--:|
| All withdrawn (n=7,067) | **100.00%** | 86.54% |
| **Students where censoring actually changed the value (n=951)** | **100.00%** | **0.00%** |

On every student for whom it mattered, the stored feature matches the censored computation
and never the uncensored one. **Post-withdrawal data is provably excluded.**

*Note on an audit artifact:* the automated check reported "20,075 VLE rows after
unregistration in the population." That query ran against **raw** `studentVle`, so it counts
rows the censor *removed*, not leakage — the verification above settles it.

### 1.3 Per-feature classification (42 features)

| Group (n) | Features | Verdict |
|---|---|---|
| **VLE behaviour (20)** | `active_days`, `clicks_per_day`, `study_spread`, `burstiness`, 5 activity-type ratios, `has_vle_activity`, `w1–w4_clicks`, `precourse_clicks`, `days_since_last`, `decay_clicks`, `engagement_decay_ratio`, `max_gap`, `active_weeks` | **Safe.** All derive from click timestamps censored at withdrawal; each is observable at the prediction point (course end). No future dependence. |
| **Coursework (13)** | `weighted_average`, `clicks_per_assessment`, `assessment_focus`, `recovery_slope`, `has_coursework`, `mean/min_submit_lead`, `submitted_count`, `first_submit_day`, `n_assess_types_submitted`, `completion_ratio_avail`, `score_slope_cw`, `score_std_cw` | **Safe.** Exam-free and censored. Deadlines are published schedule facts (known in advance); scores/submission timing are observed the day they occur. `completion_ratio_avail` uses the corrected denominator (coursework with deadline ≤ the student's own horizon). |
| **Cohort ranks (3)** | `rank_clicks`, `rank_wa`, `rank_active_days` | **Safe.** Percentile ranks of ≤ prediction-point behaviour within (module, presentation). Peers' *labels* are never used; a deployed system has exactly this information. |
| **Enrolment/static (5)** | `registration_lead`, 4 `highest_education` dummies | **Safe.** Fixed at registration. |
| **`late_submissions` (1)** | count of submissions after their deadline | **Safe.** Lateness is observed on the submission date. |

**Indirect target encoding — the honest finding.** No feature encodes the label
*mechanically*, but participation features are *definitionally* related to the Withdrawn
class: at course end, withdrawal **is** cessation of participation. This is intrinsic to the
prediction task, not a feature defect — confirmed by 006c, where removing the suspect
features simply redistributed the signal onto others. It is why the standing caveat
(course-end Withdrawn detection is **description, not forecasting**) must remain in any
publication.

**Verdict: V3 is leakage-free.** No future information, no post-withdrawal information, no
exam information, no unavailable-at-prediction-time information.

## 2. Robustness audit

### 2.1 Feature ablation — no shortcut dependence

Baseline (repeated grouped splits, seeds 0–4): **acc 0.8392 ± 0.0048, macro-F1 0.8000 ± 0.0043.**

| Removed | Acc | Macro-F1 | Δ Macro-F1 |
|---|--:|--:|--:|
| — (baseline) | 0.8392 | 0.8000 | — |
| `submitted_count` (rank 1, imp 0.23) | 0.8390 | 0.7996 | **−0.0004** |
| `decay_clicks` (rank 2, imp 0.11) | 0.8379 | 0.7979 | −0.0021 |
| `days_since_last` (rank 3, imp 0.06) | 0.8377 | 0.7984 | −0.0016 |
| **All top-3 simultaneously** | 0.8362 | 0.7963 | **−0.0037** |

Removing the three most important features together costs **0.004 macro-F1** — inside
split noise (±0.0043). The model relies on **distributed, redundant evidence**, not a
shortcut. This is the strongest single result of the audit.

### 2.2 Trivial predictors — the task is not trivial

| Model | Acc | Macro-F1 |
|---|--:|--:|
| Majority class | 0.412 | 0.146 |
| `submitted_count` alone (top feature) | 0.606 | 0.361 |
| `decay_clicks` alone | 0.581 | 0.420 |
| **Full V3 model** | **0.839** | **0.800** |

The best single feature reaches less than half the full model's macro-F1.

### 2.3 Subgroup robustness — no single module carries the result

Per-module macro-F1 on the headline test split: FFF 0.846, CCC 0.828, AAA 0.796, BBB 0.774,
EEE 0.752, DDD 0.745, **GGG 0.699**. Spread ≈ 0.15, all far above the 0.146 majority
baseline. GGG is the weakest and has an explanation in the data: it is the module whose
coursework carries **zero assessment weight** (all graded weight is exam-only), so the
coursework-performance features are least informative there.

### 2.4 Class imbalance

Balance is moderate: Pass 12,361 / Withdrawn 7,067 / Fail 7,044 / Distinction 3,024
(4.1:1 max ratio). Per-class F1: Withdrawn 0.936, Pass 0.836, Fail 0.803,
**Distinction 0.585**. Distinction is the weak class — the smallest (10.3%) and separated
from Pass by a grade margin whose determinant (the exam) is deliberately excluded. This is
an honest information limit, not an artifact; it should be stated rather than tuned away.

## 3. Feature overlap audit

Pairs with |r| ≥ 0.85 (of 861 pairs):

| Pair | \|r\| | Nature |
|---|--:|---|
| `recovery_slope` ~ `score_slope_cw` | **0.986** | **Genuine redundancy** — two measures of score trajectory (mean consecutive diff vs OLS slope) |
| `has_coursework` ~ `first_submit_day` | 0.985 | Sentinel artifact — no coursework ⇒ sentinel submit day |
| `rank_clicks` ~ `rank_active_days` | 0.949 | Both cohort-normalised volume |
| `has_vle_activity` ~ `days_since_last` | 0.946 | Sentinel artifact — no activity ⇒ sentinel recency |
| `rank_active_days` ~ `active_weeks` | 0.929 | Engagement breadth, raw vs ranked |
| `weighted_average` ~ `rank_wa` | 0.896 | Same quantity, raw vs cohort-ranked |
| `active_days` ~ `rank_active_days` | 0.879 | Same quantity, raw vs ranked |
| `rank_clicks` ~ `active_weeks` | 0.879 | Engagement volume vs breadth |

**Recommendation: keep all, with one optional simplification.** Only
`recovery_slope`/`score_slope_cw` (r=0.986) is true duplication and one could be dropped for
parsimony; §2.1 shows the model is insensitive to such removals, so the gain is
presentational, not statistical. The remaining pairs are raw/ranked or level/indicator
encodings that tree models exploit differently. Given the mandate ("do not remove unless
strongly justified") and the demonstrated redundancy tolerance, **no removal is required**.

## 4. Cross-validation (StratifiedGroupKFold, 5 folds, grouped by `id_student`)

| Fold | Acc | Macro-F1 |
|--:|--:|--:|
| 0 | 0.8349 | 0.7900 |
| 1 | 0.8346 | 0.7970 |
| 2 | 0.8385 | 0.7957 |
| 3 | 0.8394 | 0.8005 |
| 4 | 0.8447 | 0.8035 |
| **Mean ± SD** | **0.8384 ± 0.0037** | **0.7973 ± 0.0046** |

Zero student overlap asserted in every fold. Cross-validation (0.8384/0.7973) agrees with
repeated grouped splits (0.8392/0.8000) to within 0.001–0.003 — **the headline is not a
favourable split.** Fold spread is 0.010 accuracy / 0.014 macro-F1.

## 5. Final publication check

**1. Is V3 leakage-free?** **Yes**, on the evidence in §1: no forbidden columns, no exam
information, no perfect separators, no target-correlated feature above 0.56, and — decisively
— stored features match the *censored* recomputation on 100% of the students where censoring
mattered and the uncensored one on 0%.

**2. Remaining methodological weakness?** Three, all documented rather than fixed:
(i) **Interpretive, not statistical** — participation features are definitionally related to
withdrawal at course end, so the Withdrawn result is description, not forecasting;
(ii) **Distinction is weak** (F1 0.585) because its determinant is the excluded exam;
(iii) **Mixed class-weighting conventions** are inherited from Baseline v4 (balanced for
LogReg/RF, none for XGB/Tree) for comparability — defensible but should be stated.

**3. Any feature to remove before publication?** **No feature must be removed.** Optionally
drop one of `recovery_slope`/`score_slope_cw` (r=0.986) for parsimony. Note the already-
corrected `completion_ratio_cw` → `completion_ratio_avail` (Experiment 006c) must be the
version reported.

**4. Reproducible and robust?** **Yes.** Pinned environment verified bit-exact against
Baseline v4; grouped protocol with zero-overlap assertions throughout; agreement across three
independent evaluation schemes (headline split, 5 repeated grouped splits, 5-fold stratified
grouped CV) within 0.003; and insensitivity to removing the top-3 features (−0.004 macro-F1).

**5. Three strongest reviewer criticisms** (each grounded in this audit, not speculation):

> **(a) The headline conflates description with prediction.** Withdrawn F1 of 0.936 largely
> reflects that a student who stopped participating has stopped participating. The
> class-level result is near-tautological at a course-end prediction point, and 006c showed
> the signal cannot be engineered away. *Response:* report it explicitly as a ceiling/
> description, keep the early-prediction pipeline as the forecasting claim.

> **(b) No intervention or causal claim is supported.** OULAD has no intervention arm; the
> work optimises targeting/timing quality, never demonstrating that acting changes outcomes.
> *Response:* already stated in the 006/004 reports; must remain prominent.

> **(c) Single-institution, seven anonymised modules, unbalanced presentations.** Per-module
> F1 varies 0.699–0.846 with a mechanistic explanation for the weakest, and modules AAA/CCC
> have only two presentations each — external validity is untested.
> *Response:* frame all results as within-OULAD; do not claim cross-institution transfer.

**Overall audit verdict: the V3 configuration is scientifically sound, leakage-free, robust
to feature ablation and subgroup variation, and reproducible. No methodological change was
made or is required.** For this (with-scores) arm the headline is
**accuracy 0.838 ± 0.004, macro-F1 0.797 ± 0.005** (stratified grouped 5-fold CV).
