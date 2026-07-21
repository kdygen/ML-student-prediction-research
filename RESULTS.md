# RESULTS — All Key Numbers in One Place

*Single reference for every headline result. All figures verified against the pipeline.
Validation: StratifiedGroupKFold(5) grouped by `id_student` unless stated otherwise.
Last updated 2026-07-21.*

**Contents:** [Full-course](#1-full-course-multiclass--headline) · [Early cutoffs](#2-early-prediction-by-cutoff) ·
[Binary](#3-binary-tasks) · [Regression](#4-regression) · [Intervention & capacity](#5-intervention-timing--capacity) ·
[Features](#6-feature-importance) · [Methodology facts](#7-methodology-quick-facts)

---

## 1. Full-course multiclass — HEADLINE

| Population | Accuracy | Macro-F1 | Weighted-F1 |
|---|--:|--:|--:|
| **Engaged (29,496) — published headline** | **0.8362 ± 0.0038** | **0.7946 ± 0.0038** | 0.8330 |
| Full registered (32,593) — for literature comparison | 0.8506 ± 0.0048 | 0.7976 ± 0.0083 | 0.8473 |

### Per-class (out-of-fold, engaged population)

| Class | Precision | Recall | F1 | Support |
|---|--:|--:|--:|--:|
| Withdrawn | 0.959 | 0.928 | **0.943** | 7,067 |
| Fail | 0.872 | 0.761 | **0.813** | 7,044 |
| Pass | 0.789 | 0.905 | **0.843** | 12,361 |
| Distinction | 0.668 | 0.514 | **0.581** | 3,024 |
| **Macro average** | **0.822** | **0.777** | **0.795** | 29,496 |

### Confusion matrix (row % — what each true class is predicted as)

| True ↓ / Predicted → | Withdrawn | Fail | Pass | Distinction |
|---|--:|--:|--:|--:|
| **Withdrawn** | **92.8%** | 5.4% | 1.8% | 0.1% |
| **Fail** | 3.8% | **76.1%** | 19.9% | 0.2% |
| **Pass** | 0.1% | 3.3% | **90.5%** | 6.1% |
| **Distinction** | 0.1% | 0.2% | 48.3% | **51.4%** |

> All residual error sits exactly where the excluded final exam decides the outcome:
> 19.9% of Fails look like Pass, 48.3% of Distinctions look like Pass.

---

## 2. Early prediction by cutoff

Accuracy / Macro-F1, all four models (Baseline v4, grouped split):

| Cutoff | LogReg | Decision Tree | Random Forest | **XGBoost** |
|--:|:--|:--|:--|:--|
| **Day 14** | 0.348 / 0.347 | 0.489 / 0.303 | 0.480 / 0.309 | **0.491 / 0.343** |
| **Day 30** | 0.401 / 0.397 | 0.516 / 0.339 | 0.524 / 0.372 | **0.527 / 0.396** |
| **Day 60** | 0.462 / 0.445 | 0.555 / 0.393 | 0.570 / 0.402 | **0.575 / 0.439** |
| **Day 90** | 0.487 / 0.462 | 0.590 / 0.418 | 0.608 / 0.433 | **0.616 / 0.464** |
| **Day 140** | 0.557 / 0.504 | 0.670 / 0.483 | 0.678 / 0.477 | **0.693 / 0.514** |

**XGBoost summary (the numbers to quote):**

| | Day 14 | Day 30 | Day 60 | Day 90 | Day 140 | Full course |
|---|--:|--:|--:|--:|--:|--:|
| Accuracy | 0.491 | 0.527 | 0.575 | 0.616 | 0.693 | **0.836** |
| Macro-F1 | 0.343 | 0.396 | 0.439 | 0.464 | 0.514 | **0.795** |

> The 0.514 → 0.795 macro-F1 gap between day 140 and full course quantifies how much
> outcome information does not yet exist during the intervention window.

---

## 3. Binary tasks

Full-course, per-student horizon, all four models:

| Task | Prevalence | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| **Withdrawn vs rest** | 0.302 | LogReg | 0.9726 | 0.950 | 0.960 | 0.955 | 0.9939 | 0.9862 |
| | | Decision Tree | 0.9530 | 0.929 | 0.915 | 0.922 | 0.9815 | 0.9551 |
| | | Random Forest | 0.9626 | 0.962 | 0.913 | 0.936 | 0.9904 | 0.9688 |
| | | **XGBoost** | **0.9743** | 0.959 | 0.955 | **0.957** | **0.9958** | 0.9907 |
| **Fail vs rest** | 0.223 | LogReg | 0.8767 | 0.691 | 0.808 | 0.745 | 0.9328 | 0.8440 |
| | | Decision Tree | 0.8943 | 0.876 | 0.613 | 0.721 | 0.9181 | 0.8062 |
| | | Random Forest | 0.9019 | 0.873 | 0.656 | 0.749 | 0.9346 | 0.8690 |
| | | **XGBoost** | **0.9175** | 0.888 | 0.721 | **0.796** | **0.9519** | 0.8941 |
| **Pass vs rest** | 0.384 | LogReg | 0.8291 | 0.707 | 0.948 | 0.810 | 0.8951 | 0.7756 |
| | | Decision Tree | 0.8419 | 0.730 | 0.932 | 0.819 | 0.9098 | 0.7894 |
| | | Random Forest | 0.8611 | 0.781 | 0.886 | 0.830 | 0.9373 | 0.8777 |
| | | **XGBoost** | **0.8639** | 0.783 | 0.893 | **0.834** | **0.9412** | 0.8870 |
| **Distinction vs rest** | 0.091 | LogReg | 0.8444 | 0.362 | 0.924 | 0.520 | 0.9425 | 0.5611 |
| | | Decision Tree | 0.9223 | 0.581 | 0.533 | 0.556 | 0.9344 | 0.5398 |
| | | Random Forest | 0.9244 | 0.642 | 0.389 | 0.485 | 0.9503 | 0.6119 |
| | | **XGBoost** | **0.9283** | 0.636 | 0.503 | **0.561** | **0.9524** | 0.6356 |
| **At-risk (W+F) vs (P+D)** | 0.525 | LogReg | 0.9286 | 0.967 | 0.895 | 0.929 | 0.9767 | 0.9830 |
| | | Decision Tree | 0.9158 | 0.946 | 0.891 | 0.917 | 0.9692 | 0.9726 |
| | | Random Forest | 0.9315 | 0.966 | 0.901 | 0.933 | 0.9803 | 0.9854 |
| | | **XGBoost** | **0.9368** | 0.969 | 0.908 | **0.938** | **0.9824** | 0.9870 |

*"Completed vs Withdrawn" is the label complement of Withdrawn-vs-rest — the same task, not
duplicated.*

---

## 4. Regression

Predicting final weighted coursework score from **behaviour only** (33 features, no grades as
inputs). GroupKFold(5), n = 23,241, target mean 70.75 (SD 16.39).

| Model | MAE | RMSE | R² |
|---|--:|--:|--:|
| Linear Regression | 11.089 ± 0.108 | 14.287 | 0.2399 ± 0.0100 |
| Random Forest | 10.376 ± 0.109 | 13.425 | 0.3288 ± 0.0090 |
| **XGBoost** | **10.296 ± 0.108** | **13.375** | **0.3339 ± 0.0094** |

> Behaviour alone explains about a third of coursework-score variance, ~10-point mean error.

---

## 5. Intervention timing & capacity

*Evaluation-layer analysis. Risk score = **ASI = P(Fail) + P(Withdrawn)**. Reference cohort:
5,635 held-out students; 46% eventually at-risk; 1,193 eventually withdraw.*

### 5.1 Risk-score quality by cutoff (and Earliest Reliable Intervention Point)

| Cutoff | At-risk ranking AUC | Top-5% list precision | Calibration error (ECE) |
|--:|--:|--:|--:|
| Day 14 | 0.721 | 0.858 | 0.015 |
| **Day 30 ← ERIP** | **0.786** | **0.978** | **0.021** |
| Day 60 | 0.829 | 0.992 | 0.015 |
| Day 90 | 0.862 | 0.996 | 0.014 |
| Day 140 | 0.914 | 1.000 | 0.012 |

**ERIP = Day 30** — the earliest checkpoint meeting a pre-registered rule (top-5% precision
≥ 0.95 AND calibration error ≤ 0.02). Day 14 fails at 0.858.

### 5.2 Budget vs reach vs lead time — THE PLANNING TABLE

| Budget | Policy | Precision (hit rate) | % of all at-risk reached | % of all withdrawals reached | Median lead (days) |
|--:|---|--:|--:|--:|--:|
| **10%** | One-shot day 30 | 0.941 | 20.6% | **22.5%** | 50 |
| | Staged day 30 + day 90 | 0.980 | 21.5% | 16.3% | 47 |
| | One-shot day 140 | 0.996 | 21.8% | 7.0% | 40 |
| **15%** | One-shot day 30 | 0.905 | 29.7% | **32.2%** | 63 |
| | Staged day 30 + day 90 | 0.966 | 31.7% | 25.9% | 49 |
| | One-shot day 140 | 0.980 | 32.2% | 11.7% | 36 |
| **20%** | One-shot day 30 | 0.851 | 37.3% | **38.6%** | 64 |
| | Staged day 30 + day 90 | 0.928 | 40.6% | 33.9% | 50 |
| | One-shot day 140 | 0.933 | 40.8% | 16.3% | 38 |

**Two conclusions:**
1. **Coverage is capacity-bound** (≈ budget ÷ prevalence): 10% budget → ~22% of at-risk
   reached; 20% → ~41%. More coverage needs more capacity, not better timing.
2. **Timing decides who you reach.** Acting at day 30 reaches **2–3× more withdrawing
   students** than waiting to day 140, at every budget. The staged day-30 + day-90 policy is
   the balanced default: matches a late one-shot on precision and total at-risk coverage while
   roughly **doubling withdrawal reach** and adding ~12 days of lead time.

### 5.3 Reachability decay — the shrinking window

| Point in course | At-risk still reachable | Withdrawals still reachable |
|---|--:|--:|
| Day 14 | 100% | 100% |
| Day 30 | 94.7% | 88.6% |
| Day 60 | 85.0% | 67.6% |
| Day 90 | 77.7% | 51.9% |
| Day 140 | 67.3% | **29.5%** |

After day 30, ranking quality improves ~0.01 AUC per 10 days while withdrawal reachability
falls ~0.07 per 10 days — **five times faster**.

### 5.4 Risk trajectories

| Trajectory | Share of students | P(adverse outcome) |
|---|--:|--:|
| Early exit (withdrew before day 60) | 6.9% | 1.000 |
| **Escalating** (unflagged → flagged) | 7.5% | **0.993** |
| Persistent high | 6.7% | 0.973 |
| Late emerging (first flagged ≥ day 90) | 4.3% | 0.942 |
| Intermittent | 7.0% | 0.657 |
| Recovering | 8.2% | 0.418 |
| Persistent low | 59.4% | 0.215 |

**Rising risk beats high risk.** Escalating students end adverse 99.3% of the time — above
persistently high-risk students (97.3%). Rapid risers (Δ score ≥ 0.15 between checkpoints) are
68–80% adverse vs 31–38% otherwise. **Median lead time first flag → unregistration: 58 days.**

---

## 6. Feature importance

Final model (XGBoost, 36 features), top 15 by gain:

| Rank | Feature | Importance | What it means |
|--:|---|--:|---|
| 1 | `submitted_count` | 0.273 | How much coursework they handed in |
| 2 | `decay_clicks` | 0.101 | Recency-weighted engagement |
| 3 | `rank_wa` | 0.072 | Coursework average vs their cohort |
| 4 | `completion_ratio_avail` | 0.058 | Share of *available* coursework done |
| 5 | `active_weeks` | 0.055 | Weekly study consistency |
| 6 | `n_assess_types_submitted` | 0.052 | Breadth of assessment engagement |
| 7 | `engagement_decay_ratio` | 0.038 | Did engagement fade over the course |
| 8 | `study_spread` | 0.036 | Share of enrolled span active |
| 9 | `max_gap` | 0.027 | Longest silence |
| 10 | `active_days` | 0.024 | Days with any activity |
| 11 | `rank_clicks` | 0.021 | Click volume vs cohort |
| 12 | `score_std_cw` | 0.020 | Score volatility |
| 13 | `first_submit_day` | 0.017 | When they first submitted |
| 14 | `highest_education_Lower Than A Level` | 0.015 | Strongest demographic |
| 15 | `w4_clicks` | 0.015 | Activity 3–4 weeks before endpoint |

**Regression top predictor:** `active_weeks` (0.162) — study consistency, leading by 3×.

**Robustness:** removing the top 3 features *simultaneously* costs only 0.004 macro-F1 (inside
split noise). The best single feature alone reaches macro-F1 0.36 vs 0.795 for the full model.

---

## 7. Methodology quick facts

| | |
|---|---|
| **Populations** | 32,593 registered → **29,496 engaged (headline)** → 23,241 regression subset |
| **Unique students** | 28,785 (why grouping matters) |
| **Classes** | Pass 12,361 (41.9%) · Withdrawn 7,067 (24.0%) · Fail 7,044 (23.9%) · Distinction 3,024 (10.3%) |
| **Features** | 36 final (42 → 36, cost 0.0026 macro-F1); regression uses 33 |
| **Feature mix** | 11 behavioural · 10 coursework · 10 temporal · 5 demographic |
| **Models** | LogReg · Decision Tree (depth 5) · Random Forest (300) · **XGBoost (300, depth 6, lr 0.05)** |
| **Validation** | StratifiedGroupKFold(5) grouped by student; zero-overlap asserted per fold |
| **Environment** | Python 3.12.6 · sklearn 1.6.1 · pandas 2.2.2 · numpy 2.0.2 · xgboost 3.3.0 |

**Key leakage numbers:**

| Finding | Value |
|---|--:|
| `date_unregistration` alone predicts Withdrawn (no model) | **F1 0.9950** (P 0.9991, R 0.9908) |
| Random split leaked test rows onto seen students | **15.1%** |
| Survivorship: at-risk population excluded at day 30 | **31%** |
| Withdrawn students with any exam record | **1 of 10,156** |
| Censoring removed | 29,440 VLE rows + 600 submissions |
| Literature best *defensible* macro-F1 (vs our 0.798) | 0.706 |

**Stability:** three independent evaluation schemes agree within 0.003 —
headline split 0.8364/0.7946 · repeated grouped splits 0.8392/0.8000 · SGKF-5 0.8384/0.7973.

---

## Where the detail lives

| Topic | File |
|---|---|
| Business-facing at-risk briefing | `AT_RISK_MODEL_BRIEFING.md` |
| Full-course methodology + audit | `reports/experiment_006_full_course.md`, `reports/experiment_007_publication_audit.md` |
| Early-prediction baselines | `reports/baseline_v4.md` |
| Intervention framework | `reports/experiment_004_intervention_framework.md` |
| Capacity numbers (JSON) | `reports/experiment_004_capacity_by_budget.json` |
| Feature parsimony | `reports/experiment_008_parsimony.md` |
| Literature comparison | `reports/literature_comparison.md` |
| Chronological record | `RESEARCH_LOG.md` |
