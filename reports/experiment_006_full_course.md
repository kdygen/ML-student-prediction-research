# Experiment 006 — Full-Course Leakage-Free Prediction (the ceiling experiment)

**Date:** 2026-07-20
**Question:** how accurately can student outcomes be predicted using *all* behavioral
information collected during the course, while remaining rigorously leakage-free?
**Relation to existing work:** a NEW experiment alongside the early-prediction pipeline —
no cutoffs, no changes to Baselines v1–v4, caches p3/p4, the notebook, or any prior output.
**Results folder:** `reports/experiment_006_full_course/` (metrics.json, dataset_meta.json,
figures/) · **Drivers:** `experiments/experiment_006_{build,run,figures}.py`
**Environment:** pinned (Python 3.12.6, sklearn 1.6.1, pandas 2.2.2, numpy 2.0.2,
xgboost 3.3.0); rebuilt this session and verified to reproduce the official Baseline v4 c30
XGB headline bit-exactly (diff 0.0) before any experiment ran.

---

## 1. Experimental definition

- **Prediction point:** end of teaching, *before any final-exam information exists*.
- **Population:** all 32,593 registered (student, module, presentation) pairs
  (Withdrawn 10,156 / Fail 7,052 / Pass 12,361 / Distinction 3,024). Includes 3,097
  enrolments that unregistered on/before day 0 ("never started"; 3,089 of them Withdrawn) —
  they are legitimately predictable from their empty behavioral signature. 93 Withdrawn rows
  lack an unregistration date (their recorded activity is used as-is).
- **Censoring rule (withdrawn students):** for every enrolment with a
  `date_unregistration`, only VLE activity with `date ≤ unregistration` and submissions with
  `date_submitted ≤ unregistration` are used. This filter is real, not theoretical — it
  removed **29,440 VLE rows and 600 coursework submissions** recorded after unregistration.
- **Split protocol:** grouped by `id_student` everywhere (GroupShuffleSplit 80/20 seed 42
  headline; repeated grouped splits seeds 0–4; GroupKFold(5) for multiclass RF/XGB); zero
  student overlap asserted in every split.
- **Models:** the four official classification models with Baseline-v4 hyperparameters,
  verbatim; regressors mirror them (Linear, RF-300, XGB with the same tree parameters).

## 2. Assessment types — inclusion decisions

| Type | Decision | Reasoning |
|---|---|---|
| **TMA** (tutor-marked) | **Included** (scores + timing), censored at unregistration | Coursework graded during teaching; observed before the prediction point |
| **CMA** (computer-marked) | **Included**, censored | Same; note CMA weight is 0 (formative) in FFF/GGG — still behavioral signal |
| **Exam** | **Excluded entirely** — scores, submission records, *and* attendance | (a) exams occur at/after the end of teaching, i.e., after the prediction point; (b) exam results directly determine `final_result`; (c) empirically, exam *attendance alone* is a near-perfect outcome encoder: only **1 of 10,156** Withdrawn students has an exam record. 4,959 exam rows removed |

## 3. Feature leakage audit (all 42 classification features)

**Safe — carried over conceptually from the official pipeline, computed on censored,
exam-free, full-course data (16):** `active_days`, `sum_click`→(`clicks_per_day`,
`clicks_per_assessment`), `burstiness`, 5 activity-type ratios, `has_vle_activity`,
`has_coursework`, `registration_lead` (from `date_registration`, enrollment-time),
4 `highest_education` dummies, `rank_clicks`, `rank_active_days` (cohort percentiles of ≤
prediction-point behavior; peers' labels never used).

**Modified — same concept as v4 but re-anchored for the full-course setting (14):**

| Feature | Modification | Why leakage-free |
|---|---|---|
| `w1..w4_clicks` | recency windows anchored at each student's **own horizon** (min(unregistration, course end)) instead of a fixed cutoff | windows end at/before the student's last legitimate observation |
| `days_since_last`, `decay_clicks` | anchored at course end | uses only observed activity dates; large recency gap for early leavers is behavior, not the label |
| `study_spread` | denominator = presentation length (max observed VLE day) | schedule fact |
| `assessment_focus` | clicks within 7 days of **any coursework deadline** over the whole course | deadlines are published schedule; clicks are censored |
| `weighted_average`, `recovery_slope` | full-course **coursework-only** (exam-free), censored | observed graded performance before the prediction point |
| `mean/min_submit_lead`, `late_submissions`, `submitted_count`, `first_submit_day`, `n_assess_types_submitted` | over all censored coursework submissions | submission timing is observed the day it happens |
| `rank_wa` | cohort percentile of the exam-free weighted average | as above |

**New features (6)** — definition / why predictive / why leakage-free / difference from existing:

| Feature | Definition | Why predictive | Leakage-free because | Differs from |
|---|---|---|---|---|
| `engagement_decay_ratio` | clicks in last third of course ÷ (clicks in first third + 1) | fading engagement is the canonical disengagement signature | censored clicks only | recency windows measure *level*; this measures *shape* over the whole course |
| `max_gap` | longest inactivity gap (days) between active days | long silences precede withdrawal/failure | dates of observed activity only | `days_since_last` sees only the final gap |
| `active_weeks` | number of distinct calendar weeks with any activity | consistency of study rhythm over months | censored activity | `active_days` counts days, not sustained rhythm |
| `completion_ratio_cw` | submitted coursework count ÷ total coursework items in module | doing the work predicts outcomes | count-based (robust to zero-weight modules), censored | `submitted_count` is unnormalized across modules |
| `score_slope_cw` | linear slope of coursework scores in deadline order | improving vs deteriorating trajectory | scores observed before prediction point | full-course version of `recovery_slope` (mean diff) |
| `score_std_cw` | dispersion of coursework scores | volatility distinguishes Fail from stable Pass | as above | no existing dispersion-of-scores feature |

**Excluded (with reasons):**
- **All Exam-derived information** (scores, submissions, sat-exam indicator) — §2.
- **`date_unregistration` as a feature** (or anything derived from it: days-enrolled,
  unregistration timing) — it *is* the Withdrawn label's timestamp. Used **only** as a
  censoring boundary.
- **Post-unregistration activity** — 29,440 + 600 rows removed.
- **`course_end` / horizon as features** — schedule constants that would proxy module
  identity mixed with unregistration timing.
- **`final_result`-derived anything** — target only.
- **For the regression only**, 5 score-derived features are additionally excluded
  (`weighted_average`, `recovery_slope`, `rank_wa`, `score_slope_cw`, `score_std_cw`)
  because the regression target *is* the final coursework score — keeping them would make
  the task circular. The regression uses 37 purely behavioral/enrollment features.

**One honestly gray area, stated openly:** for withdrawn students, feature values are
computed on a horizon truncated at withdrawal. The *values* are all legitimately observable,
but the truncation pattern itself (e.g., a huge `days_since_last`) is highly informative
about withdrawal. We classify this as *the honest signal of the setting* — at course end, a
student who left in week 6 genuinely looks like this — not leakage: no post-outcome
information is used, and the timestamp of unregistration never enters a feature. It does
mean Withdrawn detection here is partly "recognizing that someone stopped," which is why the
binary Withdrawn task should be read as near-saturated rather than as a modeling triumph.

## 4. Results

### 4.1 Multiclass (held-out grouped test, seed 42; repeats seeds 0–4)

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | Repeats acc | Repeats F1 |
|---|--:|--:|--:|--:|:--|:--|
| LogReg | 0.7453 | 0.716 | 0.734 | 0.7216 | 0.758±0.004 | 0.734±0.004 |
| Decision Tree | 0.7736 | — | — | 0.7294 | 0.789±0.003 | 0.718±0.010 |
| Random Forest | 0.8036 | — | — | 0.7373 | 0.811±0.004 | 0.750±0.004 |
| **XGBoost** | **0.8234** | 0.795 | 0.755 | **0.7704** | 0.824±0.002 | 0.774±0.002 |

GroupKFold(5) agrees (XGB F1 0.7725±0.0036; RF 0.7465±0.0058). XGB per-class:
Withdrawn **P 0.946 / R 0.882 / F1 0.913**; Fail 0.778/0.723/0.749; Pass 0.792/0.912/0.848;
Distinction 0.663/0.503/0.572. Confusions for all models: `figures/fig1`.
The residual confusion is concentrated exactly where information is genuinely absent without
exams: Fail↔Pass (the pass mark depends on the excluded exam) and Distinction↔Pass.

### 4.2 Binary tasks (XGB shown; all 4 models in metrics.json)

| Task | prevalence | Acc | P | R | F1 | ROC-AUC | PR-AUC | AUC repeats |
|---|--:|--:|--:|--:|--:|--:|--:|:--|
| Withdrawn vs rest | 0.302 | 0.950 | 0.946 | 0.883 | 0.914 | **0.987** | 0.974 | 0.9866±0.0007 |
| Fail vs rest | 0.223 | 0.891 | 0.778 | 0.717 | 0.746 | 0.930 | 0.775 | 0.9314±0.0024 |
| Pass vs rest | 0.384 | 0.869 | 0.790 | 0.898 | 0.840 | 0.944 | 0.890 | 0.9472±0.0019 |
| Distinction vs rest | 0.091 | 0.929 | 0.663 | 0.489 | 0.563 | 0.952 | 0.638 | 0.9565±0.0028 |
| At-risk (W+F) vs (P+D) | 0.525 | 0.942 | 0.945 | 0.944 | 0.943 | **0.984** | 0.988 | 0.9853±0.0005 |

("Completed vs Withdrawn" is the label complement of Withdrawn-vs-rest — the identical
task — so it is not duplicated.) ROC and PR curves: `figures/fig2`. Distinction remains the
hardest target at the default threshold (small class, subtle margin), though its *ranking*
is strong (AUC 0.95): threshold tuning would trade its precision/recall freely.

### 4.3 Score regression (behavior-only → final weighted coursework score)

Target: full-course weighted coursework average (weight>0 assessments; excludes module GGG,
whose graded weight is exam-only). n = 23,241. Features: 37 behavioral/enrollment only —
**no score-derived features** (see audit).

| Model | MAE | RMSE | R² | R² repeats |
|---|--:|--:|--:|:--|
| Linear | 11.02 | 14.18 | 0.233 | — |
| RF | 10.26 | 13.23 | 0.333 | — |
| **XGB** | **10.15** | **13.16** | **0.340** | 0.331±0.015 |

Scatter + residuals: `figures/fig3`. Reading: *behavior alone* (clicks, rhythm, submission
timing — no grades) explains about a third of coursework-score variance with ~10-point mean
error. Residuals show the classic pattern: scores of weak-engagement/low-score students are
over-predicted — behavioral effort does not fully determine achievement.

### 4.4 Feature importance (top of the rankings; full top-20 in `figures/fig4`)

Multiclass XGB: `engagement_decay_ratio` (0.168, **a new feature is #1**),
`days_since_last` (0.132), `decay_clicks`, `w1_clicks`, `w4_clicks`, `rank_wa`,
`weighted_average`, `completion_ratio_cw` (new, #8), then remaining recency windows.
Interpretation: over a full course, the **shape of engagement over time** — did it fade, and
when did it stop — dominates everything, ahead of graded performance itself. Regression
(behavior-only): `active_weeks` (0.162) leads by 3× — **consistency of weekly study rhythm
is the strongest behavioral predictor of achievement** — followed by cohort activity rank,
`study_spread`, prior education, `completion_ratio_cw`, and submission-timing leads
(punctuality), a nice separation: *when/whether* you work predicts scores even with *how
well* excluded.

## 5. Comparison with the early-prediction pipeline (`figures/fig5`)

| XGB | c14 | c30 | c60 | c90 | c140 | **full course** |
|---|--:|--:|--:|--:|--:|--:|
| Accuracy | 0.491 | 0.527 | 0.575 | 0.617 | 0.693 | **0.823** |
| Macro-F1 | 0.343 | 0.396 | 0.439 | 0.464 | 0.514 | **0.770** |

- **Headline:** +13 pp accuracy and +26 pp macro-F1 over the best early cutoff (c140).
- **The transformed class is Withdrawn:** F1 0.09 (c140) → **0.91**. The reason is
  structural, not just more data: at day-140 the early pipeline predicts only for
  *still-enrolled* students (withdrawal hasn't shown its full signature yet, and most
  eventual withdrawals have already left that population), whereas the full-course frame
  contains every withdrawal with its complete truncated trace — and a completed
  disengagement arc is nearly unmistakable (binary AUC 0.987). Distinction improves least
  (0.50→0.57 F1): without exam information, excellence remains genuinely hard to certify.
- **Features shift accordingly:** at early cutoffs, cohort performance rank (`rank_wa`)
  dominates; at full course, *trajectory-shape* features take over
  (`engagement_decay_ratio`, `days_since_last`) — early prediction is about "how are they
  doing so far," end-of-course classification is about "how did their engagement end."
- **Population caveat (important):** the comparison is *not* like-for-like. The full-course
  population (all 32,593 registered, including 3,097 never-starters) differs from the
  still-enrolled risk populations at each cutoff, and the full-course task has no
  intervention value — the course is over. This experiment measures the **information
  ceiling** of leakage-free behavioral data, not a deployable predictor. The gap between
  0.77 (ceiling) and 0.51 (c140) macro-F1 quantifies how much predictive information about
  outcomes simply *does not exist yet* during the intervention window — supporting the
  project's turn toward intervention timing rather than chasing early-cutoff accuracy.

## 6. Assumptions and limitations

1. "End of teaching" is operationalized per presentation as the last observed VLE day;
   exam-period VLE activity (browsing during revision) is included as behavior.
2. The Withdrawn signature partly encodes truncation (§3, gray area) — legitimate at this
   prediction point, but it means the multiclass headline should not be read as
   early-warning capability.
3. Regression target is coursework-only (exam scores are mostly absent from OULAD and
   excluded by design); GGG is out of the regression population.
4. Single institution, 7 anonymized modules; no cross-institution generalization claim.
5. Class-imbalance handling inherits the mixed v4 conventions (balanced weights for
   LogReg/RF, none for XGB/Tree) for comparability with the official baselines.

**No leakage was detected and silently passed through**; the two discovered leakage surfaces
(post-unregistration records, exam attendance) are documented above and removed, with row
counts. All numbers, per-class reports, confusions, and importances:
`reports/experiment_006_full_course/metrics.json`.
