# RESEARCH_LOG.md

This file records the evolution of the research.

Every completed experiment should append a new entry.

---

## Experiment Template

Date:

Objective:

Hypothesis:

Implementation:

Features Added:

Features Removed:

Models Evaluated:

Validation Strategy:

Results:

Observations:

Decision:

Next Ideas:

---

## Research Principles

- Record experiments in chronological order.
- Do not delete previous experiments.
- Negative results are valuable.
- Every conclusion should be supported by evidence.
- Baselines should only be updated after successful verification.
- Keep research history reproducible and auditable.

---

## Current Status

Baseline v2 established (2026-07-02). Official reference is now the leakage-corrected
pipeline: three assessment features were switched from due-date to submission-date filtering
to remove confirmed future-information leakage. See `reports/baseline_v2.md`,
`reports/baseline_v2_results.json`, and `reports/leakage_audit.md`. Baseline v1 remains as the
immutable prior baseline (`reports/baseline_v1.md`).

---

## Baseline v2

Date: 2026-07-02

Objective: Create a leakage-corrected early-prediction benchmark by auditing every active
feature and fixing any that use information not available at the cutoff.

Hypothesis: The assessment-derived features use future information (scores of work not yet
submitted at the cutoff). Removing that leakage yields a more valid benchmark without
destroying predictive value.

Implementation: Full leakage audit of all 17 active features (`reports/leakage_audit.md`).
Root cause: `weighted_average`, `clicks_per_assessment` (via `assessment_count`), and
`recovery_slope` filtered assessments by DUE date (`assessments.date <= CUTOFF`), which leaks
work submitted after the cutoff and discards work submitted before it. Fix: filter by
`date_submitted <= CUTOFF` (the day a score first exists). The ONLY change is the filter
predicate — grouping, weighting, fillna, row membership, split, models, and environment are
unchanged. Implemented as an additive notebook section inserted before the split (v1 cells
preserved as history, then overwritten in `mlData`); single notebook, still driven by only the
`CUTOFF` variable. Evidence at cutoff 30: 473 leaked late scores (2.1%), 4,962 discarded early
submissions; `weighted_average` changed for 1,419 students (428 → 0), `clicks_per_assessment`
for 3,098, `recovery_slope` for 2,642.

Features Added: None. Features Redesigned (3): `weighted_average`, `clicks_per_assessment`,
`recovery_slope` (due-date → submission-date filter). 14 features unchanged/SAFE (9
VLE/behavioural, `assessment_focus`, 4 education dummies).

Features Removed: None (removal would destroy real early signal; every feature justified).

Models Evaluated: same as v1 (LogReg, Decision Tree, Random Forest, XGBoost multiclass; binary
RF pairs). Environment pinned identical to v1 (sklearn 1.6.1 / pandas 2.2.2 / numpy 2.0.2).

Validation Strategy: unchanged — `train_test_split(test_size=0.2, random_state=42)`, 80/20, no
stratification (multiclass); binary pairs stratified. Membership identical to v1.

Results (multiclass accuracy / macro-F1; pinned sklearn 1.6.1, macOS/arm64):

| Cutoff | mlData | LogReg | Tree | RF | XGB |
|-------:|:------|:------|:-----|:---|:----|
| 14  | (1188, 78)  | 0.3403 / 0.3135 | 0.5210 / 0.3350 | 0.4832 / 0.3003 | 0.4874 / 0.3646 |
| 30  | (19300, 78) | 0.3904 / 0.3857 | 0.5044 / 0.3167 | 0.5122 / 0.3731 | 0.5189 / 0.3948 |
| 60  | (23411, 78) | 0.4281 / 0.4206 | 0.5253 / 0.3645 | 0.5447 / 0.4161 | 0.5499 / 0.4425 |
| 90  | (23452, 78) | 0.4498 / 0.4394 | 0.5543 / 0.4285 | 0.5670 / 0.4458 | 0.5760 / 0.4735 |
| 140 | (23478, 78) | 0.4977 / 0.4858 | 0.5730 / 0.4859 | 0.6078 / 0.5143 | 0.6180 / 0.5399 |

Observations: Removing leakage was performance-neutral overall and clearly positive at the
earliest cutoffs (cutoff 14: XGB +0.050 acc / +0.041 F1; cutoff 30: RF/XGB/LogReg up), because
v1 was starved of early data and v2 recovers legitimately-available early submissions. Cutoffs
60/90/140 are neutral (±0.004 acc). Binary score-driven tasks at cutoff 14 dropped (Pass-vs-Fail
−0.034, Distinction-vs-Fail −0.051) — those were inflated by the leak. XGBoost remains strongest.

Decision: Register as official Baseline v2 (leakage-corrected). Compare future experiments
against these numbers. Baseline v1 is immutable.

Next Ideas: (v3) fix sample-membership survivorship leakage — seed `mlData` from all registered
students (studentInfo/studentRegistration) rather than only those who submitted coursework, so
membership no longer depends on future behaviour. Also audit `study_spread`/ratio features'
cutoff-dependence and consider student-level grouped splits.

---

## Baseline v1

Date: 2026-07-02

Objective: Establish the official baseline by running the existing notebook
(`notebook/OULAD_early_prediction_v1 (1).ipynb`) exactly as written across every
available cutoff.

Hypothesis: N/A (baseline registration, not a hypothesis test). Expectation: later
cutoffs expose more information and should yield better predictions.

Implementation: Executed the notebook's code cells verbatim, in order, once per cutoff.
The only change was the permitted one — setting `CUTOFF` (cell 5) to each value in the
notebook's own list `[14, 30, 60, 90, 140]`. No preprocessing, feature engineering,
models, hyperparameters, evaluation, or reporting logic was modified. Run with cwd set to
`data/raw/` so the bare CSV filenames resolve; the 453 MB raw load was executed once and
deep-copied per cutoff (numerically identical to five independent runs). Completed with no
errors.

Environment (pinned to project Colab stack): Python 3.12.6, scikit-learn 1.6.1,
pandas 2.2.2, numpy 2.0.2, xgboost 3.3.0, imbalanced-learn 0.13.0; this run on macOS/arm64.
scikit-learn is pinned deliberately: RandomForest metrics are sklearn-version-dependent
(see reproducibility note below).

Features Added: None (baseline). Active feature set = 17 features
(`base_features` [13] + `edu_features` [4 highest_education one-hot]). Full `mlData` has 78
columns; only these 17 feed the models.

Features Removed: None.

Models Evaluated: Logistic Regression (balanced), Decision Tree (depth 5), Random Forest
(300 trees, balanced), XGBoost (300, depth 6, lr 0.05) on 4-class `target_multi`; plus a
binary Random Forest section on 4 class pairs. RF and XGB duplicate "rerun" cells reproduced
the first runs exactly.

Validation Strategy: `train_test_split(test_size=0.2, random_state=42)`, 80/20, no
stratification (multiclass); binary pairs stratified. No student-level grouping.

Results (multiclass accuracy / macro-F1; pinned sklearn 1.6.1, macOS/arm64):

| Cutoff | mlData | LogReg | Tree | RF | XGB |
|-------:|:------|:------|:-----|:---|:----|
| 14  | (1188, 78)  | 0.3151 / 0.2950 | 0.5042 / 0.2773 | 0.4664 / 0.2969 | 0.4370 / 0.3234 |
| 30  | (19300, 78) | 0.3886 / 0.3851 | 0.5083 / 0.3383 | 0.5062 / 0.3619 | 0.5142 / 0.3836 |
| 60  | (23411, 78) | 0.4273 / 0.4187 | 0.5274 / 0.3809 | 0.5471 / 0.4127 | 0.5509 / 0.4421 |
| 90  | (23452, 78) | 0.4507 / 0.4388 | 0.5508 / 0.4435 | 0.5675 / 0.4476 | 0.5781 / 0.4756 |
| 140 | (23478, 78) | 0.5002 / 0.4868 | 0.5771 / 0.4914 | 0.6105 / 0.5135 | 0.6220 / 0.5407 |

Observations: All models improve monotonically with later cutoffs. XGBoost is the strongest
overall — highest accuracy at cutoffs 30/60/90/140 and highest macro-F1 at 14/60/90/140
(exceptions: Decision Tree has top accuracy at the data-starved cutoff 14; LogReg edges XGB on
macro-F1 at cutoff 30). Random Forest sits just below XGB on accuracy with lower macro-F1
(majority-leaning under sklearn 1.6.1). Cutoff 14 is data-starved (1,188 rows) and least
reliable; row counts plateau (~23.4k) by cutoff 60. Feature set and split are identical across
cutoffs, so results are directly comparable. Full details, binary-pair results, and caveats are
in `reports/baseline_v1.md`.

Reproducibility note: An initial run on scikit-learn 1.9.0 gave RF cutoff-30 accuracy 0.4508 vs
0.5098 from Colab; the difference was fully diagnosed (not a pipeline bug). ~96% is the
scikit-learn version (RF is not cross-version reproducible; 1.9.0→0.4508, 1.6.1→0.5062) — hence
pinning sklearn 1.6.1. ~4% is platform floating-point: the split is byte-identical and 16/17
active features are bit-identical across macOS/arm64 vs Colab/Linux-x86_64; only `burstiness`
(a `.std()` reduction) differs below 1e-9, flipping 14/3860 RF predictions (local 0.5062 vs
Colab 0.5098). LogReg/Tree/XGBoost were byte-identical across the sklearn versions tested. Data
pipeline reproduces to <1e-9; RF metrics carry a ~±0.4% cross-platform tolerance and are
sklearn-version-dependent.

Decision: Register this as the official Baseline v1 under the pinned environment above. Future
experiments are measured against these per-cutoff numbers, using the same sklearn version.

Next Ideas: (not started) audit potential leakage in cutoff-dependent features and the
train/test split (student-level grouping / stratification); focus early-prediction work on
cutoffs 14 and 30; prefer XGBoost or seeded/averaged reporting where small RF differences would
change a conclusion.