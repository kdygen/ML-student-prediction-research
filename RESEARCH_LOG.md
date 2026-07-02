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

Pipeline frozen and cached (2026-07-02): canonical v3 datasets live in `data/processed/p3/`
(one parquet per cutoff + hash/env/schema manifests; parquet gitignored, manifests tracked).
The cache is the mandatory input for all experiments. Experiment 001 complete: best valid
model = baseline XGBoost + class-prior threshold adjustment (τ=0.75), macro-F1 +4–6 pp over
Baseline v3 at every cutoff (25/25 paired seeds), Withdrawn recall 3–18× baseline; 0.70
macro-F1 shown unrealistic under the leak-free protocol (practical ceiling ≈0.45–0.48 at
cutoff 30). See `reports/experiment_001.md`. Baselines v1/v2/v3 immutable.

---

## Experiment 001 — Macro-F1 Optimization (Phase 8)

Date: 2026-07-02

Objective: Using the frozen v3 cache, find the strongest scientifically valid model under
the official protocol, optimizing macro-F1, then Withdrawn recall, then accuracy.

Hypothesis: Baseline v3's unweighted XGBoost is majority-biased; imbalance-aware techniques
(class/sample weights, in-fold SMOTE, threshold adjustment) should raise macro-F1 materially.

Implementation: Part A — cache freeze: `data/processed/p3/c{014..140}/mlDataV3.parquet`
built by executing the notebook verbatim through the v3 cell; manifests with raw sha256s,
pipeline-code sha256, env, schema, bit-exact + round-9 frame hashes; reload-verified; no
splits/standardization/resampling/model outputs cached; .gitignore updated (parquet out,
manifests in). Part B — experiment (code: `experiments/experiment_001_macro_f1.py`):
official seed-42 grouped test held out untouched; all tuning via GroupKFold(3) inside the
seed-42 train, including class-prior threshold τ (p·(1/π_c)^τ, τ∈{0,.25,.5,.75,1});
SMOTE/SMOTETomek inside training folds only; 13 configs screened at cutoff 30; top-3
re-selected per cutoff by inner CV; winner refit and evaluated once on the held-out test,
then on repeated grouped splits (seeds 0–4, everything frozen) with paired baseline
comparison on identical splits.

Features Added/Removed: none (cache is canonical and unmodified).

Models Evaluated: 13 configs — reference logreg/RF/XGB (baseline v3), sample-weighted XGB
(4 hyperparameter variants), RF variants (balanced_subsample, min_samples_leaf=5),
SMOTE/SMOTETomek in-fold (XGB, RF), all × τ grid.

Validation Strategy: official v3 protocol (grouped, seed 42) + repeated grouped splits
(seeds 0–4); paired per-seed deltas; sign test.

Results (winner per cutoff; macro-F1 on held-out test, repeats mean±std, baseline-XGB repeats):

| Cutoff | winner | test F1 | repeats | baseline XGB | Withdrawn recall (win/base) |
|--:|:--|--:|:--|:--|:--|
| 14  | xgb+τ.75 | 0.3615 | 0.3622±0.0061 | 0.3030±0.0028 | 0.193 / 0.060 |
| 30  | rf_leaf5 | 0.4133 | 0.4241±0.0076 | 0.3804±0.0055 | 0.209 / 0.149 |
| 60  | xgb+τ.75 | 0.4612 | 0.4645±0.0039 | 0.4233±0.0040 | 0.245 / 0.068 |
| 90  | xgb+τ.75 | 0.4686 | 0.4787±0.0058 | 0.4392±0.0073 | 0.196 / 0.031 |
| 140 | xgb+τ.75 | 0.5130 | 0.5130±0.0054 | 0.4743±0.0053 | 0.180 / 0.011 |

All 25 paired seed-deltas positive (sign test p≈0.031 per cutoff); deltas 5–15× split noise.
Accuracy cost 3–6 pp (declared trade-off; τ=0 recovers baseline accuracy from the same model).

Observations: Simple class-prior threshold adjustment on the untouched baseline XGBoost beat
every retraining-time imbalance technique. Negative results: XGB sample-weighting, SMOTE,
SMOTETomek, and all hyperparameter variants ≤ baseline+τ. Dominant residual error is
Withdrawn↔Fail↔Pass confusion; best Withdrawn F1 ≈0.26 (c30) — a data limit, not tuning.
0.70 macro-F1 unrealistic: 13 diverse configs span 0.375–0.419 inner F1 at c30; estimated
practical ceiling ≈0.45–0.48 (c30) / ≈0.55 (c140) with current features.

Decision: Adopt xgb_base+τ=0.75 as the recommended model (τ = deployment precision/recall
knob). Keep Baseline v3 as the protocol reference. Cache is canonical from now on.

Next Ideas: temporal/trajectory features targeting early-withdrawal signal (weekly click
deltas, submission-timing dynamics); dedicated grouped binary "withdraw-within-k-weeks"
early-warning task; per-cutoff τ calibration as standard practice.

---

## Baseline v3 — Methodology Hardening (Phases 1–6)

Date: 2026-07-02

Objective: Harden the pipeline into a scientifically defensible early-prediction benchmark:
audit evaluation leakage, survivorship bias, target validity, temporal validity, and overall
experimental methodology; fix only what evidence proves broken.

Hypothesis: (a) the random row split leaks student identity across train/test
(multi-enrollment); (b) mlData membership (submitted coursework due ≤ cutoff) is
outcome-correlated survivorship; (c) fixing both changes reported metrics but yields the
honest deployment estimate.

Implementation: Evidence collected per cutoff on the v2 pipeline (notebook cells verbatim,
pinned env). Phase 1: overlap quantified — 464–763 overlapping students, 12–17% of test rows
share a student with train (cutoffs 30–140); overlap-conditional accuracy measured (overlap
rows are mostly harder — multi-enrollment students struggle more, so the flaw is protocol
mismatch + identity leakage, not naive memorization inflation). Grouped evaluation
(GroupShuffleSplit/GroupKFold/repeated GSS, identical models) costs ≤ ~2 acc points early,
within split noise late. Phase 2: survivorship quantified against all 32,593 registered
pairs — missing 96.4% (c14), 40.8% (c30), 28% (c60–140); of the still-enrolled risk
population, 31% missing at c30; exclusions are 47% Withdrawn vs 20% in the represented
sample → CONFIRMED bias. Phase 3: 0 duplicate keys, 0 duplicate columns, 0 label mismatches
at every cutoff; up to 4 rows/student = legitimate multi-enrollment. Phase 4: all active
features re-audited SAFE; v3 adds only has_vle_activity / has_coursework (presence of
≤cutoff data → SAFE) and a membership rule using registration facts observable at the
cutoff. Phase 5: full methodology review written (imbalance-handling inconsistency, legacy
binary section, execution-order contract, hidden assumptions made explicit; NaN-free frame
and complete studentVle load verified). Phase 6: Baseline v3 implemented as an additive
notebook section (single notebook, CUTOFF-driven): prediction cases = all registered pairs
still enrolled at the cutoff (exclude date_unregistration ≤ CUTOFF and
date_registration > CUTOFF); leakage-free v2 feature tables merged in; missing info kept as
explicit indicators + zero fill; GroupShuffleSplit(group=id_student, 0.2, seed 42); identical
models/hyperparameters; repeated grouped splits (seeds 0–4) for RF/XGB.

Features Added (v3, 2): has_vle_activity, has_coursework (availability indicators — required
to keep previously-dropped students). Features Redesigned: none. Features Removed: none.

Models Evaluated: unchanged (LogReg balanced, Tree depth5, RF 300 balanced, XGB 300/d6/lr.05).

Validation Strategy (official from v3): GroupShuffleSplit, group=id_student, test_size=0.2,
random_state=42; no student in both train and test (asserted in-notebook); robustness =
repeated grouped splits seeds 0–4 (mean±std).

Results (v3 official; accuracy / macro-F1; pinned sklearn 1.6.1, macOS/arm64):

| Cutoff | cases | LogReg | Tree | RF | XGB |
|-------:|------:|:------|:-----|:---|:----|
| 14  | 28,061 | 0.3512/0.3344 | 0.4782/0.2510 | 0.4626/0.2999 | 0.4810/0.3151 |
| 30  | 27,450 | 0.3831/0.3747 | 0.4817/0.2883 | 0.5048/0.3565 | 0.5099/0.3705 |
| 60  | 26,353 | 0.4368/0.4199 | 0.5383/0.3468 | 0.5591/0.3888 | 0.5614/0.4066 |
| 90  | 25,558 | 0.4391/0.4175 | 0.5713/0.3737 | 0.5874/0.4049 | 0.5947/0.4244 |
| 140 | 24,289 | 0.4995/0.4581 | 0.6238/0.4133 | 0.6530/0.4474 | 0.6627/0.4713 |

Repeated grouped splits: RF/XGB acc std ≈ ±0.004–0.009 → gains < ~1–2 points on a single
split are not evidence.

Observations: v2→v3 is a population + protocol change, not a model change — numbers are not
comparable to v1/v2. Grouping alone costs ~0–2 acc points (tested on unseen students).
The full population raises accuracy at late cutoffs (majority share grows as decided
withdrawals leave the risk pool) but lowers macro-F1 (remaining late-withdrawers are
genuinely hard; c140 Withdrawn test support = 372). v2's higher macro-F1 was earned on an
easier survivor-filtered task. Withdrawn class shrinks with cutoff by design (deployment
semantics). Legacy v1/v2 notebook sections still reproduce committed numbers exactly
(verified every cutoff).

Decision: Adopt grouped evaluation + registered risk population as the OFFICIAL protocol
(Baseline v3). Freeze the pipeline; caching designed in reports/caching_plan.md, execution
deferred until explicitly instructed. v1/v2 remain immutable.

Next Ideas: controlled imbalance-handling experiment (XGB sample_weight / consistent class
weights); grouped binary tasks on the v3 population; per-class recall tracking for Withdrawn
(early-warning core metric); optional stratified-grouped splitter (StratifiedGroupKFold).

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