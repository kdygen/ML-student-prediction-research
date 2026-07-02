# Baseline v1 — OULAD Early Prediction (verified)

**Date:** 2026-07-02
**Notebook:** `notebook/OULAD_early_prediction_v1 (1).ipynb`
**Status:** Official verified baseline. Established by running the existing notebook
implementation unchanged across every available cutoff, under a pinned environment matching
the project's Colab stack.

---

## 1. Purpose

Establish the official baseline by executing the notebook **exactly as written**. No
preprocessing, feature engineering, model, hyperparameter, evaluation, or reporting logic
was modified by hand. The **only** change was the one explicitly permitted: setting the
`CUTOFF` value (cell 5) to each of the cutoffs already defined in the notebook
(`cutoffs = [14, 30, 60, 90, 140]`), and running the same pipeline once per cutoff.

## 2. Environment (pinned)

The baseline is pinned to the project's Colab stack, because RandomForest results depend on
the scikit-learn version (see §6):

| Package | Version |
|---|---|
| Python | 3.12.6 |
| scikit-learn | **1.6.1** |
| pandas | 2.2.2 |
| numpy | 2.0.2 |
| xgboost | 3.3.0 |
| imbalanced-learn | 0.13.0 |
| Platform (this run) | macOS / arm64 |
| Reference platform | Colab Linux / x86_64 |

> `imblearn`/`SMOTE` are imported in cell 2 but never used in the active pipeline.

## 3. How it was run (reproducibility of the run)

- The notebook reads its CSVs by bare filename, so it was executed with the working
  directory set to `data/raw/` (where the six OULAD CSVs live). No read paths were changed.
- Every code cell's source was pulled directly from the `.ipynb` and executed in order in a
  fresh namespace, once per cutoff. The cutoff line `CUTOFF = 30  # temporary single test`
  was programmatically set to each cutoff value; nothing else was altered.
- Efficiency note (no effect on results): the raw-CSV load (cell 4, a 453 MB read) was run
  **once** and the pristine frames deep-copied into each per-cutoff run. Because the cutoff
  only affects cells from #5 onward and the raw inputs are identical, this is numerically
  identical to running the whole notebook independently five times.
- The notebook's two duplicate model sections were both captured: the Random Forest rerun
  (cell 129) reproduces the first RF (cell 115) **exactly**, and the XGBoost rerun (cell 130)
  reproduces the first XGB (cell 119) **exactly**.
- Execution completed with **no errors** at any cutoff. One non-fatal `FutureWarning`
  (pandas `replace` downcasting, from the `target_multi` mapping in cell 23) is emitted but
  does not affect results.

## 4. Fixed configuration (identical across all cutoffs)

### Active feature list — 17 features (`base_features + edu_features`)

`base_features` (13):
`weighted_average`, `active_days`, `clicks_per_day`, `clicks_per_assessment`,
`study_spread`, `burstiness`, `resource_ratio`, `oucontent_ratio`, `homepage_ratio`,
`forumng_ratio`, `quiz_ratio`, `assessment_focus`, `recovery_slope`

`edu_features` (4, one-hot of `highest_education`, `drop_first=True`):
`highest_education_HE Qualification`, `highest_education_Lower Than A Level`,
`highest_education_No Formal quals`, `highest_education_Post Graduate Qualification`

> The full `mlData` has 78 columns, but the active model input `X` is only these 17.
> Demographic/region/imd dummies, `completion_ratio`, `deadline_panic_score`,
> `disappearance_ratio`, etc. are computed but **not** in the active feature set (commented
> out in the `base_features` list). Several exploratory features remain as commented-out
> cells (preserved research history).

### Target

`target_multi` — 4-class: `0 = Withdrawn`, `1 = Fail`, `2 = Pass`, `3 = Distinction`.

### Train/test split setup

- `train_test_split(X, Y, test_size=0.2, random_state=42)` — **80/20, no stratification**
  (multiclass models). One split reused for LogReg, Decision Tree, Random Forest, XGBoost.
- Binary-RF section uses per-pair `train_test_split(..., test_size=0.2, random_state=42,
  stratify=Y_pair)`.

### Models & hyperparameters (as in notebook)

- **Logistic Regression:** `Pipeline(StandardScaler → LogisticRegression(max_iter=5000,
  class_weight="balanced", random_state=42))`.
- **Decision Tree:** `DecisionTreeClassifier(max_depth=5, random_state=42)`.
- **Random Forest:** `RandomForestClassifier(n_estimators=300, class_weight="balanced",
  random_state=42)`.
- **XGBoost:** `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, objective="multi:softmax", num_class=4,
  eval_metric="mlogloss", random_state=42)`.
- **Binary RF (paper-style pairs):** `RandomForestClassifier(random_state=42)` (defaults,
  `n_estimators=100`) on 4 one-vs-one class pairs.

## 5. Dataset shape per cutoff

`mlData` is 78 columns at every cutoff; the row count and class balance grow with the cutoff.

| Cutoff | mlData shape | X shape | Train | Test | Withdrawn(0) | Fail(1) | Pass(2) | Distinction(3) |
|-------:|:-------------|:--------|------:|-----:|-----:|-----:|------:|-----:|
| 14  | (1188, 78)  | (1188, 17)  | 950   | 238  | 148  | 316  | 558   | 166  |
| 30  | (19300, 78) | (19300, 17) | 15440 | 3860 | 3900 | 4406 | 8906  | 2088 |
| 60  | (23411, 78) | (23411, 17) | 18728 | 4683 | 4400 | 5163 | 11220 | 2628 |
| 90  | (23452, 78) | (23452, 17) | 18761 | 4691 | 4414 | 5180 | 11230 | 2628 |
| 140 | (23478, 78) | (23478, 17) | 18782 | 4696 | 4423 | 5192 | 11235 | 2628 |

> The jump from 1,188 rows (cutoff 14) to ~19,300 (cutoff 30) is because a student only enters
> `mlData` once they have graded coursework with `weight > 0` on/before the cutoff; almost no
> coursework is due by day 14. Row counts then plateau (~23.4k) by cutoff 60.

## 6. Model results per cutoff

### 6.1 Multiclass (4-class `target_multi`) — accuracy / macro-F1

Environment: pinned stack (§2), this run on macOS/arm64. **Bold** = best per column-group per row.

| Cutoff | LogReg acc | LogReg F1 | Tree acc | Tree F1 | RF acc | RF F1 | XGB acc | XGB F1 |
|-------:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 14  | 0.3151 | 0.2950 | **0.5042** | 0.2773 | 0.4664 | 0.2969 | 0.4370 | **0.3234** |
| 30  | 0.3886 | **0.3851** | 0.5083 | 0.3383 | 0.5062 | 0.3619 | **0.5142** | 0.3836 |
| 60  | 0.4273 | 0.4187 | 0.5274 | 0.3809 | 0.5471 | 0.4127 | **0.5509** | **0.4421** |
| 90  | 0.4507 | 0.4388 | 0.5508 | 0.4435 | 0.5675 | 0.4476 | **0.5781** | **0.4756** |
| 140 | 0.5002 | 0.4868 | 0.5771 | 0.4914 | 0.6105 | 0.5135 | **0.6220** | **0.5407** |

(Random Forest and XGBoost each have a duplicate "rerun" cell in the notebook; both reproduce
the values above exactly and are therefore not listed separately.)

**RandomForest — cross-platform reference (cutoff 30):** this macOS/arm64 run = **0.5062**
(1954/3860); the same code on Colab Linux/x86_64 = **0.5098** (1968/3860). See §7.

### 6.2 Binary Random Forest (paper-style pairs) — accuracy

(Identical under sklearn 1.6.1 and 1.9.0; these smaller stratified pairs were not
version-sensitive.)

| Cutoff | Pass vs Fail | Distinction vs Fail | Distinction vs Pass | Withdrawn vs Pass |
|-------:|-----:|-----:|-----:|-----:|
| 14  | 0.6571 | 0.7732 | 0.7793 | 0.7676 |
| 30  | 0.7169 | 0.8214 | 0.8136 | 0.7459 |
| 60  | 0.7522 | 0.8525 | 0.8249 | 0.7910 |
| 90  | 0.7629 | 0.8796 | 0.8377 | 0.8345 |
| 140 | 0.8065 | 0.9239 | 0.8395 | 0.8560 |

## 7. Reproducibility & sensitivity (investigated 2026-07-02)

A discrepancy was found and fully diagnosed: an initial run on scikit-learn **1.9.0** gave RF
cutoff-30 accuracy **0.4508**, versus **0.5098** from a manual Colab run. Root cause,
decomposed:

1. **scikit-learn version (dominant, ~96% of the gap).** With the split, features, and RF
   hyperparameters held identical, `RandomForestClassifier` produces different trees across
   sklearn versions (sklearn does not guarantee cross-version reproducibility for a fixed
   `random_state`): sklearn 1.9.0 → 0.4508, sklearn 1.6.1 → 0.5062. **We pin sklearn 1.6.1**
   (the Colab version) for the baseline. This affects RF's accuracy **and** macro-F1 (e.g.
   cutoff 30 macro-F1 0.4071 on 1.9.0 → 0.3619 on 1.6.1) and changes the "best model"
   conclusion (see §8). LogReg, Decision Tree, and XGBoost were **byte-identical** across the
   two sklearn versions.
2. **Platform floating-point (residual, ~4% of the gap).** With the exact Colab stack
   (sklearn 1.6.1 / pandas 2.2.2 / numpy 2.0.2), macOS/arm64 gives 0.5062 vs Colab
   Linux/x86_64 0.5098. Verified by cross-environment hashing:
   - The train/test **split is byte-identical** (`X_test.index` sha256 `99a04ba0…c6c0b`,
     `Y_test = {W:774, F:878, P:1810, D:398}`, same first-10 indices) — so this is **not** a
     split or pipeline difference.
   - **16 of the 17 active features are bit-identical** across platforms. **Only `burstiness`
     differs, and only below 1e-9** (its round-9 hash matches exactly; full-precision hash
     differs). `burstiness = std/(mean+1)` — the `.std()` reduction rounds differently on
     arm64 (Apple Accelerate) vs x86_64 (OpenBLAS). That single sub-nanoscale difference,
     amplified by RF's tie-sensitive split thresholds, flips **14 of 3860** predictions.
   - pandas 2.2.2↔2.2.3 and numpy 2.0.2↔2.5.0 produced **byte-identical** `mlData` on the same
     machine, so those are not a factor.

**Reproducibility guarantees for Baseline v1:**
- Data pipeline is reproducible to **<1e-9** (round-9 feature hashes identical) and the split
  is **byte-identical** across platforms/pandas/numpy.
- **RandomForest** absolute metrics are reproducible only up to **~±0.4%** across CPU
  architecture, and are **sklearn-version-dependent** — always compare RF against a baseline
  produced with the same sklearn version.
- LogReg / Decision Tree / XGBoost were stable across the sklearn versions tested; their
  cross-platform variation was not separately measured but is expected to be ≤ the RF
  tolerance.

## 8. Observations

1. **Monotonic improvement with cutoff.** Every model improves as the cutoff moves later, for
   both accuracy and macro-F1 — more information available → better prediction.
2. **XGBoost is the strongest overall.** It has the highest accuracy at cutoffs 30/60/90/140
   and the highest macro-F1 at 14/60/90/140. Exceptions: at the data-starved cutoff 14 the
   Decision Tree has the highest accuracy (0.5042, largely majority-class), and at cutoff 30
   Logistic Regression edges XGBoost on macro-F1 (0.3851 vs 0.3836).
3. **Random Forest sits just below XGBoost** on accuracy and has notably lower macro-F1 under
   the pinned sklearn 1.6.1 — i.e. it leans toward the majority (Pass) class. (Under the newer
   sklearn 1.9.0 the RF was more balanced/higher macro-F1; that behavior is version-specific
   and not the pinned baseline.)
4. **Cutoff 14 is data-starved** (1,188 rows) and least reliable. Cutoff 30 is the notebook's
   default and the first full-size dataset. Row counts plateau (~23.4k) by cutoff 60.
5. **Feature set and split are identical across cutoffs** (17 features, 80/20, seed 42), so
   results are directly comparable cutoff-to-cutoff.
6. **Class imbalance** is stable from cutoff 60 onward (~48% Pass, ~22% Fail, ~19% Withdrawn,
   ~11% Distinction).

### Caveats to revisit in later experiments (not changed here — baseline only)
- The 80/20 split is **not stratified** and is **not grouped by student**; the same
  `id_student` can appear in multiple module/presentation rows, so per-student leakage across
  train/test is possible.
- Later cutoffs (90, 140) fall well past the intended "early" prediction window; 14/30 are the
  genuinely "early" settings for the research goal.
- `study_spread = active_days / CUTOFF` and several ratio features depend directly on the
  cutoff — worth auditing for leakage in future work.
- **RandomForest metrics are sklearn-version- and platform-fragile** (§7); prefer XGBoost or
  seeded/averaged reporting when small RF differences would change a conclusion.

## 9. Artifacts

- Raw per-cutoff metrics (machine-readable): `reports/baseline_v1_results.json`
- This document: `reports/baseline_v1.md`
