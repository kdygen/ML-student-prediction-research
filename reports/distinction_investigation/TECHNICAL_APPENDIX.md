# Technical Appendix — All 21 Experiments

Complete specification of every arm in the Distinction investigation.
**Invariants across all 21 experiments** (never varied, so differences are attributable only
to the stated change):

| Constant | Value |
|---|---|
| Population | Engaged: 29,496 enrolments / 26,358 students (excludes 3,097 unregistered ≤ day 0) |
| Class counts | Withdrawn 7,067 · Fail 7,044 · Pass 12,361 · Distinction 3,024 |
| Horizon | h = min(`date_unregistration`, `course_end`) per student |
| Censoring | All VLE rows and submissions with date > h deleted before any aggregation |
| Excluded always | exams (scores/submissions/attendance), `final_result`, `date_unregistration` as a feature, coursework **scores** (`rank_wa`, `score_slope_cw`, `score_std_cw`) |
| Outer CV | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped by `id_student`, zero-overlap asserted every fold |
| Inner CV (tuning only) | `GroupKFold(n_splits=3)` on the training half, grouped by `id_student` |
| Base learner | `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, objective="multi:softmax", num_class=4, eval_metric="mlogloss", n_jobs=-1)` |
| Reproduction check | Baseline re-run matched Experiment 009 Arm B to **Δaccuracy = 0.000000, Δmacro-F1 = 0.000000** before any experimentation |

**A note on SHAP.** SHAP is not installed in the pinned environment (Python 3.12.6,
scikit-learn 1.6.1, pandas 2.2.2, numpy 2.0.2, xgboost 3.3.0). Installing it risks upgrading
numpy/scipy and breaking the bit-exact reproducibility the project depends on. Instead this
appendix reports **XGBoost gain/cover/weight** and **permutation importance** (model-agnostic,
and for a ranking question arguably more faithful than SHAP, since it measures the actual
AUC an input is responsible for). All values in `appendix_importance.json`.

---

# PART A — Iteration 1: decision architecture (experiments 1–6)

Code: `code/02_models.py` (+ `code/02b_ordinal_ceiling.py` for arms 5–6 after a bin-monotonicity fix).
Feature set fixed at the 33 score-free features throughout.

## Experiment 1 — M0 `baseline_argmax`

- **Hypothesis:** none (reference point). Establishes what the score-free model does under the
  default decision rule.
- **Code changes:** none — the Experiment 009 Arm B configuration, re-run to capture
  out-of-fold probabilities (`dx_oof_proba.npy`) for diagnosis.
- **Features:** 33 (the audited 36 minus the 3 score-derived).
- **Class weights:** none (uniform).
- **Threshold:** `argmax` over the 4 class probabilities.
- **Model config:** base learner, unmodified.
- **CV metrics:** accuracy **0.8053 ± 0.0028** · macro-F1 **0.6951 ± 0.0041** ·
  D P/R/F1 **0.563 / 0.142 / 0.227** · W 0.941 · F 0.791 · P 0.822.
- **Diagnostic output:** 85.3% of true Distinctions predicted Pass; mean p(D) for true
  Distinction = 0.289 (median 0.263); only **13.3%** exceed p(D) > 0.5. P-vs-D ranking
  AUC = **0.746**, PR-AUC 0.423 at 19.7% prevalence.
- **Interpretation:** the model *ranks* Distinction students well but almost never *selects*
  them. With Pass 4.09× more prevalent, p(Pass) beats p(D) at argmax for 87% of true
  Distinctions. This split the problem into a fixable operating-point component and an
  information component, and set the agenda for everything that followed.

## Experiment 2 — M1 `inverse_freq_weights` ★ the decisive fix

- **Hypothesis:** Distinction F1 is suppressed by the argmax operating point under class
  imbalance. Re-weighting the training loss by inverse class frequency should recover F1
  **without any new information**.
- **Code changes:** added `sample_weight=np.vectorize(cw.get)(y[tr])` to `.fit()`; nothing else.
- **Class weights** — `w_c = N / (4 · n_c)`, computed on the full engaged population, fixed
  a priori (never tuned):

  | Class | n | weight |
  |---|--:|--:|
  | Withdrawn | 7,067 | 1.043441 |
  | Fail | 7,044 | 1.046848 |
  | Pass | 12,361 | **0.596554** |
  | Distinction | 3,024 | **2.438492** |

  Effective Distinction:Pass leverage = 2.438 / 0.597 = **4.09×**, exactly cancelling the
  prevalence ratio.
- **Threshold:** argmax (unchanged) — the weights alone move the decision boundary.
- **CV metrics:** accuracy 0.7569 · macro-F1 **0.7288 ± 0.0036** ·
  D P/R/F1 **0.364 / 0.595 / 0.452** · W 0.941 · F 0.789 · P 0.733.
- **Why it worked:** Distinction recall rose 0.142 → 0.595 (4.2×) while precision fell
  0.563 → 0.364 — a pure movement along the existing ROC curve, confirming the ranking
  information was already present. Macro-F1 *increased* by +0.034 because the Distinction gain
  (+0.225) far exceeded the Pass loss (−0.089). Withdrawn and Fail were untouched (±0.002),
  since their prevalences are near the mean and their weights ≈ 1.04.
- **Cost:** accuracy −4.8 pp. This is the honest trade: accuracy rewards the majority class,
  macro-F1 does not.

## Experiment 3 — M2 `D_threshold_rescue`

- **Hypothesis:** an explicit per-fold decision threshold on p(D) should match or beat implicit
  re-weighting, and allows a controlled macro-F1 guard.
- **Code changes:** per outer fold, fit 3 inner GroupKFold models to obtain unbiased inner OOF
  probabilities; scan τ ∈ [0.10, 0.50] step 0.02; accept the τ maximising inner Distinction F1
  **subject to** inner macro-F1 ≥ (argmax macro-F1 − 0.005); apply to the outer test fold.
- **Class weights:** none (this arm isolates thresholding).
- **CV metrics:** accuracy 0.7570 · macro-F1 0.7286 ± 0.0034 ·
  D P/R/F1 0.357 / 0.600 / **0.447** · W 0.941 · F 0.791 · P 0.737.
- **Why it neither improved nor failed:** statistically indistinguishable from M1
  (ΔD-F1 = −0.005, within fold noise). Re-weighting and thresholding are two parameterisations
  of the same operating-point move. M1 is preferred for simplicity — one hyperparameter-free
  change versus a nested tuning loop.

## Experiment 4 — M3 `hierarchical_PvsD`

- **Hypothesis:** a dedicated binary classifier trained **only** on Pass+Distinction students
  will learn a sharper boundary than a 4-class model that must also separate W/F.
- **Code changes:** two-stage. Stage 1: the standard 4-class model assigns W/F/P/D. Stage 2:
  every case predicted P or D is re-decided by a binary XGBoost trained on the P+D subset with
  `scale_pos_weight = n_Pass/n_Distinction` (≈ 4.09), threshold tuned on inner OOF for
  Distinction F1 over τ ∈ [0.20, 0.80] step 0.02.
- **CV metrics:** accuracy 0.7399 · macro-F1 0.7196 ± **0.0110** ·
  D P/R/F1 0.331 / 0.652 / **0.439** · W 0.941 · F 0.791 · P 0.708.
- **Why it failed (relative to M1):** worse D F1 (−0.013), worse macro-F1 (−0.009), and **3×
  the fold variance** (±0.0110 vs ±0.0036). Specialising on P+D discards the W/F students as
  training signal, so the binary model sees only 15,385 rows instead of 29,496 and overfits the
  boundary. The extra recall (0.652) is bought with precision (0.331) — the same trade as M1,
  reached less stably. **Conclusion: the P/D boundary is not improved by isolating it.**

## Experiment 5 — M4 `ordinal_regression`

- **Hypothesis:** the outcomes are ordered (Withdrawn < Fail < Pass < Distinction), so an
  ordinal formulation should exploit structure that nominal multiclass ignores.
- **Code changes:** `XGBRegressor` (same tree hyperparameters) on the integer-coded target;
  three cut points tuned by exhaustive grid on inner OOF maximising macro-F1; prediction by
  `np.digitize`. *Bug found and fixed:* the initial grid produced non-monotonic bins
  (`ValueError: bins must be monotonically increasing`); corrected to enforce
  c1 < c2−0.25 < c3−0.25 (`code/02b_ordinal_ceiling.py`).
- **CV metrics:** accuracy 0.7751 · macro-F1 0.7181 ± 0.0016 ·
  D P/R/F1 0.433 / 0.393 / **0.412** · W **0.920** · F **0.751** · P 0.789.
- **Why it failed:** worst macro-F1 of the working arms, and uniquely it **damaged the other
  classes** — Withdrawn 0.941 → 0.920, Fail 0.791 → 0.751. Forcing a single latent scale means
  errors propagate across the whole ordering: a student mispredicted between Pass and
  Distinction shifts the same axis that separates Withdrawn from Fail. **Evidence the 4-class
  problem is not usefully ordinal**, because Withdrawn is a behavioural/administrative state,
  not a low rung on an achievement ladder.

## Experiment 6 — M5 `PvsD_binary_ceiling` ★ the ceiling probe

- **Hypothesis:** measuring the P-vs-D boundary in isolation, with balanced weighting and an
  optimal threshold, gives the **maximum extractable** Distinction performance from the
  33-feature set — a ceiling independent of the multiclass decision rule.
- **Code changes:** subset to P+D (15,385 rows), binary XGBoost with
  `scale_pos_weight = n_Pass/n_D` per fold, `eval_metric="logloss"`; threshold tuned on inner
  GroupKFold(3) OOF; report AUC/PR-AUC (threshold-free) and F1 at the tuned threshold.
- **CV metrics:** **AUC 0.7370 ± 0.0112** · PR-AUC 0.4162 ·
  D-F1 at tuned τ **0.449 ± 0.021**.
- **Why this matters more than any single arm:** the dedicated, balanced, optimally-thresholded
  binary reaches F1 0.449 — **statistically identical to M1's 0.452 and M2's 0.447**. Three
  independent routes converge on ≈ 0.45. That convergence is the first hard evidence that
  0.45 is a property of the *feature set*, not of any decision architecture, and it defined
  the target for Iterations 2–3: to beat it, new **information** was required.

---

# PART B — Iteration 2: feature hypotheses (experiments 7–16)

Code: `code/03_build_features.py` (construction), `code/04_feature_eval.py` (evaluation).
**Metric choice:** P-vs-D binary AUC/PR-AUC on the P+D subset, same folds. Rationale — AUC is
threshold-free, so it measures *information gained* rather than operating-point movement,
which Iteration 1 showed dominates F1.
**Pre-registered acceptance rule (stated before running):** a group is retained iff it improves
AUC by > **+0.002** over base_33.

## Feature definitions (22 new features, all leakage-audited)

| Group | Feature | Definition | Raw source |
|---|---|---|---|
| **H1** regularity | `weekly_cv` | std ÷ (mean+1) of weekly click totals | studentVle ≤ h |
| | `inactive_week_share` | 1 − (active weeks ÷ enrolled weeks), clipped [0,1] | studentVle ≤ h |
| | `max_week_streak` | longest run of consecutive active weeks | studentVle ≤ h |
| | `week_entropy` | −Σp·log p over weekly click shares, ÷ log(n weeks) | studentVle ≤ h |
| | `phase_concentration` | max share of clicks falling on one day-of-7 phase | studentVle ≤ h |
| **H2** depth ★ | `unique_sites` | count of distinct `id_site` touched | studentVle+vle ≤ h |
| | `revisit_ratio` | mean number of distinct days per visited site | studentVle ≤ h |
| | `top3_site_share` | clicks on the 3 most-used sites ÷ total clicks | studentVle ≤ h |
| | `content_admin_ratio` | clicks on learning content ÷ clicks on navigation (homepage, subpage) | + `vle.activity_type` |
| | `enrichment_share` | clicks on optional types (glossary, ouwiki, dataplus, htmlactivity, collaborate, externalquiz…) ÷ total | + `vle.activity_type` |
| | `n_activity_types` | count of distinct activity types used (of 20) | + `vle.activity_type` |
| **H3** proactivity | `early_work_share` | clicks 8–21 days before a coursework deadline ÷ clicks 0–21 days before | + published deadlines |
| | `first_active_day` | first day of any VLE activity (negative = pre-course) | studentVle ≤ h |
| **H4** spacing | `mean_gap`, `gap_std` | mean and std of gaps between consecutive active days | studentVle ≤ h |
| | `weekly_slope` | OLS slope of weekly clicks against week index | studentVle ≤ h |
| | `late_new_content_share` | resources first opened after h/2 ÷ all resources opened | studentVle ≤ h |
| **H5** cohort-weekly | `mean_weekly_z` | mean z-score of weekly clicks vs (module, presentation, week) cohort | peers' censored behaviour |
| | `share_topq_weeks` | share of weeks in the cohort's top activity quartile | peers' censored behaviour |
| | `weekly_z_slope` | OLS slope of the weekly z-score over time | peers' censored behaviour |
| **H6** enrolment | `studied_credits`, `num_of_prev_attempts` | registration-time records | studentInfo |

**Univariate effect sizes (Cohen's d, Distinction − Pass), top 8 of the new features:**
`share_topq_weeks` +0.475 · `mean_weekly_z` +0.399 · `inactive_week_share` −0.377 ·
`max_week_streak` +0.375 · `gap_std` −0.335 · `mean_gap` −0.318 · `revisit_ratio` +0.312 ·
`week_entropy` +0.301. All real and all in the expected direction — but note none exceeds
d = 0.5, and every one correlates with engagement volume.

## Experiments 7–14 — P-vs-D AUC per feature set

Model for all eight: binary XGBoost, `scale_pos_weight = n_Pass/n_D` per fold,
`eval_metric="logloss"`, no threshold (AUC is threshold-free).

| # | Arm | Features | AUC | PR-AUC | ΔAUC vs base | Verdict |
|--:|---|--:|--:|--:|--:|---|
| 7 | `base_33` | 33 | 0.7370 ± 0.0112 | 0.4162 | — | reference |
| 8 | `+H1_regularity` | 38 | 0.7381 ± 0.0087 | 0.4172 | +0.0011 | ✗ below rule |
| 9 | **`+H2_depth`** | **39** | **0.7488 ± 0.0098** | **0.4369** | **+0.0118** | ✓ **retained** |
| 10 | `+H3_proactivity` | 35 | 0.7383 ± 0.0095 | 0.4173 | +0.0013 | ✗ |
| 11 | `+H4_spacing` | 37 | 0.7351 ± 0.0108 | 0.4121 | −0.0019 | ✗ (hurt) |
| 12 | `+H5_cohort_week` | 36 | 0.7371 ± 0.0088 | 0.4163 | +0.0001 | ✗ |
| 13 | `+H6_enrolment` | 35 | 0.7379 ± 0.0103 | 0.4169 | +0.0009 | ✗ |
| 14 | `+ALL_new` | 55 | 0.7466 ± 0.0112 | 0.4332 | +0.0096 | ✗ (below H2 alone) |

**Why H2 succeeded.** Depth/revisit features measure a *different axis* from volume: how a
student distributes attention across resources, not how much they click. `revisit_ratio`
(distinct days per site) captures returning to material — the behavioural signature that
self-regulated-learning theory (Kizilcec et al. 2017, *Computers & Education*) associates
with goal attainment. `content_admin_ratio` and `enrichment_share` separate substantive study
from navigation and capture use of optional material. This is the only construct tested that
is not a monotone transform of "clicked more."

**Why H1, H3, H4, H5, H6 failed.** Despite healthy univariate effect sizes, they are
**conditionally redundant** — once the 33 base features (which already include `active_days`,
`active_weeks`, `study_spread`, `max_gap`, `decay_clicks`, `rank_clicks`, submit-lead features)
are in the model, these add no orthogonal signal. `share_topq_weeks` has d = +0.475 alone but
is nearly a deterministic function of `rank_clicks` (already present). H4 actively *hurt*
(−0.0019): four correlated gap statistics added variance without signal, and tree models
split noise when given redundant continuous inputs.

**Why ALL-22 < H2 alone** (0.7466 < 0.7488): adding 16 uninformative features dilutes
`colsample_bytree=0.8` sampling — each tree is less likely to see the useful features. This is
direct evidence that the failures are not merely neutral but mildly harmful, and validates the
per-group testing design over a kitchen-sink approach.

## Experiments 15–16 — the 4-class check

Model: weighted 4-class XGBoost (M1 weights above), argmax.

| # | Arm | Features | Accuracy | Macro-F1 | D F1 (P/R) | W / F / P |
|--:|---|--:|--:|--:|:--|:--|
| 15 | `weighted_base_33` | 33 | 0.7569 | 0.7288 ± 0.0036 | 0.452 (0.36/0.60) | 0.941 / 0.789 / 0.733 |
| 16 | `weighted_best_feats` | 39 | 0.7616 | 0.7314 ± 0.0055 | 0.455 (0.37/0.58) | 0.940 / 0.789 / 0.742 |

**Interpretation:** H2's +0.0118 AUC translates to only **+0.003 D F1** — real but inside fold
noise (±0.005). Information gained at the ranking level does not automatically convert to
decision-level gains when the operating point is already the binding constraint. Honest
reading: **H2 is a genuine but small information gain**, and the F1 difference alone would not
justify the features; the AUC evidence does.

---

# PART C — Iteration 3: representation and model family (experiments 17–21)

Code: `code/05_final_iteration.py`.

## Experiments 17–20 — module identity and an alternative GBDT

- **Hypothesis (module):** per-module Distinction F1 ranged 0.00 (AAA) to 0.40 (CCC) and
  Distinction prevalence varies by course, so module dummies should let the model calibrate
  per-course priors and decision boundaries. Module code is static course metadata —
  leakage-free.
- **Hypothesis (HistGB):** if the ≈0.75 AUC ceiling is an XGBoost artefact, a different
  gradient-boosting implementation should exceed it.
- **Code changes:** `pd.get_dummies(code_module, drop_first=True)` → 6 dummies;
  `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42, class_weight={0:1.0, 1:scale_pos_weight})`.
- **LightGBM/CatBoost:** deliberately **not** tested — neither is in the pinned environment, and
  installing them risks dependency upgrades that would invalidate the project's bit-exact
  reproducibility guarantee. Documented as a limitation rather than silently skipped.

| # | Arm | Model | AUC | PR-AUC | Verdict |
|--:|---|---|--:|--:|---|
| 17 | `33+H2` | XGBoost | 0.7488 ± 0.0098 | 0.4369 | reference |
| 18 | `33+H2+module` | XGBoost | 0.7491 ± 0.0115 | 0.4373 | +0.0003 — **nothing** |
| 19 | `33+H2` | HistGradientBoosting | 0.7418 ± 0.0110 | 0.4188 | −0.0070 — worse |
| 20 | `33+H2+module` | HistGradientBoosting | 0.7440 ± 0.0110 | 0.4265 | −0.0048 — worse |

**Why module dummies failed at the P/D boundary:** the per-module F1 variation was driven by
*prevalence and sample size* (AAA has 748 students and few Distinctions), not by
course-specific behavioural signatures. Cohort-relative features already in the base set
(`rank_clicks`) absorb between-module differences. Notably, module dummies do carry
substantial **gain** in the final 4-class model (12.5% — see Part D) because they help
calibrate priors across all four classes; they simply do not separate Pass from Distinction.
**A feature can be globally important and locally useless** — a distinction the AUC probe
made visible and gain alone would have hidden.

**Why the alternative GBDT failed:** HistGradientBoosting scored *below* XGBoost on both
feature sets. Two independent boosting implementations agreeing within ~0.007 AUC is strong
evidence the ceiling is a property of the data, not of one library's inductive bias.

## Experiment 21 — FINAL stack ★

Configuration: 33 base + 6 H2 + 6 module dummies = **45 features**; inverse-frequency class
weights; per-fold Distinction threshold tuned on inner GroupKFold(3) OOF.

- **CV metrics:** accuracy 0.7554 · macro-F1 **0.7306 ± 0.0066** ·
  D P/R/F1 **0.36 / 0.63 / 0.458** · W **0.943** · F **0.795** · P 0.726.
- **Per-fold Distinction F1:** 0.451, 0.491, 0.453, 0.456, 0.439 — every fold ≫ baseline 0.227.
- **Per-fold tuned thresholds τ:** 0.44, 0.46, 0.48, 0.44, 0.44 (mean 0.452, range 0.04 —
  remarkably stable, indicating a well-conditioned optimum rather than noise-fitting).

---

# PART D — The winning model, in full

## D.1 How the class weights were computed

Inverse class frequency, normalised so total weight equals sample count:

```
w_c = N / (K · n_c)        N = 29,496,  K = 4
```

| Class | n_c | w_c | Effect |
|---|--:|--:|---|
| Withdrawn | 7,067 | **1.043441** | ≈ neutral |
| Fail | 7,044 | **1.046848** | ≈ neutral |
| Pass | 12,361 | **0.596554** | down-weighted 1.68× |
| Distinction | 3,024 | **2.438492** | up-weighted 2.44× |

Applied via `sample_weight` at `.fit()` time. **Computed once from the full population and
fixed a priori — never tuned, never fitted per fold, never touched by test data.** The
Distinction:Pass leverage ratio is 4.087, which is exactly the prevalence ratio
12,361/3,024 = 4.088 — so the weighting precisely neutralises the prior, no more.

## D.2 How the threshold was optimised

Per outer fold, with **no test data involved at any point**:

1. Split the outer training half into 3 inner folds with `GroupKFold(3)` (grouped by student).
2. Train a weighted 4-class model on each inner train, predict on each inner validation →
   unbiased inner out-of-fold probability matrix.
3. Compute the argmax baseline's inner macro-F1 as a guard value.
4. Scan τ ∈ [0.20, 0.60] step 0.02. For each τ, override the argmax label with Distinction
   wherever p(D) > τ.
5. Accept the τ maximising **inner Distinction F1**, subject to the constraint
   `inner macro-F1 ≥ argmax macro-F1 − 0.005` (prevents buying Distinction at the expense of
   overall balance).
6. Refit on the full outer training half; apply the selected τ once to the outer test fold.

Selected τ per fold: **[0.44, 0.46, 0.48, 0.44, 0.44]**. Note τ ≈ 0.45 is *below* the naive
0.5 but far above the 0.25 that pure prior-correction would imply — the weighting has already
done most of the work, and the threshold provides a small final adjustment.

## D.3 Which revisit features were added, and why they help

The six **H2 depth** features (definitions in Part B). Their contribution measured two ways:

**Gain importance in the final model:** H2 total = **5.64%** of gain (individually modest,
none in the top 5).

**Permutation importance on the P-vs-D task specifically** (AUC drop, base AUC 0.7425):

| Feature | AUC drop |
|---|--:|
| `unique_sites` | +0.0069 |
| `enrichment_share` | +0.0053 |
| `revisit_ratio` | +0.0048 |
| **H2 group total** | **+0.0194** |

The mismatch is the key insight: H2 carries only 5.6% of global gain but **+0.019 of the
P-vs-D AUC** — these features matter disproportionately for the boundary in question relative
to their overall model contribution. That is precisely why the AUC probe (rather than gain
inspection or overall F1) was the correct instrument for this investigation.

**Why they improve Distinction, mechanistically.** Every other feature family answers "how
much did this student engage?" Distinction and Pass students differ on that axis only
modestly (largest d = 0.58) and with heavy overlap. H2 answers a different question: "how did
they distribute their attention?" A Distinction student returns to the same material across
multiple days (`revisit_ratio` ↑, d = +0.312), covers more distinct resources
(`unique_sites` ↑), spends proportionally more time on learning content than on navigation
(`content_admin_ratio` ↑), and engages optional/enrichment material a Pass student skips
(`enrichment_share` ↑). This matches the self-regulated-learning literature, where reviewing
previously-studied material is the behaviour that distinguishes goal-attaining learners.
It is the only tested construct genuinely orthogonal to volume — which is why it is the only
one that survived.

## D.4 Complete pipeline

```
INPUT: studentInfo, studentRegistration, studentVle, vle, assessments (studentAssessment for
       submission timing only — scores never read)

1. POPULATION
   all 32,593 registered (student, module, presentation) triples
   drop date_unregistration ≤ 0                                  → 29,496

2. HORIZON            h = min(date_unregistration, course_end)    per enrolment
   course_end = max observed VLE day in that (module, presentation)

3. CENSOR             delete VLE rows with date > h
                      delete submissions with date_submitted > h
                      delete ALL exam records (type == "Exam")

4. FEATURES (45)
   33 base score-free   engagement volume/rhythm/recency, activity-type shares,
                        submission timing/completion, cohort click rank, registration lead,
                        prior-education dummies
                        (excludes rank_wa, score_slope_cw, score_std_cw)
    6 H2 depth          unique_sites, revisit_ratio, top3_site_share,
                        content_admin_ratio, enrichment_share, n_activity_types
    6 module dummies    pd.get_dummies(code_module, drop_first=True)
   sentinel: first_submit_day = 999 when never submitted

5. OUTER CV           StratifiedGroupKFold(5, shuffle=True, random_state=42), groups=id_student
                      assert no student in both sides of any fold
   FOR EACH FOLD:
     5a. inner GroupKFold(3) on training half → OOF probabilities
     5b. scan τ ∈ [0.20, 0.60] step 0.02, maximise inner Distinction F1
         subject to inner macro-F1 ≥ argmax macro-F1 − 0.005
     5c. fit XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, random_state=42,
             objective="multi:softmax", num_class=4, eval_metric="mlogloss")
         with sample_weight = {W:1.0434, F:1.0468, P:0.5966, D:2.4385}
     5d. predict test fold; pred = argmax(P); pred[P[:,3] > τ] = Distinction

6. OUTPUT   accuracy 0.7554 · macro-F1 0.7306 ± 0.0066
            Withdrawn 0.943 · Fail 0.795 · Pass 0.726 · Distinction 0.458
```

## D.5 Final model gain importance (top 12 of 45)

| # | Feature | Gain | Group |
|--:|---|--:|---|
| 1 | `submitted_count` | 0.1847 | base |
| 2 | `decay_clicks` | 0.0809 | base |
| 3 | `rank_clicks` | 0.0592 | base |
| 4 | `active_weeks` | 0.0551 | base |
| 5 | `completion_ratio_avail` | 0.0530 | base |
| 6 | `study_spread` | 0.0423 | base |
| 7 | `n_assess_types_submitted` | 0.0359 | base |
| 8 | `mod_GGG` | 0.0335 | module |
| 9 | `highest_education_Lower Than A Level` | 0.0261 | base |
| 10 | `engagement_decay_ratio` | 0.0261 | base |
| 11 | `mod_FFF` | 0.0255 | module |
| 12 | `max_gap` | 0.0246 | base |

Group totals: base 82.9% · module dummies 12.5% · H2 depth 5.6%.

## D.6 What the appendix does not claim

The winning configuration improves the **decision rule** and adds a **small genuine
information gain** (H2). It does not overcome the underlying ceiling: across 21 experiments
the P-vs-D ranking moved only 0.737 → 0.749 AUC. Distinction precision remains 0.36 — roughly
two in three flagged students are not Distinctions. The single untested lever is a
fine-grained temporal/sequence representation (per Al-azazi's high-precision, low-recall
result), which the pinned aggregate pipeline could not evaluate.
