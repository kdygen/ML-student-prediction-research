# Experiment 008 — Feature Redundancy Elimination (parsimony study of V3)

> **📌 These figures describe the WITH-SCORES comparison arm, not the official baseline.**
> As of 2026-07-21 the official baseline is the **assessment-free** pipeline: accuracy 0.739,
> macro-F1 0.715, per-class F1 W 0.940 / F 0.779 / P 0.709 / D 0.430
> (`reports/official_baseline_results.json`). The numbers below remain the correct record
> for the with-scores configuration and for comparison against papers that use assessment
> scores.


**Date:** 2026-07-20
**Objective:** remove redundant features from the V3 configuration without sacrificing
performance. No new features, no pipeline redesign — elimination only.
**Method:** greedy backward elimination. Each round takes the highest |r| pair among the
*remaining* features (threshold 0.85), evaluates dropping **each** member, and accepts the
better drop iff it costs less than 0.002 in both macro-F1 and accuracy. Comparisons are
**paired** — identical grouped splits (seeds 0–4) for every candidate — so per-seed deltas
remove split variance and resolve effects well below the 0.002 threshold. Correlations are
recomputed after every removal. Model, hyperparameters, population, and protocol unchanged.
**Artifacts:** `reports/experiment_008_parsimony/parsimony_metrics.json`,
`experiments/experiment_008_parsimony.py`.

---

## 1. Elimination trace (8 rounds, 10 initial pairs ≥ 0.85)

| Round | Pair (r) | Drop A: ΔF1 | Drop B: ΔF1 | Decision |
|--:|---|--:|--:|---|
| 1 | `recovery_slope` ~ `score_slope_cw` (0.986) | **+0.0004** | −0.0014 | **remove `recovery_slope`** |
| 2 | `has_coursework` ~ `first_submit_day` (0.985) | **−0.0007** | −0.0021 | **remove `has_coursework`** |
| 3 | `rank_clicks` ~ `rank_active_days` (0.949) | +0.0006 | **+0.0019** | **remove `rank_active_days`** |
| 4 | `has_vle_activity` ~ `days_since_last` (0.946) | −0.0018 | **−0.0012** | **remove `days_since_last`** |
| 5 | `weighted_average` ~ `rank_wa` (0.896) | **−0.0010** | −0.0030 | **remove `weighted_average`** |
| 6 | `rank_clicks` ~ `active_weeks` (0.879) | −0.0022 | −0.0020 | **keep both** (best drop −0.0020) |
| 7 | `active_days` ~ `active_weeks` (0.876) | −0.0039 | −0.0020 | **keep both** (best drop −0.0020) |
| 8 | `w1_clicks` ~ `decay_clicks` (0.868) | **−0.0018** | −0.0036 | **remove `w1_clicks`** |

Two removals actually *improved* macro-F1 (`recovery_slope` +0.0004, `rank_active_days`
+0.0019) — redundant inputs were adding variance.

## 2. Features removed (6) — and what survives in their place

| Removed | Redundant with (r) | Why this member was the one dropped |
|---|---|---|
| `recovery_slope` | `score_slope_cw` (0.986) | Both measure score trajectory; the OLS slope is the better-defined estimator, and dropping the mean-diff version *improved* F1 |
| `has_coursework` | `first_submit_day` (0.985) | The indicator is implied by the sentinel value of the submission day — pure duplication |
| `rank_active_days` | `rank_clicks` (0.949) | Two cohort-normalised engagement ranks; removing this one *improved* F1 by 0.0019 |
| `days_since_last` | `has_vle_activity` (0.946) | Under per-student anchoring this became near-inert (correlation with any class only 0.031 — see §3 note) |
| `weighted_average` | `rank_wa` (0.896) | Same quantity raw vs cohort-ranked; the **ranked** version is retained because it is comparable across modules with different grading scales |
| `w1_clicks` | `decay_clicks` (0.868) | The exponential-decay recency measure subsumes the final-week window |

## 3. Redundant features kept (and why)

| Pair kept | r | Best-drop cost | Reason |
|---|--:|--:|---|
| `rank_clicks` ~ `active_weeks` | 0.879 | −0.0020 F1 | At the tolerance boundary; the pair encodes *volume* (clicks) vs *breadth* (distinct weeks) — related but not interchangeable |
| `active_days` ~ `active_weeks` | 0.876 | −0.0020 F1 | Same distinction at day vs week granularity; removing either costs measurably more than the accepted removals |

No further pairs ≥ 0.85 remain in the final set.

*Note on `days_since_last`:* it was the **#2 most important feature in Experiment 006**
(global anchoring) but is removable at −0.0012 cost here. This is a direct consequence of
006b's per-student anchoring, which stripped it of the global-timeline information — the
audit measured its correlation with any class at just 0.031. Its removal is therefore
evidence *for* the anchoring correction, not a loss.

## 4. Performance before vs after

**Repeated grouped splits (seeds 0–4, paired):**

| Set | Accuracy | Macro-F1 |
|---|--:|--:|
| Full (42 features) | 0.8392 | 0.8000 |
| **Reduced (36 features)** | 0.8365 | 0.7975 |
| Paired cumulative Δ | **−0.0027** | **−0.0025** |

**Stratified grouped 5-fold CV (the decision-relevant comparison):**

| Set | Accuracy | Macro-F1 |
|---|--:|--:|
| Full (42) | 0.8384 ± 0.0037 | 0.7973 ± 0.0046 |
| Intermediate (37) | 0.8359 ± 0.0039 | 0.7947 ± 0.0054 |
| **Reduced (36)** | **0.8360 ± 0.0029** | **0.7947 ± 0.0048** |

Per-class F1 (headline split) is essentially unchanged: Withdrawn 0.950→0.945,
Fail 0.824→0.820, Pass 0.841→0.841, **Distinction 0.580→0.585** (slightly better).

**Honest note on cumulative drift.** Every individual removal satisfied the < 0.002
criterion, but greedy elimination re-baselines after each step, so small costs accumulate:
the total is **−0.0025 macro-F1**, marginally above the per-step tolerance. It is, however,
roughly half the cross-validation fold standard deviation (±0.0048), so by the stated rule
("or within normal cross-validation noise") the reduction qualifies. The 37- and 36-feature
sets are statistically indistinguishable (0.7947 both), so the sixth removal is free — take
it for the extra parsimony.

## 5. Final recommended feature set for publication — 36 features

**Engagement volume & rhythm (10):** `active_days`, `active_weeks`, `clicks_per_day`,
`clicks_per_assessment`, `study_spread`, `burstiness`, `max_gap`, `engagement_decay_ratio`,
`decay_clicks`, `has_vle_activity`
**Recency windows (4):** `w2_clicks`, `w3_clicks`, `w4_clicks`, `precourse_clicks`
**Activity-type composition (5):** `resource_ratio`, `oucontent_ratio`, `homepage_ratio`,
`forumng_ratio`, `quiz_ratio`
**Coursework behaviour (8):** `submitted_count`, `completion_ratio_avail`, `first_submit_day`,
`mean_submit_lead`, `min_submit_lead`, `late_submissions`, `n_assess_types_submitted`,
`assessment_focus`
**Coursework performance (2):** `score_slope_cw`, `score_std_cw`
**Cohort-normalised (2):** `rank_clicks`, `rank_wa`
**Enrolment/static (5):** `registration_lead`, 4 × `highest_education_*`

**This arm (with coursework scores): accuracy 0.836 ± 0.003, macro-F1 0.795 ± 0.005**
(stratified grouped 5-fold CV, 36 features) — a 14% reduction in feature count for a
macro-F1 cost of 0.0026, well inside fold noise. If a reviewer prefers zero measured cost,
the 42-feature set (0.8384/0.7973) remains valid and both are reported here; the reduced set
is easier to justify and describe, which is the stated objective.
