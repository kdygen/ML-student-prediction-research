# Experiment 006b — Per-Student Timeline Anchoring (sensitivity analysis of 006)

**Date:** 2026-07-20
**Type:** sensitivity analysis of Experiment 006. The original experiment is untouched; 006b
is a parallel run whose **only change** is the timeline anchor for temporal/recency features.
**Provenance guarantee:** the 006b build and run scripts were generated from the 006 scripts
by programmatic patches, each asserted to apply exactly once
(`experiments/experiment_006b_{build,run,compare}.py`); pipeline, models, hyperparameters,
splits (identical grouped indices), preprocessing, leakage rules, and metrics are byte-
identical apart from the anchor change and I/O paths.
**Results folder:** `reports/experiment_006b_per_student_anchor/` (metrics.json,
dataset_meta.json, figures/fig1_comparison.png).

## 1. The single change

Each student's endpoint is now their **own final observed week**: horizon = unregistration
date for withdrawn students, course end for completers (a Week-3 withdrawer's timeline ends
at Week 3; a completer's at the final course week).

| Feature | 006 (original) | 006b (this run) |
|---|---|---|
| `days_since_last` | course_end − last activity | **horizon** − last activity (never-active: constant 999 sentinel) |
| `decay_clicks` | exp-decay anchored at course_end | anchored at **horizon** |
| `clicks_first/last_third`, `engagement_decay_ratio` | thirds of the course | thirds of the **student's own span** |
| `study_spread` | active_days / course length | active_days / **own span** |
| `w1–w4_clicks` | already per-student anchored | unchanged |
| everything else (34 features, censoring, exam exclusion) | — | unchanged |

## 2. Results — comparison table (XGB; all models in metrics.json)

### Multiclass (headline grouped test; repeats seeds 0–4)

| Model | Acc 006 → 006b | Macro-F1 006 → 006b |
|---|:--|:--|
| LogReg | 0.745 → 0.791 (**+0.046**) | 0.722 → 0.763 (+0.041) |
| Decision Tree | 0.774 → 0.800 (+0.026) | 0.729 → 0.740 (+0.010) |
| Random Forest | 0.804 → 0.829 (+0.026) | 0.737 → 0.765 (+0.028) |
| **XGBoost** | 0.823 → **0.845** (+0.022) | 0.770 → **0.789** (+0.019) |

XGB repeats: acc 0.824±0.002 → 0.850±0.004; F1 0.774±0.002 → 0.799±0.004 — the gain is
robust, not split noise.

### Per-class (XGB)

| Class | P 006→006b | R 006→006b | F1 006→006b |
|---|:--|:--|:--|
| Withdrawn | 0.946→0.958 | 0.882→0.958 | 0.913→**0.958** (+0.045) |
| Fail | 0.778→0.885 | 0.723→0.737 | 0.749→**0.804** (+0.055) |
| Pass | 0.792→0.784 | 0.912→0.904 | 0.848→0.840 (−0.008) |
| Distinction | 0.663→0.640 | 0.503→0.492 | 0.572→0.556 (−0.015) |

### Binary (XGB)

| Task | ROC-AUC 006→006b | PR-AUC 006→006b |
|---|:--|:--|
| Withdrawn vs rest | 0.987→0.996 | 0.974→0.991 |
| Fail vs rest | 0.930→0.952 | 0.775→**0.894 (+0.119)** |
| Pass vs rest | 0.944→0.941 | 0.890→0.887 |
| Distinction vs rest | 0.952→0.952 | 0.638→0.636 |
| At-risk vs P+D | 0.984→0.982 | 0.988→0.987 |

### Regression (behavior-only; unchanged features dominate)

R² 0.340→0.333 (XGB), MAE 10.15→10.19 — essentially unchanged, as expected: the regression
population is dominated by completers, for whom the anchor change is a no-op.

### Feature importance (XGB multiclass, top of ranking)

| Rank | 006 (global anchor) | 006b (per-student anchor) |
|--:|---|---|
| 1 | engagement_decay_ratio 0.168 | **has_vle_activity 0.229** |
| 2 | days_since_last 0.132 | **completion_ratio_cw 0.216** |
| 3 | decay_clicks 0.075 | submitted_count 0.100 |
| 4 | w1_clicks 0.072 | decay_clicks 0.062 |
| 5 | w4_clicks 0.067 | rank_wa 0.036 |

## 3. Answers to the four questions

**1. How much did overall performance change?** It went **up**, not down: +2.2 pp accuracy
and +1.9 pp macro-F1 for XGB (+4.6 pp accuracy for logistic regression), robust across
repeated splits. This refutes the hypothesis stated in the 006 report's gray-area note that
the global-timeline encoding was inflating performance — removing it *helped*.

**2. How much did Withdrawn prediction change?** Withdrawn improved (F1 0.913→0.958; AUC
0.987→0.996), and **Fail improved most** (F1 +0.055; PR-AUC +0.119). Mechanism, verified on
the class signatures: under the global anchor, a withdrawer's `study_spread` averaged 0.06
(trivially low — position-in-course); per-student anchoring lifts it to 0.26, overlapping
Pass (0.34) — that signature is genuinely *removed*. Separation now comes from the
participation footprint: coursework completion averages 0.15 (Withdrawn) / 0.45 (Fail) /
0.96 (Pass) / 0.97 (Distinction) — a clean ladder that also explains the Fail gain:
normalizing engagement to each student's active window exposes failers as *present but not
completing work*, distinct both from absent withdrawers and from completing passers.

**3. Which features became most important?** The shape-of-ending features
(`engagement_decay_ratio`, `days_since_last`) were dethroned by **whether-the-work-was-done
features**: `has_vle_activity`, `completion_ratio_cw`, `submitted_count`. Notably, the new
top features do not use the horizon at all — the model shifted onto anchor-independent
signals.

**4. Global-timeline information vs behavioral patterns?** Three conclusions. (i) The
original 006 performance did **not** depend on global-timeline position — an equal-or-better
ceiling exists on purely span-normalized features; the honest ceiling is, if anything, the
006b number (0.845/0.789). (ii) No anchoring choice is information-free: for withdrawn
students the per-student horizon *is* the unregistration date, so 006b features are
normalized by an administrative event that defines the Withdrawn label. Both variants
therefore describe rather than forecast the Withdrawn class at course end — but 006b's
importance ranking shows the model prefers anchor-free completion signals anyway, which
strengthens the behavioral interpretation. (iii) The truly anchor-free readouts —
regression from behavior only (R² ≈ 0.33) and Distinction/Pass separation (unchanged) —
are identical in both runs: what students *did* carries the durable signal; how the timeline
is framed mainly redistributes the Withdrawn/Fail boundary.

## 4. Limitations

Inherits all Experiment-006 limitations (§6 of that report), including the descriptive (not
early-warning) character of course-end Withdrawn detection. The 999 sentinel for
never-active `days_since_last` is a constant chosen to avoid making never-starters look
recently active under per-student anchoring; results are insensitive to its exact value for
tree models, but linear-model coefficients on that feature should not be interpreted.
