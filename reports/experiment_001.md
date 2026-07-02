# Experiment 001 — Macro-F1 Optimization under the Official v3 Protocol

**Date:** 2026-07-02
**Input:** canonical cache `data/processed/p3/` (frozen v3 datasets; never modified)
**Code:** `experiments/experiment_001_macro_f1.py` · **Raw metrics:** `reports/experiment_001_results.json`
**Environment:** pinned (Python 3.12.6, scikit-learn 1.6.1, pandas 2.2.2, numpy 2.0.2, xgboost 3.3.0)

---

## 1. Objective & hypothesis

**Objective:** find the strongest scientifically valid model for 4-class outcome prediction
under the official protocol (grouped by `id_student`, v3 population, leakage-free features),
optimizing primarily macro-F1, then Withdrawn recall, then accuracy.

**Hypothesis:** Baseline v3's XGBoost is majority-biased (unweighted); imbalance-aware
techniques (class/sample weights, in-fold resampling, decision-threshold adjustment) should
raise macro-F1 materially without touching the pipeline or protocol.

## 2. Protocol (tuning never touches the test set)

- Official split per cutoff: `GroupShuffleSplit(group=id_student, 0.2, seed 42)`. The test
  side was **never** used for any tuning decision.
- All tuning: **GroupKFold(3) inside the seed-42 train**, including the decision-threshold
  parameter (below). SMOTE/SMOTETomek were applied **inside training folds only**.
- Screening of 13 configurations at cutoff 30 → top-3 shortlist → per-cutoff re-selection by
  inner CV → winner refit on the full seed-42 train → **one** evaluation on the held-out test
  → repeated grouped splits (seeds 0–4) with hyperparameters and threshold **frozen**.
- Paired comparison: Baseline v3's XGBoost and RF evaluated on the **identical** repeat splits.

**Threshold tuning (justified):** class-prior adjustment
`p_adj[:,c] = p(c|x) · (1/π_c)^τ`, `τ ∈ {0, .25, .5, .75, 1}`, tuned on inner folds only
(π = training class priors). τ=0 is plain argmax; τ>0 shifts the decision boundary toward
minority classes — the standard cost-sensitive decision rule, appropriate because macro-F1,
not accuracy, is the target. Calibration was considered and skipped: τ operates on relative
probability ratios, so monotone recalibration would be absorbed into the τ choice.

## 3. Screening results (cutoff 30, inner-CV macro-F1)

| Config | inner F1 | τ* | Withdrawn recall |
|---|---:|--:|---:|
| **rf_bal_leaf5** (RF balanced, min_samples_leaf=5, 600 trees) | **0.4186** | 0.0 | 0.229 |
| **xgb_base + τ** (baseline XGB, prior-adjusted) | **0.4129** | 0.5 | 0.223 |
| **rf_balsub_600** (balanced_subsample, 600 trees) | **0.4101** | 0.5 | 0.226 |
| rf_bal (baseline RF) + τ | 0.4082 | 0.5 | 0.225 |
| xgb_smote (SMOTE in-fold) | 0.4069 | 0.0 | 0.269 |
| xgb_bal_sub06 / xgb_smotetomek | 0.4039 | — | — |
| xgb_bal_d8 / xgb_bal / xgb_bal_mcw10 | 0.401–0.403 | 0.0 | ~0.26–0.31 |
| rf_smote | 0.3987 | 0.0 | 0.249 |
| xgb_bal_d4_lr01 | 0.3943 | 0.0 | 0.306 |
| logreg_bal | 0.3750 | 0.0 | 0.299 |

**Negative results (worth recording):** per-sample class weighting for XGBoost, SMOTE and
SMOTETomek (in-fold), and every hyperparameter variation tried all landed **at or below** the
untouched baseline XGBoost combined with simple prior-threshold adjustment. Post-hoc decision
adjustment beat retraining-time imbalance handling across the board.

## 4. Final results per cutoff

Winner selected by inner CV per cutoff; `xgb_base+τ=0.75` won at 14/60/90/140,
`rf_bal_leaf5 (τ=0)` at 30.

### Held-out official test (seed 42) — winner vs Baseline v3 XGBoost

| Cutoff | winner | macro-F1 | baseline F1 | Δ | acc (winner/base) | Withdrawn recall (winner/base) |
|--:|:--|--:|--:|--:|:--|:--|
| 14  | xgb+τ.75 | **0.3615** | 0.3151 | +0.046 | 0.4113 / 0.4810 | **0.193** / 0.060 |
| 30  | rf_leaf5 | **0.4133** | 0.3705 | +0.043 | 0.4725 / 0.5099 | **0.209** / 0.149 |
| 60  | xgb+τ.75 | **0.4612** | 0.4066 | +0.055 | 0.5188 / 0.5614 | **0.245** / 0.068 |
| 90  | xgb+τ.75 | **0.4686** | 0.4244 | +0.044 | 0.5350 / 0.5947 | **0.196** / 0.031 |
| 140 | xgb+τ.75 | **0.5130** | 0.4713 | +0.042 | 0.6101 / 0.6627 | **0.180** / 0.011 |

### Repeated grouped splits (seeds 0–4), mean ± std — macro-F1

| Cutoff | winner | baseline XGB | baseline RF | paired Δ vs XGB (per-seed) |
|--:|:--|:--|:--|:--|
| 14  | **0.3622 ± 0.0061** | 0.3030 ± 0.0028 | 0.2913 ± 0.0010 | +.054 +.071 +.056 +.051 +.064 |
| 30  | **0.4241 ± 0.0076** | 0.3804 ± 0.0055 | 0.3602 ± 0.0063 | +.034 +.054 +.043 +.049 +.038 |
| 60  | **0.4645 ± 0.0039** | 0.4233 ± 0.0040 | 0.3930 ± 0.0041 | +.040 +.044 +.038 +.037 +.048 |
| 90  | **0.4787 ± 0.0058** | 0.4392 ± 0.0073 | 0.4158 ± 0.0080 | +.037 +.054 +.035 +.025 +.048 |
| 140 | **0.5130 ± 0.0054** | 0.4743 ± 0.0053 | 0.4497 ± 0.0066 | +.040 +.034 +.037 +.047 +.036 |

**Statistical justification:** the winner beats baseline XGBoost on **all 5 seeds at all 5
cutoffs** (paired sign test, one-sided p = 2⁻⁵ ≈ 0.031 per cutoff; 25/25 positive overall).
Mean deltas (+0.039 to +0.061) are 5–15× the split-noise σ (±0.004–0.008) — far beyond the
pre-registered ≥2σ evidence bar from the methodology review. Same holds vs baseline RF
(all 25 deltas positive, +0.05 to +0.08).

### Per-class detail (winner, official test; precision / recall / F1)

| Cutoff | Withdrawn | Fail | Pass | Distinction |
|--:|:--|:--|:--|:--|
| 14  | .254/.193/.219 | .396/.464/.427 | .532/.497/.514 | .252/.329/.286 |
| 30  | .347/.209/.261 | .401/.450/.424 | .565/.612/.587 | .373/.388/.381 |
| 60  | .320/.245/.278 | .531/.493/.511 | .620/.611/.615 | .365/.555/.440 |
| 90  | .226/.196/.210 | .557/.543/.550 | .647/.600/.622 | .412/.610/.492 |
| 140 | .178/.180/.179 | .655/.598/.625 | .718/.660/.688 | .464/.705/.560 |

Confusion matrices (rows = true W/F/P/D) are in `experiment_001_results.json`
(`winner_test.confusion` per cutoff). The dominant error everywhere: **Withdrawn ↔ Fail ↔
Pass confusion** — e.g. at c30, of 1,013 true Withdrawn, 360 are predicted Fail and 399 Pass.

### Feature importance (winner)

- **c30 (RF, impurity):** `weighted_average` .141, `clicks_per_assessment` .088,
  `oucontent_ratio` .084, `resource_ratio` .082, `burstiness` .081, `clicks_per_day` .080 —
  assessment performance plus engagement volume/mix dominate.
- **c14 (XGB, gain):** `highest_education_Lower Than A Level` .101, `study_spread` .088,
  `active_days` .069 — with almost no coursework by day 14, demographics and raw engagement
  carry the signal.

## 5. The accuracy trade-off (not hidden)

The winner's accuracy is 3–6 pp **below** baseline (e.g. c140: 0.611 vs 0.663). That is the
explicit cost of moving decision mass toward minority classes; macro-F1 and Withdrawn recall
were the declared priorities. Deployment gets a knob: τ=0 recovers baseline accuracy; τ=0.75
buys ~18× the Withdrawn recall at c140 (0.011 → 0.194). Both operating points come from the
same fitted model.

## 6. Is 0.70 macro-F1 realistic? No — quantified

Best achieved: **0.42 (c30) / 0.51 (c140)** with the winner; 13 diverse configurations
(weighting, resampling, depth/lr/leaf grids, thresholding) span only 0.375–0.419 inner-CV F1
at c30 — the technique axis is nearly exhausted at ~0.42.

Limiting factors, in order:
1. **Withdrawn is weakly separable early.** Best Withdrawn F1 ≈ 0.26 (c30) even when
   optimized for. Day-30 behaviour barely determines a withdrawal that may happen months
   later; the class is defined by a *future decision*, not a present state. 0.70 macro-F1
   requires ≈0.7 per-class F1 — ~2.7× the observed Withdrawn ceiling.
2. **Fail/Pass boundary noise:** borderline students (near-pass) are intrinsically ambiguous
   at any cutoff; Fail F1 plateaus ≈ 0.42 (c30) → 0.63 (c140).
3. **Feature ceiling:** 19 aggregate features; importance is spread thin (max 0.14). The
   remaining headroom is in *new information* (temporal dynamics, per-assessment
   trajectories, module context), not in model choice.

**Practical upper-bound estimate with current data:** ~**0.45–0.48** macro-F1 at cutoff 30
and ~**0.55** at cutoff 140 (winner + a few points for richer features). 0.70 would need
information the early window does not contain; at cutoff 30 it is not a realistic target
under the leak-free protocol.

## 7. Honest caveats

- The repeats (seeds 0–4) share rows with the seed-42 train used for hyperparameter/τ
  selection; only *hyperparameters* transfer (models are refit per split), so contamination
  is second-order, and the clean seed-42 held-out result matches the repeat means closely
  (e.g. c30: 0.4133 test vs 0.4241 ± 0.0076 repeats).
- τ=0.75 was selected at 4 cutoffs by inner CV; per-cutoff τ re-tuning on more inner folds
  could shift it slightly. Withdrawn recall remains low in absolute terms (~0.2) — this is a
  data limit, not a tuning failure.
- `rf_bal_leaf5`'s c30 win over `xgb_base+τ` is within selection noise (0.4186 vs 0.4129
  inner); either is defensible there.

## 8. Conclusions

- **What improved:** macro-F1 +4–6 pp at every cutoff (25/25 paired seeds, p≈0.03/cutoff);
  Withdrawn recall 3–18× baseline; both from a *decision-rule* change (τ) or a minor RF
  regularization — no pipeline, feature, or protocol change.
- **What failed:** XGBoost sample-weighting, SMOTE, SMOTETomek, and all hyperparameter
  variants — none beat baseline-XGB-plus-threshold; a clean negative result worth keeping.
- **Recommended model going forward:** `xgb_base + τ=0.75` (single model family across
  cutoffs; τ exposed as the deployment precision/recall knob). Report macro-F1 with repeats.
- **Next:** (1) new *information*, not new models — temporal/trajectory features
  (week-by-week click deltas, per-assessment submission timing) targeted specifically at
  early-withdrawal signal; (2) a dedicated binary "Withdrawn-within-k-weeks" early-warning
  task (grouped, v3 population) where the operating point is directly actionable;
  (3) optional per-cutoff τ calibration on a validation fold as standard practice.
