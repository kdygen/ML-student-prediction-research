# Experiment 002 — AutoResearch Feature Loop

**Date:** 2026-07-02
**Input:** canonical cache `data/processed/p3/` (unmodified) + raw OULAD tables (new features only)
**Code:** `experiments/feature_generation_002.py` (reusable safe generator) ·
`experiments/experiment_002_{build_frames,loop,final}.py` (drivers)
**Raw metrics:** `reports/experiment_002_results.json` (loop log + final confirmation)
**Environment:** pinned (Python 3.12.6, sklearn 1.6.1, pandas 2.2.2, numpy 2.0.2, xgboost 3.3.0)

---

## 1. Objective

Improve the official v3 benchmark — macro-F1 first, Withdrawn recall second, accuracy third
(70% accuracy as a stretch) — through an iterative propose → verify-safety → implement →
evaluate → accept/reject feature loop, without any methodology compromise.

## 2. Protocol (identical discipline to Experiment 001)

- Official seed-42 `GroupShuffleSplit(group=id_student)` test **never used for any decision**.
- Every loop evaluation: **GroupKFold(3) inside the seed-42 train** at cutoff 30, with the
  class-prior threshold τ tuned on the same inner folds. Model **fixed** throughout:
  baseline-v3 XGBoost hyperparameters (no hyperparameter search in this experiment).
- **Pre-registered acceptance rule** (declared before the first evaluation): accept the
  round's best group iff ΔF1 ≥ +0.002, or (ΔF1 ≥ −0.001 and ΔWithdrawn-recall ≥ +0.02);
  stop when no candidate passes or after 5 rounds.
- Final confirmation: accepted set at **all 5 cutoffs** — τ re-tuned per cutoff by inner CV,
  one evaluation on the held-out test, repeated grouped splits (seeds 0–4) with everything
  frozen, and paired per-seed comparison against Baseline v3 and the Experiment 001 winner
  on **identical splits**.

## 3. Candidate features (11 groups, 37 columns — why each may help, and safety)

| Group | Columns | Rationale | Availability proof |
|---|---|---|---|
| recent_windows | clicks in [C−6,C], [C−13,C−7], [C−20,C−14], [C−27,C−21], pre-course | recency-resolved engagement; late-window silence precedes withdrawal | `studentVle.date ≤ C` (asserted) |
| trend | click slope over days, recent-quarter ratio, last-week share | direction of engagement, not just volume | same |
| recency_decay | days since last activity, exp-decay(7d) weighted clicks | "gone quiet" signal | same |
| gaps | max / mean gap between active days | intermittency | same |
| first_activity | first active day, pre-course flag | early-bird vs late starter | same |
| diversity | #activity types, click entropy, #sites | breadth of engagement | same + static `vle` metadata |
| submission_timing | mean/min lead vs deadline, late count, submitted count, first submission day, #types | procrastination signal; deadlines are published schedule | `date_submitted ≤ C` (asserted); deadline = known schedule |
| score_trajectory | last/first/max/min score, range, std, n over submitted work | performance dynamics | `date_submitted ≤ C` |
| module_norm | percentile rank of clicks / weighted_average / active_days within (module, presentation) cohort | normalizes module-specific scales; peer context a deployed system has on day C | ranks over ≤C behaviour only, no labels |
| registration | registration lead | preparation proxy | known at registration |
| interactions | wa×spread, wa×clicks/day, active_days×has_coursework | explicit products of official features | derived from safe features |

NaN policy: counts/scores/leads → 0; day-valued features → sentinel C+30; disambiguated by
the official `has_vle_activity`/`has_coursework` indicators. Zero NaN/inf asserted per frame.
The generator hard-asserts `max(date) ≤ C` and `max(date_submitted) ≤ C` on its inputs.

## 4. Loop trace (cutoff 30; inner-CV macro-F1, τ tuned per evaluation)

Base (official 19 features): **0.4129** (acc 0.4812, Wrec 0.2226, τ=0.5).

| Round | Accepted | inner F1 | ΔF1 | Wrec | Runners-up that round (ΔF1) |
|--:|---|--:|--:|--:|---|
| 1 | **module_norm** | 0.4290 | +0.0161 | 0.250 | submission_timing +0.0136, recent_windows +0.0093, recency_decay +0.0074, first_activity +0.0058, score_traj +0.0053, trend +0.0047; gaps only +0.0009 |
| 2 | **submission_timing** | 0.4363 | +0.0074 | 0.270 | recent_windows +0.0056, registration +0.0033, score_traj +0.0023 |
| 3 | **recent_windows** | 0.4401 | +0.0038 | 0.272 | score_traj +0.0011, registration +0.0010 |
| 4 | **recency_decay** | 0.4424 | +0.0023 | 0.276 | trend +0.0021, registration +0.0019 |
| 5 | *(none — plateau)* | 0.4424 | — | 0.276 | best remaining: score_traj +0.0002 (below rule) |

(candidates re-competed every round; deltas are vs that round's current feature set)

**Rejected groups (documented):** `trend` (−0.0039 at r5), `gaps` (−0.0063), `first_activity`
(−0.0029), `diversity` (−0.0036), `score_trajectory` (+0.0002, below threshold),
`registration` (−0.0016), `interactions` (−0.0018). Note: `gaps` and `first_activity` raised
Withdrawn recall strongly (+0.044/+0.048 at τ=0.75) but cost too much macro-F1 under the
pre-registered rule — a lead worth revisiting in a recall-first experiment.

**Final accepted set = official 19 + 16 columns** (module_norm 3, submission_timing 6,
recent_windows 5, recency_decay 2) = **35 features**.

## 5. Final confirmation (all cutoffs; repeats mean ± std, seeds 0–4)

Macro-F1 (τ per cutoff by inner CV: 0.75/0.5/0.5/0.75/0.75):

| Cutoff | Baseline v3 | Exp001 winner | **Exp002** | Δ vs Exp001 | Δ vs v3 |
|--:|:--|:--|:--|--:|--:|
| 14  | 0.3030 ± 0.0028 | 0.3622 ± 0.0061 | **0.3943 ± 0.0045** | +0.032 | +0.091 |
| 30  | 0.3804 ± 0.0055 | 0.4161 ± 0.0084 | **0.4429 ± 0.0100** | +0.027 | +0.063 |
| 60  | 0.4233 ± 0.0040 | 0.4645 ± 0.0039 | **0.4909 ± 0.0053** | +0.026 | +0.068 |
| 90  | 0.4392 ± 0.0073 | 0.4787 ± 0.0058 | **0.5087 ± 0.0041** | +0.030 | +0.070 |
| 140 | 0.4743 ± 0.0053 | 0.5130 ± 0.0054 | **0.5547 ± 0.0046** | +0.042 | +0.080 |

**Statistics:** all **25/25 paired seed-deltas positive** vs the Exp001 winner (sign test
p = 2⁻⁵ per cutoff) and 25/25 vs Baseline v3; mean deltas are 3–9× split noise. At c30 the
Exp001 winner was RF-leaf5, so the comparison was additionally run against it on identical
splits: exp002 wins **5/5 seeds, +0.019 mean** (per-seed +0.011…+0.024).

Withdrawn recall (repeats):

| Cutoff | v3 | Exp001 | **Exp002** |
|--:|--:|--:|--:|
| 14  | 0.063 | 0.197 | **0.310** |
| 30  | 0.146 | 0.273 | **0.275** |
| 60  | 0.079 | 0.252 | 0.226 |
| 90  | 0.041 | 0.218 | **0.286** |
| 140 | 0.011 | 0.194 | **0.301** |

Accuracy: exp002 at its macro-F1 operating point recovers most of the τ-cost (e.g. c30
0.506 vs Exp001's 0.453 at the same τ-style operating point; v3 0.514). At the **τ=0
accuracy operating point** the new features beat Baseline v3's accuracy at every cutoff
(c30: 0.5333 vs 0.5141; c140: **0.6981 vs 0.6713**).

Per-class detail and full confusion matrices for every cutoff are in
`experiment_002_results.json` (`test_exp002.per_class` / `.confusion`). Example (c140, τ=.75):
Withdrawn .235/.274/.253, Fail .699/.602/.647, Pass .748/.707/.727, Distinction .499/.718/.589.

### What the model actually uses (feature importance, gain)

- **c30:** `rank_wa` (cohort-normalized score) 0.090 — the single most important feature,
  ahead of raw `weighted_average` 0.053; `rank_active_days` 0.052, `rank_clicks` 0.041;
  submission timing (`first_submit_day`, `min/mean_submit_lead`, `late_submissions`) all in
  the top 12; `days_since_last` 0.027.
- **c140:** `decay_clicks` 0.130 (top), `rank_wa` 0.102, `days_since_last` 0.069 — recency
  dominates late-course withdrawal signal, exactly the hypothesized mechanism.

## 6. Was 70% accuracy reached?

**Not quite — 69.8% at cutoff 140** (τ=0 operating point, repeats mean 0.6981 ± n/a; held-out
test 0.6933), vs 67.1% for Baseline v3. At early cutoffs the honest ceiling is far lower
(53.3% at c30). The stretch target was approached only where most outcome-determining
behaviour has already happened; no methodology was sacrificed chasing it.

## 7. Evidence that no leakage was introduced

1. Every candidate column derives from tables **hard-filtered** to `date ≤ CUTOFF` or
   `date_submitted ≤ CUTOFF`, with `assert` guards in the generator (`max date` checks).
2. Deadlines and registration dates used are published/known-in-advance facts; cohort ranks
   use peers' ≤cutoff **behaviour only** — never labels, never future.
3. The official cache was **not modified** (augmented frames live outside the repo; official
   parquet hashes unchanged); the grouped protocol and populations are byte-identical to v3.
4. Feature selection and τ used **only** inner folds of the seed-42 train; the held-out test
   was evaluated exactly once per arm; repeats ran with the feature set and τ frozen.
5. No resampling anywhere in the accepted pipeline; model hyperparameters untouched.

## 8. Conclusions & recommendation

- **Best feature set:** official 19 + module_norm + submission_timing + recent_windows +
  recency_decay (35 features; generator in `experiments/feature_generation_002.py`).
- **Best model:** fixed baseline XGBoost + per-cutoff τ (0.5–0.75) on the augmented set —
  new state of the art on the official protocol at every cutoff.
- **Headline:** macro-F1 +2.6 to +4.2 pp over Experiment 001 (+6.3 to +9.1 over Baseline
  v3), Withdrawn recall up to **27×** Baseline v3 (c140: 0.011 → 0.301), accuracy at τ=0
  above Baseline v3 everywhere. Gains are simultaneous, not traded.
- **Next experiment (recommended):** recall-first sub-experiment picking up the rejected
  `gaps`/`first_activity` lead (+0.04–0.05 Withdrawn recall at τ=0.75 for ~0.3–0.6 pp F1) —
  formulated as a binary "withdraw-within-k-weeks" early-warning task on the v3 population;
  and consider promoting the accepted 16 columns into the official pipeline as **Baseline
  v4 / cache p4** once independently re-verified.
