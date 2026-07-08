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

Baseline v4 established (2026-07-02) — the official benchmark. The 16 Experiment-002
features passed a full scientific promotion audit (independent recomputation to 2.3e-13,
leakage proofs, redundancy scan, pre-registered leave-one-group-out ablations at two
cutoffs) and were promoted into the pipeline: 35 active features, same v3 population /
grouped protocol / models. Every model improves on both accuracy and macro-F1 at every
cutoff (50/50 paired repeat-seeds positive); GroupKFold(5) agrees with repeated splits.
New canonical cache `data/processed/p4/` (rebuild-deterministic, reload-verified); p3 and
Baselines v1/v2/v3 immutable (verified by hash). See `reports/baseline_v4.md`,
`reports/baseline_v4_feature_audit.md`. Future experiments start from p4 and compare
within-cutoff against `baseline_v4_results.json`.

Experiment 003 (2026-07-05) recommends the deployment output: **ASI = P(Fail)+P(Withdrawn)**
from the official v4 XGB (+ train-only isotonic calibration), evaluated as an intervention
ranking (AUC 0.73→0.90, top-5% list 97-100% truly at-risk from c30). Evaluation-layer only —
the v4 benchmark itself is unchanged. See `reports/experiment_003_intervention_index.md`.

Experiment 004 (2026-07-05) adds the official intervention methodology: ERIP = day 30 first
pass (pre-stated reliability criterion met: red-band precision 0.978, ECE 0.018) with staged
capacity top-ups at day 60/90 — staged 10%+10% doubles withdrawal reach vs a late one-shot
at equal budget and precision. Decision rules (immediate / monitor closely / continue
monitoring / low risk) each carry measured outcome rates. Evaluation/decision layer only.
See `reports/experiment_004_intervention_framework.md`.

---

## Experiment 004 — Early Intervention Decision Framework

Date: 2026-07-05

Objective: Move beyond scores to an evidence-based intervention methodology: when to
intervene, whom to prioritize, how confidence evolves, and how to use progressively
available information — Baseline v4 and the Experiment-003 ASI unchanged.

Hypothesis: Combining progressive prediction quality, calibrated probabilities, individual
risk trajectories, and capacity constraints yields defensible decision rules and an
earliest-reliable-intervention-point methodology.

Implementation: Evaluation/decision layer only. Fixed student panel (global grouped 80/20
split of the union of student IDs, seed 42; out-of-sample at every cutoff; sanity-matched
official GSS-42 within ±0.014). Phase 2: per-cutoff acc/F1/AUC/AP/ECE/entropy/top-K for
ASI+WRI. Phase 3: band trajectories (red top-5% / amber 5-20% / green) across cutoffs,
categories, transition matrices, lead times to unregistration. Phase 5: reachability decay +
one-shot vs staged capacity policies at equal budget. Timing-literature review (weekly OU
Analyse, OAAI 25/50/75% checkpoints, NTU event triggers, RTI escalation, Howard 2018
"optimal time", survival/hazard work, acknowledged gaps).

Features Added: none. Features Removed: none. Models Evaluated: official v4 XGB (verbatim).

Results: ASI AUC 0.721→0.914, ECE ≤ 0.021 everywhere (c14→c140); marginal AUC gain after c30
(+0.010-0.014/10d) is 5x slower than withdrawal-reachability loss (-0.070/10d). Earliest
reliable point = day 30 by pre-stated criterion (red-band precision ≥ 0.95 & ECE ≤ 0.02: c14
fails at 0.858, c30 passes at 0.978). Trajectories: escalating students 99.3% adverse (above
persistent-high 97.3%); rapid risers (ΔASI ≥ +0.15) 68-80% adverse vs 31-38% others; red
band's imminent-withdrawal rate 7.6x green's; median first-flag→unregistration lead 58 days.
Policies at equal 20% budget: staged 10%@c30+10%@c90 matches one-shot-c140 precision (0.928
vs 0.933) and adverse recall (40.7% vs 40.8%) while doubling withdrawal reach (34.0% vs
16.3%, lead 50d vs 38d) — staged dominates. Decision framework: immediate (red or rapid
riser) / monitor closely (amber) / continue monitoring (persistent amber, recovering,
high-entropy manual check) / low risk (stable green), every rule with measured outcome
rates; green explicitly framed as capacity allocation, not safety certification.

Decision: Recommended official methodology = ERIP (pre-stated reliability criterion +
earliest satisfying checkpoint + remaining-opportunity weighing + staged capacity): first
pass day 30 (~half capacity), top-ups day 60/90, day-14 for provisional triage only;
calibrated P ≥ ~0.92 = immediate band; top-quartile entropy ⇒ manual verification.
Literature gap analysis supports novelty of the integration (components exist separately).
Intervention treatment effects not claimed (not measurable in OULAD). Deliverables:
reports/experiment_004_intervention_framework.md, reports/experiment_004_results.json,
reports/figures/experiment_004/ (5 figures incl. decision-flow diagram),
experiments/experiment_004_*.py.

Next Ideas: per-module ERIP (assessment calendars differ); weekly-cadence scoring between
checkpoints (needs new cache cutoffs — requires explicit instruction); subgroup fairness
audit of red/amber composition; urgency score (predicted time-to-withdrawal) alongside ASI.

---

## Experiment 003 — Early Intervention Index

Date: 2026-07-05

Objective: Pivot from outcome prediction toward early-intervention support — determine
whether Baseline v4 model outputs can be transformed into a useful intervention ranking for
educators, and whether one index should become the official deployment output.

Hypothesis: Some transformation of the v4 class posteriors (additive, severity-weighted,
confidence-adjusted, calibrated, cohort-normalized, or ensemble) prioritizes truly at-risk
students well enough for capacity-limited intervention.

Implementation: Evaluation-layer only — official v4 models/features/cache verbatim.
Literature review of deployed EWS (Course Signals, OU Analyse, Wisconsin DEWS, Chicago
On-Track, Balfanz, Lakkaraju KDD'15) and ranking/calibration methodology. Ten candidate
indices from official posteriors; truths = at-risk {W,F}, withdrawn-only, fail-only; grouped
splits (headline seed 42, repeats 0-4), tunables (combination weight, isotonic) fitted on
inner GroupKFold(3) train OOF only. Metrics: ROC-AUC, AP, precision/recall/lift/hits at
5/10/20%, Brier, ECE, reliability; permutation importance per index; 5 comparison figures.

Features Added: none. Features Removed: none. Models Evaluated: official 4 (index layer).

Results: ASI = P(Fail)+P(Withdrawn) (XGB) wins at-risk ranking at every cutoff — AUC
0.733/0.782/0.829/0.857/0.903, AP 0.701→0.883 (c14→c140); repeats ±0.003-0.006. The tuned
weighted combination independently converged to w=0.5 (≡ ASI) at all five cutoffs; entropy
adjustment, severity weights, ensembles, disagreement penalties, and course-percentile all
ranked at-or-below plain ASI (documented negative results). Top-5% list purity: 96.7% at c30,
100% at c90+; lift at 5% near theoretical max. WRI best for withdrawn-only truth (AUC
0.656→0.766); FRI for fail-only. Raw ASI already calibrated (ECE ≤ 0.02 everywhere; isotonic
layer tightens slightly). Permutation importance: rank_wa dominates every index; promoted v4
features hold most top slots (submission timing drives WRI; recency grows by c90).
Literature: ASI is exactly the field-standard merged "at-risk" P(adverse) recovered from the
4-class posterior; precision@K under capacity is the canonical evaluation.

Decision: Recommend ASI (official v4 XGB + train-only isotonic layer) as the official
deployment output, with WRI as the scoped variant for withdrawal-specific campaigns. Baseline
v4 unchanged. Deliverables: reports/experiment_003_intervention_index.md,
reports/experiment_003_results.json, reports/figures/experiment_003/ (5 figures),
experiments/experiment_003_*.py.

Next Ideas: subgroup fairness audit of top-K composition (ABROCA/Aequitas-style) before any
rollout; per-presentation recalibration test (train earlier presentations → score later);
weekly-cadence score smoothing (He et al. 2015) between cutoffs.

---

## Binary Evaluation under the v4 Protocol

Date: 2026-07-02

Objective: Replace the notebook's original `#Binary RF` results (pre-v2 leaky features,
survivorship-biased population, random row split with student overlap) with the four pairwise
tasks evaluated under the official v4 protocol. Secondary view only — no change to the
official 4-class benchmark.

Implementation: One additive markdown + code section appended to the end of the notebook
(original binary cell preserved as history; 0 committed cells modified). Per pair: v4
population (`mlDataV4`), `GroupShuffleSplit(group=id_student, test_size=0.2, seed=42)` with a
zero-overlap assert, `RandomForestClassifier(random_state=42)` unchanged from the original
binary cell. Both the v3 (19) and v4 (35) feature sets run on identical splits to isolate the
promoted-feature effect. Validated by executing the exact cell source against the p4 cached
frames for all cutoffs.

Features Added: none. Features Removed: none. Models Evaluated: RF only (matching the
original binary section).

Results (v4 accuracy, c14→c140): Pass/Fail 0.698→0.843; Distinction/Fail 0.787→0.928;
Distinction/Pass 0.799→0.836; Withdrawn/Pass 0.716→0.917. Promoted features improve 19/20
task×cutoff cells (+1–3 pp; sole exception Distinction/Pass c14 −0.002, within noise).
Caveat: Distinction/Pass and Withdrawn/Pass are imbalanced — positive-class F1 (0.08→0.47 and
0.37→0.59 respectively) must be read alongside accuracy; default-threshold RF is
precision-heavy on the minority class.

Decision: Recorded as the official binary view of Baseline v4 (`reports/binary_v4.md`,
`reports/binary_v4_results.json`). The 4-class task in `baseline_v4.md` remains the primary
benchmark.

Next Ideas: threshold tuning (τ-style) on the Withdrawn/Pass task for recall-oriented
operating points; the withdraw-within-k-weeks binary experiment already proposed.

---

## Baseline v4 — Verified Feature Promotion

Date: 2026-07-02

Objective: Promote the Experiment-002 accepted features into the official baseline only
after independent scientific verification of every feature.

Hypothesis: The 16 features are implementation-correct, leakage-free, non-redundant, and
consistently beneficial; promoting them yields a strictly better official benchmark without
touching population, protocol, or models.

Implementation: (1) Independent recomputation of all 16 features from raw OULAD tables via
deliberately different code paths (raw-row masks vs daily aggregates, reversed merges,
scipy rankdata vs pandas rank, sort/drop_duplicates vs groupby.min) — all matched at
cutoffs 30 and 90, worst-case diff 2.27e-13. (2) Dependency trace + leakage proof per
feature; hard max-date asserts (all ≤ cutoff). (3) Redundancy scan: no new feature reaches
|r|≥0.9 vs any official feature; one within-new pair (w1_clicks~decay_clicks, r=0.948) —
both survive leave-one-feature-out (−0.0038/−0.0010 inner F1 at c30 when removed).
(4) Pre-registered leave-one-group-out ablations (train-only GroupKFold(3), c30+c90): every
group's removal costs F1 at BOTH cutoffs (c30: −0.0088/−0.0096/−0.0086/−0.0023) → all 4
groups promoted; nothing rejected. (5) Additive notebook v4 section (1 new code cell, 0
committed cells modified; in-cell asserts: zero overlap, no dup keys, no NaN, temporal
guards); notebook features match Experiment-002 frames to 2.3e-13. (6) Official run all
cutoffs + repeats (seeds 0–4) + GroupKFold(5); p4 cache built with manifests, reload
verification, and a from-scratch c030 rebuild hash match (True). p3 verified untouched.

Features Added (16, verified): rank_clicks, rank_wa, rank_active_days; mean/min_submit_lead,
late_submissions, submitted_count, first_submit_day, n_assess_types_submitted; w1–w4_clicks,
precourse_clicks; days_since_last, decay_clicks. Features Removed: none (all candidates
passed). Active set: 35.

Models Evaluated: unchanged (LogReg balanced, Tree d5, RF 300 balanced, XGB 300/d6/lr.05).

Validation Strategy: official grouped protocol; repeats seeds 0–4 + GroupKFold(5) both
reported; all overlap asserts pass in every split.

Results (GSS-42 test acc/macro-F1; XGB repeats F1 mean±std; GKF5 in report):

| Cutoff | LogReg | Tree | RF | XGB | XGB rep F1 (v3 → v4) |
|--:|:--|:--|:--|:--|:--|
| 14  | .3478/.3468 | .4890/.3028 | .4799/.3093 | .4914/.3428 | .3030 → .3325±.006 |
| 30  | .4014/.3972 | .5161/.3392 | .5244/.3719 | .5265/.3955 | .3804 → .4081±.005 |
| 60  | .4621/.4451 | .5555/.3932 | .5697/.4024 | .5752/.4387 | .4233 → .4502±.005 |
| 90  | .4866/.4624 | .5896/.4180 | .6081/.4328 | .6165/.4637 | .4392 → .4701±.008 |
| 140 | .5570/.5040 | .6699/.4831 | .6783/.4765 | .6933/.5144 | .4743 → .5083±.003 |

Observations: Every model improves on both metrics at every cutoff (only exception: LogReg
c14 accuracy −0.003 with F1 +0.012); all 50/50 paired repeat-seed accuracy deltas positive
(RF+XGB × 5 cutoffs × 5 seeds); GKF5 agrees with repeats within ~1σ. All four classes'
F1 improve (c140 XGB: W .021→.090, F .650→.690, P .748→.776, D .467→.502). Plain-argmax
Withdrawn recall remains low by construction — the Experiment-001 τ knob stays the
deployment tool for recall-oriented operating points.

Decision: Baseline v4 is the official benchmark; p4 the canonical cache. v1/v2/v3 + p3
immutable history.

Next Ideas: recall-first grouped binary "withdraw-within-k-weeks" experiment (picking up
the rejected gaps/first_activity lead); consistent class-weighting sub-experiment; per-
cutoff τ calibration as standard reporting.

---

## Experiment 002 — AutoResearch Feature Loop

Date: 2026-07-02

Objective: Improve the official v3 benchmark (macro-F1 first, Withdrawn recall second,
accuracy third; 70% accuracy stretch) via an iterative propose→verify→implement→evaluate→
accept/reject feature loop, with zero methodology compromise.

Hypothesis: Temporal engagement dynamics (recency, windows), submission-timing behaviour,
and cohort-normalized peer context carry withdrawal/failure signal the 19 aggregate v3
features miss.

Implementation: Reusable leakage-safe generator (`experiments/feature_generation_002.py`) —
11 candidate groups / 37 columns, every column hard-filtered to date ≤ CUTOFF or
date_submitted ≤ CUTOFF with asserts; deadlines/registration = known-in-advance facts;
cohort ranks use peers' ≤cutoff behaviour only (never labels). Augmented frames built from
the frozen p3 cache + raw tables (official cache untouched). Greedy loop at cutoff 30:
GroupKFold(3) inside the seed-42 train only, τ tuned on inner folds, model fixed (baseline
XGBoost hyperparameters), pre-registered acceptance rule (ΔF1 ≥ +0.002, or ΔF1 ≥ −0.001 with
ΔWrec ≥ +0.02), max 5 rounds. Final: accepted set at all cutoffs — τ per cutoff by inner CV,
one held-out-test evaluation per arm, repeats seeds 0–4 frozen, paired per-seed comparison vs
Baseline v3 and the Exp001 winner on identical splits.

Features Added (accepted, 16): module_norm (rank_clicks, rank_wa, rank_active_days),
submission_timing (mean/min_submit_lead, late_submissions, submitted_count,
first_submit_day, n_assess_types_submitted), recent_windows (w1–w4_clicks,
precourse_clicks), recency_decay (days_since_last, decay_clicks).

Features Rejected (documented with deltas in the report): trend, gaps, first_activity,
diversity, score_trajectory, registration, interactions. Note: gaps/first_activity raised
Withdrawn recall (+0.044/+0.048) at too high an F1 cost — recall-first lead for later.

Models Evaluated: fixed baseline XGBoost + τ ∈ {0,…,1} only (no hyperparameter search).

Validation Strategy: official v3 protocol; loop plateaued at round 5 (inner F1
0.4129 → 0.4424 at c30).

Results (repeats seeds 0–4, macro-F1; v3 / Exp001 / Exp002):

| Cutoff | Baseline v3 | Exp001 | Exp002 | Wrec v3→e2 | acc@τ0 e2 (v3) |
|--:|:--|:--|:--|:--|:--|
| 14  | 0.3030±.003 | 0.3622±.006 | 0.3943±.005 | 0.063→0.310 | 0.480 (0.465) |
| 30  | 0.3804±.006 | 0.4161±.008 | 0.4429±.010 | 0.146→0.275 | 0.533 (0.514) |
| 60  | 0.4233±.004 | 0.4645±.004 | 0.4909±.005 | 0.079→0.226 | 0.587 (0.568) |
| 90  | 0.4392±.007 | 0.4787±.006 | 0.5087±.004 | 0.041→0.286 | 0.624 (0.605) |
| 140 | 0.4743±.005 | 0.5130±.005 | 0.5547±.005 | 0.011→0.301 | 0.698 (0.671) |

All 25/25 paired deltas positive vs both references (sign test p=2⁻⁵ per cutoff); at c30
also verified vs Exp001's actual winner rf_leaf5: 5/5 seeds, +0.019 mean. 70% accuracy not
reached: best 69.8% at c140 (τ=0). Importance: rank_wa top at c30 (beats raw
weighted_average); decay_clicks + days_since_last dominate at c140 — the hypothesized
recency mechanism.

Decision: Accept the 16-feature augmentation as the new best-known configuration under the
official protocol. Cache p3 remains canonical/unmodified; promotion of accepted features to
Baseline v4 / cache p4 deferred to a dedicated verification pass.

Next Ideas: recall-first sub-experiment on the gaps/first_activity lead as a grouped binary
"withdraw-within-k-weeks" task; Baseline v4 promotion; per-cutoff τ calibration standard.

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