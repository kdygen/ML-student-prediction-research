# Experiment 004 — Early Intervention Decision Framework

**Date:** 2026-07-05
**Type:** evaluation/decision-layer experiment. Baseline v4 is used verbatim: official XGB
hyperparameters, official 35-feature set, p4 cache. No model changes, no new features, no
cache changes. (The official model is *fitted* per the official protocol to produce
out-of-sample predictions — as in Experiments 001–003 — never altered or tuned.)
**Protocol:** fixed student panel — one global grouped 80/20 split of the union of student
IDs across all cutoffs (rng seed 42; 25,262 students, 5,052 test). Every test student is
out-of-sample at every cutoff, enabling leakage-free individual trajectories. Grouped by
construction; sanity-checked against the official GSS-42 baseline (all metrics within
±0.014, i.e. split noise). Isotonic calibration fitted on inner GroupKFold(3) train-side
out-of-fold predictions only.
**Raw results:** `reports/experiment_004_results.json` · **Figures:** `reports/figures/experiment_004/`
**Environment:** pinned (Python 3.12.6, sklearn 1.6.1, pandas 2.2.2, numpy 2.0.2, xgboost 3.3.0).

**Bands used throughout** (within-cutoff test ranking of ASI = P(Fail)+P(Withdrawn), the
Experiment-003 official score): **red** = top 5%, **amber** = 5–20%, **green** = rest.

---

## Phase 1 — Literature: how existing systems decide when to intervene

Deployed systems use four timing mechanisms, none derived from a formal reliability
analysis (details and URLs in the review notes; key sources cited inline):

1. **Fixed high-frequency re-scoring** — OU Analyse re-scores weekly against the *next
   assessment* (Kuzilek et al. 2015; the Open University, source of OULAD), maintaining dual
   horizons (next-TMA and final outcome) — the main deployed multi-horizon precedent.
2. **Fixed sparse calendar checkpoints** — Marist OAAI at 25/50/75% of semester with four
   probability bands (<50 / 50–75 / 75–90 / >90%) and *escalating message tone* on repeat
   flags (Jayaprakash et al. 2014); Wisconsin DEWS twice yearly; EAB Navigate campaign waves.
3. **Event triggers** — Nottingham Trent's StREAM alerts tutors after **14 consecutive days
   of zero engagement** (10 for first-years); RTI/MTSS tier escalation on documented
   non-response to a lower tier.
4. **Human-discretionary timing** — Course Signals was instructor-run on demand (with an
   explicitly acknowledged lack of consensus on timing/frequency); a 2025 Georgia State study
   (Schechtman et al., arXiv:2505.13325) found ~2/3 of advisor interventions were targeted
   using non-algorithmic context — lists are a starting point, not the decision rule.

**Multiple horizons & dynamic risk:** Taylor et al. 2014 (MOOCs) built the canonical
lead×lag accuracy matrix (AUC ~0.9 one week ahead vs ~0.7 for end-of-course from week 1);
survival/hazard models move from *whether* to *when* (Ameri et al. CIKM'16; Masci et al.
2024, whose title names the trade-off: "accurate *and* early predictions"); Aguiar et al.
LAK'15 add an **urgency score** (predicted time-to-off-track) alongside risk, noting risk
magnitude is not a valid proxy for urgency — the direct descendant of Lakkaraju's
"identify before off-track" metric.

**Thresholds:** OAAI's fixed probability bands; DEWS's uncertainty-aware rule (high-risk if
score + margin < 78.5); capacity-anchored top-K everywhere else (a 2025 workload-aware OULAD
system fixes the list at 15% of students). Howard, Meehan & Parnell (2018) is the closest
"earliest reliable point" precedent: they name the earliness–accuracy tension and pick weeks
5–6 of 12 *by inspection*; Adnan et al. (2021) draw OULAD accuracy-vs-progress curves at
0–100% of course length but attach no decision criterion.

**Acknowledged gaps** (which Phase 6 targets): (i) of 689 learning-analytics papers surveyed
by Larrabee Sønderlund et al. 2019, only 11 evaluate interventions at all; (ii) the
earliness–accuracy trade-off is named but not formalized in education (the early-time-series
classification literature has the cost machinery, unapplied); (iii) re-flagging/escalation
policy is essentially unstudied; (iv) accurate flags demonstrably fail to move outcomes
without capacity and process (Perdomo et al. FAccT'25 on Wisconsin DEWS; the EWIMS RCT's
fidelity problems); (v) no published framework reconciles short- and long-horizon signals
into a single act/monitor/wait decision.

## Phase 2 — Progressive prediction analysis (fixed panel, out-of-sample at every cutoff)

| Cutoff | Acc | Macro-F1 | ASI AUC | ASI AP | ECE raw→iso | Mean entropy | P@5% / R@5% | P@10% / R@10% | P@20% / R@20% | WRI AUC |
|--:|--:|--:|--:|--:|:--|--:|:--|:--|:--|--:|
| 14  | 0.477 | 0.332 | 0.721 | 0.685 | 0.015→0.012 | 0.816 | 0.858 / 0.094 | 0.821 / 0.180 | 0.746 / 0.327 | 0.661 |
| 30  | 0.529 | 0.407 | 0.786 | 0.773 | 0.021→0.018 | 0.753 | 0.978 / 0.110 | 0.945 / 0.213 | 0.855 / 0.386 | 0.710 |
| 60  | 0.581 | 0.446 | 0.829 | 0.810 | 0.015→0.010 | 0.676 | 0.992 / 0.119 | 0.966 / 0.232 | 0.896 / 0.430 | 0.719 |
| 90  | 0.618 | 0.466 | 0.862 | 0.837 | 0.014→0.012 | 0.625 | 0.996 / 0.126 | 0.986 / 0.249 | 0.918 / 0.464 | 0.739 |
| 140 | 0.703 | 0.521 | 0.914 | 0.892 | 0.012→0.012 | 0.506 | 1.000 / 0.138 | 1.000 / 0.276 | 0.966 / 0.534 | 0.801 |

(Truths: ASI vs at-risk {W,F}; WRI vs withdrawn. Timeline plots: `fig1_progressive_timeline.png`.)

**Uncertainty is informative and falls steadily**: mean posterior entropy 0.82→0.51; the
share of confident cases (entropy < 0.5) grows 0.3%→44%; 4-class accuracy in the
lowest-entropy quartile is 0.52/0.60/0.68/0.76/0.87 (c14→c140) versus 0.43–0.55 in the
highest — model confidence is a valid advisor signal at every cutoff.

**Earliest reliable prediction point.** Reliability criterion (stated before inspection,
from the deployment requirements): *red-band precision ≥ 0.95 and calibrated ECE ≤ 0.02*.
Day 14 fails (red precision 0.858); **day 30 passes** (0.978; iso-ECE 0.018) and every later
cutoff passes. Day 14 is usable for provisional triage (top-5% still 86% correct) but not
for confident automatic flagging.

**Marginal benefit of waiting** (ASI AUC per 10 days): +0.041 (c14→c30), +0.014 (c30→c60),
+0.011 (c60→c90), +0.010 (c90→c140) — steep early gains, then flat diminishing returns.
The cost side of waiting is in Phase 5: withdrawal reachability falls ~0.070 per 10 days
over c30→c60 — **five times faster than the AUC gain over the same window**.

## Phase 3 — Student trajectories (test panel, cases present at day 14; n = 5,639)

Band trajectories across cutoffs classify into seven categories with sharply separated
outcome rates (`fig3_trajectories.png`):

| Trajectory | n | share | P(adverse) | P(withdrawn) | P(fail) |
|---|--:|--:|--:|--:|--:|
| early_exit (withdrew before day 60) | 387 | 6.9% | 1.000 | 1.000 | — |
| escalating (green → flagged by c60) | 425 | 7.5% | **0.993** | 0.379 | 0.614 |
| persistent_high (flagged at every cutoff) | 375 | 6.7% | 0.973 | 0.333 | 0.640 |
| late_emerging (first flagged c90+) | 242 | 4.3% | 0.942 | 0.211 | 0.731 |
| intermittent | 394 | 7.0% | 0.657 | 0.190 | 0.467 |
| recovering (flagged early → green) | 464 | 8.2% | 0.418 | 0.162 | 0.256 |
| persistent_low (never flagged) | 3,348 | 59.4% | 0.215 | 0.095 | 0.119 |

- **Rising risk is the single strongest signal**: escalating students end adverse at 99.3% —
  *above* persistent-high (97.3%). Rapid risers (Δ calibrated ASI ≥ +0.15 between
  checkpoints) run 68–80% adverse versus 31–38% for everyone else, at every step.
- **Early vs late identification** (eventual-adverse cases, top-20% flag budget per cutoff):
  47.3% are first flagged by day 30; 12.8% at c60–c90; 5.7% only at c140; **34.1% are never
  flagged** at this budget (40.1% for withdrawn — withdrawal stays the hardest target).
- **Bands are sticky and prognostic**: 71% of c30-red stays red at c60; green is 92% stable.
  Red's *imminent-withdrawal* rate (unregisters before the next checkpoint) is 18.1%
  (c30→c60) versus 2.4% for green — a 7.6× gradient that justifies treating red as urgent.
- **Intervention lead time** (withdrawn students, first flag → actual unregistration day):
  median **58 days** (IQR 24–120; never negative). Flagged at c14/c30 the median is 61–64
  days; first-flagged at c140 it shrinks to 38. Early flags buy two months of contact window.
- **Recovering students** (8.2%) still end adverse at 42% — de-escalation is real but not a
  discharge; they fit "continue monitoring", not "low risk".

## Phase 4 — The decision framework

Combines cutoff, calibrated probability, ranking, confidence, trajectory, and capacity.
Diagram: `fig5_decision_flow.png`. Every rule carries its measured evidence:

| Recommendation | Rule (at each checkpoint) | Evidence (test panel) |
|---|---|---|
| **Immediate intervention** | red band (top 5%; calibrated P(adverse) ≥ 0.92 at c30, ≥ 0.99 from c60) **or** rapid riser (Δ calibrated ASI ≥ +0.15 since last checkpoint) | red: P(adverse) 0.98–1.00 from c30 on; 18% of c30-red unregisters before c60; escalating trajectory ends 99.3% adverse; rapid risers 68–80% adverse |
| **Monitor closely** | amber band (5–20%; calibrated P ≈ 0.65–0.79) | amber: P(adverse) 0.71–0.95 across cutoffs; 12% of c30-amber escalates to red by c60; 11.6% unregisters before c60 |
| **Continue monitoring** | amber at 2+ consecutive checkpoints without escalation; recovering cases; green with rising ASI below the amber line; high-entropy cases (top quartile) get manual advisor verification | recovering still ends 42% adverse; top-entropy-quartile accuracy is 0.43–0.55 (vs 0.52–0.87 in the most confident quartile) — model confidence is not uniform |
| **Low risk** | green band, stable or falling ASI | green: P(adverse) 0.38 (c14) → 0.21 (c140); 92% band-stable; 1.5–2.4% near-term unregistration |

Honest caveat baked into the framework: **green is a capacity statement, not a safety
certificate** — at c30 the green band still carries a 34% adverse rate (prevalence is 44%).
The framework allocates scarce attention; it does not certify students as safe.

## Phase 5 — Capacity-aware evaluation (reference cohort = test-panel cases at day 14)

Reference cohort: 5,635 cases; 2,574 eventually adverse (45.7%); 1,193 eventually withdrawn.
A student can only be reached **while still enrolled** — and the reachable pool decays:

| Cutoff | adverse still reachable | withdrawn still reachable |
|--:|--:|--:|
| 14  | 100% | 100% |
| 30  | 94.7% | 88.6% |
| 60  | 85.0% | 67.6% |
| 90  | 77.7% | 51.9% |
| 140 | 67.3% | 29.5% |

**One-shot policies** (flag top-20% once; `fig2_timing_tradeoff.png`):

| Intervene at | precision | adverse reached | withdrawn reached | median lead (days) |
|--:|--:|--:|--:|--:|
| c30  | 0.851 | 37.3% | **38.6%** | **64** |
| c90  | 0.895 | 39.2% | 24.5% | 59 |
| c140 | **0.933** | **40.8%** | 16.3% | 38 |

Waiting buys +8 points of precision and +3.5 points of adverse recall — and costs **22
points of withdrawal recall and 26 days of lead time**. For failure-support, timing barely
matters (failing students remain enrolled); for withdrawal prevention, late intervention is
self-defeating: by day 140, 70% of eventual withdrawals are already gone.

**Staged policies at the identical total budget (20% of cohort; `fig4_policies.png`):**

| Policy | precision | adverse reached | withdrawn reached | median lead |
|---|--:|--:|--:|--:|
| one-shot c140 (benchmark) | 0.933 | 40.8% | 16.3% | 38d |
| **staged 10% @c30 + 10% @c90 (top-up)** | 0.928 | 40.7% | **34.0%** | 50d |
| staged 10% @c30 + 5% @c60 + 5% @c90 | 0.927 | 40.6% | 35.2% | 50d |
| staged 5% @ each of c14/c30/c60/c90 | 0.916 | 40.1% | 35.3% | 49d |

**The staged policy dominates the late one-shot**: it matches precision (−0.005) and
adverse recall (−0.001) while **more than doubling withdrawal reach** (34.0% vs 16.3%) and
adding 12 days of median lead time. There is no meaningful cost to intervening early *if*
capacity is staged — the early stage catches withdrawals while they are still reachable, and
the later stage adds precision and late-emerging cases (the 4.3% late-emerging trajectory).

**Best intervention value:** day 30 for the first (largest) stage; day 90 for the top-up;
day 60 optionally splits the difference. Day 14 adds withdrawal reach only marginally
(35.3% vs 35.2%) at visibly lower precision (0.916) — consistent with c14 failing the
reliability criterion.

## Phase 6 — Recommendation: the official intervention methodology

1. **Score of record:** calibrated ASI from the official Baseline v4 XGB (Experiment 003),
   re-computed at each checkpoint; WRI retained for withdrawal-specific campaigns.
2. **Earliest Reliable Intervention Point (ERIP) methodology** — defined and applied:
   *(i)* fix a reliability criterion from deployment requirements **before** inspection
   (ours: red-band precision ≥ 0.95 and calibrated ECE ≤ 0.02); *(ii)* find the earliest
   checkpoint satisfying it (**day 30** here; day 14 fails on red precision 0.858);
   *(iii)* weigh it against the remaining-opportunity curve (at day 30: 94.7% of adverse and
   88.6% of withdrawals still reachable, median 64-day lead) and confirm the waiting
   trade-off is unfavorable (AUC gains +0.014/10d vs withdrawal-reachability losses
   −0.070/10d after day 30). All three components are measurable on any course with cached
   cutoff datasets, so ERIP transfers beyond OULAD.
3. **Deployment workflow:** first full advising pass at day 30 with ~half the term's
   outreach capacity; top-up passes at day 60/90 with the remainder, flagging only
   not-yet-contacted students; day-14 scores used for provisional triage/preparation, not
   automatic outreach. Between checkpoints, the rapid-riser rule (Δ calibrated ASI ≥ +0.15)
   escalates immediately at the next scoring.
4. **Advisor decision rules:** the Phase-4 table (immediate / monitor closely / continue
   monitoring / low risk), each rule with measured outcome rates; risk shown to staff, not
   students (OU Analyse practice; labeling-harm literature).
5. **Confidence thresholds:** calibrated P(adverse) ≥ ~0.92 = red/immediate; ~0.65–0.79 =
   amber/monitor; posterior entropy in the top quartile ⇒ manual verification regardless of
   band (accuracy there is 0.43–0.55). List sizes remain capacity-set (top-K), thresholds
   are the communicated meaning of the bands, not the selection mechanism.

### What is known / reproduced / improved / genuinely new

- **Already known (literature):** timing in deployed systems is set by organizational rhythm,
  not reliability analysis; risk bands, escalation-on-repeat-flag, and capacity-anchored
  top-K are standard; accuracy-vs-progress curves exist for OULAD (Adnan et al. 2021);
  survival models argue "when" matters as much as "whether"; accurate flags without capacity
  and process do not move outcomes (Wisconsin DEWS, EWIMS).
- **Reproduced here:** the rising accuracy/AUC-vs-progress curve on OULAD (now under a
  leakage-hardened grouped protocol with calibration measured at every point); band/tier
  decision structures (OAAI-style); the finding that early identification of eventual
  withdrawals is the binding constraint.
- **Improved over prior practice:** the earliness–accuracy trade-off is *jointly quantified*
  on one cohort — both sides (prediction quality AND reachable at-risk pool with real
  unregistration dates and lead times), where Howard et al. balanced by inspection and Adnan
  et al. reported only the quality side; trajectory categories carry measured outcome rates
  (escalating = 99.3% adverse); the rapid-riser escalation rule is quantified (68–80% vs
  31–38%); staged-vs-one-shot capacity policies are compared at equal budget with
  unique-reach accounting.
- **Genuinely new (supported by the review's gap analysis):** the integrated **ERIP
  methodology** — a pre-stated reliability criterion + earliest satisfying checkpoint +
  remaining-opportunity weighing + capacity staging, evaluated end-to-end on OULAD; and the
  headline empirical result that **a staged 10%+10% policy dominates the late one-shot at
  equal budget** (equal precision/adverse recall, 2.1× withdrawal reach, +12 days lead).
  The review found the components exist separately but "the integration does not."
- **Not claimed:** any intervention *treatment effect*. This framework optimizes targeting
  and timing; whether contacted students improve is unmeasurable in OULAD (no intervention
  arm) and the literature (Perdomo et al.) warns effects cannot be assumed. That is the
  correct next validation step, not a modeling one.

## Rules compliance

- Baseline v4 untouched (no notebook/cache/feature/model/hyperparameter changes); official
  XGB fitted per protocol only; drivers archived as `experiments/experiment_004_*.py`.
- No retraining beyond the official protocol fit; no new features (bands, velocities, and
  trajectories are transformations of model outputs, never model inputs).
- Grouped evaluation throughout: the panel split is at student level by construction;
  overlap asserted zero at every cutoff; inner calibration folds grouped by student.
- Every reported number is in `reports/experiment_004_results.json`
  (phase2_progressive / phase3_trajectories / phase5_policies).
