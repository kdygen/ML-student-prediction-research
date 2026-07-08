# Experiment 003 — Early Intervention Index

**Date:** 2026-07-05
**Type:** evaluation-layer experiment. Baseline v4 (population, features, models,
hyperparameters, p4 cache) is used verbatim and unmodified; every score below is a
transformation of the official models' predicted probabilities.
**Data:** p4 cache, all cutoffs. **Split:** GroupShuffleSplit(group=`id_student`,
test_size=0.2), headline seed 42 + repeated seeds 0–4; zero student overlap asserted in every
split. Any tunable choice (combination weight, isotonic calibration) was fitted on inner
GroupKFold(3) out-of-fold predictions **within the training side only**.
**Raw metrics:** `reports/experiment_003_results.json` · **Figures:** `reports/figures/experiment_003/`
**Environment:** pinned (Python 3.12.6, sklearn 1.6.1, pandas 2.2.2, numpy 2.0.2, xgboost 3.3.0).

---

## 1. Literature review — do intervention indices like this already exist?

Yes. Reviewing deployed early-warning systems and the methodological literature (full notes
below), risk scores in education come in four families:

| Family | Examples | Construction |
|---|---|---|
| Predicted probability of a binary *adverse* outcome | Lakkaraju et al. KDD'15; Wisconsin DEWS (Knowles 2015, JEDM); Marist OAAI (Jayaprakash et al. 2014, JLA); Macfadyen & Dawson 2010 | The adverse outcome is a **pre-merged union of bad end-states** ("not graduating on time", "grade < C", "fail OR not submit"); students ranked by P(adverse) |
| Ensemble vote counts | OU Analyse v1 (Kuzilek et al. 2015 — the Open University's own system, on the data OULAD comes from) | 4 models vote; >2 votes ⇒ at-risk flag, weekly lists to tutors |
| Weighted composite of heterogeneous components | Purdue Course Signals (Arnold & Pistilli, LAK'12) | Proprietary weighted mix of performance, effort, history, demographics → traffic light |
| Transparent rule-based flags | Chicago On-Track (Allensworth & Easton 2005/07); Balfanz ABC flags (2007) | No model; thresholds on raw indicators, OR-combined |

Findings that directly shaped this experiment:

- **The Academic Support Index P(Fail)+P(Withdrawn) is precedented, not a novel heuristic.**
  Most OULAD studies merge {Fail, Withdrawn} into a single "at-risk" label before modelling
  (e.g. the OULAD systematic-review literature; OU Analyse's own target is the union
  "not submit OR score < 40"). Because the four classes are mutually exclusive,
  P(Fail)+P(Withdrawn) from the 4-class posterior **is exactly the field-standard P(adverse)**,
  recovered post-hoc instead of by label merging. Unequal severity weights (e.g. 2·P(W)+P(F))
  are the expected-cost ranking of Elkan (IJCAI'01) and need explicit institutional cost
  justification — no deployed system uses them.
- **Precision@K / recall@K under real intervention capacity is the canonical evaluation**
  (Lakkaraju et al., KDD'15: "school officials pick the top k% to match resources"; DSSG/Ghani
  Triage practice; a 2025 workload-aware OULAD system fixes the alert list at 15% of students
  for instructor workload, reporting 84% precision / 35% recall at day 14). K is an
  operational input, not a modelling output — report curves over K.
- **Calibration is near-mandatory when the score gates "act if risk > x%" decisions**
  (Van Calster et al. 2019, BMC Medicine; He et al. AAAI'15 require MOOC risk probabilities
  "well-calibrated and smoothed across weeks"); ranking itself is invariant to monotone
  recalibration. Wisconsin DEWS publishes its score *as* a calibrated frequency with a margin
  of error (high-risk if score + margin < 78.5).
- **Presentation norms:** traffic-light banding (red/amber/green) shown to educators, not
  students (OU Analyse; Scholes 2016 on labeling ethics); tutor-group size effectively sets K
  at the OU.
- **Known pitfalls:** ranking disagreement between comparable models (Bird et al. 2021, AERA
  Open: only ~60% bottom-decile overlap between OLS and XGBoost — report cross-model
  stability); fairness bias amplifies at small K (Bird & Castleman 2024); calibration drifts
  across presentations (recalibrate per cohort); separating ranking quality from intervention
  efficacy (the Course Signals retention-claim controversy; Perdomo et al. FAccT'25 found
  Wisconsin DEWS ranks accurately yet measurable intervention benefit was elusive).

Key sources: Lakkaraju et al. 2015 (https://dl.acm.org/doi/10.1145/2783258.2788620);
Kuzilek et al. 2015, OU Analyse (https://analyse.kmi.open.ac.uk/); Herodotou et al. 2019,
BJET 50(6) (teachers using OUA predictions → better student outcomes); Knowles 2015, JEDM 7(3);
Arnold & Pistilli 2012 (https://dl.acm.org/doi/10.1145/2330601.2330666); Allensworth & Easton
2005/2007; Balfanz et al. 2007; Macfadyen & Dawson 2010, C&E 54(2); Elkan 2001; Van Calster
et al. 2019, BMC Medicine 17:230; Guo et al. 2017 (ECE/reliability); Bird et al. 2021, AERA
Open 7; Gardner et al. 2019 (ABROCA); Hlosta et al. 2017 "Ouroboros" (LAK'17);
He et al. 2015 (AAAI); Perdomo et al. 2025 (FAccT).

## 2. Candidate indices

All built from the official Baseline v4 model posteriors P(W), P(F), P(P), P(D) on the
held-out grouped test set. Classes: 0=Withdrawn, 1=Fail, 2=Pass, 3=Distinction.

| Index | Formula | Rationale |
|---|---|---|
| **ASI** (Academic Support Index) | P(W) + P(F) | field-standard P(adverse) |
| **WRI** (Withdrawal Risk) | P(W) | withdrawal-specific campaigns |
| **FRI** (Failure Risk) | P(F) | failure-specific support |
| **SEV2** (severity-weighted) | 2·P(W) + P(F) | expected-cost ranking, withdrawal costlier |
| **WSEL** (tuned combination) | w·P(W) + (1−w)·P(F), w selected on inner OOF (grid 0…1) | let the data pick the weights |
| **ENT_ASI** (confidence-adjusted) | ASI · (1 − H/H_max), H = posterior entropy | penalize uncertain predictions |
| **ISO_ASI** (calibrated) | isotonic(ASI), fitted on inner OOF only | deployment probability |
| **ASI_COURSE_PCT** (cohort-normalized) | percentile rank of ASI within (module, presentation) | per-course tutor lists |
| **ENS_ASI** (model ensemble) | mean ASI over the 4 official models | cross-model averaging |
| **ENS_ASI−STD** (disagreement-penalized) | mean − std of ASI across models | uncertainty-aware variant |

Truth definitions evaluated against: **at-risk** = {Withdrawn, Fail} (prevalence 0.45→0.38
across cutoffs), **withdrawn** = {W} only, **fail** = {F} only.

## 3. Evaluation

### 3.1 Ranking quality — XGB indices vs at-risk truth (headline seed 42)

ROC-AUC:

| Index | c14 | c30 | c60 | c90 | c140 |
|---|--:|--:|--:|--:|--:|
| WRI | 0.6905 | 0.7413 | 0.7822 | 0.7995 | 0.8357 |
| FRI | 0.7064 | 0.7572 | 0.8158 | 0.8461 | 0.8964 |
| **ASI** | **0.7327** | **0.7817** | **0.8287** | **0.8571** | **0.9030** |
| SEV2 | 0.7284 | 0.7791 | 0.8255 | 0.8545 | 0.9000 |
| ENT_ASI | 0.6885 | 0.7655 | 0.8086 | 0.8441 | 0.8893 |
| WSEL (tuned) | 0.7327 | 0.7817 | 0.8287 | 0.8571 | 0.9030 |
| ISO_ASI | 0.7325 | 0.7812 | 0.8284 | 0.8562 | 0.9027 |
| ASI_COURSE_PCT | 0.7134 | 0.7658 | 0.8232 | 0.8471 | 0.8903 |
| ENS_ASI | 0.7317 | 0.7818 | 0.8263 | 0.8533 | 0.8995 |
| ENS_ASI−STD | 0.7289 | 0.7775 | 0.8223 | 0.8487 | 0.8943 |

Average precision (PR-AUC) shows the same ordering — ASI: 0.7008 / 0.7555 / 0.8062 / 0.8377 /
0.8827 (c14→c140); full table in the results JSON. Two decisive observations:

- **The tuned combination converges to ASI.** The inner-fold weight search picked w = 0.5 —
  i.e. exactly proportional to P(W)+P(F) — at **all five cutoffs** independently. Unequal
  weighting has no empirical support here.
- **Every "cleverer" variant ranks at or below plain ASI**: entropy adjustment *hurts*
  (−0.014 to −0.044 AUC), ensembling adds nothing over XGB alone, disagreement penalties and
  course-percentile normalization cost a little. Isotonic calibration leaves the ranking
  unchanged (as theory says it must, up to ties).

### 3.2 Top-K performance — how many truly at-risk students are in the flag list (ASI, XGB)

| Cutoff | prevalence | top 5%: P / R / lift (hits) | top 10%: P / R / lift (hits) | top 20%: P / R / lift (hits) |
|--:|--:|:--|:--|:--|
| 14  | 0.447 | 0.896 / 0.100 / 2.01 (251/280) | 0.847 / 0.190 / 1.90 (475/561) | 0.778 / 0.348 / 1.74 (872/1121) |
| 30  | 0.434 | 0.967 / 0.111 / 2.23 (265/274) | 0.929 / 0.214 / 2.14 (509/548) | 0.839 / 0.386 / 1.93 (919/1096) |
| 60  | 0.418 | 0.989 / 0.118 / 2.36 (261/264) | 0.966 / 0.231 / 2.31 (511/529) | 0.888 / 0.425 / 2.12 (939/1058) |
| 90  | 0.406 | 1.000 / 0.123 / 2.47 (256/256) | 0.990 / 0.244 / 2.44 (508/513) | 0.922 / 0.455 / 2.27 (946/1026) |
| 140 | 0.378 | 1.000 / 0.132 / 2.65 (243/243) | 0.998 / 0.264 / 2.64 (485/486) | 0.964 / 0.510 / 2.55 (938/973) |

Reading: at cutoff 30 (one month in), **96.7% of the top-5% list and 92.9% of the top-10%
list are genuinely at-risk students**; by cutoff 90 the top-5% list is pure. Lift at 5% is
near its theoretical maximum (1/prevalence): 2.23 of 2.30 at c30, 2.65 of 2.65 at c140 — the
top of the ranking is essentially as good as a ranking can be at this prevalence. The
constraint is capacity, not model quality: a 10% list can only ever contain ~23% of a 43%-
prevalence at-risk population (recall ceiling = K/prevalence), and ASI achieves 21.4% of that
23% at c30. Precision@K, recall@K and lift curves over K ∈ [1%, 50%]:
`fig2_topk_curves.png`, `fig4_lift.png`.

### 3.3 Targeted truths — one index does not fit all deployment goals

Best XGB index by ROC-AUC per truth (all cutoffs):

| Truth | Best index | AUC range c14→c140 | ASI for comparison |
|---|---|---|---|
| at-risk (W∪F) | **ASI** (every cutoff) | 0.733 → 0.903 | — |
| withdrawn only | **WRI** (every cutoff) | 0.656 → 0.766 | 0.638 → 0.734 |
| fail only | **FRI** (c14–c90; ASI ties at c140) | 0.695 → 0.870 | 0.690 → 0.871 |

The inner-fold weight search agrees: targeting withdrawn it picks w ≈ 0.8–0.9 (≈ pure WRI);
targeting fail, w ≈ 0.2–0.4 (≈ FRI). So the index must match the campaign: general support
outreach → ASI; withdrawal-prevention calls → WRI. Note withdrawal is intrinsically harder to
rank (WRI AUC ≤ 0.77 everywhere) — consistent with everything since Experiment 001.

### 3.4 Robustness — repeated grouped splits (seeds 0–4) and cross-model stability

ROC-AUC vs at-risk, mean ± std over 5 grouped splits:

| Cutoff | XGB ASI | RF ASI | LogReg ASI | 4-model ensemble |
|--:|:--|:--|:--|:--|
| 14  | 0.7291 ± 0.0029 | 0.7183 ± 0.0034 | 0.7192 ± 0.0030 | 0.7284 ± 0.0035 |
| 30  | 0.7890 ± 0.0058 | 0.7821 ± 0.0049 | 0.7774 ± 0.0054 | 0.7876 ± 0.0056 |
| 60  | 0.8366 ± 0.0053 | 0.8298 ± 0.0057 | 0.8217 ± 0.0061 | 0.8318 ± 0.0062 |
| 90  | 0.8621 ± 0.0057 | 0.8547 ± 0.0063 | 0.8429 ± 0.0050 | 0.8567 ± 0.0054 |
| 140 | 0.9116 ± 0.0039 | 0.9069 ± 0.0051 | 0.9002 ± 0.0047 | 0.9095 ± 0.0042 |

XGB ASI is best or tied-best at every cutoff with split noise ±0.003–0.006; the index
ordering of §3.1 (ASI ≥ SEV2 > ENT_ASI, WRI worst) reproduces on every seed. The ensemble
never beats plain XGB — no reason to pay 4× the serving complexity.

### 3.5 Calibration — can ASI be read as a probability?

Yes, essentially as-is (XGB posteriors; reliability plots in `fig3_reliability.png`):

| Cutoff | raw ASI Brier / ECE | isotonic ASI Brier / ECE |
|--:|:--|:--|
| 14  | 0.2058 / 0.0189 | 0.2062 / 0.0207 |
| 30  | 0.1852 / 0.0169 | 0.1851 / 0.0147 |
| 60  | 0.1617 / 0.0151 | 0.1619 / 0.0127 |
| 90  | 0.1451 / 0.0131 | 0.1455 / 0.0111 |
| 140 | 0.1144 / 0.0168 | 0.1144 / 0.0146 |

Raw ASI is already well-calibrated (ECE ≤ 0.02 at every cutoff; the reliability curve hugs
the diagonal), because the official XGB is trained without class re-weighting. Train-only
isotonic calibration nudges ECE down from c30 onward and is a cheap, safe deployment layer;
it changes no ranking decisions. (Caution for variants: LogReg with `class_weight="balanced"`
should *not* be read as a probability without recalibration.)

## 4. Feature contribution — what drives the index

Permutation importance on the headline grouped test (official v4 XGB; 5 repeats; metric =
AUC drop of the index against its truth; `fig5_feature_contribution.png`; full values in the
results JSON under `feature_contribution`). Promoted v4 features marked ★.

Top contributors at cutoff 30 — ASI: ★rank_wa (0.071, ~4.5× the runner-up),
highest_education_Lower-than-A-Level (0.016), ★rank_active_days (0.016), ★rank_clicks (0.011),
weighted_average (0.007), assessment_focus, ★first_submit_day, ★mean_submit_lead.
WRI adds ★first_submit_day/★mean_submit_lead higher (submission timing signals withdrawal);
FRI leans on ★rank_active_days and education background.

At cutoff 90, promoted features take nearly every top slot — ASI: ★rank_wa (0.049),
★submitted_count (0.019), ★decay_clicks (0.016), weighted_average, ★days_since_last,
clicks_per_assessment, ★mean_submit_lead, ★rank_active_days; for WRI the top seven are all
promoted features.

Conclusion: the intervention ranking is powered chiefly by the Baseline v4 promotion —
**cohort-normalized performance (`rank_wa`) is the single dominant signal for every index**,
with submission timing (`first_submit_day`, `mean_submit_lead`, `submitted_count`) driving
the withdrawal component and recency/engagement (`decay_clicks`, `days_since_last`) growing
in importance as the course progresses.

## 5. Deployment perspective — how an advisor would use this

**The ranked list, not a threshold, is the primary interface** (per Lakkaraju et al. and OU
Analyse practice). Each week/cutoff: score every enrolled student with ASI from the official
XGB, sort descending, and work down the list until intervention capacity is exhausted.

- **Threshold selection = capacity selection.** K should come from advising resources, not
  from the model (the literature is unanimous). Concretely at cutoff 30, the full enrolled
  population is 27,450: a top-5% policy means contacting ~1,370 students with ~97% of them
  genuinely at-risk; top-10% ≈ 2,745 students at ~93% precision; top-20% ≈ 5,490 at ~84%.
  Per-course, OU-style tutor groups (~20 students) imply 1–4 flagged students per tutor at
  the 10% operating point — use `ASI_COURSE_PCT` only for *allocating* within tutor groups;
  keep global ASI as the score of record (its global ranking is strictly better, §3.1).
- **False positives vs false negatives.** At small K the list is nearly pure (FP rate 0–7%
  at 5–10% from c30 on), so the practical cost is false *negatives*: a 10% list reaches only
  ~21–26% of all at-risk students — that is a capacity ceiling (recall ceiling = K/prevalence
  ≈ 23–26%), not a model deficiency. Institutions wanting recall must raise K: catching half
  of the at-risk population at c30 requires K ≈ 26% at 84%-ish precision (fig2). Deployed
  systems accept high-K imprecision because interventions are cheap and benign relative to a
  missed dropout — but two documented FP harms (labeling effects; discouragement-triggered
  withdrawal, seen at Purdue and Marist) argue for showing risk to staff, never to students.
- **Confidence estimates.** ASI is calibrated (ECE ≤ 0.02), so "this student's probability of
  an adverse outcome is 78%" is a faithful statement; add the isotonic layer to tighten it.
  For per-student confidence, the DEWS pattern (score ± margin) can be reproduced from
  seed-variance (±0.003–0.006 AUC ⇒ rank stability is high) or model disagreement; the
  disagreement-penalized ranking performed slightly *worse* (§3.1), so use disagreement as a
  secondary "verify manually" flag, not as a rank modifier.
- **Banding.** Traffic-light presentation maps directly: red = top 5% (near-certain risk),
  amber = 5–20%, green = rest, with the calibrated probability shown to staff alongside the
  band and the top contributing features (§4) as the "why".
- **Cadence and drift.** Re-score at each cutoff; recalibrate per presentation (calibration
  drift across cohorts is documented in the literature); audit top-K subgroup composition
  (ABROCA / Aequitas-style) before institutional rollout — fairness bias concentrates at
  small K.

## 6. Recommendation

**Adopt ASI = P(Fail) + P(Withdrawn) from the official Baseline v4 XGBoost, with a train-only
isotonic calibration layer, as the official deployment output** ("Academic Support Index").
One clear winner:

1. **Best ranker for the intervention population at every cutoff** (AUC 0.73→0.90, AP
   0.70→0.88), best-or-tied on every seed, near-ceiling lift at deployable K.
2. **The data independently chose it**: the tuned weighted combination collapsed to w = 0.5 at
   all five cutoffs; severity weighting, entropy adjustment, ensembling, disagreement
   penalties and cohort normalization all ranked at-or-below plain ASI.
3. **It is the field-standard construction** — the marginalized P(adverse) that OULAD
   literature and deployed EWS (OU Analyse, DEWS, Lakkaraju) all use, so results are
   comparable to prior work.
4. **It is already calibrated** (ECE ≤ 0.02), so the same number serves ranking, banding, and
   honest probability communication.

Scoped exception, not a competitor: for a **withdrawal-specific** campaign, rank by WRI =
P(Withdrawn) (clearly better on that truth at every cutoff, §3.3). ENT_ASI, SEV2, ensembles
and percentile variants are documented negative results.

## 7. Rules compliance

- Baseline v4 untouched: no notebook change, no cache change, no model/feature/hyperparameter
  change; all scores are post-hoc transformations of official-model posteriors (drivers
  archived in `experiments/experiment_003_*.py`).
- Grouped evaluation throughout: every split (headline, repeats, inner folds) grouped by
  `id_student` with zero-overlap assertions.
- No test-set tuning: WSEL weights and isotonic calibration fitted on inner out-of-fold
  train predictions only; fixed-formula indices involve no fitting at all.
- Every metric documented: `reports/experiment_003_results.json` (per cutoff × model × index
  × truth: AUC, AP, precision/recall/lift/hits at 5/10/20%, Brier, ECE, selected weights,
  repeat-seed distributions, permutation importances).
