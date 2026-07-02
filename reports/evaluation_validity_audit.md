# Evaluation Validity Audit — Phases 1–4 (Baseline v2 → v3)

**Date:** 2026-07-02
**Environment:** pinned Colab stack (Python 3.12.6, scikit-learn 1.6.1, pandas 2.2.2,
numpy 2.0.2, xgboost 3.3.0), macOS/arm64.
**Subject:** the Baseline v2 pipeline (leakage-free features, survivor sample, random 80/20
split). All measurements were produced by executing the notebook cells verbatim per cutoff and
instrumenting the result; models and hyperparameters are the notebook's own throughout.

---

## Phase 1 — Evaluation leakage (student overlap between train and test)

### 1.1 The problem

Deployment predicts outcomes for **students the model has never seen**. In `mlData`, the same
`id_student` appears in up to **4 rows** (multi-enrollment across module-presentations:
1,456 students at cutoff 30; 2,302 at cutoff 140). The v1/v2 random row split
(`train_test_split(random_state=42)`) therefore places the *same student* on both sides.

### 1.2 Quantified overlap (v2, random split)

| Cutoff | train students | test students | overlapping students | % of test students | test rows w/ student in train | % of test rows |
|-------:|------:|------:|------:|------:|------:|------:|
| 14  | 950    | 238   | **0**   | 0.0   | 0   | 0.0   |
| 30  | 14,463 | 3,799 | **464** | 12.2  | 467 | 12.1  |
| 60  | 17,217 | 4,581 | **763** | 16.7  | 777 | 16.6  |
| 90  | 17,202 | 4,585 | **723** | 15.8  | 734 | 15.7  |
| 140 | 17,228 | 4,603 | **752** | 16.3  | 760 | 16.2  |

### 1.3 Expected impact on reported metrics (measured, not assumed)

Test accuracy conditioned on whether the row's student also appears in train (v2 random split):

| Cutoff | RF all | RF overlap rows | RF non-overlap | XGB all | XGB overlap | XGB non-overlap |
|-------:|-----:|-----:|-----:|-----:|-----:|-----:|
| 30  | 0.5122 | 0.4561 | 0.5199 | 0.5189 | 0.4690 | 0.5258 |
| 60  | 0.5447 | 0.5122 | 0.5512 | 0.5499 | 0.5148 | 0.5568 |
| 90  | 0.5670 | 0.5409 | 0.5719 | 0.5760 | **0.5804** | 0.5752 |
| 140 | 0.6078 | 0.5474 | 0.6194 | 0.6180 | 0.5684 | 0.6275 |

**Honest reading:** overlap rows are mostly *harder*, not easier — multi-enrollment students
are disproportionately repeat/struggling students. So the random split's inflation is **not**
a large per-row memorization effect in aggregate. The violation is nonetheless real: (a) the
protocol does not match the deployment question (unseen students); (b) within-student
correlation still leaks identity information into the model; (c) the measured metric mixes two
populations (seen/unseen students) in proportions that are an artifact of the split seed.

### 1.4 Grouped evaluation (identical models and hyperparameters)

`GroupShuffleSplit(group=id_student, test_size=0.2, random_state=42)` on the same v2 data:

| Cutoff | LogReg acc/F1 | Tree acc/F1 | RF acc/F1 | XGB acc/F1 | RF Δacc vs random | XGB Δacc |
|-------:|:---|:---|:---|:---|---:|---:|
| 14  | 0.3403/0.3135 | 0.5210/0.3350 | 0.4664/0.2735 | 0.4748/0.3524 | +0.000 | −0.013 |
| 30  | 0.3962/0.3935 | 0.5012/0.3227 | 0.4981/0.3537 | 0.4988/0.3744 | −0.014 | −0.020 |
| 60  | 0.4175/0.4086 | 0.5274/0.3912 | 0.5377/0.4029 | 0.5535/0.4462 | −0.007 | +0.004 |
| 90  | 0.4379/0.4279 | 0.5410/0.4259 | 0.5570/0.4418 | 0.5613/0.4610 | −0.010 | −0.015 |
| 140 | 0.4851/0.4706 | 0.5653/0.4811 | 0.6069/0.5102 | 0.6190/0.5349 | −0.001 | +0.001 |

Robustness (RF/XGB, same models):

| Cutoff | GroupKFold(5) RF acc | GKF5 XGB acc | Repeated GSS (seeds 0–4) RF | XGB |
|-------:|:---|:---|:---|:---|
| 14  | 0.4562 ± 0.0056 | 0.4512 ± 0.0102 | 0.4655 ± 0.0198 | 0.4588 ± 0.0142 |
| 30  | 0.5031 ± 0.0057 | 0.5070 ± 0.0079 | 0.5007 ± 0.0060 | 0.5071 ± 0.0108 |
| 60  | 0.5462 ± 0.0050 | 0.5549 ± 0.0043 | 0.5461 ± 0.0045 | 0.5547 ± 0.0045 |
| 90  | 0.5678 ± 0.0098 | 0.5764 ± 0.0101 | 0.5615 ± 0.0066 | 0.5721 ± 0.0052 |
| 140 | 0.6068 ± 0.0066 | 0.6174 ± 0.0045 | 0.6108 ± 0.0051 | 0.6205 ± 0.0047 |

### 1.5 Verdict and recommendation

Grouped evaluation costs at most ~2 accuracy points (early cutoffs) and is within split noise
(±0.005–0.02) at later ones. It is a **more realistic deployment estimate** because the test
set contains only students absent from training — exactly the population the model will face —
and because repeated grouped splits give an honest variance estimate instead of one arbitrary
seed. **Recommendation: grouped evaluation (group = `id_student`) becomes the official
protocol** (adopted in Baseline v3; single GSS seed 42 as the headline, repeated GSS seeds 0–4
reported as mean±std).

---

## Phase 2 — Sample membership / survivorship bias

### 2.1 How prediction cases were created in v1/v2

`mlData` is seeded from `courseworkPerformance` — only (student, module, presentation) pairs
with **submitted, weighted coursework due before the cutoff**. Membership therefore depends on
the student's own (future-correlated) behaviour.

### 2.2 Quantification (registered population = `studentInfo`, 32,593 pairs)

| Cutoff | registered | represented | missing | missing % | withdrawn ≤ cutoff | registered > cutoff | risk population* | risk missing | risk missing % |
|-------:|------:|------:|------:|-----:|-----:|----:|------:|------:|-----:|
| 14  | 32,593 | 1,188  | 31,405 | **96.4%** | 4,474 | 58 | 28,061 | 26,881 | **95.8%** |
| 30  | 32,593 | 19,300 | 13,293 | **40.8%** | 5,127 | 16 | 27,450 | 8,518  | **31.0%** |
| 60  | 32,593 | 23,411 | 9,182  | 28.2% | 6,232 | 8  | 26,353 | 4,042  | 15.3% |
| 90  | 32,593 | 23,452 | 9,141  | 28.1% | 7,031 | 4  | 25,558 | 3,846  | 15.0% |
| 140 | 32,593 | 23,478 | 9,115  | 28.0% | 8,303 | 1  | 24,289 | 3,637  | 15.0% |

\* risk population = still enrolled at the cutoff (not unregistered ≤ cutoff, not registered
after it) — the students a deployed model would actually score.

### 2.3 The exclusion is outcome-correlated (this is what makes it bias)

Missing vs represented students by final result, cutoff 30:

| final_result | missing | represented | share of missing | share of represented |
|---|------:|------:|-----:|-----:|
| Withdrawn   | 6,256 | 3,900 | **47.1%** | 20.2% |
| Pass        | 3,455 | 8,906 | 26.0% | 46.1% |
| Fail        | 2,646 | 4,406 | 19.9% | 22.8% |
| Distinction | 936   | 2,088 | 7.0%  | 10.8% |

Students disappear from v1/v2 precisely because they have not submitted coursework — and
non-submission is itself a withdrawal signal. The reported v1/v2 metrics therefore describe a
**conditioned, easier population** ("students who already engaged with assessment"), not the
population a deployed early-warning system must score. **CONFIRMED methodological flaw.**

### 2.4 Redesign (implemented as Baseline v3)

Prediction cases originate from **all registered pairs still enrolled at the cutoff**:

- **preserve every registered student** in the risk population (e.g. 27,450 at cutoff 30 vs
  19,300 in v2 — +8,150 previously-dropped students, 73% of them Withdrawn/Fail);
- **exclude decided outcomes**: pairs unregistered on/before the cutoff (withdrawal already
  observed — not a prediction task; deployment would not score them) and pairs registered
  after the cutoff (not yet enrolled on prediction day). Both facts are known at the cutoff,
  so this filter is temporally valid;
- **represent unavailable information explicitly** instead of dropping rows:
  `has_vle_activity`, `has_coursework` indicator features + zero-filled feature values;
- **preserve comparability**: identical models/hyperparameters, identical leakage-free v2
  feature tables, same seed, same 80/20 proportion.

Documented consequence: the Withdrawn class now (correctly) shrinks with later cutoffs
(5,034 at c30 → 1,860 at c140) because withdrawals that already happened are known outcomes,
not prediction targets. Class balance varies across cutoffs **by design** — as it does in
deployment.

---

## Phase 3 — Target validity

Checked at every cutoff (14/30/60/90/140) on the final v2 `mlData`:

| Check | Result (all cutoffs) |
|---|---|
| duplicate (id_student, code_module, code_presentation) rows | **0** |
| duplicate columns after merges | **0** |
| leftover merge-suffix columns (`*_x`/`*_y`) | **none** |
| `final_result` label mismatch vs `studentInfo` | **0** |
| rows = unique key combinations | yes (one row = one valid prediction case) |
| same student in several rows | up to 4 (legit multi-enrollment; 1,456 students at c30, 2,302 at c140) → handled by grouped split |
| target derivation | `target_multi` is a pure recode of `final_result` (W=0, F=1, P=2, D=3); no feature in `X` derives from `final_result` |

**Verdict:** no duplicate-merge errors, no label inconsistencies, no hidden target leakage at
the row level. The multi-row students are legitimate enrollment records, not pipeline errors;
their only methodological consequence (split contamination) is fixed by grouping.

---

## Phase 4 — Temporal validity (re-audit of every active feature)

Re-audit of the v2 feature set plus the two v3 additions. Full derivations in
`reports/leakage_audit.md`; classifications re-verified against the current notebook.

| # | Feature | Raw source → filter | Classification |
|--:|---|---|---|
| 1 | `active_days` | `studentVle.date ≤ CUTOFF` (activity day) | **SAFE** |
| 2 | `clicks_per_day` | same + `sum_click` | **SAFE** |
| 3 | `study_spread` | `active_days / CUTOFF` (constant divisor) | **SAFE** |
| 4 | `burstiness` | per-day clicks ≤ cutoff, `std/(mean+1)` | **SAFE** (platform FP < 1e-9 noted) |
| 5–9 | `resource/oucontent/homepage/forumng/quiz_ratio` | `studentVleEarly ⋈ vle.activity_type` (static metadata) | **SAFE** |
| 10 | `assessment_focus` | clicks ≤ cutoff in 7-day windows before **deadlines ≤ cutoff**; deadlines are published schedule | **SAFE** |
| 11 | `weighted_average` (v2) | scores with `date_submitted ≤ CUTOFF` | **SAFE** (fixed in v2) |
| 12 | `clicks_per_assessment` (v2) | count with `date_submitted ≤ CUTOFF` | **SAFE** (fixed in v2) |
| 13 | `recovery_slope` (v2) | score changes, submissions ≤ cutoff, ordered by (known) due date | **SAFE** (fixed in v2) |
| 14–17 | `highest_education_*` | `studentInfo`, recorded at registration | **SAFE** |
| 18 | `has_vle_activity` (v3) | presence of any ≤cutoff VLE row | **SAFE** |
| 19 | `has_coursework` (v3) | presence of any ≤cutoff submitted weighted coursework | **SAFE** |
| — | v3 membership rule | `date_registration`, `date_unregistration` compared to cutoff — both facts observable on the day they occur | **SAFE** |

**No POTENTIAL or CONFIRMED leakage remains among active features.** No feature was modified
in this phase (the v2 fixes already stand; nothing new was proven leaky).

Assumptions carried (explicit): `date_submitted` proxies score availability (OULAD has no
grading date; conservative). `date_unregistration = NaN` is treated as "did not withdraw
early" (these students remain in the risk population). Deadline days are treated as known in
advance (published course schedule).

---

## Bottom line

- Phase 1: random-split student overlap **confirmed** (12–17% of test rows) → grouped
  evaluation adopted; measured cost ≤ ~2 points, more honest deployment estimate.
- Phase 2: survivorship bias **confirmed and severe** (31% of the still-enrolled population
  missing at cutoff 30; exclusions 47% Withdrawn) → v3 rebuilds prediction cases from the
  registered risk population.
- Phase 3: row/target integrity **clean** at all cutoffs — no fixes needed.
- Phase 4: all active features **SAFE** — no changes made.

Results of the combined fixes: `reports/baseline_v3.md`.
