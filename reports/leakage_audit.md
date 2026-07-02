# Leakage Audit — OULAD Early Prediction (Baseline v1 → v2)

**Date:** 2026-07-02
**Scope:** every active model feature (the 17 columns in `X = mlData[base_features + edu_features]`).
**Method:** trace each feature to its raw OULAD columns and cutoff filter, then judge — at
each cutoff (14/30/60/90/140) — whether it uses only information available on/before the
cutoff. Claims are backed by the notebook cells and by empirical measurement on the data.
**Environment for measurements:** pinned Colab stack (Python 3.12, pandas 2.2.2, numpy 2.0.2).

---

## 1. OULAD data-flow primer (what "available at cutoff" means)

Relevant raw columns:

- `studentVle.date` — the **day a click happened** (relative to course start). A click with
  `date <= C` genuinely occurred on/before day `C`. **Safe to use up to the cutoff.**
- `assessments.date` — the **due date (deadline)** of an assessment. This is part of the
  published course schedule, **known in advance**. Knowing a deadline is not leakage.
- `studentAssessment.date_submitted` — the **day a student submitted** an assessment. This is
  the earliest day the corresponding **score** exists / could be known.
- `studentAssessment.score` — the mark. Only meaningful **after** submission (`date_submitted`).

**Key rule:** a *score* (or the fact a piece of work was completed) is available at cutoff `C`
**iff `date_submitted <= C`** — not iff the deadline `date <= C`. Filtering scores by the
**due date** is leakage in two directions:

1. **Leaks the future:** includes work due before `C` but **submitted after `C`** (late
   submissions) — the score did not exist at `C`.
2. **Discards the present:** excludes work due after `C` but **submitted on/before `C`**
   (early submissions) — the score *did* exist at `C` but is thrown away.

## 2. Empirical evidence for the due-date-filter leakage

Measured on the v1 pipeline (mirrors the notebook), overwriting only the filter predicate:

**Cutoff 30 (coursework, `assessment_type != "Exam"`):**
- v1 coursework rows (due ≤ 30): **22,014**
- of these, `date_submitted > 30` → **473 (2.1%) leaked in** (score not available at day 30)
- of these, `date_submitted` is NaN (never submitted): **0**
- submissions with `date_submitted ≤ 30` but `due > 30` → **4,962 discarded by v1** (available
  early, wrongly excluded)
- Example leaked-in rows (module AAA/2013J, due day 19, submitted days 32–54, scores 45–85).

**Per-cutoff feature-value impact** (rows = students in `mlData`; a change = v1 value differs
from the leakage-free value):

| Cutoff | rows | `weighted_average` changed | of which → 0 in v2 | `clicks_per_assessment` changed | `recovery_slope` changed |
|-------:|-----:|-----:|-----:|-----:|-----:|
| 14  | 1188  | 133  | 100 | 106  | 31   |
| 30  | 19300 | 1419 | 428 | 3098 | 2642 |
| 60  | 23411 | 1158 | 115 | 4206 | 4138 |
| 90  | 23452 | 2394 | 20  | 5344 | 5372 |
| 140 | 23478 | 799  | 2   | 4342 | 4345 |

Reading: at cutoff 30, 428 students had a **non-zero** v1 `weighted_average` that becomes 0
under the correct filter — i.e. their *entire* score signal came from work submitted after the
cutoff (pure leakage). The larger `clicks_per_assessment` / `recovery_slope` change counts are
driven mostly by the **early submissions v1 discarded** (the 4,962 above), which the fix
restores. The magnitude of the `weighted_average` leak shrinks at later cutoffs (by day 140
almost everything due is already submitted), but the discarded-early-signal effect is present
at every cutoff.

---

## 3. Feature-by-feature audit (17 active features)

Legend: **SAFE** = uses only ≤cutoff info; **CONFIRMED LEAKAGE** = provably uses future info.

### 3.1 VLE / behavioural features — all SAFE

All derive from `studentVleEarly = studentVle[studentVle["date"] <= CUTOFF]`, i.e. clicks that
happened on/before the cutoff day. `studentVle.date` is the activity day, so the filter is
correct. `vle` (activity-type metadata) is static.

| Feature | Computation | Raw deps | Class |
|---|---|---|---|
| `active_days` | count of distinct active days per student in `studentVleEarly` (cells #19–#20) | `studentVle.date`, `sum_click` | **SAFE** |
| `clicks_per_day` | `sum_click / (active_days + 1)` (#22) | `studentVle.sum_click`, `date` | **SAFE** |
| `study_spread` | `active_days / CUTOFF`, clipped [0,1] (#44) | `active_days`, `CUTOFF` | **SAFE** |
| `burstiness` | `std/(mean+1)` of per-day clicks in `studentVleEarly` (#48) | `studentVle.sum_click`, `date` | **SAFE** |
| `resource_ratio` | `resource` clicks / total activity clicks (≤cutoff) (#50–#56) | `studentVleEarly ⋈ vle.activity_type` | **SAFE** |
| `oucontent_ratio` | as above for `oucontent` | same | **SAFE** |
| `homepage_ratio` | as above for `homepage` | same | **SAFE** |
| `forumng_ratio` | as above for `forumng` | same | **SAFE** |
| `quiz_ratio` | as above for `quiz` | same | **SAFE** |

### 3.2 `assessment_focus` — SAFE

Computed (#88) from `studentVleEarly` clicks in the 7-day window **before each assessment
deadline** that is `<= CUTOFF`, divided by total clicks (≤cutoff). It uses **deadline days**
(`assessments.date`, known schedule) and **clicks** (≤cutoff) only — no scores, no submission
timing. The deadline set is taken from `courseworkData` (deadlines ≤ cutoff that appear in the
data); this is known-in-advance schedule information. → **SAFE.** (Unchanged in v2.)

### 3.3 Demographic education features — all SAFE

`highest_education_*` (4 one-hot columns, #59–#63) come from `studentInfo.highest_education`,
recorded at **registration**, before the course starts. → **SAFE.**

| Feature | Raw dep | Class |
|---|---|---|
| `highest_education_HE Qualification` | `studentInfo.highest_education` | **SAFE** |
| `highest_education_Lower Than A Level` | same | **SAFE** |
| `highest_education_No Formal quals` | same | **SAFE** |
| `highest_education_Post Graduate Qualification` | same | **SAFE** |

### 3.4 Assessment-derived features — CONFIRMED LEAKAGE (fixed in v2)

All three filter assessments by **due date** (`assessments.date <= CUTOFF`) rather than
submission date, so they use scores/counts not available at the cutoff (§1–§2).

| Feature | Computation (v1) | Raw deps | Class | Why |
|---|---|---|---|---|
| `weighted_average` | `courseworkData = finalData[(type!=Exam) & (date<=CUTOFF)]`; `Σ(score·weight/100)/Σweight` per student (#17–#18) | `assessments.date` (due), `score`, `weight` | **CONFIRMED** | Uses `score` of coursework filtered by **due date**; includes late-submitted scores, drops early-submitted ones. 428 students at cutoff 30 have an all-leaked value. |
| `clicks_per_assessment` | `sum_click / (assessment_count + 1)`, where `assessment_count = #coursework with due<=CUTOFF` (#26, #36) | `assessments.date` (due), `studentVle.sum_click` | **CONFIRMED** | `sum_click` is safe, but `assessment_count` counts submissions by **due date** → same leakage; changes for 3,098 students at cutoff 30. |
| `recovery_slope` | mean consecutive `score` change over `studentAssessmentEarly` (due<=CUTOFF), ordered by due date (#100) | `assessments.date` (due), `score` | **CONFIRMED** | Uses `score` sequence filtered by **due date**; changes for 2,642 students at cutoff 30. |

---

## 4. Redesign (Baseline v2)

**Single change, applied identically to all three features:** replace the filter predicate
`date <= CUTOFF` (due date) with **`date_submitted <= CUTOFF`** (submission date). Everything
else — grouping keys, weighting formula, ordering, `fillna(0)`, `mlData` row membership, the
train/test split, and the models — is **unchanged**. Implemented as an additive notebook
section ("Baseline v2 — Leakage-Free Assessment Features") inserted before the split; the
original v1 cells are preserved as history and then overwritten in `mlData`.

- `weighted_average` (v2): `courseworkData_v2 = finalData[(type!=Exam) & (date_submitted<=CUTOFF)]`,
  then the same weighted mean; missing → 0.
- `clicks_per_assessment` (v2): `assessment_count_v2 = #coursework with date_submitted<=CUTOFF`,
  then `sum_click/(assessment_count_v2+1)`.
- `recovery_slope` (v2): same consecutive-score-change mean over submissions with
  `date_submitted<=CUTOFF`, ordered by deadline; missing → 0.

**Assumption stated:** OULAD provides no grading date, so `date_submitted` is used as the proxy
for "score is available." This is the standard, and it is conservative (a score cannot be known
before it is submitted).

**Why not remove the features:** they carry real early-signal once filtered correctly (and the
fix actually *adds* legitimately-available early submissions v1 discarded). Removal would
destroy predictive information without justification.

## 5. Known residual (out of scope for v2, flagged for v3)

**Sample-membership / survivorship leakage — CONFIRMED, pipeline-level (not a feature).**
`mlData` is seeded from `courseworkPerformance`, i.e. only (student, module, presentation)
rows that have **submitted** weighted coursework with due date ≤ cutoff. This means:

1. Students who **never submit any coursework** are excluded entirely — a selection defined by
   future behaviour (correlated with not-withdrawing).
2. The seed still uses the **due-date** filter for membership.

This was **not** changed in v2 because it alters the sample (row count, class balance) and would
break the clean feature-by-feature v1↔v2 comparison; it is a pipeline redesign, not a feature
redesign. v2 therefore keeps identical membership to v1 (same rows, same split) and removes only
the **feature-value** leakage. Making membership itself leakage-free (e.g. seeding from all
registered students via `studentInfo`/`studentRegistration`, then left-joining features) is the
recommended **Baseline v3** step.

## 6. Summary

- **SAFE (14):** `active_days`, `clicks_per_day`, `study_spread`, `burstiness`,
  `resource_ratio`, `oucontent_ratio`, `homepage_ratio`, `forumng_ratio`, `quiz_ratio`,
  `assessment_focus`, and the 4 `highest_education_*`.
- **CONFIRMED LEAKAGE, fixed in v2 (3):** `weighted_average`, `clicks_per_assessment`,
  `recovery_slope` — all via due-date filtering; fixed by switching to `date_submitted<=CUTOFF`.
- **CONFIRMED, deferred to v3 (1, pipeline-level):** sample membership / survivorship.

Result and metric comparison: see `reports/baseline_v2.md`.
