# Baseline v3 — Registered Risk Population + Grouped Evaluation (official)

**Date:** 2026-07-02
**Notebook:** `notebook/OULAD_early_prediction_v1 (1).ipynb` (single notebook, additive v3
section; v1/v2 history preserved and still reproducing exactly)
**Builds on:** Baseline v2 (immutable) · Evidence: `reports/evaluation_validity_audit.md`
**Environment:** pinned (Python 3.12.6, scikit-learn 1.6.1, pandas 2.2.2, numpy 2.0.2,
xgboost 3.3.0), macOS/arm64.

---

## 1. What v3 changes — and what it deliberately does not

| Aspect | v2 | v3 | Type of change |
|---|---|---|---|
| Prediction cases | students with submitted coursework due ≤ cutoff (survivors) | **all registered pairs still enrolled at the cutoff** | methodological |
| Split | random rows, same student on both sides (12–17% of test rows) | **GroupShuffleSplit, group = `id_student`** (0 overlap, asserted in-notebook) | methodological |
| Missing info | students dropped | **kept**, with explicit `has_vle_activity`, `has_coursework` indicators + zero-filled features | methodological |
| Feature values | leakage-free (v2 tables) | identical v2 tables reused | none |
| Feature list | 17 | **19** (= 17 + 2 indicators) | pipeline (required by population change) |
| Models / hyperparameters / seed | LogReg·Tree·RF·XGB, seed 42 | **identical** | none |
| Robustness reporting | single split | + repeated grouped splits (seeds 0–4, RF/XGB, mean±std) | methodological |

No change was made for performance reasons; there are **no performance-motivated changes** in
v3 at all.

Membership rule (all facts observable on the cutoff day): keep every
(student, module, presentation) in `studentInfo` **except** pairs with
`date_unregistration ≤ CUTOFF` (withdrawal already observed — a known outcome, not a
prediction case) and pairs with `date_registration > CUTOFF` (not yet enrolled).

## 2. The population v3 actually predicts (per cutoff)

| Cutoff | prediction cases | vs v2 rows | Withdrawn | Fail | Pass | Distinction | note |
|-------:|------:|:---|-----:|-----:|------:|-----:|---|
| 14  | 28,061 | +26,873 (24×) | 5,673 | 7,027 | 12,339 | 3,022 | v2 kept only 4% of the risk population |
| 30  | 27,450 | +8,150 | 5,034 | 7,037 | 12,356 | 3,023 | |
| 60  | 26,353 | +2,942 | 3,931 | 7,040 | 12,358 | 3,024 | |
| 90  | 25,558 | +2,106 | 3,132 | 7,042 | 12,360 | 3,024 | |
| 140 | 24,289 | +811   | 1,860 | 7,044 | 12,361 | 3,024 | most withdrawals already observed |

The Withdrawn class **shrinks with later cutoffs by design**: withdrawals that already
happened are decided outcomes and leave the risk population — exactly as in deployment.
Consequence: **cross-cutoff numbers are different tasks**; compare experiments only against
the same cutoff's baseline.

## 3. Baseline v3 results (official protocol)

Headline: GroupShuffleSplit(group=`id_student`, test_size=0.2, seed 42), accuracy / macro-F1.

| Cutoff | LogReg | Tree | RF | XGB |
|-------:|:------|:------|:------|:------|
| 14  | 0.3512 / 0.3344 | 0.4782 / 0.2510 | 0.4626 / 0.2999 | **0.4810 / 0.3151** |
| 30  | 0.3831 / 0.3747 | 0.4817 / 0.2883 | 0.5048 / 0.3565 | **0.5099 / 0.3705** |
| 60  | 0.4368 / 0.4199 | 0.5383 / 0.3468 | 0.5591 / 0.3888 | **0.5614 / 0.4066** |
| 90  | 0.4391 / 0.4175 | 0.5713 / 0.3737 | 0.5874 / 0.4049 | **0.5947 / 0.4244** |
| 140 | 0.4995 / 0.4581 | 0.6238 / 0.4133 | 0.6530 / 0.4474 | **0.6627 / 0.4713** |

(macro-F1: LogReg leads at cutoffs 14–90 — its balanced class weights favour minority
classes; XGB leads accuracy everywhere and macro-F1 at 140.)

Robustness — repeated grouped splits (seeds 0–4), mean ± std:

| Cutoff | RF acc | RF macro-F1 | XGB acc | XGB macro-F1 |
|-------:|:---|:---|:---|:---|
| 14  | 0.4519 ± 0.0038 | 0.2913 ± 0.0010 | 0.4654 ± 0.0036 | 0.3030 ± 0.0028 |
| 30  | 0.5044 ± 0.0059 | 0.3602 ± 0.0063 | 0.5141 ± 0.0063 | 0.3804 ± 0.0055 |
| 60  | 0.5589 ± 0.0047 | 0.3930 ± 0.0041 | 0.5680 ± 0.0038 | 0.4233 ± 0.0040 |
| 90  | 0.5978 ± 0.0058 | 0.4158 ± 0.0080 | 0.6052 ± 0.0063 | 0.4392 ± 0.0073 |
| 140 | 0.6581 ± 0.0085 | 0.4497 ± 0.0066 | 0.6713 ± 0.0067 | 0.4743 ± 0.0053 |

Split noise is ~±0.005 acc: future experiments claiming gains below ~1–2 points on a single
split are not evidence.

## 4. v1 vs v2 vs v3 (XGBoost and RF, accuracy / macro-F1)

Reminder: v1→v2 is the **same population** (deltas attributable to the leakage fix);
v2→v3 changes **population + protocol** (deltas describe the honest task, not model quality).

| Cutoff | v1 XGB | v2 XGB | v2-grouped XGB* | v3 XGB | | v1 RF | v2 RF | v2-grouped RF* | v3 RF |
|-------:|:---|:---|:---|:---|---|:---|:---|:---|:---|
| 14  | .4370/.3234 | .4874/.3646 | .4748/.3524 | .4810/.3151 | | .4664/.2969 | .4832/.3003 | .4664/.2735 | .4626/.2999 |
| 30  | .5142/.3836 | .5189/.3948 | .4988/.3744 | .5099/.3705 | | .5062/.3619 | .5122/.3731 | .4981/.3537 | .5048/.3565 |
| 60  | .5509/.4421 | .5499/.4425 | .5535/.4462 | .5614/.4066 | | .5471/.4127 | .5447/.4161 | .5377/.4029 | .5591/.3888 |
| 90  | .5781/.4756 | .5760/.4735 | .5613/.4610 | .5947/.4244 | | .5675/.4476 | .5670/.4458 | .5570/.4418 | .5874/.4049 |
| 140 | .6220/.5407 | .6180/.5399 | .6190/.5349 | .6627/.4713 | | .6105/.5135 | .6078/.5143 | .6069/.5102 | .6530/.4474 |

\* v2-grouped = grouped split on the v2 survivor sample (isolates the split effect;
from `evaluation_validity_audit.md` §1.4).

Decomposition of the v2→v3 movement:
- **Grouping alone** (v2 → v2-grouped): −1 to −2 acc points at cutoffs 30/90, ~0 at 60/140 —
  the price of testing on genuinely unseen students.
- **Population alone** (v2-grouped → v3): accuracy **up** at later cutoffs (e.g. c140 XGB
  .6190 → .6627) but macro-F1 **down** (.5349 → .4713). Both are honest: the risk population
  has a higher majority share (Pass ≈ 51% at c140 since decided withdrawals left), and the
  remaining Withdrawn are the genuinely hard late-withdrawers (support 372 in the c140 test
  set). v2's higher macro-F1 was earned on an easier, survivor-filtered version of the task.

**No performance drop is hidden here:** minority-class (Withdrawn) detection is substantially
harder in v3 — that is the real difficulty of early prediction once survivorship is removed.

## 5. Why v3 is the more valid benchmark

1. Its test population is the deployment population: every still-enrolled student, including
   the 31% (c30) that v2 silently dropped — of whom 47% end up Withdrawn, the very students an
   early-warning system exists for.
2. Its test students are unseen at training time (grouped split), matching the deployment
   question and eliminating identity leakage.
3. Its features remain the leakage-free v2 features; missing information is an explicit,
   temporally-valid signal (`has_coursework` is itself a strong early indicator) rather than a
   silent row filter.
4. Its headline comes with a variance estimate (repeated grouped splits).

## 6. Consequences and limitations

- **v3 numbers are not comparable to v1/v2 numbers.** Different population, different split.
  The v1→v2→v3 lineage is preserved precisely so each effect is attributable.
- Class balance varies across cutoffs by design; compare within-cutoff only.
- Withdrawn support shrinks at late cutoffs (372 test rows at c140) — macro-F1 there is noisy;
  interpret with the ±std bands.
- Known documented limitations (unchanged models: XGB/Tree unweighted; non-stratified grouped
  split; legacy binary section) — see `methodology_review.md` §§1–5.
- v1/v2 sections in the notebook still run and **reproduce their committed numbers exactly**
  (verified at every cutoff in this run).

## 7. Artifacts

- Raw metrics incl. repeats + legacy-reproduction check: `reports/baseline_v3_results.json`
- Evidence for both fixes: `reports/evaluation_validity_audit.md`
- Full methodology pass: `reports/methodology_review.md`
- Immutable history: `reports/baseline_v1.md`, `reports/baseline_v2.md` (+ JSONs)
