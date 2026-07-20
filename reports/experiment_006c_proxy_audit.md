# Experiment 006c — Proxy-Feature Leakage & Robustness Audit (addendum to 006b)

**Date:** 2026-07-20
**Trigger:** reviewer challenge to the two features that topped the 006b importance ranking —
`completion_ratio_cw` (suspected "fraction of the whole course completed" ≈ label proxy) and
`has_vle_activity` (suspected "withdrew early" proxy).
**Scope:** diagnosis + robustness only. Experiments 006 and 006b are unmodified.
**Artifacts:** `reports/experiment_006b_per_student_anchor/robustness_proxy_audit.json`,
`experiments/experiment_006c_robustness.py`.

---

## 1. Diagnosis — the challenge was correct

`completion_ratio_cw` as built in 006/006b is:

```python
n_cw_total = count of ALL non-exam assessments in the (module, presentation)   # whole course
completion_ratio_cw = submitted_count / n_cw_total
```

The denominator is the **entire course's** coursework, not the coursework a student had the
opportunity to attempt. The distortion is confined to withdrawers and is large:

| Class | ratio (whole-course denom.) | ratio (fair: deadline ≤ own horizon) | avg denominator total → available |
|---|--:|--:|--:|
| **Withdrawn** | **0.148** | **0.348** | **8.9 → 2.0** |
| Fail | 0.446 | 0.446 | 9.1 → 9.1 |
| Pass | 0.956 | 0.956 | 8.7 → 8.7 |
| Distinction | 0.968 | 0.968 | 8.7 → 8.7 |

A Week-3 withdrawer is charged for ~7 assessments that did not exist yet when they left.
Single-feature AUC against the Withdrawn label falls **0.916 → 0.795** under the fair
denominator: roughly that much of its power was *opportunity censoring*, not diligence.
Non-withdrawers are unaffected (identical denominators) — which is precisely what makes it a
Withdrawn-specific proxy rather than a neutral effort measure.

**Verdict:** not temporal leakage (everything is observable at course end, no post-outcome
information), but a **mis-specified feature** that conflates "did not do the available work"
with "was not present to do the work." The fair-denominator version is the defensible one.

`has_vle_activity` is **partly, but not simply,** a withdrew-early flag: where it is 0
(n = 3,554), 89.3% are Withdrawn and 2,536 never started (unregistered by day 0) — yet
**6,981 withdrawers do have activity**, and its standalone AUC is only 0.648. It isolates
the never-starter block rather than proxying withdrawal generally.

## 2. Robustness — four variants (XGB, identical grouped splits, all else fixed)

| Variant | n | Acc | Macro-F1 | Repeats (acc / F1) | Withdrawn F1 | Fail F1 | Top feature |
|---|--:|--:|--:|:--|--:|--:|---|
| **V0** 006b as published | 32,593 | 0.8452 | 0.7894 | 0.8497 / 0.7989 | 0.958 | 0.804 | has_vle_activity 0.23 |
| **V1** fair denominator | 32,593 | 0.8478 | **0.7928** | 0.8525 / 0.8013 | 0.960 | 0.809 | has_vle_activity 0.40 |
| **V2** V1 − never-starter flags | 32,593 | 0.8476 | 0.7921 | 0.8524 / 0.8012 | 0.962 | 0.809 | submitted_count 0.28 |
| **V3** V1, engaged population only | 29,496 | 0.8295 | 0.7901 | 0.8392 / 0.8000 | 0.936 | 0.803 | submitted_count 0.23 |

**Findings.**

1. **The flagged proxy was not propping up the result.** Correcting the denominator changes
   macro-F1 by **+0.003** (0.7894 → 0.7928) — it slightly *helps*. Removing the
   never-starter flags entirely changes it by **−0.001**. The 006b conclusion is unaffected.
2. **Dropping the trivially-easy never-starters costs accuracy, not macro-F1.** V3 loses
   1.8 pp accuracy (those 3,097 rows are ~free) but macro-F1 holds at 0.790 and Withdrawn F1
   remains 0.936 — the class is still highly separable among students who actually engaged.
3. **The signal is redundantly encoded, which is the deeper point.** Remove the flags and
   `submitted_count` (0.28) and `decay_clicks` (0.10) simply absorb the role. This is not a
   feature-engineering defect: **at course end, the Withdrawn class *is* defined by ceasing
   participation**, so every participation feature will proxy it to some degree. No feature
   set can remove that, which is why the standing interpretive caveat holds — course-end
   Withdrawn detection is *description*, not forecasting.

## 3. Recommendations adopted

- **Use V1 (fair denominator) as the corrected specification** for any future full-course
  work: `completion_ratio_avail = submitted_count / (coursework with deadline ≤ own horizon)`.
  It is strictly better specified and marginally better performing.
- **Report V3 (0.830 acc / 0.790 macro-F1, engaged population, fair denominator) as the
  conservative headline ceiling** when the claim must survive the strongest objection; V1
  (0.848 / 0.793) when the full registered cohort is the intended population.
- **Do not treat any variant's Withdrawn metric as early-warning capability** — unchanged
  from the 006/006b caveat.
- 006 and 006b remain published as-run; this audit is additive.
