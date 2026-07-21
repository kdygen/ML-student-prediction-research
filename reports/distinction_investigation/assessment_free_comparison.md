# Assessment-Free Comparison with Al-azazi & Ghurab (2023)

> **📌 OFFICIAL BASELINE NOTE (2026-07-21).** The AF3 configuration described here was
> **promoted to the project's official baseline**. The numbers in this document come from the
> research pipeline, which computes cohort percentile features (`rank_clicks`) within all
> 32,593 registered students. The official notebook baseline ranks within the **engaged**
> population (29,496) — the population the model actually predicts for — giving
> **accuracy 0.739, macro-F1 0.715, Distinction F1 0.430, Withdrawn 0.940, Fail 0.779**.
> The difference is ≤ 0.003 on every metric, well inside fold noise (Distinction fold
> SD = 0.019). Official artifact: `reports/official_baseline_results.json`. The figures below
> are preserved as the experimental record.

**Why this document exists.** The "score-free" 33-feature set removes coursework *scores* but
**retains 9 assessment-derived features** — including `submitted_count`, the single most
important feature in the winning model (18.5% of gain). Al-azazi & Ghurab use demographics +
clickstream **only**, with no access to the assessment tables at all. Comparing our 0.458 to
their 0.59 would therefore have been invalid. This document reports the genuinely
assessment-free replication.

## What was removed

The 9 assessment-derived features (all use `studentAssessment` / `assessments`):
`submitted_count`, `mean_submit_lead`, `min_submit_lead`, `late_submissions`,
`first_submit_day`, `n_assess_types_submitted`, `completion_ratio_avail`,
`clicks_per_assessment`, `assessment_focus`.

Leaving **24 features** = engagement volume/rhythm/recency, activity-type click shares,
cohort click rank, registration lead, prior-education dummies. The 6 H2 depth features are
clickstream/`vle`-derived and therefore retained as assessment-free. An in-code assertion
verifies no assessment or score feature enters any assessment-free arm.

## Results (identical protocol: engaged 29,496, StratifiedGroupKFold-5 by student)

| Arm | Feats | Accuracy | Macro-F1 | **Distinction F1** | D P/R | W / F / P |
|---|--:|--:|--:|--:|:--|:--|
| AF0 assessment-free, argmax | 24 | 0.7956 | 0.6748 ± 0.0042 | **0.172** | 0.52 / 0.10 | 0.938 / 0.775 / 0.814 |
| AF1 + class weights | 24 | 0.7408 | 0.7110 ± 0.0030 | **0.416** | 0.33 / 0.55 | 0.937 / 0.772 / 0.719 |
| AF2 + H2 depth | 30 | 0.7476 | 0.7170 ± 0.0063 | **0.427** | 0.35 / 0.56 | 0.938 / 0.776 / 0.727 |
| **AF3 + module + tuned τ** | **36** | 0.7388 | 0.7156 ± 0.0066 | **0.433** | 0.33 / 0.62 | 0.941 / 0.781 / 0.708 |
| *(reference)* with-assessment winner | 45 | 0.7554 | 0.7306 ± 0.0066 | *0.458* | 0.36 / 0.63 | 0.943 / 0.795 / 0.726 |

AF3 per-fold Distinction F1: **0.431, 0.464, 0.441, 0.422, 0.406** (std 0.019) — stable.
Tuned thresholds τ: 0.46, 0.44, 0.42, 0.44, 0.42.

**P-vs-D ranking (threshold-free information measure):**

| Feature set | AUC | PR-AUC |
|---|--:|--:|
| Assessment-free 24 | 0.7079 ± 0.0097 | 0.3741 |
| Assessment-free 24 + H2 | **0.7228 ± 0.0099** | 0.4024 |
| With-assessment 33 + H2 | 0.7488 ± 0.0098 | 0.4369 |

Assessment submission behaviour is worth **+0.026 AUC** on the Pass/Distinction boundary —
real, but small. The H2 depth features are worth +0.015 within the assessment-free set,
slightly *more* than in the with-assessment set (+0.012), consistent with them being the only
non-volume construct available once submission metadata is gone.

## Head-to-head with Al-azazi & Ghurab

*Their setup: OULAD, 4-class, day 270 (full course), demographics + daily clickstream,
ANN-LSTM, 70/30 holdout, unit unstated, **no student grouping**.*

| Metric | Al-azazi & Ghurab | **Ours (AF3, assessment-free)** | Difference |
|---|--:|--:|--:|
| Distinction precision | **0.82** | 0.33 | −0.49 |
| Distinction recall | 0.47 | **0.62** | +0.15 |
| **Distinction F1** | **0.59** | **0.433** | **−0.157** |
| Overall accuracy | 0.72 | **0.739** | +0.019 |
| Macro-F1 | 0.66 | **0.716** | +0.056 |
| Withdrawn F1 | 0.70 | **0.941** | +0.241 |
| Fail F1 | 0.65 | **0.781** | +0.131 |
| Pass F1 | 0.70 | 0.708 | +0.008 |

**On the like-for-like feature access, they beat us on Distinction F1 (0.59 vs 0.433) and we
beat them on everything else** (macro-F1 +0.056, Withdrawn +0.241, Fail +0.131), under
stricter validation (student-grouped 5-fold vs a single ungrouped split).

The shape of their advantage is diagnostic: **precision 0.82 / recall 0.47**. They identify a
small, confidently-separable subset of Distinction students and miss the majority. We do the
opposite — recall 0.62, precision 0.33. Two plausible explanations, which we cannot separate
from published information:

1. **Representation.** Their day-wise cumulative clickstream sequences fed to an LSTM may
   capture temporal shape that our aggregate features discard. This is the substantive
   hypothesis, and it is untested here.
2. **Validation.** Their single 70/20 ungrouped split permits the repeat-enrolment leakage we
   measured at 15.1% of test rows in our own data. High precision on a minority class is
   exactly what identity leakage produces.

We report both and assume neither.

## What can now be claimed

✅ **"Under genuinely assessment-free feature access (demographics + clickstream only),
our student-grouped model reaches Distinction F1 0.433, macro-F1 0.716, accuracy 0.739."**

✅ **"Fixing the operating point is the dominant intervention in the assessment-free regime
too: Distinction F1 rises 0.172 → 0.433 (2.5×) with no new information."**

✅ **"Assessment submission behaviour — timing, counts, completion, independent of any score —
is worth +0.026 P-vs-D AUC and +0.025 Distinction F1."** A distinct, quantified finding: it is
not only grades that carry Distinction signal, but grades carry most of it (scores were worth
+0.123 D F1: 0.458 → 0.581).

✅ **"We outperform the closest published assessment-free comparator on macro-F1 (0.716 vs
0.66), Withdrawn (+0.24), Fail (+0.13) and accuracy, under stricter validation."**

✅ **"The information ceiling holds in the assessment-free regime: P-vs-D AUC 0.708 → 0.723
across all engineered features."**

## What must NOT be claimed

❌ **"We beat Al-azazi & Ghurab on Distinction."** We do not — 0.433 vs 0.59. State it plainly.

❌ **"Our score-free model is assessment-free."** The 33-feature headline uses 9
assessment-derived features. Use "score-free" and "assessment-free" as distinct terms and
report the right number for each: **0.458 score-free, 0.433 assessment-free.**

❌ **"Aggregate features are sufficient for Distinction."** Al-azazi's precision 0.82 is
evidence a sequence representation may capture something ours does not, even if their
validation is weaker.

❌ **"Their result is invalid because of leakage."** Their features are genuinely
assessment-free (verified in full text). Their *validation* is weaker, which is a caveat, not
a refutation. Junejo et al. is the invalid result; Al-azazi is not.

❌ Any claim that the Distinction ceiling is proven absolutely — it is demonstrated for
**aggregate** representations under grouped validation. Sequence models remain untested.

## Honest summary line for a paper

> Under matched assessment-free feature access, our student-grouped model achieves macro-F1
> 0.716 versus 0.66 for the closest published comparator, with substantially better Withdrawn
> and Fail detection, but lower Distinction F1 (0.433 vs 0.59). The comparator attains higher
> Distinction F1 through high precision on a small identified subset, using a day-wise
> sequence representation under a single ungrouped split; whether that advantage is
> representational or an artefact of validation cannot be determined from published
> information, and is the primary open question this work leaves.
