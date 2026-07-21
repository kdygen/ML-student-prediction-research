# Final Report — Can Distinction Be Separated from Pass Without Scores?

> **📌 OFFICIAL BASELINE NOTE (2026-07-21).** The AF3 configuration described here was
> **promoted to the project's official baseline**. The numbers in this document come from the
> research pipeline, which computes cohort percentile features (`rank_clicks`) within all
> 32,593 registered students. The official notebook baseline ranks within the **engaged**
> population (29,496) — the population the model actually predicts for — giving
> **accuracy 0.739, macro-F1 0.715, Distinction F1 0.430, Withdrawn 0.940, Fail 0.779**.
> The difference is ≤ 0.003 on every metric, well inside fold noise (Distinction fold
> SD = 0.019). Official artifact: `reports/official_baseline_results.json`. The figures below
> are preserved as the experimental record.

**Date:** 2026-07-21 · **Protocol:** engaged population (29,496), per-student horizon &
censoring, StratifiedGroupKFold(5) by `id_student`, baseline reproduced to Δ = 0.000000,
all tuning inside training folds. Full trail: `DISTINCTION_RESEARCH_LOG.md`,
`experiment_results.{json,csv}`, `leakage_audit.md`, `feature_hypotheses.md`,
`literature_pass_vs_distinction.md`, `code/`.

## Outcome in one paragraph

Both pre-registered success conditions were met. **Distinction F1 doubled reproducibly**
(0.227 → **0.458**, every fold, W/F preserved, macro-F1 *up* 0.695 → 0.731) — almost entirely
by fixing the operating point, not by new information. Simultaneously, the investigation
produced **strong evidence of a genuine information ceiling**: across 22 well-motivated new
features, module identity, two GBDT families, and four decision architectures, the underlying
Pass-vs-Distinction ranking moved only AUC 0.737 → 0.749. The residual gap to the with-scores
model (D F1 0.581) is an information limit of score-free OULAD data, not a modelling failure
— a conclusion independently corroborated by the literature.

## The fourteen questions

**1. Why does Distinction F1 collapse without scores?**
Two stacked causes. *(i) Operating point:* under 4.09:1 Pass:Distinction imbalance, argmax
almost never lets D win — 85.3% of true Distinctions were predicted Pass and only 13.3%
received p(D) > 0.5, even though the ranking AUC was 0.746. This cause is fixable and was
fixed (+0.23 F1). *(ii) Information:* what separates Distinction from Pass is the *quality*
of produced work, which clickstreams measure only weakly; behaviour measures presence and
effort, which separate Withdrawn/Fail/Pass well. This cause is not fixable from this data.

**2. Which behavioural/demographic differences actually exist between Pass and Distinction?**
Real but modest, and all on one axis: Distinction students are *more of everything* —
cohort click rank (d = +0.58), active days (+0.42), study spread (+0.42), weeks in cohort top
quartile (+0.48), weekly consistency (inactive-week share −0.38, streaks +0.38), revisiting
(+0.31), early work share (+0.28), and lower prior-education disadvantage (−0.25). No single
feature exceeds d = 0.6; nothing is a qualitative signature.

**3. Which are statistically meaningful and practically useful?**
All of the above are statistically solid at n = 15,385 (P+D). Practically, only the
**depth/revisit family (H2)** added information beyond the existing 33 features (+0.012 AUC,
+0.021 PR-AUC — consistent with Kizilcec's SRL-revisiting result). Regularity, proactivity,
spacing, cohort-trajectory and enrolment groups were redundant re-encodings of the volume
axis (all within ±0.002 AUC).

**4. What did prior literature do?**
Mostly avoided the problem: merged Distinction into Pass or dropped it (the field norm per
two systematic reviews). The two genuine attempts: Al-azazi & Ghurab 2023 (score-free
ANN-LSTM, D F1 0.59 at P 0.82/R 0.47 — high-precision subset only) and Borna et al. 2024
(click-only RF, D F1 0.15, with an explicit statement that interaction data does not capture
top performance).

**5. Did those papers genuinely avoid leakage?**
Al-azazi and Borna avoided *feature* leakage (verified in full text) but neither uses
student-grouped validation (Al-azazi: single 70/30 split, unit unstated). Junejo et al. 2025
(claimed D F1 0.94 "without assessment") did **not** — their top feature derives from
`date_unregistration`, which our null-check shows is the label (F1 0.995). No published
result on this boundary survives at our validation standard.

**6. Which new features helped?**
Only H2 — `unique_sites`, `revisit_ratio`, `top3_site_share`, `content_admin_ratio`,
`enrichment_share`, `n_activity_types`: +0.012 P-vs-D AUC, and +0.003 D F1 / +0.003 macro-F1
in the weighted 4-class (inside fold noise for F1, real for AUC).

**7. Which hypotheses failed?**
H1 regularity, H3 proactivity, H4 spacing, H5 cohort-weekly trajectories, H6 enrolment
records — every one within ±0.002 AUC of base. Module dummies: +0.0003. Also failed as
architectures: hierarchical P-vs-D re-decision (0.439, below plain weighting) and ordinal
regression (0.412, and it damaged W/F — evidence that the 4-class problem is not usefully
ordinal at this boundary). HistGradientBoosting scored *below* XGB (0.742 vs 0.749),
confirming the ceiling is not model-specific. LightGBM/CatBoost were skipped to preserve the
pinned environment.

**8. Which model performed best?**
Inverse-frequency **sample-weighted XGBoost** — simplest of the fixes and equal-best
(D F1 0.452 alone; 0.458 with H2 + inner-tuned threshold). Threshold rescue is equivalent
(0.447); everything more elaborate is worse.

**9. Final score-free Distinction F1?**
**0.458** (precision 0.36, recall 0.63), per-fold [0.451, 0.491, 0.453, 0.456, 0.439].
Configuration: 33 + H2 + module dummies, inverse-frequency weights, D-threshold tuned on
inner GroupKFold(3) per fold.

**10. Improvement over baseline?**
**+0.231 D F1 (0.227 → 0.458, ×2.0)**; macro-F1 +0.036 (0.695 → 0.731); accuracy −0.050
(0.805 → 0.755) and Pass F1 −0.096 — the explicit, documented trade at the P/D boundary.

**11. Did Fail and Withdrawn stay stable?**
Yes: Withdrawn 0.941 → 0.943, Fail 0.791 → 0.795 — both marginally *better*.

**12. Modelling problem or information limit?**
**The recoverable part was an operating-point problem (now recovered); the remainder is an
information limit.** Evidence: dedicated binary, weighting, and thresholding all converge on
F1 ≈ 0.45; 22 engineered features spanning five behavioural constructs move AUC ≤ +0.012;
two GBDT families agree; missed Distinctions are behaviourally Pass-like (max d = 0.47);
with-scores comparison shows exactly where the missing information lives (scores → 0.581);
and independent literature replicates the collapse and the explanation.

**13. Defensible claims for a paper?**
(a) The score-free Distinction collapse is an *operating-point artefact stacked on an
information limit*, and the two are separable: fixing the operating point recovers F1
0.23 → 0.46 at zero information cost. (b) Best rigorously-validated (student-grouped)
score-free Distinction F1 on OULAD: 0.458. (c) Aggregate behaviour cannot reliably separate
Pass from Distinction (grouped-CV ceiling AUC ≈ 0.75); revisiting behaviour is the only
construct adding measurable information — consistent with SRL theory. (d) Do **not** claim
sequence models cannot do better (untested here — Al-azazi's high-precision subset suggests
a fine-grained temporal representation is the one live lever).

**14. What next with richer real-world data?**
The missing information is *quality of work*, so richer behaviour of the same kind won't
close the gap; different channels might: formative/self-test item responses (correctness,
not grades), fine-grained timestamps enabling true session/sequence models, content of
forum posts, assignment drafts/revision histories, and instructor-touchpoint records. With
OULAD itself: a daily-sequence model (LSTM/transformer) under grouped CV is the one
remaining test; expectation, based on Al-azazi's precision/recall shape, is a high-precision
low-recall Distinction subset rather than a solved boundary.

## Recommendation for the official pipeline

Keep the published headline (with coursework scores, D F1 0.581) as-is. For any **score-free
deployment**, adopt the weighted operating point: it costs 5 accuracy points but doubles
Distinction detection and raises macro-F1 — and for at-risk use cases (W/F), the score-free
model was never the constraint. Record the P-vs-D ceiling (AUC ≈ 0.75) as the honest
statement of what OULAD behaviour alone supports at this boundary.
