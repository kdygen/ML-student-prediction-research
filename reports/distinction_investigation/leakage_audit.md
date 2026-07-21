# Leakage Audit — Distinction Investigation

**Scope:** every feature and procedure used in this investigation, against the
non-negotiable rules. Protocol constants: engaged population (29,496, excludes
unregistration ≤ day 0), per-student horizon h = min(unregistration, course_end),
censoring of all VLE/submission data to ≤ h, StratifiedGroupKFold(5, shuffle, rs=42)
grouped by `id_student`, zero-overlap asserted per fold.

## Base features (33)

The score-free set = the audited final 36 (Experiment 007 audit) minus the three
score-derived features (`rank_wa`, `score_slope_cw`, `score_std_cw`). All 33 previously
classified **Safe** in the publication audit; the audit trail (censored-recompute
verification: stored features match censored recomputation 100% / uncensored 0% on the 951
affected students) carries over unchanged.

## New features (22) — verdicts

| Group | Features | Verdict | Reasoning |
|---|---|---|---|
| H1 regularity | weekly_cv, inactive_week_share, max_week_streak, week_entropy, phase_concentration | **Safe** | Pure functions of censored click timestamps. `inactive_week_share` denominates by enrolled weeks = h/7 — h is the same censor boundary used everywhere; the *value* of h is not exposed beyond what every censored feature already reflects |
| H2 depth | unique_sites, revisit_ratio, top3_site_share, content_admin_ratio, enrichment_share, n_activity_types | **Safe** | Censored clicks joined to static `vle` metadata (activity_type is course-design metadata, fixed before the course) |
| H3 proactivity | early_work_share, first_active_day | **Safe** | Censored clicks + published deadlines (schedule facts, known in advance). Only non-exam deadlines used |
| H4 spacing | mean_gap, gap_std, weekly_slope, late_new_content_share | **Safe** | Censored clicks only; `late_new_content_share` uses h/2 as a midpoint — same censor-boundary consideration as H1 |
| H5 cohort-weekly | mean_weekly_z, share_topq_weeks, weekly_z_slope | **Safe** | Cohort statistics computed from peers' **censored behaviour** per (module, presentation, week); labels never touched. Identical justification to the accepted `rank_clicks` |
| H6 enrolment | studied_credits, num_of_prev_attempts | **Safe** | Registration-time records; validated in Experiment 005 |

## Grey area, stated openly

Features normalised by the student's own horizon (`inactive_week_share`, `late_new_content_share`,
and the pre-existing `study_spread`) use h in a denominator/midpoint. For withdrawn students
h *is* the unregistration date. This was examined in Experiment 006b/006c: such normalisation
does not constitute target leakage for the P-vs-D question (both classes have h = course_end
— **withdrawn students are irrelevant to the Pass/Distinction boundary**), and the 006c audit
showed the model's signal is redundantly encoded regardless. For the P-vs-D analysis
specifically, h is constant within the compared classes, so no horizon information can
separate them.

## Procedures

| Rule | Status |
|---|---|
| Grouped validation by `id_student` | ✅ every fold, asserted zero overlap |
| Same folds across all compared arms | ✅ fixed StratifiedGroupKFold(5, rs=42); baseline reproduced to Δ=0.000000 before any experiment |
| Tuning inside training folds only | ✅ thresholds (M2/M3/M5), ordinal cuts (M4) tuned on inner GroupKFold(3) OOF within each training fold |
| Class weights | fixed a priori (inverse frequency) — no tuning |
| No test-fold feature selection | ✅ feature groups defined a priori from hypotheses; group inclusion decided on the AUC metric with a pre-stated +0.002 rule |
| `date_unregistration` / `final_result` / exams / grades as features | ❌ never — verified programmatically for the base 33; new features audited above |
