# Feature Hypotheses — Pass vs Distinction Separation (score-free)

Every feature is traced to its raw source and its availability time. All are computed on
data censored at each student's horizon **h = min(date_unregistration, course_end)**;
`date_unregistration` is used only as the censor boundary. Deadlines are published schedule
facts. Cohort statistics use peers' *behaviour* only, never labels. Demographics are
enrolment-time records.

## H1 — Regularity ("Distinction students study consistently, Pass students in bursts")

| Feature | Definition | Raw source | Available when |
|---|---|---|---|
| `weekly_cv` | std/mean of weekly clicks | studentVle ≤ h | any time ≥ data seen |
| `inactive_week_share` | share of enrolled weeks with zero activity | studentVle ≤ h | same |
| `max_week_streak` | longest run of consecutive active weeks | studentVle ≤ h | same |
| `week_entropy` | normalised entropy of clicks across weeks (low = cramming) | studentVle ≤ h | same |
| `phase_concentration` | max share of clicks on one day-of-7-cycle phase | studentVle ≤ h | same |

*Rationale:* self-regulated-learning literature associates distributed practice with high
achievement. Timestamps are day-granularity only, so time-of-day features are impossible;
the mod-7 phase is the honest weekday proxy (calendar weekday labels are unknown).

## H2 — Depth vs volume ("Distinctions revisit and go deeper, not just more")

| Feature | Definition | Raw source |
|---|---|---|
| `unique_sites` | distinct VLE resources touched | studentVle+vle ≤ h |
| `revisit_ratio` | mean distinct days per visited site | same |
| `top3_site_share` | click concentration on top-3 sites | same |
| `content_admin_ratio` | learning-content clicks ÷ navigation clicks | same |
| `enrichment_share` | share of clicks on optional/enrichment types (glossary, ouwiki, dataplus, htmlactivity, collaborate…) | same |
| `n_activity_types` | breadth across the 20 activity types | same |

## H3 — Proactivity ("Distinctions work ahead of deadlines")

| Feature | Definition | Raw source |
|---|---|---|
| `early_work_share` | clicks 8–21 days before coursework deadlines ÷ clicks 0–21 days before | studentVle ≤ h + published deadlines |
| `first_active_day` | first day of any VLE activity (incl. negative = pre-course) | studentVle ≤ h |

## H4 — Spacing & trajectory

| Feature | Definition |
|---|---|
| `mean_gap`, `gap_std` | statistics of gaps between active days |
| `weekly_slope` | OLS slope of weekly clicks (rising/falling engagement) |
| `late_new_content_share` | share of resources first opened in the second half of the student's own span (continued exploration vs early-only coverage) |

## H5 — Cohort-relative weekly behaviour

| Feature | Definition |
|---|---|
| `mean_weekly_z` | mean weekly z-score of clicks vs (module, presentation, week) cohort |
| `share_topq_weeks` | share of weeks in the cohort's top activity quartile |
| `weekly_z_slope` | trend of cohort-relative position over the course |

*Leakage note:* cohort mean/std/quantile per week are computed from all students' censored
behaviour in that presentation-week — identical justification to the accepted `rank_clicks`.

## H6 — Enrolment records

`studied_credits`, `num_of_prev_attempts` — fixed at registration (validated leakage-free in
Experiment 005, where they passed cross-model promotion tests).

## Explicitly rejected as unavailable or unsafe

- Time-of-day patterns — OULAD has no intra-day timestamps.
- Weekday/weekend labels — day 0 anchors to presentation start, calendar weekday unknown;
  only the mod-7 *cycle* is usable.
- Resource-to-quiz transition sequences — no intra-day ordering exists.
- Post-assessment recovery around *scores* — would require assessment results; only
  click-timing relative to deadlines is used.
- Anything derived from `date_unregistration`, `final_result`, exams, or grades.
