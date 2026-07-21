# OULAD Student Outcome Prediction — Leakage-Free Early & Full-Course Modelling

A reproducible research pipeline for predicting student outcomes (Withdrawn / Fail / Pass /
Distinction) from the **Open University Learning Analytics Dataset (OULAD)**, built around one
principle: **prevent data leakage above all else, then measure honestly.**

The project spans two questions:

1. **Early prediction (forecasting)** — how accurately can we predict outcomes at fixed points
   during a course (days 14/30/60/90/140), using only information available at each cutoff?
2. **Full-course prediction (ceiling)** — how accurately can outcomes be predicted using *all*
   behavioural data from the course, while remaining rigorously leakage-free?

---

## Headline results

**⭐ OFFICIAL BASELINE — assessment-free pipeline** (StratifiedGroupKFold-5 grouped by
student, 29,496 enrolments, 36 features, **no data from `studentAssessment.csv` or
`assessments.csv`**):

| Metric | Value |
|---|--:|
| Accuracy | **0.739 ± 0.004** |
| Macro-F1 | **0.715 ± 0.005** |
| Weighted-F1 | 0.753 |
| Per-class F1 | Withdrawn **0.940** · Fail **0.779** · Pass 0.709 · Distinction **0.430** |

**Documented comparison arms** (same folds and population, not the baseline):

| Arm | Features | Accuracy | Macro-F1 | Distinction F1 |
|---|--:|--:|--:|--:|
| ⭐ **OFFICIAL — assessment-free** | 36 | **0.739** | **0.715** | **0.430** |
| + assessment *behaviour* (no scores) | 45 | 0.753 | 0.729 | 0.457 |
| + coursework *scores* (argmax) | 36 | 0.836 | 0.795 | 0.579 |
| Regression (coursework score, behaviour-only) | 33 | R² 0.334 | MAE 10.30 | — |

Assessment behaviour is worth ≈ +0.014 macro-F1; coursework scores a further ≈ +0.066.

**Early prediction** (XGBoost, accuracy / macro-F1): day 30 → 0.527 / 0.396; day 140 → 0.693 /
0.514. The gap between the day-140 forecast (macro-F1 0.51) and the full-course with-scores
ceiling (0.795) quantifies how much outcome information does not yet exist during the
intervention window.

Note the *with-scores arm* uses coursework performance (no exam data); its feature access is
closest to Althibyani (2024) and Shou et al. (2024), who also use assessment scores — it
exceeds their macro-F1 by 0.09 and 0.12 under stricter student-grouped validation. **That arm
is no longer the baseline**, because it depends on assessment records a deploying institution
may not have.

Against the closest published **assessment-free** work (Al-azazi & Ghurab 2023 —
demographics + clickstream only), our official baseline leads on **macro-F1 (0.715 vs 0.66)**,
**Withdrawn (0.940 vs 0.70)** and **Fail (0.779 vs 0.65)** under stricter student-grouped
validation, but **they lead on Distinction (0.59 vs 0.430)**. See
`reports/experiment_009_feature_access.md` and
`reports/distinction_investigation/assessment_free_comparison.md`.

---

## Key methodological decisions

| Decision | Why |
|---|---|
| **Grouped cross-validation** (by `id_student`) | 32,593 enrolments cover only 28,785 students; a random split leaked 15.1% of test rows onto students also in training |
| **Exams excluded** (scores, submissions, attendance) | Exams occur at/after the prediction point and determine the label; only 1 of 10,156 withdrawn students has any exam record |
| **`date_unregistration` used only as a censor** | It *is* the Withdrawn label — a one-line null-check gives F1 0.9950 with no model |
| **Per-student horizon** (`h = min(unreg, course_end)`) | Temporal features anchored to each student's own endpoint, not the global course end |
| **Fair completion denominator** | Counts only coursework whose deadline fell before `h`, so early leavers are not charged for assessments that did not yet exist |
| **Survivorship correction** | The original population silently excluded 31% of the still-enrolled risk pool at day 30 |
| **Pinned environment** | RandomForest is not reproducible across scikit-learn versions |

Full leakage audit and per-feature classification: [`reports/experiment_007_publication_audit.md`](reports/experiment_007_publication_audit.md).

---

## Repository structure

```
notebook/    OULAD_early_prediction_v1 (1).ipynb   ← the single reproducible artifact
experiments/ Archived Python drivers (experiment_001 … 009)   research history
reports/     Markdown reports + JSON metrics + figures for every experiment
             official_baseline_results.json  ← ⭐ the official baseline artifact
             distinction_investigation/      Pass-vs-Distinction research loop
data/raw/    OULAD CSVs (gitignored — see "Data" below)
data/processed/  p3/, p4/  cached datasets (parquet gitignored; manifests tracked)
RESEARCH_LOG.md   Chronological log of every experiment (newest first)
OBJECTIVE.md      Research goals and optimisation priorities
AGENTS.md         Working conventions for this repository
RESULTS.md        All headline numbers in one reference
AT_RISK_MODEL_BRIEFING.md   Business-facing briefing (intervention targeting)
```

### The notebook is the single reproducible artifact
Running `notebook/OULAD_early_prediction_v1 (1).ipynb` from start to finish reproduces:
1. **Baselines v1–v4** — the early-prediction pipeline across all cutoffs (immutable history);
2. **Full-course methodology** — the with-scores comparison arm (36 features);
3. **Supplementary regression** — final coursework score from behaviour only (33 features);
4. **⭐ OFFICIAL BASELINE** — the assessment-free pipeline plus the assessment-behaviour arm.

Development experiments (006–008) are **not** in the notebook — they remain archived under
`experiments/` and `reports/` as research history.

---

## The experiments (research arc)

| # | Report | What it established |
|---|---|---|
| Baseline v1–v4 | `reports/baseline_v*.md` | Early-prediction pipeline; fixed temporal leakage (v2), survivorship + grouped evaluation (v3), verified 35-feature set (v4) |
| Exp 001–002 | `reports/experiment_001*`, `_002*` | Macro-F1 optimisation; automated feature search |
| Exp 003–004 | `reports/experiment_003*`, `_004*` | Intervention index and timing framework (evaluation-layer) |
| Exp 005 | `reports/experiment_demographic_features.md` | Demographic feature families (v5 candidate) |
| **Exp 006** | `reports/experiment_006_full_course.md` | **Full-course leakage-free ceiling** |
| Exp 006b | `reports/experiment_006b_per_student_anchor.md` | Per-student horizon (sensitivity) |
| Exp 006c | `reports/experiment_006c_proxy_audit.md` | Proxy-feature leakage audit + fair denominator |
| Exp 007 | `reports/experiment_007_publication_audit.md` | Publication audit (leakage, robustness, CV) |
| Exp 008 | `reports/experiment_008_parsimony.md` | Feature redundancy → final 36-feature set |
| **Exp 009** | `reports/experiment_009_feature_access.md` | **Feature-access ablation → the assessment-free baseline** |
| Distinction | `reports/distinction_investigation/` | Pass-vs-Distinction research loop; operating-point fix; information ceiling |
| Literature | `reports/literature_comparison.md` | Comparison vs ~25 published OULAD papers |

Every report ships with a JSON metrics file; figures are under `reports/*/figures/`.

---

## Data

The six OULAD CSVs are **not committed** (`.gitignore` excludes `data/raw/**/*.csv`).
Download them from the official source and place them in `data/raw/`:

- Kuzilek, J., Hlosta, M. & Zdrahal, Z. *Open University Learning Analytics dataset.*
  *Scientific Data* 4, 170171 (2017). https://doi.org/10.1038/sdata.2017.171
- Dataset: https://analyse.kmi.open.ac.uk/open_dataset

Required files: `studentInfo.csv`, `studentRegistration.csv`, `studentVle.csv`, `vle.csv`,
`studentAssessment.csv`, `assessments.csv`.

Processed parquet caches (`data/processed/p3`, `p4`) are gitignored; their `manifest.json`
files (raw-input hashes, environment, schema, git commit) are tracked for provenance.

---

## Environment (pinned)

RandomForest metrics are not reproducible across scikit-learn versions, so the stack is pinned:

```
Python 3.12.6
scikit-learn 1.6.1
pandas 2.2.2
numpy 2.0.2
xgboost 3.3.0
```

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install scikit-learn==1.6.1 pandas==2.2.2 numpy==2.0.2 xgboost==3.3.0 pyarrow matplotlib scipy
```

The notebook reads CSVs by bare filename, so run it with the working directory set to
`data/raw/` (or adjust the load paths).

---

## What is and is not claimed

**Claimed:** first student-grouped evaluation of OULAD; the `date_unregistration` label-leak
finding; the best *defensible* 4-class macro-F1 under strict feature/validation constraints;
quantified corrections for temporal, survivorship, and identity leakage.

**Not claimed:** unqualified state of the art (a leaked 0.98 exists in the literature and is
rebutted, not beaten); superiority over deep-learning architectures (untested under this
protocol); that full-course Withdrawn detection is *forecasting* (at course end it is
descriptive); any causal or intervention benefit (OULAD has no intervention arm).

See [`MEETING_PREP_GUIDE.md`](MEETING_PREP_GUIDE.md) for the full discussion, results tables,
per-feature explanations, and anticipated questions.

---

## Citation

If you use OULAD, cite the dataset paper (Kuzilek et al. 2017, above). This repository is a
research project; see `RESEARCH_LOG.md` for the full experimental record.
