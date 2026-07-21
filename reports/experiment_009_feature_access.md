# Experiment 009 — Feature-Access Ablation and Like-for-Like Comparison with Al-azazi & Ghurab (2023)

**Date:** 2026-07-21
**Motive:** the literature comparison described Al-azazi & Ghurab (2023) as the "fairest
comparator" under a "same no-scores constraint." **That was wrong.** They use demographics +
daily clickstream only — *no assessment data at all* — whereas our 36-feature model uses
coursework scores and coursework submission behaviour. This experiment measures what our model
achieves under *their* feature access, so the comparison is genuinely like-for-like.
**Protocol unchanged:** StratifiedGroupKFold(5) grouped by `id_student`, official XGBoost
hyperparameters, exams excluded, per-student horizon. Only the feature set varies.
**Results JSON:** `reports/experiment_009_feature_access_results.json`

---

## 1. The three arms

| Arm | Features | Contents |
|---|--:|---|
| **A** — full (published headline) | 36 | Everything: clickstream + demographics + coursework submission behaviour + coursework scores |
| **B** — no grades | 33 | A minus the 3 score-derived features (`rank_wa`, `score_slope_cw`, `score_std_cw`) |
| **C** — clickstream + demographics only | 24 | B minus all assessment-derived features (submissions, submit leads, late count, completion ratio, first submit day, assessment types, clicks-per-assessment, assessment focus) — **matches Al-azazi's feature access** |

Arm C contains only: engagement volume/rhythm/recency, activity-type click shares, cohort
click rank, registration lead, and prior-education dummies.

## 2. Results

### Engaged population (29,496 — our published population)

| Arm | Feats | Accuracy | Macro-F1 | W | F | P | **D** |
|---|--:|--:|--:|--:|--:|--:|--:|
| **A** full | 36 | **0.8362 ± 0.0038** | **0.7949 ± 0.0048** | 0.943 | 0.813 | 0.843 | **0.581** |
| **B** no grades | 33 | 0.8053 ± 0.0028 | 0.6951 ± 0.0041 | 0.941 | 0.791 | 0.822 | **0.227** |
| **C** clickstream+demo | 24 | 0.7956 ± 0.0041 | 0.6748 ± 0.0042 | 0.938 | 0.775 | 0.814 | **0.172** |

### Full registered population (32,593 — matches Al-azazi)

| Arm | Feats | Accuracy | Macro-F1 | W | F | P | **D** |
|---|--:|--:|--:|--:|--:|--:|--:|
| **A** full | 36 | 0.8506 ± 0.0048 | 0.7976 ± 0.0083 | 0.959 | 0.808 | 0.843 | **0.581** |
| **B** no grades | 33 | 0.8225 ± 0.0042 | 0.6995 ± 0.0067 | 0.958 | 0.785 | 0.822 | **0.233** |
| **C** clickstream+demo | 24 | 0.8124 ± 0.0055 | 0.6756 ± 0.0076 | 0.956 | 0.768 | 0.812 | **0.166** |

## 3. The headline finding — coursework scores exist almost entirely to identify Distinction

Removing the 3 score features (A → B) costs **0.100 macro-F1** (0.795 → 0.695). Decomposed by
class:

| Class | Δ F1 when scores removed | Share of the total loss |
|---|--:|--:|
| Withdrawn | −0.002 | 0.5% |
| Fail | −0.022 | 5.5% |
| Pass | −0.021 | 5.3% |
| **Distinction** | **−0.354** | **88.7%** |

**Nearly 89% of the value of coursework scores goes to one class.** Withdrawn, Fail and Pass
are almost entirely predictable from behaviour — engagement volume, recency and submission
patterns. But *excellence* is invisible in behaviour: without grades the model cannot tell a
Distinction student from an ordinary Pass (F1 collapses from 0.581 to 0.227).

This is intuitive in hindsight. Disengagement has a strong behavioural signature; academic
excellence does not — high-performing and merely-passing students look similar in clickstream
terms. Grades are the only channel that carries it.

## 4. Like-for-like comparison with Al-azazi & Ghurab (2023)

*Their setup: OULAD 32,593, 4-class, day 270 (full course), demographics + daily clickstream
(no assessment data), ANN-LSTM, **70/30 holdout with no student grouping**.*

Matched arm: **C on the full 32,593 population.**

| Metric | Al-azazi & Ghurab | **Ours (arm C)** | Difference |
|---|--:|--:|--:|
| Accuracy | 0.72 | **0.8124** | **+0.092** |
| **Macro-F1** | **0.66** | **0.6756** | **+0.016** |
| F1 Withdrawn | 0.70 | **0.956** | +0.256 |
| F1 Fail | 0.65 | **0.768** | +0.118 |
| F1 Pass | 0.70 | **0.812** | +0.112 |
| **F1 Distinction** | **0.59** | 0.166 | **−0.424** ← *they win decisively* |

### Honest reading

1. **On macro-F1 the like-for-like result is essentially a tie** — +0.016, roughly two
   standard deviations of our fold noise (±0.0076). The previously claimed "+0.14 macro-F1
   advantage" was an artefact of comparing our score-using model against their score-free one.
   **That claim must be withdrawn.**
2. **We win clearly on accuracy (+9.2 points)** and on three of four classes — Withdrawn
   (+0.26), Fail (+0.12), Pass (+0.11). These are real and substantial.
3. **They beat us decisively on Distinction (0.59 vs 0.17)** in the no-assessment regime. Two
   plausible explanations, and we cannot separate them from published information alone:
   (a) their day-wise ANN-LSTM sequence model may extract excellence signal from temporal
   clickstream patterns that our aggregate features discard; or (b) their 70/30 split with no
   student grouping may inflate the minority class, since repeat-enrolment students appear on
   both sides. We flag both rather than assume the favourable one.
4. **Our validation is stricter throughout** — grouped, 5-fold, with zero student overlap
   asserted. Their holdout permits the identity leakage we measured at 15.1% of test rows. So
   a marginal macro-F1 lead under a harder protocol is a modest but genuine result.

## 5. Corrected claims

**Withdraw these claims:**
- ❌ "Al-azazi & Ghurab is the fairest comparator, same no-scores constraint" — they are *more*
  constrained than our headline model.
- ❌ "We beat the fairest comparator by +0.14 macro-F1 under equal constraints."
- ❌ Any statement implying our headline 0.795 is achieved without assessment data.

**Defensible replacements:**
- ✅ "Our headline model uses coursework performance but excludes all exam data. Its feature
  access is closest to Althibyani (2024) and Shou et al. (2024), who also use assessment
  scores — we exceed their macro-F1 by 0.09 and 0.12 respectively while using stricter
  student-grouped validation."
- ✅ "Under Al-azazi & Ghurab's exact feature access (clickstream + demographics only) our
  model reaches accuracy 0.812 vs their 0.72 and macro-F1 0.676 vs their 0.66 — a clear
  accuracy advantage and a marginal macro-F1 advantage, achieved under stricter validation.
  They outperform us on the Distinction class in this regime."
- ✅ "Coursework scores contribute 0.10 macro-F1, of which 89% is the Distinction class alone —
  behaviour predicts failure and withdrawal well, but not excellence."

## 6. What this changes about the project's story

It **strengthens** the honest narrative rather than weakening it:

- The behavioural signal for **at-risk detection** (Withdrawn + Fail) is robust — it barely
  moves when grades are removed (Withdrawn −0.002, Fail −0.022). The early-warning use case,
  which is the practically important one, does **not** depend on having grades.
- The dependence on grades is confined to grading the *top* of the distribution — which matters
  little for intervention, since nobody intervenes on Distinction students.
- It gives a clean, quotable scientific finding: **failure is behaviourally visible; excellence
  is not.**

## 7. Limitations

Al-azazi & Ghurab's figures are taken from their published tables; we did not re-run their
ANN-LSTM under our protocol, so the Distinction comparison in particular cannot be attributed
to architecture versus validation. Running a sequence model under grouped CV remains the
outstanding check (also flagged in the publication audit).
