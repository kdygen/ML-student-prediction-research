# Binary Evaluation — Baseline v4 Protocol

**Date:** 2026-07-02
**Notebook section:** "Binary evaluation — Baseline v4 protocol" (2 additive cells at the end
of `notebook/OULAD_early_prediction_v1 (1).ipynb`; nothing above modified)
**Data:** official p4 cache (`data/processed/p4/`) — the run below executed the exact
notebook-cell source against the cached `mlDataV4` frames (bit-identical to notebook output).
**Environment:** pinned (Python 3.12.6, sklearn 1.6.1, pandas 2.2.2, numpy 2.0.2, xgboost 3.3.0).

---

## 1. What this is

The four pairwise binary tasks from the notebook's original `#Binary RF` section, re-run
under the **official v4 protocol**. The original binary cell (preserved as history) used the
pre-v2 leaky features, the survivorship-biased population, and a random row split with
student overlap; these results supersede it.

| Component | Setting |
|---|---|
| Population | `mlDataV4` — registered still-enrolled risk population at each cutoff |
| Split | `GroupShuffleSplit(group=id_student, test_size=0.2, random_state=42)`; zero-overlap asserted per task |
| Model | `RandomForestClassifier(random_state=42)` — unchanged from the original binary cell; **no model or feature changes** |
| Feature sets | official v3 (19) and v4 (35) evaluated on **identical splits** — the delta isolates the 16 promoted features |
| Positive class | first-named class of each pair |

Raw per-task metrics (accuracy, positive-class precision/recall/F1, split sizes) for every
cutoff are printed by the notebook cell and archived in `reports/binary_v4_results.json`.

## 2. Accuracy — v4 feature set (35), held-out grouped test

| Binary task | c14 | c30 | c60 | c90 | c140 |
|---|--:|--:|--:|--:|--:|
| Pass vs Fail        | 0.6984 | 0.7297 | 0.7769 | 0.7877 | 0.8431 |
| Distinction vs Fail | 0.7870 | 0.8323 | 0.8848 | 0.9045 | 0.9278 |
| Distinction vs Pass | 0.7994 | 0.8091 | 0.8218 | 0.8304 | 0.8360 |
| Withdrawn vs Pass   | 0.7158 | 0.7797 | 0.8307 | 0.8553 | 0.9172 |

Test sizes per task: ~2,000–3,900 rows depending on pair and cutoff (e.g. c30: Pass/Fail
3,896, Distinction/Fail 2,004, Distinction/Pass 3,086, Withdrawn/Pass 3,468).

## 3. Effect of the promoted features (Δ accuracy, v4 − v3, identical splits)

| Binary task | c14 | c30 | c60 | c90 | c140 |
|---|--:|--:|--:|--:|--:|
| Pass vs Fail        | +0.0240 | +0.0313 | +0.0188 | +0.0198 | +0.0297 |
| Distinction vs Fail | +0.0125 | +0.0140 | +0.0118 | +0.0129 | +0.0110 |
| Distinction vs Pass | −0.0019 | +0.0003 | +0.0036 | +0.0078 | +0.0065 |
| Withdrawn vs Pass   | +0.0203 | +0.0320 | +0.0188 | +0.0178 | +0.0189 |

19 of 20 task×cutoff cells improve (the one exception, Distinction-vs-Pass at c14, is −0.002 —
within single-split noise). The promoted features help most exactly where they were designed
to: engagement-recency and submission-timing signal for Pass/Fail and Withdrawn/Pass
(+2–3 pp), less for the Distinction/Pass boundary, which is driven by score quality rather
than activity volume.

## 4. Positive-class F1 (v4 feature set) — read this alongside accuracy

Both Distinction-vs-Pass (~20% positive) and Withdrawn-vs-Pass (29% → 13% positive as
withdrawals leave the risk population by c140) are imbalanced, so accuracy overstates them:

| Binary task (positive) | c14 | c30 | c60 | c90 | c140 |
|---|--:|--:|--:|--:|--:|
| Pass vs Fail (Pass)               | 0.7881 | 0.8047 | 0.8386 | 0.8455 | 0.8830 |
| Distinction vs Fail (Distinction) | 0.6072 | 0.6978 | 0.8037 | 0.8404 | 0.8734 |
| Distinction vs Pass (Distinction) | 0.0802 | 0.1943 | 0.3276 | 0.3849 | 0.4667 |
| Withdrawn vs Pass (Withdrawn)     | 0.3743 | 0.4796 | 0.5154 | 0.5077 | 0.5903 |

At the default 0.5 threshold the RF is precision-heavy on the minority class: Withdrawn
recall runs 0.27 → 0.45 (precision 0.59 → 0.85) from c14 to c140; Distinction-vs-Pass recall
0.04 → 0.37. For recall-oriented deployment the threshold is the knob to move (cf. the τ
adjustment of Experiment 001) — not part of this baseline evaluation.

## 5. Caveats

- Single grouped split (seed 42), matching the original binary section's single-split design;
  headline multiclass robustness (repeated seeds + GroupKFold) lives in `baseline_v4.md`.
  Deltas of ~±0.005 in §3 should be read as noise.
- Each pair's population is the still-enrolled-at-cutoff subset holding those two outcomes;
  pair populations shrink/shift with cutoff (especially Withdrawn), so cells are comparable
  within a cutoff, not across cutoffs as a cohort.
- These binary numbers are a **secondary view** of the official benchmark. The official
  Baseline v4 remains the 4-class task in `baseline_v4.md` / `baseline_v4_results.json`.
