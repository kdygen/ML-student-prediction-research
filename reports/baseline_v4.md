# Baseline v4 — Promoted, Independently Verified Features (official)

**Date:** 2026-07-02
**Notebook:** `notebook/OULAD_early_prediction_v1 (1).ipynb` (single notebook; one additive
v4 section; v1/v2/v3 cells byte-identical to the committed versions)
**Cache:** `data/processed/p4/` (canonical for future experiments) · p3 preserved immutable
**Promotion evidence:** `reports/baseline_v4_feature_audit.md` · **Raw metrics:** `reports/baseline_v4_results.json`
**Environment:** pinned (Python 3.12.6, sklearn 1.6.1, pandas 2.2.2, numpy 2.0.2, xgboost 3.3.0), macOS/arm64.

---

## 1. What Baseline v4 is

Identical to Baseline v3 in **population** (registered still-enrolled risk population),
**protocol** (GroupShuffleSplit, group=`id_student`, test_size=0.2, seed 42), **models, and
hyperparameters**. The only change: the active feature set grows 19 → **35** with the 16
Experiment-002 features that survived a full scientific promotion audit — independent
recomputation (worst-case deviation 2.27e-13), per-feature dependency trace and leakage
proof, redundancy scan, and pre-registered leave-one-group-out ablations at two cutoffs
(all documented in the feature audit). **Nothing was removed: all 16 candidates passed.**

Promoted groups: `module_norm` (cohort percentile ranks of ≤cutoff behaviour),
`submission_timing`, `recent_windows`, `recency_decay`.

## 2. Official results (held-out GSS-42 test; accuracy / macro-F1)

| Cutoff | LogReg | Tree | RF | XGB |
|--:|:--|:--|:--|:--|
| 14  | 0.3478/0.3468 | 0.4890/0.3028 | 0.4799/0.3093 | **0.4914/0.3428** |
| 30  | 0.4014/0.3972 | 0.5161/0.3392 | 0.5244/0.3719 | **0.5265/0.3955** |
| 60  | 0.4621/0.4451 | 0.5555/0.3932 | 0.5697/0.4024 | **0.5752/0.4387** |
| 90  | 0.4866/0.4624 | 0.5896/0.4180 | 0.6081/0.4328 | **0.6165/0.4637** |
| 140 | 0.5570/0.5040 | 0.6699/0.4831 | 0.6783/0.4765 | **0.6933/0.5144** |

Per-class precision/recall/F1 and confusion matrices for every model and cutoff are in
`baseline_v4_results.json` (`v4.models.<name>.per_class` / `.confusion`).

### Robustness — repeated grouped splits (seeds 0–4) and GroupKFold(5)

macro-F1, mean ± std:

| Cutoff | RF repeats | RF GKF5 | XGB repeats | XGB GKF5 |
|--:|:--|:--|:--|:--|
| 14  | 0.3059 ± 0.0077 | 0.3065 ± 0.0018 | 0.3325 ± 0.0057 | 0.3364 ± 0.0042 |
| 30  | 0.3836 ± 0.0059 | 0.3788 ± 0.0071 | 0.4081 ± 0.0053 | 0.4009 ± 0.0068 |
| 60  | 0.4143 ± 0.0086 | 0.4106 ± 0.0085 | 0.4502 ± 0.0048 | 0.4449 ± 0.0058 |
| 90  | 0.4401 ± 0.0085 | 0.4339 ± 0.0036 | 0.4701 ± 0.0075 | 0.4685 ± 0.0042 |
| 140 | 0.4758 ± 0.0055 | 0.4818 ± 0.0105 | 0.5083 ± 0.0033 | 0.5113 ± 0.0115 |

The two independent protocols agree within ~1σ everywhere — the estimates are stable.

## 3. v3 → v4 (same protocol, same models; the feature effect isolated)

GSS-42 test deltas (acc / macro-F1):

| Cutoff | LogReg | Tree | RF | XGB |
|--:|:--|:--|:--|:--|
| 14  | −0.003/+0.012 | +0.011/+0.052 | +0.017/+0.009 | +0.010/+0.028 |
| 30  | +0.018/+0.023 | +0.034/+0.051 | +0.020/+0.015 | +0.017/+0.025 |
| 60  | +0.025/+0.025 | +0.017/+0.046 | +0.011/+0.014 | +0.014/+0.032 |
| 90  | +0.048/+0.045 | +0.018/+0.044 | +0.021/+0.028 | +0.022/+0.039 |
| 140 | +0.058/+0.046 | +0.046/+0.070 | +0.025/+0.029 | +0.031/+0.043 |

Every model improves on **both** metrics at every cutoff (single exception: LogReg c14
accuracy −0.003 while its macro-F1 gains +0.012). Repeated-split paired evidence (identical
splits, seeds 0–4): **all 50/50 paired accuracy deltas positive** (RF and XGB × 5 cutoffs ×
5 seeds); e.g. XGB c140 accuracy 0.6713 → 0.6981, macro-F1 0.4743 → 0.5083.

Balanced improvement across classes (XGB, GSS-42 test, plain argmax, F1 by class v3→v4 at
c140): Withdrawn 0.021→0.090, Fail 0.650→0.690, Pass 0.748→0.776, Distinction 0.467→0.502 —
**all four classes improve**; gains are not bought from any single class. (Withdrawn remains
weak at plain argmax — recall-oriented deployments should use the τ operating point, §6.)

## 4. Validity re-verification (executed as part of this run)

| Check | Result |
|---|---|
| Zero student overlap train/test | asserted in-notebook for GSS-42 **and** in the runner for every repeat seed and every GKF fold — all passed |
| Temporal leakage | hard `assert max(date) ≤ CUTOFF` / `max(date_submitted) ≤ CUTOFF` inside the v4 cell — passed at all cutoffs |
| Duplicate prediction cases | `assert dup(id_student, module, presentation) == 0` in-cell — passed |
| NaN/inf entering models | `assert X.isna().sum() == 0` in-cell — passed |
| Cache hash reproducibility | every p4 parquet reload-verified bit-exact; **c030 rebuilt from scratch → identical frame hash** (`c030_rebuild_hash_match: true` in the global manifest) |
| p3 cache immutable | all 5 p3 parquets still match their manifests (sha256) |
| v1/v2/v3 notebook history | 0 committed code cells modified; exactly 1 new cell added |

## 5. The p4 cache (canonical from now on)

```
data/processed/p4/{manifest.json, c014..c140/{mlDataV4.parquet, manifest.json}}
```
Rows per cutoff: 28,061 / 27,450 / 26,353 / 25,558 / 24,289 (46 columns; 35 active features
listed in each manifest). Manifests carry raw-input sha256s, `pipeline_code_sha256` (all code
cells through the v4 cell), environment, schema, bit-exact + round-9 frame hashes, git
commit. No splits, standardization, resampling, or model outputs are cached. Existing
`.gitignore` rules already cover p4 (parquet ignored, manifests tracked).

## 6. Notes and standing caveats

- Baseline v4 reports **plain-argmax** model outputs (like v1–v3). The class-prior threshold
  τ from Experiment 001 remains the deployment knob for recall-oriented operating points
  (e.g. XGB c140 τ=0.75: macro-F1 0.5540, Withdrawn recall 0.27 — Experiment 002 report §5);
  it is an evaluation-layer choice, not part of the baseline definition.
- v4 numbers are comparable to v3 (same population/protocol) but **not** to v1/v2.
- Inherited documented limitations (methodology review §§1–6): mixed class-weighting across
  models, non-stratified grouped split, RF environment sensitivity.

**Baseline v4 is the official benchmark.** Future experiments start from the p4 cache and
compare within-cutoff against `baseline_v4_results.json`.
