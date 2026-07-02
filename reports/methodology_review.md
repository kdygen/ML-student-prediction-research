# Methodology Review — Phase 5 (senior-reviewer pass over the full experimental setup)

**Date:** 2026-07-02
**Scope:** everything that is not a feature: splits, imbalance handling, metrics, binary
experiments, reproducibility, seeds, execution order, hidden assumptions. Companion documents:
`leakage_audit.md` (features), `evaluation_validity_audit.md` (splits/membership evidence),
`baseline_v3.md` (resulting official protocol).

Verdicts: ✅ sound · 🔧 fixed in v3 · ⚠️ documented limitation (change not justified now).

---

## 1. Train/test strategy — 🔧 fixed in v3

- v1/v2: `train_test_split(test_size=0.2, random_state=42)`, unstratified, ungrouped, on a
  survivor sample. Two confirmed flaws (overlap: 12–17% of test rows share a student with
  train; membership: 31% of the at-risk population missing at cutoff 30) — quantified in
  `evaluation_validity_audit.md`.
- v3 (official): `GroupShuffleSplit(group=id_student, test_size=0.2, random_state=42)` on the
  registered still-enrolled population, with repeated grouped splits (seeds 0–4) reported as
  mean±std for RF/XGB.
- Not stratified: GroupShuffleSplit cannot stratify exactly (groups carry mixed labels).
  Measured test-set class proportions stay within ~1pp of the population at every cutoff
  (sample sizes ≥ 5,400 rows), so this is acceptable; repeats absorb residual variation. ⚠️
  minor, documented.

## 2. Class imbalance handling — ⚠️ inconsistent by design, kept for comparability

- LogReg and RF use `class_weight="balanced"`; Decision Tree and XGBoost use none. This is an
  inconsistency inherited from v1: model comparisons partly confound algorithm with weighting.
- Not changed in v3: altering weights would break v1↔v2↔v3 attribution ("what did the
  methodology fixes cost?"). It biases *model-vs-model* claims, not *baseline-vs-baseline*
  claims. Flagged as the first candidate for a controlled follow-up experiment (e.g. XGBoost
  with `sample_weight`).
- `imblearn` (SMOTE/SMOTETomek) is imported in cell 2 but **never used** — no resampling
  happens anywhere. Left as-is (imports are harmless history). Documented so nobody assumes
  SMOTE was applied.

## 3. Metric selection — ✅ (accuracy + macro-F1, both always reported)

- Accuracy alone is misleading at ~46–51% majority share; macro-F1 weights the four classes
  equally and exposes majority-collapse (e.g. Decision Tree c14 v1: 0.504 acc / 0.277 F1).
  Both are reported for every model/cutoff in all baselines; neither is optimized for.
- Per-class precision/recall remain visible via `classification_report` in the notebook.
- Class distributions now vary across cutoffs in v3 (by design — decided outcomes leave the
  risk population), so **cross-cutoff comparisons must compare against each cutoff's own
  baseline**, not raw numbers across cutoffs. Documented in `baseline_v3.md`.

## 4. Multiclass evaluation — ✅ after v3

Single-split point estimates are now accompanied by GroupKFold(5) and repeated-GSS mean±std
(see audit §1.4). Observed split noise (±0.005–0.02 acc) defines the minimum effect size worth
claiming in future experiments: **differences under ~2σ ≈ 1–2 points are not evidence**.

## 5. Binary experiments — ⚠️ legacy, clearly demarcated

The pairwise-RF section (Pass/Fail etc.) predates the audits: stratified but **ungrouped**
split, **survivor** sample, default RF. It remains in the notebook as preserved history and
for continuity with prior literature-style numbers, but it is **not part of the official v3
protocol**. Any future use should re-run it grouped on the v3 population.

## 6. Reproducibility — ✅ with known, quantified tolerances

- Environment pinned: Python 3.12.6, scikit-learn 1.6.1, pandas 2.2.2, numpy 2.0.2,
  xgboost 3.3.0 (`baseline_v1.md` §7 for the full investigation).
- Data pipeline reproduces **bit-exact** across pandas 2.2.2↔2.2.3 and numpy 2.0.2↔2.5.0 on
  one platform; across platforms (arm64↔x86_64) all frames match except `burstiness`, which
  differs below 1e-9 (verified by round-9 hashing against Colab).
- Model tolerances: RF is sklearn-version-dependent (1.6.1→1.9.0 shifted cutoff-30 accuracy
  by +5.5pp) and platform-sensitive (~±0.4%). Rule: **never compare RF numbers produced under
  different sklearn versions.**
- The notebook is the single source of truth; all official numbers were produced by executing
  its cells verbatim in order.

## 7. Random seeds — ✅

`random_state=42` for every split and every model in every baseline; robustness repeats use
declared seeds 0–4. No undeclared randomness remains (no `np.random` calls outside seeded
APIs; SMOTE unused).

## 8. Preprocessing / feature-engineering order — ✅ verified, with one sharp edge

- Verified end-state correctness at every cutoff: no duplicate keys/columns, no merge
  suffixes, labels consistent (audit Phase 3). The blanket `mlData.fillna(0, inplace=True)`
  (cell ~56) executes before the later feature merges, so each later feature's own
  `fillna` governs — checked: total NaNs in the final frame = 0, and each fill rule is the
  one documented per feature.
- v3 is order-robust by construction: it assembles from named intermediate tables
  (`behaviorFeatures`, `courseworkPerformance_v2`, …) rather than mutating `mlData`.
- ⚠️ Sharp edge: the v1/v2 sections *do* mutate `mlData` sequentially, so cells must run
  top-to-bottom exactly once (see §10). This is inherent to the preserved history.

## 9. Data contamination — ✅ none found

- Target: `target_multi` is a pure recode of `final_result`; no active feature derives from
  it (audit Phase 3/4). Registration-status fields used for v3 membership are compared to the
  cutoff only (temporally valid) and are **not** features.
- Test-time information: scalers/models are fit on train folds only (`Pipeline` fits
  StandardScaler inside the split); no global normalization leaks test statistics.
- Duplicate rows that could straddle a split: none (0 duplicate keys); same-student rows are
  handled by grouping.

## 10. Notebook execution order — ✅ documented contract

The notebook is history-preserving and **must run top-to-bottom once per cutoff** (set
`CUTOFF` in the early-prediction-setup cell, Run All). Later sections overwrite earlier
feature values in `mlData` by design (v2 overwrites v1's three leaky features; v3 builds its
own `mlDataV3` without touching `mlData`). Running cells out of order or twice is unsupported
— this is the standing execution contract for all official numbers.

## 11. Hidden assumptions — now explicit

1. `date_submitted ≤ CUTOFF` proxies "score is known" (no grading date in OULAD; conservative).
2. Assessment deadlines (`assessments.date`) are known in advance (published schedule).
3. `date_unregistration = NaN` ⇒ student did not withdraw early (stays in risk population).
4. `date_registration = NaN` (45 rows) ⇒ treated as enrolled (present in `studentInfo`).
5. Students who withdrew on/before the cutoff are **known outcomes, not prediction cases**
   (excluded from v3 membership) — the deployment framing.
6. `registration_lead` median-fill exists in the v1 history but the feature is not active.
7. OULAD click counts (`sum_click`) are trusted as-is; `on_bad_lines="skip"` on the
   `studentVle.csv` read could silently drop malformed lines — measured: row count matches the
   published dataset (10,655,280), so nothing is actually skipped.

## Readiness verdict

With v3 (grouped evaluation + registered risk population + leakage-free features), no known
**major** methodological issue remains. Remaining limitations — imbalance-handling
inconsistency (§2), non-stratified grouped split (§1), legacy binary section (§5), RF
environment fragility (§6) — are documented, quantified, and none undermines the validity of
the benchmark. **The pipeline is ready to freeze; processed-data caching may proceed when
explicitly instructed** per `reports/caching_plan.md`.
