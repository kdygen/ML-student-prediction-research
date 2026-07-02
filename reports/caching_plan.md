# Caching Plan — Processed-Data Caching Architecture

**Date:** 2026-07-02
**Status:** **IMPLEMENTED** (Phase 8 Part A, same date) — see the addendum in §9. Sections
1–8 are the original design; §9 records what was actually built and the one deliberate
narrowing versus the design.

---

## 1. Goals

- Skip the ~30 s raw load + ~60–90 s feature pipeline per cutoff when iterating on models.
- Guarantee that a cached artifact is **provably** the product of a specific (raw data,
  notebook pipeline, environment) triple — no silent staleness.
- Keep the cache **out of git** (large binaries) while keeping its **manifests in git**
  (small JSON) so reproducibility information is versioned.

## 2. What to cache (three layers)

| Layer | Artifact | Why | Format |
|---|---|---|---|
| L0 (already present) | raw OULAD CSVs (`data/raw/*.csv`) | immutable inputs | CSV (gitignored) |
| L1 feature tables (per cutoff) | `studentVleEarly`-derived aggregates: `behaviorFeatures`, `activityClicks`, `burst`, `focus`; assessment tables: `courseworkPerformance_v2`, `assessmentCount_v2`, `recoverySlope_v2` | most expensive computations; reusable by any future baseline | Parquet |
| L2 model-ready frames (per cutoff) | `mlData` (v2 features, survivor sample — needed to reproduce v1/v2 baselines) and `mlDataV3` (official); plus `X`/`y`/`groups` column lists in the manifest | what experiments actually consume | Parquet |

Model artifacts and predictions are **not** cached (cheap to refit, environment-fragile — RF
is sklearn-version-dependent).

## 3. Where + naming convention

```
data/processed/
  p3/                                # pipeline version (v3 = current official)
    manifest.json                    # global manifest (see §4)
    c014/                            # zero-padded cutoff
      mlData.parquet
      mlDataV3.parquet
      behaviorFeatures.parquet
      activityClicks.parquet
      burst.parquet
      focus.parquet
      courseworkPerformance_v2.parquet
      assessmentCount_v2.parquet
      recoverySlope_v2.parquet
      manifest.json                  # per-cutoff manifest
    c030/ ... c060/ ... c090/ ... c140/
```

- `p<N>` = **pipeline version**, bumped manually whenever any feature/membership logic
  changes (v1→v2→v3 lineage). Multiple pipeline versions may coexist for comparison.
- Zero-padded cutoffs keep lexicographic = numeric ordering.
- Parquet: exact dtypes, fast, compact; pinned pandas 2.2.2 reads/writes it stably.

## 4. Manifest contents (per cutoff)

```json
{
  "pipeline_version": "p3",
  "cutoff": 30,
  "created_at": "<ISO timestamp>",
  "git_commit": "<repo HEAD when generated>",
  "notebook": {
    "file": "notebook/OULAD_early_prediction_v1 (1).ipynb",
    "pipeline_code_sha256": "<sha256 of concatenated source of all code cells up to and including the v3 cell>"
  },
  "raw_inputs": {"studentVle.csv": {"bytes": 453836331, "sha256": "..."}, "...": {}},
  "environment": {"python": "3.12.6", "pandas": "2.2.2", "numpy": "2.0.2",
                   "scikit-learn": "1.6.1", "xgboost": "3.3.0",
                   "platform": "macOS-arm64 | Linux-x86_64"},
  "artifacts": {
    "mlDataV3.parquet": {"rows": 27450, "cols": 30,
                          "frame_sha256": "<pd.util.hash_pandas_object sha256>",
                          "round9_sha256": "<hash of values rounded to 1e-9>"},
    "...": {}
  },
  "active_features_v3": ["weighted_average", "..."],
  "split_spec": {"strategy": "GroupShuffleSplit", "group": "id_student",
                  "test_size": 0.2, "random_state": 42}
}
```

Two hashes per frame: `frame_sha256` (bit-exact, platform-specific) and `round9_sha256`
(rounded to 1e-9, **platform-independent** — the `burstiness` `.std()` reduction differs below
1e-9 between arm64/x86_64, so cross-platform verification must use the round-9 hash; see
Baseline v1 report §7).

## 5. Cache invalidation rules

A cached cutoff directory is **valid** iff ALL of:
1. every `raw_inputs` sha256 matches the current `data/raw` files;
2. `pipeline_code_sha256` matches the current notebook (code cells up to and including the
   v3 cell — markdown and later model/analysis cells do not invalidate);
3. `environment.pandas` and `environment.numpy` match exactly (bit-level frame reproducibility
   was only verified within these versions); sklearn/xgboost versions do NOT invalidate the
   cache (they affect models, not frames) but must be recorded;
4. `pipeline_version` matches the requested version.

If any check fails → regenerate that cutoff and rewrite its manifest. Never partially update
artifacts under an existing manifest. Loader behaviour: hard-fail with a clear message rather
than silently recomputing, so invalidation is always a conscious event.

## 6. gitignore additions (when caching is implemented)

```
# processed cache — artifacts out, manifests in
data/processed/**/*.parquet
!data/processed/**/manifest.json
```

## 7. Generation procedure (when instructed)

1. Run the notebook (or the finder-based runner) once per cutoff in the pinned env.
2. After the v3 cell, dump the L1/L2 frames listed in §2 with `DataFrame.to_parquet`.
3. Compute hashes + write per-cutoff `manifest.json`, then the global manifest.
4. Verify: reload every parquet, recompute `frame_sha256`, compare. Fail loudly on mismatch.
5. Commit manifests only (with the rest of the repo state that produced them).

## 8. Readiness gate

Caching should only be executed once the methodology is frozen (see
`reports/methodology_review.md` §Readiness). Caching before freezing bakes a moving pipeline
into binary artifacts and invites silent mismatch between cache and code.

## 9. Implementation addendum (executed 2026-07-02, Phase 8 Part A)

Built exactly per §§3–7 with **one deliberate narrowing** per the freeze instruction
("cache one processed dataset per cutoff"): only the **L2 canonical frame `mlDataV3`** is
cached — the L1 intermediate tables from §2 were not materialized (they can be regenerated
from the notebook and would add cache surface without a current consumer).

What exists now:

```
data/processed/p3/{manifest.json, c014..c140/{mlDataV3.parquet, manifest.json}}
```

- One dataset per cutoff (28,061 / 27,450 / 26,353 / 25,558 / 24,289 rows; 30 columns,
  19 active features listed in each manifest). Target distributions match the official v3
  run exactly.
- Contents are model-independent: **no splits, no standardization, no resampling, no model
  outputs** (split spec is recorded declaratively in the manifest for recomputation).
- Every parquet was **reload-verified** (bit-exact `frame_sha256` match after round-trip);
  manifests carry raw-input sha256s (all six CSVs), `pipeline_code_sha256` over all notebook
  code cells up to and including the v3 cell, environment versions, schema (per-column
  dtypes), `round9_sha256` for cross-platform checks, and the generating git commit.
- `.gitignore` updated per §6 (parquet ignored, manifests tracked — verified with
  `git check-ignore`).
- The cache is the canonical input for experiments from Experiment 001 onward.
