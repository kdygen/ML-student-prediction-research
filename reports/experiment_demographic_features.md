# Experiment 005 — Enrollment-Time Demographic Features (additive on Baseline v4)

**Date:** 2026-07-08
**Question:** do enrollment-time demographics add statistically significant predictive value
*beyond* the official v4 features? Every arm = the complete official v4 35-feature set + added
demographic columns; nothing removed, Baseline v4 unmodified.
**Protocol (official):** GroupShuffleSplit(group=`id_student`, seed 42) headline with
per-class metrics + confusion; repeated grouped splits seeds 0–4; GroupKFold(5); official
XGB hyperparameters identical across arms; RF (official params) as cross-model robustness.
Splits depend only on groups, so row indices are identical across arms — all comparisons are
**paired**. No target-dependent feature selection: arms 1–4 fixed a priori (per task spec);
arms 5–6 are post-hoc and labeled as such.
**Environment note:** the pinned venv was rebuilt (tmp cleanup had removed it) and verified to
reproduce the official v4 XGB headline **bit-exactly at every cutoff** (diff = 0.0) before any
experiment ran. A0 sanity in the results JSON re-confirms this.
**Raw results + all per-class/confusion/statistics:** `reports/experiment_demographic_results.json`

---

## 1. Variables and leakage audit

All from `studentInfo.csv` — a per-(student, module, presentation) **enrollment snapshot**:
fixed at registration, before any cutoff, independent of course behavior and of
`final_result`. Coverage 100% at every cutoff, zero missing after encoding, no duplicate keys.

| Family | Columns added | Encoding |
|---|---|---|
| academic history | `studied_credits`, `num_of_prev_attempts` | numeric as-is (prev_attempts counts strictly *earlier* attempts) |
| socioeconomic | `imd_band`, `region` | IMD ordinal decile 1–10 + missing flag ('?' = 3.4%); region 12 drop-first dummies |
| personal background | `gender`, `age_band`, `disability` | binary / 2 dummies / binary |

(`highest_education` is already in v4.) Verdict: all leakage-free — enrollment-time by
construction. 20 added columns total.

## 2. Arms and headline results (XGB, GSS-42 test; accuracy / macro-F1)

| Arm | c14 | c30 | c60 | c90 | c140 |
|---|:--|:--|:--|:--|:--|
| A0 v4 (official reference) | .4914/.3428 | .5265/.3955 | .5752/.4387 | .6165/.4637 | .6933/.5144 |
| A1 + all 20 demographics | .5051/.3684 | .5271/.4014 | .5755/.4430 | .6161/.4682 | .6931/.5123 |
| A2 + academic history (2) | .4976/.3601 | .5240/.3907 | .5780/.4442 | .6155/.4628 | .6964/.5149 |
| A3 + socioeconomic (14) | .4983/.3496 | .5286/.3994 | .5763/.4390 | .6155/.4605 | .6913/.5088 |
| A4 + personal (4) | .4915/.3437 | .5302/.3992 | .5789/.4408 | .6151/.4650 | .6913/.5108 |
| A5 + academic+personal (6)* | .4996/.3611 | .5304/.4006 | .5774/.4423 | .6190/.4687 | .6921/.5144 |
| A6 + top-3 demo (3)* | .4972/.3588 | .5280/.3965 | .5786/.4460 | .6179/.4661 | .6974/.5199 |

\* post-hoc arms, defined after inspecting A1–A4 and importance ranks (A6 = `prev_attempts`,
`studied_credits`, `disability` — the three demographics ranking in the global top-16 at
c14/c30). A0 headline matches the official baseline to 0.0 at every cutoff. Per-class
precision/recall/F1 and confusion matrices for every arm × cutoff are in the results JSON;
e.g. at c14, A1 lifts Withdrawn F1 0.222→0.292 and Distinction 0.120→0.134 with no class
degrading.

## 3. Repeated grouped evaluation — paired deltas vs A0 (pooled over 5 cutoffs × 5 obs)

XGB, seeds 0–4 (GroupKFold(5) shown where it changes the conclusion):

| Arm | Δ macro-F1 (seeds) | Δ macro-F1 (GKF5) | Δ accuracy | Δ Withdrawn recall | Δ at-risk AUC |
|---|:--|:--|:--|:--|:--|
| A1 all demo | **+0.0094** (23/25, p<10⁻⁴) | **+0.0084** (22/25, p=10⁻⁴) | +0.0046 (p=10⁻⁴) | **+0.0231** (p<10⁻⁴) | +0.0052 (p=10⁻⁴) |
| A2 academic | **+0.0041** (19/25, p=0.001) | **+0.0038** (16/25, p=0.006) | +0.0017 (p=0.017) | **+0.0136** (p=2·10⁻⁴) | +0.0019 (p=0.002) |
| A3 socioeconomic | +0.0016 (14/25, p=0.18) ✗ | +0.0010 (p=0.30) ✗ | +0.0016 (p=0.07) ✗ | +0.0025 (p=0.18) ✗; GKF −0.0019 | +0.0030 (p=2·10⁻⁴) |
| A4 personal | +0.0034 (19/25, p=0.002) | +0.0021 (p=0.10) ✗ | +0.0009 (p=0.09) ✗ | **+0.0114** (p<10⁻⁴) | +0.0005 (p=0.09) ✗ |
| A5 acad+personal* | **+0.0068** (21/25, p<10⁻⁴) | **+0.0070** (23/25, p<10⁻⁴) | +0.0024 (p=0.003) | **+0.0216** (p<10⁻⁴) | +0.0024 (p=0.001) |
| A6 top-3 demo* | **+0.0046** (22/25, p=7·10⁻⁴) | **+0.0056** (21/25, p=10⁻⁴) | +0.0013 (p=0.09) | **+0.0190** (p=10⁻⁴) | +0.0020 (p=0.004) |

Gains concentrate at early cutoffs (XGB Δ macro-F1 by cutoff, A1: +0.021 / +0.009 / +0.009 /
+0.006 / +0.002 for c14→c140) — demographics matter most exactly when behavioral data is
scarce, and fade as behavior accumulates.

### The cross-model test (RF, paired repeats, pooled) — where most arms fail

| Arm | RF Δ macro-F1 | RF Δ Withdrawn recall | Verdict |
|---|:--|:--|---|
| A1 all demo | **−0.0028** (8/25 pos, p=0.045) | negative at every cutoff | ✗ XGB-only gain; RF *degrades* |
| A5 acad+personal | +0.0008 (15/25, p=0.47) | +0.0007 (p=0.76) | ✗ RF flat — gender/age dilute the signal |
| **A2 academic** | **+0.0036** (19/25, p=6·10⁻⁴) | **+0.0050** (p=0.011) | ✓ improves both models |
| **A6 top-3 demo** | **+0.0036** (18/25, p=0.006) | **+0.0052** (p=0.008) | ✓ improves both models |

This is the decisive discriminator: the region/IMD block (14 columns) and the gender/age
dummies add variance that RF turns into noise, while `studied_credits`,
`num_of_prev_attempts`, and `disability` carry signal both model families can use.

## 4. Feature importance (XGB gain, A1 headline)

Among all 55 features: at c14, `disability` ranks **5th**, `prev_attempts` **7th**,
`studied_credits` **9th** (IMD decile 13th); at c30 `prev_attempts`/`studied_credits` rank
8th/10th; by c140 the best demographic has fallen to 13th and behavioral features dominate.
Region and age dummies never enter the top 20 at any cutoff — consistent with A3/A4's weak
arm-level results. Importance thus corroborates the arm structure: a small
academic-history + disability core carries essentially all the demographic signal.

## 5. Promotion decision against the pre-stated rules

| Rule | A1 all | A3 socio | A4 personal | **A2 academic** | **A6 top-3** |
|---|---|---|---|---|---|
| Consistent improvement (repeats) | XGB ✓ / RF ✗ | ✗ | XGB ✓ / RF n.t. | **✓ both models** | **✓ both models** |
| Leakage-free | ✓ | ✓ | ✓ | ✓ | ✓ |
| Significance vs v4 | XGB ✓ / RF ✗ (worse) | ✗ | partial | **✓ (all p ≤ 0.02)** | **✓ (all p ≤ 0.01)** |
| Improves macro-F1 + early-risk | XGB only | ✗ | partial | **✓ (Wrec +0.014 XGB / +0.005 RF)** | **✓ (Wrec +0.019 / +0.005)** |
| No generalization loss | GKF ✓ (XGB) | — | GKF ✗ | **GKF confirms (+0.0038)** | **GKF confirms (+0.0056)** |

## 6. Recommendation

**Promote the academic-history pair — `studied_credits` and `num_of_prev_attempts` — into a
proposed Baseline v5.** It is the only *pre-registered* arm that passes every promotion rule
on **both** model families and both split protocols: pooled macro-F1 +0.0041 (XGB) / +0.0036
(RF), Withdrawn recall +0.0136 / +0.0050, at-risk AUC +0.0019, all significant, GKF-confirmed,
gains largest at the early cutoffs where the intervention framework (Experiments 003/004)
operates. These are enrollment records of prior academic behavior, not protected attributes.

**`disability` is evidence-positive but held pending governance.** Adding it (arm A6) passes
every statistical rule and gives the largest early Withdrawn-recall gain — but (a) A6 was
post-hoc (importance-guided), so its evidence is one rung below A2's pre-registered status,
and (b) disability is a protected characteristic: including it in a model that drives the
Experiment-004 intervention ranking changes who gets flagged by disability status, which
requires a subgroup-fairness audit (top-K composition, ABROCA) and an explicit institutional
decision, not just a metrics test. Recommendation: keep it out of v5; document it as a
candidate contingent on that audit.

**Do not promote:** region + IMD (A3 — no significant gain on any primary metric, 14 noisy
columns), gender + age (fail cross-model consistency; they are what drag A5's RF gain to
zero), or the all-in bundle (A1 — significantly *degrades* RF macro-F1, −0.0028, p=0.045).

Perspective on effect size: v5 would be a small, honest upgrade (+0.4 pp pooled macro-F1,
+1.0–1.3 pp at c14, +1.4 pp early Withdrawn recall on XGB) — meaningful at the early cutoffs
that matter for intervention, invisible by c140. If the project prefers a
single-feature-family baseline unchanged, keeping v4 is defensible; but by the pre-stated
rules, A2 qualifies.

**Formal promotion path (not executed here, per project convention):** a v5 promotion would
follow the v4 protocol — independent recomputation of the two features, additive notebook
section, full official run, p5 cache build — and requires explicit instruction. Baseline v4
remains official and untouched by this experiment.
