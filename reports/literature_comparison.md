# Literature Comparison — Final Methodology vs Published OULAD Research

**Date:** 2026-07-20
**Basis:** the notebook methodology (per-student horizon, fair denominator, constant sentinel,
StratifiedGroupKFold-5 grouped by `id_student`).

> **BASELINE UPDATE (2026-07-21).** The project's official baseline is now the
> **assessment-free** pipeline — 36 features, **no data from `studentAssessment.csv` or
> `assessments.csv`** — at **accuracy 0.739, macro-F1 0.715, per-class F1 W 0.940 / F 0.779 /
> P 0.709 / D 0.430** (`reports/official_baseline_results.json`). The with-scores model
> (0.836 / 0.795) referenced throughout this document is now a **documented comparison arm**,
> retained because it is what most cited papers are comparable to. Where a paper is
> assessment-free, compare against the official baseline; where a paper uses assessment
> scores, compare against the with-scores arm.
**Evidence:** ~25 papers located across six literature sweeps; every paper cited below was
read in full text unless explicitly marked otherwise. Two papers could not be retrieved
(noted in §7).

## 0. Our reference numbers

**⭐ OFFICIAL BASELINE — assessment-free** (29,496; no assessment-table data):

| Metric | Value |
|---|--:|
| Accuracy | **0.7392 ± 0.0042** |
| Macro-F1 | **0.7147 ± 0.0053** |
| Weighted-F1 | 0.7528 |
| Per-class F1 (W/F/P/D) | .940 / .779 / .709 / .430 |

**Comparison arm — with coursework scores** (the arm most cited papers are comparable to):

| Result | Engaged pop. (29,496) | Full pop. (32,593 — matches the literature) |
|---|--:|--:|
| Accuracy | 0.8362 ± 0.0038 | 0.8506 ± 0.0048 |
| Macro-F1 | 0.7949 ± 0.0048 | 0.7976 ± 0.0083 |
| Weighted-F1 | 0.8330 | 0.8473 |
| Per-class F1 (W/F/P/D) | .943 / .813 / .843 / .581 | .959 / .808 / .843 / .581 |

Both populations are reported for the arm because **no published paper uses our engaged
population**; the 32,593 run exists solely to make comparisons like-for-like. Supplementary regression (33 behaviour-only
features, GroupKFold-5, n=23,241): **R² 0.334 ± 0.009, MAE 10.30, RMSE 13.38**. Archived
binary results (Exp 006b): Withdrawn-vs-rest AUC 0.9958; at-risk AUC 0.9824.

---

## 1. The finding that reframes the whole comparison

**`date_unregistration` is a near-perfect encoding of the Withdrawn label.** Verified
directly against `data/raw/`:

> Rule: *"`date_unregistration` is not null → predict Withdrawn"* — no model, one null-check.
> **TP 10,063 · FP 9 · FN 93 · TN 22,428 → precision 0.9991, recall 0.9908, F1 0.9950,
> accuracy 0.9969.**

This is definitional, and the dataset paper says so: *"Students who unregistered have
Withdrawal as the value of the final_result"* (Kuzilek, Hlosta & Zdrahal 2017, *Scientific
Data* 4:170171).

**Consequence:** the current headline claim in the OULAD literature —
**Junejo et al. 2025, *Scientific Reports* 15:16251, accuracy 0.98 / macro-F1 0.98** — uses
`total_reg_days`, derived from `date_unregistration`, as its **top-ranked feature**. The
paper states the mechanism itself: *"Only the unreg_date for the Withdrawn class is
accessible in the OULAD."* Their own trajectory table is the proof: Withdrawn F1 = 0.95 at
**5%** of course elapsed, rising only to 0.99, while Fail climbs 0.66 → 0.96. A class at
ceiling before any behaviour has accumulated is being read off a static field.

**Our pipeline uses this field only as a censoring boundary and to define the horizon; zero
of our 36 features derive from it** (verified programmatically).

The controlled contrast on the same dataset and task: **Al-azazi & Ghurab (2023) do 4-class
OULAD without that field and get 72% accuracy / 0.66 macro-F1.** Same data, same task,
26-point gap. That gap *is* the leak.

---

## 2. Tier 1 — directly comparable (4-class, full OULAD, full-course)

### 2.1 Al-azazi & Ghurab (2023) — **the assessment-free comparator**

> **⚠️ CORRECTION (2026-07-21).** An earlier version of this document described this paper as
> "the fairest comparator" under a "same no-scores constraint" and claimed a +0.14 macro-F1
> advantage. **That was wrong and is withdrawn.** Our headline model uses coursework scores
> *and* assessment submission behaviour; Al-azazi use demographics + clickstream only, with no
> access to the assessment tables at all. They are **more constrained than our headline model**,
> so the headline numbers are not comparable to theirs. The like-for-like comparison, run in
> Experiment 009 and extended in the Distinction investigation, is given below.
*ANN-LSTM: A deep learning model for early student performance prediction in MOOC.*
*Heliyon* 9(4):e15382. DOI 10.1016/j.heliyon.2023.e15382

| | |
|---|---|
| Objective | Day-wise 4-class prediction with ANN-LSTM |
| Dataset | OULAD, 32,593 enrolments / 28,785 unique students / 22 presentations |
| Task | **4-class** D/F/P/W, Withdrawn kept (they explicitly criticise papers that drop it) |
| Prediction point | Day 1 → day 270; **day 270 = full course** |
| Features | 71 after encoding; **assessment/exam scores explicitly NOT used** — demographics + daily clickstream |
| Models | ANN-LSTM vs RNN, GRU, DFFNN, RF, AML |
| Validation | 70/30 holdout, no stratification, **no grouping** |
| Metrics @ day 270 | acc **0.72**; **macro-F1 0.66**; micro-F1 0.63; weighted-F1 0.68. Per-class F1: D 0.59, F 0.65, P 0.70, W 0.70. No AUC |
| Limitations | Class imbalance unaddressed; 43% accuracy at day 1 |

**Metric-by-metric — the LIKE-FOR-LIKE comparison.** Our matched arm is the assessment-free
replication (24 base features = demographics + clickstream only, all 9 assessment-derived and
3 score-derived features removed; same 32,593 population; verified programmatically to touch
neither `studentAssessment.csv` nor `assessments.csv`):

| Metric | Theirs | ⭐ **OFFICIAL baseline (assessment-free)** | arm: with scores | Directly valid? |
|---|--:|--:|--:|---|
| Accuracy | 0.72 | **0.739** | 0.836 | ✅ vs the official column only |
| Macro-F1 | 0.66 | **0.715** | 0.795 | ✅ — **+0.055** |
| F1 Distinction | **0.59** | 0.430 | 0.579 | ✅ — **they win** |
| F1 Fail | 0.65 | **0.779** | 0.814 | ✅ |
| F1 Pass | 0.70 | 0.709 | 0.844 | ✅ |
| F1 Withdrawn | 0.70 | **0.940** | 0.942 | ✅, but see caveat below |
| AUC | not reported | — | — | n/a |

**Only the assessment-free column may be compared to theirs.** The headline column is shown
solely to quantify what assessment data is worth (+0.123 Distinction F1 from scores; +0.025
from submission behaviour).

**Distinction, after operating-point correction.** The Distinction investigation showed the
0.166 above is largely an argmax artefact under 4.09:1 imbalance. With inverse-frequency class
weights, depth/revisit features and an inner-tuned threshold, the assessment-free arm reaches
**Distinction F1 0.430** (precision 0.33 / recall 0.60) and **macro-F1 0.715** — the
configuration now adopted as the official baseline. Al-azazi still lead on Distinction (0.59), with the opposite profile —
**precision 0.82 / recall 0.47**, i.e. a small confidently-separable subset. Whether their
advantage is representational (day-wise LSTM sequences vs our aggregates) or an artefact of
their single ungrouped split cannot be determined from published information.
Full detail: `reports/distinction_investigation/assessment_free_comparison.md`.

**Caveat on the one metric where we look strongest:** our Withdrawn advantage (0.959 vs 0.70)
is partly structural. At a course-end prediction point the Withdrawn class is *defined* by
ceasing participation, and our per-student horizon makes that signature crisp. This is an
honest modelling choice, not a leak (we never touch `date_unregistration` as a feature) — but
it should not be sold as forecasting skill. On matched assessment-free access the Withdrawn
gap is 0.956 vs 0.70 and Fail 0.768 vs 0.65 — **Fail is the defensible headline**, not
Withdrawn and not macro-F1 (which is a near-tie at +0.016).

**Verdict: mixed, but favourable on balance.** On matched assessment-free access we lead on accuracy
(+0.09), Fail (+0.12), Pass (+0.11) and Withdrawn (+0.26), lead on macro-F1 (+0.055), and
**lose on Distinction** (0.430 vs their 0.59) — under stricter validation
throughout. Not "same constraints, better result": we lead on three classes and lose the
fourth.

### 2.2 Althibyani (2024)
*Predicting student success in MOOCs.* *PeerJ Computer Science* 10:e2221. DOI 10.7717/peerj-cs.2221

4-class + 2-class; full 32,593; test = 8,149 rows; **uses assessment `score` as a core
feature**; LR + RF; **random 75/25 split**, no grouping. Reports **weighted F1 only**:
RF 4-class acc 0.746 / weighted-F1 0.742; LR 0.721 / 0.706. Binary RF 0.957.
**Macro-F1 derived from their published confusion matrices: RF 0.706, LR 0.656** (per-class
RF F1: W 0.792, F **0.519**, P 0.840, D 0.672).

| Metric | Theirs (RF) | Ours | Directly valid? |
|---|--:|--:|---|
| Accuracy (4-class) | 0.746 | **0.8506** | **Partly** — same task/population, but they use assessment scores (easier) while we use grouped CV (harder). Two opposing biases; not clean |
| Weighted-F1 | 0.742 | **0.8473** | Same caveat |
| Macro-F1 | 0.706 (derived) | **0.7976** | Same caveat |
| F1 Fail | 0.519 | 0.808 | Yes — **our single largest advantage** |
| F1 Distinction | 0.672 | 0.581 | **They win** — almost certainly *because* assessment scores make Distinction separable |
| Binary acc | 0.957 | (Exp 006b at-risk 0.937) | **No** — different binary definition and our binary is an archived experiment, not the published headline |

Their own stated limitation — *"similar classes such as withdrawn and fail and pass and
distinction are often confused"* — is precisely the failure mode our result addresses.

**Verdict: improvement on the hard classes; not a clean win overall** because their score
features and our grouped CV pull in opposite directions. Report it as such.

### 2.3 Shou, Xie, Mo & Zhang (2024)
*Predicting Student Performance in Online Learning: A Multidimensional Time-Series Data
Analysis Approach.* *Applied Sciences* 14(6):2522. DOI 10.3390/app14062522

MTAPSP (LSTM + multi-head attention + ANN); full 32,593; **uses assessment scores**; random
80/20, no grouping. Usefully reports macro **and** weighted separately:
acc 0.74, **macro-F1 0.67**, weighted-F1 0.73. Per-class F1: D 0.47, F 0.56, P 0.84, W 0.80.
RF baseline: acc 0.72, macro-F1 0.67.

| Metric | Theirs | Ours | Directly valid? |
|---|--:|--:|---|
| Accuracy | 0.74 | **0.8506** | Partly (they use scores; we use grouped CV) |
| Macro-F1 | 0.67 | **0.7976** | Partly, same reason |
| F1 Fail | 0.56 | 0.808 | Yes |
| F1 Distinction | 0.47 | 0.581 | Yes |

**Verdict: improvement**, with the same two-sided-bias caveat.

### 2.4 Junejo et al. (2025) — the claim to rebut, not to beat
*Scientific Reports* 15:16251. DOI 10.1038/s41598-025-00256-3

4-class, full population, 1D-CNN, stratified 80/20, class weights (no SMOTE — the resampling
is fine). Reports acc **0.98**, macro-F1 **0.98**, macro-AUC 0.9958.

**Not comparable — three independent red flags:** (i) target leakage via `total_reg_days`
(§1); (ii) macro-F1 ≡ weighted-F1 ≡ 0.98 at 9.3% Distinction prevalence, which is a signature
of a degenerate evaluation; (iii) their own RF baseline scores 0.66 on the same features — a
32-point gap to a small CNN on 15 tabular features is not a credible architecture effect.

**Do not claim to beat this. Rebut it** with the F1 = 0.9950 null-check and the 5%-of-course
trajectory, then cite Al-azazi's leak-free 72% as the controlled contrast.

### 2.5 Adnan et al. (2021)
*IEEE Access* 9:7519–7539. DOI 10.1109/ACCESS.2021.3049446 *(abstract/secondary sources; full
text not retrievable — IEEE returned HTTP 418)*

Starts 4-class, **collapses to binary** ({W+F} vs {P+D}) for headline results at 0/20/40/60/
80/100% of course length; uses assessment scores; no grouping. Binary RF at 100%: precision
0.92, recall/F1/accuracy 0.91. Genuine **4-class DFFNN end-of-course accuracy: 0.72** (43%
demographics only → 63% +clickstream → 71% +assessment → 72% all).

| Metric | Theirs | Ours | Directly valid? |
|---|--:|--:|---|
| 4-class accuracy | 0.72 | **0.8506** | **Yes** — same task and cutoff (they use scores, we don't) |
| Binary 0.91 | 0.91 | — | **No** — routinely mis-cited as multiclass; different task |

**Verdict: improvement on the genuine 4-class number.** Be explicit that the widely-quoted
0.91 is binary.

---

## 3. Tier 2 — important but **not** metric-comparable

| Paper | Why not comparable | Their headline |
|---|---|---|
| **Lê, Abel & Laforge 2026** — LEAP, arXiv:2605.25794 | **Binary** (P+D vs F+W) at weekly cutoffs ≤ week 8 — early prediction, not full-course | ROC-AUC 0.7151 (wk 1) → **0.8602** (wk 8); leakage ablation: 0.7151 strict → **0.9669 leaky** |
| **Hlosta et al. 2017** — Ouroboros, LAK'17 | **Different target entirely**: will the student submit assessment A1 by its deadline | PR-AUC 0.78 at deadline → ~0.35 eleven days out |
| **Hlosta et al. 2018**, *Knowledge-Based Systems* 160:278–295 | Same A1-submission target; reports *loss* vs a self-test gold standard, not raw AUC | best PR-AUC loss 0.1714 (RF + SMOTE-ENN) |
| **Wolff et al. 2013**, LAK'13 | Final outcome **but** uses TMA scores; 10-fold CV, not temporal; students with no VLE activity filtered out (survivorship) | precision 0.62–0.70, recall 0.23–0.37 |
| **Jha et al. 2019**, CSEDU | **Two sequential binaries** (dropout, then pass/fail); AUC only. Notably **excludes `exam_score`** — precedent for our exclusion | AUC 0.91 dropout / 0.93 result (GBM) |
| **Tertulino & Almeida 2026**, arXiv:2508.18316 | **Drops Withdrawn entirely** (n = 22,437, reproducible); federated learning | ROC-AUC 0.86 centralised / 0.84 federated |
| **Qiu et al. 2022**, *Sci Rep* 12 | **Silently folds Withdrawn into "failed"** (reverse-engineered from N = 6,272 = DDD module); single module | acc 95.4–97.4%, F1 0.969–0.982 |
| **Hao et al. 2022**, *Educ. Inf. Technol.* 27 | **Grade-band task** (fail/pass/good/distinction) derived from scores; **Withdrawn excluded** | avg F1 ≈ 0.866 (averaging unstated) |
| **EL Habti et al. 2025**, *IJIET* 15(2) | **3-class** (Withdrawn dropped, n = 4,950); uses **`Score_exam`** — severe leakage | RF acc 0.91, macro-F1 0.86 |
| **Ismanto et al. 2024**, CLESS | **Binary Withdrawn vs Pass with Fail dropped**; SMOTE appears applied before splitting | LSTM+SMOTE 94.90% / F 0.96 at week 25 |
| **Winarsih et al. 2025**, JSINBIS 15(2) | Binary dropout, **demographics only**, no VLE; internally inconsistent metrics (F1 94.45% vs P 88.75/R 93.86) | RF acc 89.37% |
| **Torkhani & Rezgui 2025**, ICODAI | **AAA module only, n = 748** (2.3% of OULAD); uses exam scores; label-encoding imputation | LSTM acc 83.41% |
| **Shayegan & Akhtari 2024**, *CSSE* 48(5) | Binary; **uses exam grades**; 98% accuracy with AUC 0.90 is internally inconsistent | acc 98%, AUC 0.90 |
| **Balachandar & Venkatesh 2024**, *MethodsX* | **OULAD blended with UCI + EdNet** — not OULAD-specific; N never stated | acc 76%, macro-F1 0.73, AUC 0.82 |
| **Martínez-Carrascal et al. 2023**, *IRRODL* 24(1) | **Survival analysis, demographics only, no predictive metrics at all** | hazard ratios only |
| **da Silva et al. 2026**, arXiv:2604.08870 / :08874 | **Survival / discrete-time hazard**; different target and metric family | row-level AUC 0.8396, ECE 0.0012; TD-concordance 0.51–0.60 |
| **Ameri et al. 2016**, CIKM'16 | **Not OULAD** (Wayne State records, semester granularity) | AUC 0.742 → 0.844 across semesters |

---

## 4. Master comparison table

| Paper | Dataset | Task | Models | Validation | Reported | Ours | Strengths | Weaknesses |
|---|---|---|---|---|---|---|---|---|
| **Ours** | OULAD 32,593 / 29,496 | 4-class + regression | LR/DT/RF/**XGB** | **StratifiedGroupKFold-5, grouped by student** | **acc .851, macro-F1 .798** | — | Only grouped eval; exams excluded; censored at withdrawal; leakage quantified | Withdrawn partly structural; single institution; no intervention arm |
| Al-azazi & Ghurab 2023 | OULAD 32,593 | 4-class | ANN-LSTM | 70/30, no grouping | acc .72, macro-F1 .66, **D F1 .59** | **.812 / .676** (assessment-free arm; headline .851/.798 uses scores) | Genuinely assessment-free; keeps Withdrawn; day-wise sequences; **beats us on Distinction** | No grouping; single split; imbalance unaddressed; features never enumerated |
| Althibyani 2024 | OULAD 32,593 | 4-class + binary | LR, RF | random 75/25 | acc .746, weighted-F1 .742 (macro .706 derived) | .851 / .847 wtd | Publishes confusion matrices (reusable) | Uses scores; no grouping; reports weighted F1 unlabelled |
| Shou et al. 2024 | OULAD 32,593 | 4-class | MTAPSP, RF, DFFNN | random 80/20 | acc .74, macro-F1 .67 | .851 / .798 | Reports macro **and** weighted; time-series design | Uses scores; no grouping |
| Junejo et al. 2025 | OULAD 32,593 | 4-class | 1D-CNN | stratified 80/20 | acc .98, macro-F1 .98 | — (rebut) | Early-cutoff curve published | **Target leakage** via `total_reg_days`; implausible baseline gap |
| Adnan et al. 2021 | OULAD | 4-class → binary | RF, DFFNN, SVM | no grouping | 4-class acc .72; binary .91 | .851 | Percent-of-course design | Headline is binary; uses scores |
| Lê et al. 2026 (LEAP) | OULAD 32,593 | binary, ≤wk 8 | RF, GBDT | stratified 80/20 × 5 seeds | AUC .8602 @wk8 | n/a | **Best leakage protocol in the field**; quantified ablation | Row-level split — ignores student identity |
| Ouroboros 2017 | OULAD 4 courses | A1 submission | XGB, RF, SVM | temporal, prev-presentation | PR-AUC .78 → .35 | n/a | Gold-standard cutoff discipline | Different target |
| Jha et al. 2019 | OULAD | 2 × binary | GBM, DRF, DL | 10-fold + holdout | AUC .91 / .93 | n/a | **Excludes exam_score** explicitly | AUC only; aggregate features |
| Qiu et al. 2022 | OULAD DDD (6,272) | binary | 6 algorithms | 70/30 | acc 95–97% | n/a | High citation count | **Withdrawn silently merged into Fail**; undisclosed |
| EL Habti et al. 2025 | OULAD 4,950 | 3-class | RF, LR, SVM, LDA | random 80/20 | acc .91, macro-F1 .86 | n/a | Full per-class tables | **Uses final exam score**; drops Withdrawn |
| Ismanto et al. 2024 | OULAD 32,593 | binary (Fail dropped) | LSTM/FFNN + SMOTE | 10-fold, no grouping | 94.90% @wk25 | n/a | Genuine early cutoffs | Fail class removed; SMOTE likely pre-split |
| Torkhani & Rezgui 2025 | OULAD AAA (748) | binary | LSTM, RF | 70/30 | acc 83.41% | n/a | — | 2.3% of dataset; exam scores; leaky imputation |

---

## 5. Critical self-assessment

**Where we are stronger**
1. **Validation.** We are the only OULAD study located that groups by student. With 32,593
   enrolments over 28,785 unique students, ~3,800 repeat enrolments are scattered across
   train and test by every other paper's random split. None mentions the issue.
2. **Feature discipline.** Exams excluded entirely; `date_unregistration` never a feature.
   Note our headline model **does** use coursework scores and submission behaviour — it is
   *score-free of exams*, not assessment-free. Against the genuinely assessment-free
   comparator (Al-azazi) the valid contrast is our assessment-free arm: we lead on macro-F1
   (0.715 vs 0.66) and accuracy (0.739 vs 0.72) but **lose on Distinction** (0.430 vs 0.59).
3. **The hard classes.** Fail F1 0.808 vs 0.519–0.65 across Tier 1 — the boundary where
   prior confusion matrices collapse, as two of those papers concede themselves.
4. **Quantified corrections.** Survivorship (31% of the at-risk population at day 30),
   evaluation leakage (15.1% of test rows), and the `date_unregistration` leak (F1 0.9950)
   are measured, not asserted.
5. **Reporting hygiene.** We report macro *and* weighted F1, both populations, per-class
   metrics, fold SDs, and a full ablation. Most of this literature reports one unlabelled F1.

**Where we are weaker**
1. **Withdrawn is partly structural, not skill.** At course end the class is defined by
   ceasing participation. Al-azazi gets 0.70 without our per-student anchoring; we get 0.959.
   Some of that gap is anchoring design, not better learning.
2. **Distinction is our weakest class and we lose it outright when matched.** Our headline
   0.581 uses coursework scores; Al-azazi reach 0.59 with **no assessment data at all**. On
   matched access the official baseline gets 0.430 vs their 0.59.
   Papers using assessment scores also beat us (Althibyani 0.672). Excellence is the one
   boundary where we do not lead.
3. **No deep-learning comparison at equal footing.** We use XGBoost; MTAPSP, ANN-LSTM and
   1D-CNN are untested under our protocol. A reviewer may ask whether our gain is protocol
   or model.
4. **Single institution, no intervention arm, no external validity test.**
5. **`is_banked`** affects 522 enrolments (1.8%, spread across all classes) — not leakage
   (the score predates the presentation and is knowable), but a disclosed data-quality note.

**Improvement or merely different design?** — **Both, and the distinction matters per
comparator.** Against Al-azazi & Ghurab, on matched assessment-free access, it is a
**modest improvement on aggregate metrics** (+0.09 accuracy, **+0.016 macro-F1 — essentially
a tie**) under stricter validation, but a **loss on Distinction** (.433 vs .59). The
previously claimed "+0.14 macro-F1 under equal constraints" compared our score-using model
against their score-free one and is withdrawn. Against Althibyani/Shou/Adnan it is **partly a different
experimental design** — they buy performance with assessment scores, we buy validity with
grouped CV and exam exclusion; the comparison has two opposing biases and should be
presented that way rather than as a clean win. Against Junejo it is **not a comparison at
all** but a refutation.

---

## 6. Reviewer verdict

**Genuinely novel**
- **Student-grouped evaluation on OULAD.** Not one of ~25 papers does it. The strongest
  framing: LEAP (2026) is the field's most rigorous leakage protocol and *still* uses a
  row-level split — **LEAP controls *when* you look; grouping controls *who* you look at.**
  We are the missing half of an argument the field has just begun making.
- **Quantified demonstration that `date_unregistration` is a label proxy (F1 0.9950)**, with
  a named, published, high-visibility paper shown to depend on it.
- **The combined correction stack** (survivorship + temporal + identity leakage) measured
  end-to-end on one dataset, with each component's effect isolated.

**Incremental**
- The 4-class task, the model family (XGBoost), the feature families (clickstream, submission
  timing, cohort ranks), and full-course prediction are all well-trodden.
- Exam exclusion is precedented (Jha et al. 2019 excluded `exam_score`; Ouroboros never used
  scores). Our contribution is enforcement and quantification, not the idea.
- The regression (R² 0.334) is squarely mid-range for behaviour-only grade prediction.

**Claim confidently**
1. Highest *defensible* 4-class macro-F1 on OULAD (0.798 vs best credible published 0.706 —
   Althibyani, who like us uses assessment scores), achieved under stricter **validation**.
   Do not add "stricter feature constraints" to this claim: our headline uses coursework
   scores. Against genuinely assessment-free work our official baseline is macro-F1 0.715 vs their 0.66.
2. First student-grouped evaluation of OULAD; quantified identity-leakage exposure.
3. `date_unregistration` as a label proxy, and that at least one published headline depends
   on it.
4. Substantial gains on the Fail/Withdrawn boundary specifically — and on matched
   assessment-free access too (Fail 0.779 vs 0.65, Withdrawn 0.940 vs 0.70).
5. Full-course leakage-free performance as an *information ceiling*, contrasted against the
   early-prediction curve (day 30: 0.527/0.396 → day 140: 0.693/0.514).

**Avoid claiming**
1. ❌ "State of the art" without qualification — Junejo reports 0.98. Rebut, don't ignore.
2. ❌ First to address leakage in OULAD — Ouroboros (2017) and LEAP (2026) precede us.
3. ❌ That Withdrawn F1 0.96 is early-warning capability — it is course-end description.
4. ❌ Beating Ouroboros/Herrmannova/LEAP — different targets and horizons.
5. ❌ Any intervention or causal benefit — no intervention arm exists in OULAD.
6. ❌ Superiority over deep learning — we did not run MTAPSP/ANN-LSTM under our protocol.
7. ❌ **That our headline model is "score-free" or "assessment-free"** — it uses coursework
   scores *and* submission behaviour. Use three distinct terms with their own numbers:
   with-scores arm **macro-F1 0.795**, assessment-behaviour arm **D-F1 0.457**, official
   assessment-free baseline **macro-F1 0.715 / D-F1 0.430**.
8. ❌ **That we beat Al-azazi on Distinction** — we do not (0.430 vs 0.59 on matched access).
   State it plainly; it is the one class where the field leads us.

**The paper to write:** not "we got a higher number," but **"most published OULAD performance
is not comparable to itself, and here is a protocol under which it becomes so."** The rigour
*is* the contribution, and this literature makes rigour look novel.

---

## 7. Coverage limitations of this review

- **IC2IT proceedings:** no OULAD-specific student-outcome papers surfaced. Reported as an
  honest negative rather than stretched to fill the requested slot.
- **Not retrieved:** Waheed et al. (2020) *Computers in Human Behavior* 104:106189 (544
  citations — the most-cited OULAD paper; OA mirror returns 403) and the full text of Adnan
  et al. (2021) (IEEE returned HTTP 418; figures above are from abstract and secondary
  sources). Both should be obtained before submission.
- Several regional-journal papers report headline numbers only in unlabelled bar charts,
  making per-model figures unrecoverable from the published text.
