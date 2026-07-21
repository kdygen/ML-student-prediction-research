# Literature — Pass vs Distinction Separation Without Assessment Scores

Web review (full texts verified where accessible; access failures flagged). Question: has
anyone separated Distinction from Pass using only demographics + behaviour?

## Best available score-free Distinction results on OULAD

| Study | Features | Validation | Distinction result | Credibility |
|---|---|---|---|---|
| **Al-azazi & Ghurab 2023** (*Heliyon*, [PMC10119763](https://pmc.ncbi.nlm.nih.gov/articles/PMC10119763/)) | demographics + daily clickstream sequences (verified score-free) | 70/30, unit unstated, **no student grouping** | **F1 0.59** (P 0.82 / R 0.47) at day 270, ANN-LSTM | Moderate-to-weak: genuinely score-free, but single ungrouped split; captures only ~half of Distinctions at high precision |
| **Borna et al. 2024** (*Frontiers in Education*, [link](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2024.1421479/full)) | click-derived engagement only, OULAD 29,570 | 3-fold CV, grouping unstated | **F1 0.15** (P 0.55 / R 0.09), RF | The independent replication of our collapse. Verbatim: interaction data *"does not fully encompass qualities associated with top academic performance"* |
| **Ours (this investigation)** | 33→39 score-free features | **StratifiedGroupKFold-5 by student** | **F1 0.458** (P 0.36 / R 0.63) | Only student-grouped result in this set |
| Junejo et al. 2025 (*Sci Rep*) | "without assessment" ablation | row-level stratified split | claims F1 0.94 | **Invalid** — `total_reg_days` from `date_unregistration` (label proxy, our F1-0.995 null-check), row-level split |
| Shou et al. 2024; Hao et al. 2022 | **use assessment scores** | — | D ≈ 0.47–0.87 with scores | Not score-free; irrelevant to this question |
| Adnan et al. 2021 (*IEEE Access*) | ablation: demo 43% → +clicks 63% → +assessment 71–72% (4-class acc) | ungrouped | per-class D not retrievable | The +8pp assessment jump is the aggregate version of our Distinction collapse |

**Published score-free range: F1 0.15–0.59; none under student-grouped validation.** Our
0.458 grouped is the strongest rigorously-validated number in this landscape.

## Field-wide avoidance as evidence

Most OULAD studies **merge Distinction into Pass** or drop it (documented by the AIED 2024
OULAD systematic review, [10.1007/978-3-031-64315-6_46](https://link.springer.com/chapter/10.1007/978-3-031-64315-6_46),
and the 2024 deep-learning-for-VLE review). The Open University's own deployed system
(OU Analyse) predicts at-risk only. Riestra-González et al. 2021 (*C&E*) target "excellent"
students from Moodle logs but only as one-vs-rest, never excellent-vs-pass. The field's
revealed preference is that this boundary is known-hard.

## Behavioural signatures of high achievement (transferable ideas → what we did)

| Literature idea | Source | Our test | Outcome |
|---|---|---|---|
| Regularity of study time | Boroujeni et al., EC-TEL 2016 | H1 group (weekly CV, entropy, streaks, phase) | no AUC gain |
| Spacing vs cramming | *Learning & Instruction* 2021; *npj Sci Learn* 2020 | H4 group (gaps, slope) | no AUC gain |
| Procrastination/proactivity | Yilmaz 2015; *Soft Computing* 2020 | H3 + existing submit-lead features | no AUC gain beyond existing |
| **Revisiting material (SRL)** | **Kizilcec et al. 2017, *C&E*** | **H2 group (revisit ratio, unique sites, depth)** | **only group that helped: +0.012 AUC** |
| Cohort-relative trajectories | Matcha/Gašević LAK 2020 (tactics) | H5 group (weekly z, rank slope) | no AUC gain |
| Sequence representations | Al-azazi's daily LSTM | **not tested** (pinned env; aggregate pipeline) | flagged as the one untested lever |
| Forum text quality | Wen & Yang | impossible — OULAD has no post text | n/a |

## Evidence for the information ceiling

- **Tempelaar et al. 2015** (CSEDU): "basic LMS data did not substantially predict learning";
  formative assessment scores dominate.
- **Conijn et al. 2017** (IEEE TLT 10(1)): across 17 courses, LMS data explain 8–37% of grade
  variance; assessment grades add most value; LMS-only models unstable.
- **Adnan 2021 ablation** and **Borna 2024** (above).
- Theoretical framing: Withdrawn/Fail are decided by *presence and persistence* (directly
  measured by clickstreams); Distinction vs Pass is decided by *quality of produced work*,
  only weakly mediated by observable platform behaviour.

## Defensible claim (with the required qualification)

> **Aggregate behavioural features cannot reliably separate Pass from Distinction in OULAD**
> (our grouped-CV ceiling: AUC ≈ 0.75, F1 ≈ 0.46; independent replication of the collapse in
> Borna 2024). Fine-grained temporal representations may recover a **high-precision,
> low-recall subset** at best (Al-azazi: P 0.82 / R 0.47 under weaker validation). No
> published work contradicts this; the field's default of merging the classes corroborates it.
