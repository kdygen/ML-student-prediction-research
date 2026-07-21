# At-Risk Student Identification — Model Results Briefing

*A behavioural early-warning model that flags students likely to **fail or withdraw**, using
only their engagement and coursework activity (no exam data). All figures below are measured on
a held-out set of students the model never trained on, using student-grouped validation (no
student appears in both training and test). Results are from the Open University Learning
Analytics Dataset (~29,500 course enrolments).*

---

## 1. The headline numbers

- **At one month into a course, the model's top-5% highest-risk list is 97.8% correct** —
  nearly every student flagged is genuinely heading for failure or withdrawal.
- By the 3-month mark the top-5% list is essentially **100% accurate**.
- **Median warning time: ~58 days** between the first risk flag and a student actually
  withdrawing — roughly two months to intervene.
- Overall ability to separate at-risk from not-at-risk students: **AUC 0.98** (full course);
  **0.79 at one month, rising to 0.91** by the end.
- **Risk scores are calibrated** — a student the model scores at "72% risk" really does end up
  at-risk about 72% of the time (calibration error under 2%). The number can be trusted as a
  probability, not just a ranking.

---

## 2. Budget vs. reach vs. warning time — the core planning table

The practical question for any advising team is: *"We can only contact so many students. How
many of the truly at-risk ones will we actually reach, and how much warning do we get?"*

Reference group: 5,635 students; **46% eventually at-risk** (fail or withdraw), of whom
1,193 eventually withdraw.

| Outreach capacity | When you contact | Hit rate (of those contacted, % truly at-risk) | % of ALL at-risk students reached | % of ALL withdrawing students reached | Median warning (days before they drop out) |
|---|---|--:|--:|--:|--:|
| **10%** | Month 1 only | 94% | 21% | **23%** | 50 |
| | Month 1 + Month 3 (split) | 98% | 22% | 16% | 47 |
| | End of course | 100% | 22% | 7% | 40 |
| **15%** | Month 1 only | 91% | 30% | **32%** | 63 |
| | Month 1 + Month 3 (split) | 97% | 32% | 26% | 49 |
| | End of course | 98% | 32% | 12% | 36 |
| **20%** | Month 1 only | 85% | 37% | **39%** | 64 |
| | Month 1 + Month 3 (split) | 93% | 41% | 34% | 50 |
| | End of course | 93% | 41% | 16% | 38 |

**How to read this:**
- "Outreach capacity" = the share of enrolled students your team can realistically contact.
- "Hit rate" = of the students you contact, how many were genuinely at-risk (i.e. not wasted
  effort).
- The last two columns are the payoff: **how many of the students who actually fail/withdraw
  you manage to reach**, and how much lead time you get.

**The key business insight:** *acting early reaches far more of the students who withdraw.* At
every budget level, contacting at Month 1 reaches **2–3× more withdrawing students** than
waiting until late in the course — because students who withdraw physically leave and become
unreachable. Waiting buys slightly cleaner lists but loses the very people you're trying to
save.

**Recommended operating policy:** a **split campaign — most outreach at Month 1, a top-up at
Month 3.** It keeps a high hit rate (~93–98%), reaches almost as many at-risk students overall
as a late list, but roughly **doubles the number of withdrawing students caught** and adds ~10
days of warning time versus a late one-shot.

---

## 3. Why timing matters — the shrinking window

A student can only be helped while still enrolled. This is how the reachable pool shrinks as a
course progresses:

| Point in course | % of eventual at-risk students still reachable | % of eventual withdrawals still reachable |
|---|--:|--:|
| 2 weeks in | 100% | 100% |
| 1 month in | 95% | 89% |
| 2 months in | 85% | 68% |
| 3 months in | 78% | 52% |
| End of course | 67% | **30%** |

By the end of a course, **70% of the students who withdraw have already left** — no model,
however accurate, can help them anymore. The prediction gets more accurate over time, but the
opportunity to act disappears faster. **The earliest point where the model is both accurate and
still leaves room to act is around the one-month mark.**

---

## 4. The most powerful signal: rising risk

The model tracks how each student's risk *changes* over time, and this turns out to be the
single strongest indicator:

| Student pattern | How they end up at-risk (fail or withdraw) |
|---|--:|
| **Risk rising** (looked fine early, then climbed) | **99.3%** |
| High risk throughout | 97.3% |
| Risk appearing late | 94.2% |
| On-and-off | 66% |
| Risk fell over time (recovering) | 42% |
| Consistently low risk | 22% |

**Students whose risk is *climbing* end badly 99% of the time — even more reliably than
students who looked bad from day one.** Practically: a student whose risk score jumps sharply
between check-ins is 68–80% likely to be at-risk, versus 31–38% for everyone else. **A rising
trajectory should trigger immediate outreach, not just a high absolute score.**

---

## 5. Other statistics worth sharing

- **Withdrawal is highly detectable:** the model separates students who withdraw from everyone
  else with AUC 0.996 — but this signal builds over time; it's descriptive at course end and
  genuinely predictive earlier.
- **Early data is thin:** at 2 weeks, prediction is weak (the model needs roughly a month of
  behaviour before it's reliable). This is honest and worth knowing — very-early alerts should
  be treated as provisional.
- **Effort predicts achievement, but only partly:** using behaviour alone (no grades as
  inputs), the model predicts a student's final coursework score with a ~10-point average
  error, explaining about a third of the variation. Engagement matters, but it isn't destiny.
- **What actually drives the risk score** (most to least important): whether the student is
  *doing the work* (submissions), *recent engagement* (are they still active?), and *study
  consistency* (steady weekly activity beats cramming). These are intuitive and explainable to
  an advisor.
- **The score is honest about uncertainty** — it's calibrated, so it supports "act if risk >
  X%" style rules directly, and can be shown to staff as a real probability.

---

## 6. Important honesty note (worth stating up front)

These results measure **how well the model targets and times outreach** — who to contact, when,
and how much warning you get. They do **not** prove that contacting a student *changes* their
outcome, because this dataset contains no intervention experiment. Demonstrating actual
outcome improvement would require a live pilot (ideally a controlled A/B test) — which is the
natural, high-value next step and something worth building together.

A second point that reflects well on rigour: much of the published work in this area reports
inflated accuracy because it accidentally uses information that gives the answer away (for
example, a student's un-enrolment date, which by definition only exists for students who
withdrew). This model was built specifically to avoid those traps — every result above is
leakage-free and validated so that no student's data appears in both training and testing. The
numbers are conservative and real.
