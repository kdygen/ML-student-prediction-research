"""Experiment 006 — full-course leakage-free dataset builder.

Prediction point: end of teaching, BEFORE any final-exam information exists.
Censoring rule: for every enrolment with a date_unregistration, only data with
  VLE date <= unreg and submission date_submitted <= unreg is used ("no information
  generated after withdrawal"). Exam assessments are excluded entirely (scores,
  submissions, and attendance — exam sitting is a near-perfect outcome encoder:
  1/10156 Withdrawn sat an exam).
date_unregistration is used ONLY as a censor boundary, never as a feature.

Population: all 32,593 registered (student, module, presentation) pairs.
"""
import json, os, time
import numpy as np
import pandas as pd

REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
SP = os.environ["SP"]
KEY = ["id_student", "code_module", "code_presentation"]
t0 = time.time()

si = pd.read_csv(f"{REPO}/data/raw/studentInfo.csv")
sr = pd.read_csv(f"{REPO}/data/raw/studentRegistration.csv")
sa = pd.read_csv(f"{REPO}/data/raw/studentAssessment.csv")
asr = pd.read_csv(f"{REPO}/data/raw/assessments.csv")
vle = pd.read_csv(f"{REPO}/data/raw/vle.csv")
sv = pd.read_csv(f"{REPO}/data/raw/studentVle.csv")
for df, c in [(sr, "date_registration"), (sr, "date_unregistration"), (asr, "date"),
              (sa, "date_submitted"), (sv, "date")]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
sa["score"] = pd.to_numeric(sa["score"], errors="coerce")

audit = {}

# ---------- population ----------
pop = si[KEY + ["final_result", "highest_education"]].merge(
    sr[KEY + ["date_registration", "date_unregistration"]], on=KEY, how="left")
assert len(pop) == 32593 and pop.duplicated(KEY).sum() == 0
audit["population"] = {"n": int(len(pop)),
                       "never_started_unreg_le0": int((pop["date_unregistration"] <= 0).sum()),
                       "withdrawn_missing_unreg": int(((pop["final_result"] == "Withdrawn")
                                                       & pop["date_unregistration"].isna()).sum())}

# course span per presentation (end of observed activity = course end)
span = sv.groupby(["code_module", "code_presentation"])["date"].max().rename("course_end").reset_index()
pop = pop.merge(span, on=["code_module", "code_presentation"], how="left")

# ---------- censoring: drop post-unregistration data ----------
sv = sv.merge(sr[KEY + ["date_unregistration"]], on=KEY, how="left")
n_before = len(sv)
sv = sv[sv["date_unregistration"].isna() | (sv["date"] <= sv["date_unregistration"])]
audit["vle_rows_dropped_post_unreg"] = int(n_before - len(sv))

sa2 = sa.merge(asr[["id_assessment", "code_module", "code_presentation",
                    "assessment_type", "date", "weight"]], on="id_assessment", how="left")
n_exam_rows = int((sa2["assessment_type"] == "Exam").sum())
sa2 = sa2[sa2["assessment_type"] != "Exam"]              # EXCLUDE EXAMS ENTIRELY
sa2 = sa2.merge(sr[KEY + ["date_unregistration"]], on=KEY, how="left")
n_before = len(sa2)
sa2 = sa2[sa2["date_unregistration"].isna() | (sa2["date_submitted"] <= sa2["date_unregistration"])]
audit["exam_rows_excluded"] = n_exam_rows
audit["cw_submissions_dropped_post_unreg"] = int(n_before - len(sa2))

# ---------- VLE behavioural features (censored) ----------
svm = sv.merge(pop[KEY + ["course_end"]], on=KEY, how="inner")
daily = svm.groupby(KEY + ["date"])["sum_click"].sum().reset_index()
g = daily.groupby(KEY)
beh = g.agg(active_days=("date", "count"), sum_click=("sum_click", "sum"),
            first_day=("date", "min"), last_day=("date", "max")).reset_index()
b2 = daily.groupby(KEY)["sum_click"].agg(["mean", "std"]).reset_index()
b2["burstiness"] = b2["std"] / (b2["mean"] + 1)
beh = beh.merge(b2[KEY + ["burstiness"]], on=KEY, how="left")

# activity-type ratios over the full (censored) course
svt = svm.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
act = svt.groupby(KEY + ["activity_type"])["sum_click"].sum().unstack(fill_value=0)
tot = act.sum(axis=1)
ratios = pd.DataFrame(index=act.index)
for a in ["resource", "oucontent", "homepage", "forumng", "quiz"]:
    ratios[f"{a}_ratio"] = (act[a] / tot.clip(lower=1)) if a in act.columns else 0.0
ratios = ratios.reset_index()

# recency windows anchored at each student's own horizon end h = min(unreg, course_end)
pop["horizon"] = pop[["date_unregistration", "course_end"]].min(axis=1)
dh = daily.merge(pop[KEY + ["horizon", "course_end"]], on=KEY, how="left")
win = {}
for w in range(1, 5):
    m = (dh["date"] >= dh["horizon"] - (7 * w - 1) - 7 * (w - 1) * 0) & True  # placeholder
win = dh.assign(
    w1=((dh["date"] >= dh["horizon"] - 6) & (dh["date"] <= dh["horizon"])) * dh["sum_click"],
    w2=((dh["date"] >= dh["horizon"] - 13) & (dh["date"] <= dh["horizon"] - 7)) * dh["sum_click"],
    w3=((dh["date"] >= dh["horizon"] - 20) & (dh["date"] <= dh["horizon"] - 14)) * dh["sum_click"],
    w4=((dh["date"] >= dh["horizon"] - 27) & (dh["date"] <= dh["horizon"] - 21)) * dh["sum_click"],
    pre=(dh["date"] < 0) * dh["sum_click"],
    decay=np.exp(-(dh["course_end"] - dh["date"]) / 7.0) * dh["sum_click"],
    third1=(dh["date"] <= dh["course_end"] / 3) * dh["sum_click"],
    third3=(dh["date"] >= 2 * dh["course_end"] / 3) * dh["sum_click"],
).groupby(KEY)[["w1", "w2", "w3", "w4", "pre", "decay", "third1", "third3"]].sum().reset_index()
win.columns = KEY + ["w1_clicks", "w2_clicks", "w3_clicks", "w4_clicks",
                     "precourse_clicks", "decay_clicks", "clicks_first_third", "clicks_last_third"]

# max inactivity gap + active weeks (within observed activity span)
def gaps(dates):
    d = np.sort(dates.values)
    return float(np.diff(d).max()) if len(d) > 1 else 0.0
gp = daily.groupby(KEY)["date"].agg(max_gap=gaps,
                                    active_weeks=lambda s: s.floordiv(7).nunique()).reset_index()

# assessment_focus: clicks within 7 days before any coursework deadline / total clicks
dl = asr[(asr["assessment_type"] != "Exam") & asr["date"].notna()][
    ["code_module", "code_presentation", "date"]].drop_duplicates()
dm = daily.merge(dl.rename(columns={"date": "deadline"}), on=["code_module", "code_presentation"], how="left")
dm["near"] = (dm["date"] >= dm["deadline"] - 7) & (dm["date"] <= dm["deadline"])
focus_clicks = dm[dm["near"]].drop_duplicates(subset=KEY + ["date"]).groupby(KEY)["sum_click"].sum().rename("focus_clicks")
foc = beh[KEY + ["sum_click"]].merge(focus_clicks, on=KEY, how="left")
foc["assessment_focus"] = foc["focus_clicks"].fillna(0) / foc["sum_click"].clip(lower=1)

# ---------- coursework features (censored, exam-free) ----------
sc = sa2.dropna(subset=["score"]).copy()
wa = sc.groupby(KEY).apply(
    lambda g: (g["score"] * g["weight"]).sum() / g["weight"].sum() if g["weight"].sum() > 0
    else g["score"].mean(), include_groups=False).rename("weighted_average").reset_index()
sub = sa2.copy()
sub["lead"] = sub["date"] - sub["date_submitted"]     # deadline - submitted (positive = early)
sub["late"] = (sub["date_submitted"] > sub["date"]).astype(int)
agg = sub.groupby(KEY).agg(submitted_count=("id_assessment", "count"),
                           mean_submit_lead=("lead", "mean"),
                           min_submit_lead=("lead", "min"),
                           late_submissions=("late", "sum"),
                           first_submit_day=("date_submitted", "min"),
                           n_assess_types_submitted=("assessment_type", "nunique")).reset_index()
# completion ratio (count-based; robust to zero-weight modules like GGG)
n_cw = asr[asr["assessment_type"] != "Exam"].groupby(
    ["code_module", "code_presentation"])["id_assessment"].count().rename("n_cw_total").reset_index()
agg = agg.merge(n_cw, on=["code_module", "code_presentation"], how="left")
agg["completion_ratio_cw"] = agg["submitted_count"] / agg["n_cw_total"].clip(lower=1)

# score trajectory (chronological by deadline): slope + std   [score-derived]
scs = sc.sort_values(KEY + ["date"])
def slope(g):
    if len(g) < 2: return 0.0
    return float(np.polyfit(np.arange(len(g)), g["score"].values, 1)[0])
traj = scs.groupby(KEY).apply(lambda g: pd.Series(
    {"score_slope_cw": slope(g), "score_std_cw": float(g["score"].std(ddof=0)),
     "recovery_slope": float(g["score"].diff().mean()) if len(g) > 1 else 0.0}),
    include_groups=False).reset_index()

# ---------- assemble ----------
df = pop.copy()
for part in [beh, ratios, win, gp, foc[KEY + ["assessment_focus"]], wa, agg, traj]:
    df = df.merge(part, on=KEY, how="left")
df["has_vle_activity"] = df["active_days"].notna().astype(int)
df["has_coursework"] = df["weighted_average"].notna().astype(int)
df["clicks_per_day"] = df["sum_click"].fillna(0) / (df["active_days"].fillna(0) + 1)
df["clicks_per_assessment"] = df["sum_click"].fillna(0) / (df["submitted_count"].fillna(0) + 1)
df["study_spread"] = (df["active_days"].fillna(0) / df["course_end"].clip(lower=1)).clip(0, 1)
df["engagement_decay_ratio"] = df["clicks_last_third"].fillna(0) / (df["clicks_first_third"].fillna(0) + 1)
df["days_since_last"] = df["course_end"] - df["last_day"]
SENT = ["days_since_last", "first_submit_day"]
for c in SENT:
    df[c] = df[c].fillna(df["course_end"] + 30)     # sentinel: never observed
zero_cols = ["active_days", "sum_click", "burstiness", "resource_ratio", "oucontent_ratio",
             "homepage_ratio", "forumng_ratio", "quiz_ratio", "w1_clicks", "w2_clicks",
             "w3_clicks", "w4_clicks", "precourse_clicks", "decay_clicks",
             "clicks_first_third", "clicks_last_third", "max_gap", "active_weeks",
             "assessment_focus", "weighted_average", "submitted_count", "mean_submit_lead",
             "min_submit_lead", "late_submissions", "n_assess_types_submitted",
             "completion_ratio_cw", "score_slope_cw", "score_std_cw", "recovery_slope"]
for c in zero_cols:
    df[c] = df[c].fillna(0)
df["registration_lead"] = -pd.to_numeric(df["date_registration"], errors="coerce")
df["registration_lead"] = df["registration_lead"].fillna(df["registration_lead"].median())

# cohort percentile ranks within (module, presentation)
coh = df.groupby(["code_module", "code_presentation"])
df["rank_clicks"] = coh["sum_click"].rank(pct=True)
df["rank_wa"] = coh["weighted_average"].rank(pct=True)
df["rank_active_days"] = coh["active_days"].rank(pct=True)

df = pd.get_dummies(df, columns=["highest_education"], drop_first=True)
df["target_multi"] = df["final_result"].map(
    {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}).astype(int)

CLS_FEATURES = (
    ["weighted_average", "active_days", "clicks_per_day", "clicks_per_assessment",
     "study_spread", "burstiness", "resource_ratio", "oucontent_ratio", "homepage_ratio",
     "forumng_ratio", "quiz_ratio", "assessment_focus", "recovery_slope",
     "has_vle_activity", "has_coursework", "registration_lead"]
    + [c for c in df.columns if c.startswith("highest_education_")]
    + ["w1_clicks", "w2_clicks", "w3_clicks", "w4_clicks", "precourse_clicks",
       "days_since_last", "decay_clicks", "mean_submit_lead", "min_submit_lead",
       "late_submissions", "submitted_count", "first_submit_day",
       "n_assess_types_submitted", "rank_clicks", "rank_wa", "rank_active_days"]
    + ["completion_ratio_cw", "engagement_decay_ratio", "max_gap", "active_weeks",
       "score_slope_cw", "score_std_cw"])
SCORE_DERIVED = ["weighted_average", "recovery_slope", "rank_wa",
                 "score_slope_cw", "score_std_cw"]
REG_FEATURES = [c for c in CLS_FEATURES if c not in SCORE_DERIVED]

# regression target: full-course coursework weighted score (weight>0 modules only)
wtar = sc[sc["weight"] > 0].groupby(KEY).apply(
    lambda g: pd.Series({"final_cw_score": (g["score"] * g["weight"]).sum() / g["weight"].sum(),
                         "cw_weight_sum": g["weight"].sum()}), include_groups=False).reset_index()
df = df.merge(wtar, on=KEY, how="left")

assert df[CLS_FEATURES].isna().sum().sum() == 0
assert df.duplicated(KEY).sum() == 0
# hard leakage asserts: no exam info, no post-unreg info anywhere upstream (filtered above)
assert "date_unregistration" not in CLS_FEATURES and "course_end" not in CLS_FEATURES

df.to_parquet(f"{SP}/exp006_frame.parquet")
meta = {"n_rows": int(len(df)), "n_cls_features": len(CLS_FEATURES),
        "n_reg_features": len(REG_FEATURES),
        "cls_features": CLS_FEATURES, "reg_features": REG_FEATURES,
        "score_derived_excluded_from_regression": SCORE_DERIVED,
        "regression_target": "final_cw_score (weighted coursework average, weight>0 only)",
        "n_regression_rows": int(df["final_cw_score"].notna().sum()),
        "audit": audit}
json.dump(meta, open(f"{SP}/exp006_meta.json", "w"), indent=1)
print(json.dumps(audit, indent=1))
print(f"frame: {df.shape} | cls features {len(CLS_FEATURES)} | reg features {len(REG_FEATURES)}"
      f" | regression rows {meta['n_regression_rows']} | {time.time()-t0:.0f}s")
print("outcome dist:", df["final_result"].value_counts().to_dict())
