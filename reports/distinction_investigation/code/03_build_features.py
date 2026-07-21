"""Distinction investigation 03 — build new leakage-free feature hypotheses from raw.

All computed on data censored at each student's horizon h = min(unregistration, course_end);
deadlines are published schedule facts; cohort statistics use peers' BEHAVIOUR only (never
labels), consistent with the existing rank_clicks precedent. Demographics are enrolment-time.

Hypothesis groups (each tested separately in dx04):
  H1 regularity   — weekly consistency, inactive weeks, streaks, mod-7 phase concentration
  H2 depth        — revisits, site concentration, content-vs-admin, enrichment usage
  H3 proactivity  — early-work share before deadlines, engagement start
  H4 spacing      — gap statistics, weekly slope, late new-content exploration
  H5 cohort_week  — weekly z-scores vs cohort, top-quartile weeks, rank trajectory
  H6 enrolment    — studied_credits, num_of_prev_attempts (enrolment-time records)
"""
import json, os, time
import numpy as np, pandas as pd

SP = os.environ["SP"]
REPO = "/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
KEY = ["id_student", "code_module", "code_presentation"]
t0 = time.time()

sv = pd.read_csv(f"{REPO}/data/raw/studentVle.csv")
sr = pd.read_csv(f"{REPO}/data/raw/studentRegistration.csv")
asr = pd.read_csv(f"{REPO}/data/raw/assessments.csv")
vle = pd.read_csv(f"{REPO}/data/raw/vle.csv")
si = pd.read_csv(f"{REPO}/data/raw/studentInfo.csv")
sv["date"] = pd.to_numeric(sv["date"], errors="coerce")
sr["date_unregistration"] = pd.to_numeric(sr["date_unregistration"], errors="coerce")
asr["date"] = pd.to_numeric(asr["date"], errors="coerce")

# censor + horizon
sv = sv.merge(sr[KEY + ["date_unregistration"]], on=KEY, how="left")
sv = sv[sv["date_unregistration"].isna() | (sv["date"] <= sv["date_unregistration"])]
span = sv.groupby(["code_module", "code_presentation"])["date"].max().rename("course_end").reset_index()
hz = sr[KEY + ["date_unregistration"]].merge(span, on=["code_module", "code_presentation"], how="left")
hz["h"] = hz[["date_unregistration", "course_end"]].min(axis=1)
sv = sv.merge(hz[KEY + ["h", "course_end"]], on=KEY, how="left")
sv = sv.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
sv["week"] = (sv["date"] // 7).astype(int)
print(f"censored click rows: {len(sv)} ({time.time()-t0:.0f}s)", flush=True)

daily = sv.groupby(KEY + ["date"])["sum_click"].sum().reset_index()
weekly = sv.groupby(KEY + ["week"])["sum_click"].sum().reset_index()

feats = hz[KEY + ["h"]].copy()

# ---------- H1 regularity ----------
g = weekly.groupby(KEY)["sum_click"]
h1 = g.agg(_wmean="mean", _wstd="std").reset_index()
h1["weekly_cv"] = h1["_wstd"].fillna(0) / (h1["_wmean"] + 1)
# share of enrolled weeks with zero activity
wk_counts = weekly.groupby(KEY)["week"].nunique().rename("_active_wk").reset_index()
h1 = h1.merge(wk_counts, on=KEY, how="left")
h1 = h1.merge(hz[KEY + ["h"]], on=KEY, how="left")
h1["enrolled_weeks"] = (h1["h"].clip(lower=7) // 7).astype(float)
h1["inactive_week_share"] = (1 - h1["_active_wk"] / h1["enrolled_weeks"].clip(lower=1)).clip(0, 1)
# longest consecutive active-week streak
def streak(wks):
    w = np.sort(wks.unique());
    if len(w) == 0: return 0
    d = np.diff(w); runs = np.split(w, np.where(d > 1)[0] + 1)
    return max(len(r) for r in runs)
st = weekly.groupby(KEY)["week"].apply(streak).rename("max_week_streak").reset_index()
h1 = h1.merge(st, on=KEY, how="left")
# entropy of clicks across weeks (normalised) - low entropy = cramming
def norm_entropy(s):
    p = s.values / s.values.sum()
    if len(p) <= 1: return 0.0
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
en = weekly.groupby(KEY)["sum_click"].apply(norm_entropy).rename("week_entropy").reset_index()
h1 = h1.merge(en, on=KEY, how="left")
# mod-7 phase concentration (same weekday-pattern regularity)
daily["_phase"] = daily["date"].astype(int) % 7
ph = daily.groupby(KEY + ["_phase"])["sum_click"].sum().reset_index()
phc = ph.groupby(KEY)["sum_click"].apply(
    lambda s: float((s.max() / s.sum())) if s.sum() > 0 else 0.0).rename("phase_concentration").reset_index()
h1 = h1.merge(phc, on=KEY, how="left")
H1 = ["weekly_cv", "inactive_week_share", "max_week_streak", "week_entropy", "phase_concentration"]
feats = feats.merge(h1[KEY + H1], on=KEY, how="left")
print(f"H1 done ({time.time()-t0:.0f}s)", flush=True)

# ---------- H2 depth ----------
site_days = sv.groupby(KEY + ["id_site"])["date"].nunique().rename("_visits").reset_index()
h2 = site_days.groupby(KEY).agg(unique_sites=("id_site", "count"),
                                revisit_ratio=("_visits", "mean")).reset_index()
top3 = sv.groupby(KEY + ["id_site"])["sum_click"].sum().reset_index()
t3 = top3.sort_values("sum_click", ascending=False).groupby(KEY).head(3).groupby(KEY)["sum_click"].sum().rename("_t3")
tot = top3.groupby(KEY)["sum_click"].sum().rename("_tot")
conc = pd.concat([t3, tot], axis=1).reset_index()
conc["top3_site_share"] = conc["_t3"] / conc["_tot"].clip(lower=1)
h2 = h2.merge(conc[KEY + ["top3_site_share"]], on=KEY, how="left")
CONTENT = {"oucontent", "resource", "quiz", "url", "dataplus", "glossary", "ouwiki", "htmlactivity"}
ADMIN = {"homepage", "subpage"}
ENRICH = {"glossary", "ouwiki", "dataplus", "htmlactivity", "ouelluminate", "oucollaborate",
          "externalquiz", "questionnaire", "sharedsubpage", "repeatactivity"}
at = sv.groupby(KEY + ["activity_type"])["sum_click"].sum().unstack(fill_value=0)
h2b = pd.DataFrame(index=at.index)
tots = at.sum(axis=1).clip(lower=1)
h2b["content_admin_ratio"] = at[[c for c in at.columns if c in CONTENT]].sum(axis=1) / \
                             at[[c for c in at.columns if c in ADMIN]].sum(axis=1).clip(lower=1)
h2b["enrichment_share"] = at[[c for c in at.columns if c in ENRICH]].sum(axis=1) / tots
h2b["n_activity_types"] = (at > 0).sum(axis=1)
h2b = h2b.reset_index()
feats = feats.merge(h2, on=KEY, how="left").merge(h2b, on=KEY, how="left")
H2 = ["unique_sites", "revisit_ratio", "top3_site_share", "content_admin_ratio",
      "enrichment_share", "n_activity_types"]
print(f"H2 done ({time.time()-t0:.0f}s)", flush=True)

# ---------- H3 proactivity ----------
dl = asr[(asr["assessment_type"] != "Exam") & asr["date"].notna()][
    ["code_module", "code_presentation", "date"]].drop_duplicates().rename(columns={"date": "deadline"})
dm = daily.merge(dl, on=["code_module", "code_presentation"], how="inner")
dm = dm[(dm["date"] >= dm["deadline"] - 21) & (dm["date"] <= dm["deadline"])]
dm["_early"] = dm["date"] <= dm["deadline"] - 8
ew = dm.groupby(KEY).apply(lambda x: float(x.loc[x["_early"], "sum_click"].sum() /
                                           max(1, x["sum_click"].sum())), include_groups=False)
feats = feats.merge(ew.rename("early_work_share").reset_index(), on=KEY, how="left")
fd = daily.groupby(KEY)["date"].min().rename("first_active_day").reset_index()
feats = feats.merge(fd, on=KEY, how="left")
H3 = ["early_work_share", "first_active_day"]
print(f"H3 done ({time.time()-t0:.0f}s)", flush=True)

# ---------- H4 spacing ----------
def gapstats(d):
    v = np.sort(d.values)
    if len(v) < 2: return pd.Series({"mean_gap": 0.0, "gap_std": 0.0})
    dd = np.diff(v); return pd.Series({"mean_gap": float(dd.mean()), "gap_std": float(dd.std())})
gp = daily.groupby(KEY)["date"].apply(gapstats).unstack().reset_index()
feats = feats.merge(gp, on=KEY, how="left")
# weekly slope of activity (trend)
def wslope(x):
    if len(x) < 2: return 0.0
    return float(np.polyfit(x["week"].values, x["sum_click"].values, 1)[0])
ws = weekly.groupby(KEY).apply(wslope, include_groups=False).rename("weekly_slope").reset_index()
feats = feats.merge(ws, on=KEY, how="left")
# new-content exploration in second half of own span
sv["_half"] = sv["date"] > sv["h"] / 2
first_seen = sv.groupby(KEY + ["id_site"])["date"].min().reset_index()
fs = first_seen.merge(hz[KEY + ["h"]], on=KEY, how="left")
late_new = fs[fs["date"] > fs["h"] / 2].groupby(KEY)["id_site"].count().rename("_late_new")
all_sites = fs.groupby(KEY)["id_site"].count().rename("_all")
ln = pd.concat([late_new, all_sites], axis=1).reset_index().fillna(0)
ln["late_new_content_share"] = ln["_late_new"] / ln["_all"].clip(lower=1)
feats = feats.merge(ln[KEY + ["late_new_content_share"]], on=KEY, how="left")
H4 = ["mean_gap", "gap_std", "weekly_slope", "late_new_content_share"]
print(f"H4 done ({time.time()-t0:.0f}s)", flush=True)

# ---------- H5 cohort-weekly ----------
cw = weekly.merge(weekly.groupby(["code_module", "code_presentation", "week"])["sum_click"]
                  .agg(["mean", "std"]).reset_index(), on=["code_module", "code_presentation", "week"])
cw["_z"] = (cw["sum_click"] - cw["mean"]) / cw["std"].replace(0, np.nan)
cw["_q75"] = cw.groupby(["code_module", "code_presentation", "week"])["sum_click"].transform(
    lambda s: s.quantile(0.75))
cw["_top"] = (cw["sum_click"] >= cw["_q75"]).astype(float)
h5 = cw.groupby(KEY).agg(mean_weekly_z=("_z", "mean"),
                         share_topq_weeks=("_top", "mean")).reset_index()
def zslope(x):
    v = x.dropna(subset=["_z"])
    if len(v) < 2: return 0.0
    return float(np.polyfit(v["week"].values, v["_z"].values, 1)[0])
zs = cw.groupby(KEY).apply(zslope, include_groups=False).rename("weekly_z_slope").reset_index()
h5 = h5.merge(zs, on=KEY, how="left")
feats = feats.merge(h5, on=KEY, how="left")
H5 = ["mean_weekly_z", "share_topq_weeks", "weekly_z_slope"]
print(f"H5 done ({time.time()-t0:.0f}s)", flush=True)

# ---------- H6 enrolment records ----------
h6 = si[KEY + ["studied_credits", "num_of_prev_attempts"]]
feats = feats.merge(h6, on=KEY, how="left")
H6 = ["studied_credits", "num_of_prev_attempts"]

ALL = H1 + H2 + H3 + H4 + H5 + H6
for c in ALL:
    feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0)
feats = feats[KEY + ALL]
assert feats.duplicated(KEY).sum() == 0
feats.to_parquet(SP + "/dx_new_features.parquet")
json.dump({"H1_regularity": H1, "H2_depth": H2, "H3_proactivity": H3,
           "H4_spacing": H4, "H5_cohort_week": H5, "H6_enrolment": H6},
          open(SP + "/dx_feature_groups.json", "w"), indent=1)
print(f"\n{len(ALL)} new features for {len(feats)} enrolments saved ({time.time()-t0:.0f}s)")
