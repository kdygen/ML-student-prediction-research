"""
Experiment 002 — reusable, leakage-safe candidate feature generation.

Every feature is computed exclusively from information available on or before
the prediction day `cutoff`:
  - VLE features filter `studentVle.date <= cutoff` (the day a click happened);
  - submission/score features filter `date_submitted <= cutoff` (the day a
    score first exists); assessment deadlines (`assessments.date`) are used
    only as known-in-advance schedule information;
  - registration lead uses `date_registration` (known at registration);
  - cohort-normalized ranks use other students' <=cutoff BEHAVIOUR only
    (never labels) within the same (module, presentation) — information a
    deployed system possesses on prediction day.

The generator returns one row per (id_student, code_module, code_presentation)
with all candidate columns; `FEATURE_GROUPS` names the groups the research
loop accepts/rejects as units. `fill_and_derive()` merges candidates onto the
official cached v3 frame, adds cohort ranks + interaction terms, and applies
the documented NaN policy.
"""
import numpy as np
import pandas as pd

KEY = ["id_student", "code_module", "code_presentation"]

FEATURE_GROUPS = {
    "recent_windows":    ["w1_clicks", "w2_clicks", "w3_clicks", "w4_clicks", "precourse_clicks"],
    "trend":             ["click_slope", "recent_ratio", "w1_share"],
    "recency_decay":     ["days_since_last", "decay_clicks"],
    "gaps":              ["max_gap", "mean_gap"],
    "first_activity":    ["first_active_day", "precourse_flag"],
    "diversity":         ["n_activity_types", "activity_entropy", "n_sites"],
    "submission_timing": ["mean_submit_lead", "min_submit_lead", "late_submissions",
                          "submitted_count", "first_submit_day", "n_assess_types_submitted"],
    "score_trajectory":  ["last_score", "first_score", "max_score", "min_score",
                          "score_range", "score_std_sub", "n_scores"],
    "module_norm":       ["rank_clicks", "rank_wa", "rank_active_days"],
    "registration":      ["registration_lead"],
    "interactions":      ["wa_x_spread", "wa_x_cpd", "active_x_hascw"],
}

# NaN policy (students with no qualifying rows): click counts/scores/leads -> 0;
# "day"-valued features -> sentinel cutoff+30 ("not observed within window");
# availability is disambiguated by the official has_vle_activity /
# has_coursework indicators plus submitted_count / n_scores.
SENTINEL_DAY_COLS = ["days_since_last", "max_gap", "mean_gap",
                     "first_active_day", "first_submit_day"]


def _grouped_slope(daily):
    """Vectorized per-group OLS slope of sum_click over date."""
    d = daily.assign(xy=daily["date"] * daily["sum_click"],
                     x2=daily["date"] ** 2)
    g = d.groupby(KEY).agg(mx=("date", "mean"), my=("sum_click", "mean"),
                           mxy=("xy", "mean"), mx2=("x2", "mean"),
                           n=("date", "size"))
    var = g["mx2"] - g["mx"] ** 2
    slope = (g["mxy"] - g["mx"] * g["my"]) / var.where(var > 0)
    return slope.where(g["n"] >= 2, 0.0).fillna(0.0).rename("click_slope")


def compute_candidate_features(studentVle, vle, studentAssessment, assessments,
                               studentRegistration, cutoff):
    C = cutoff
    sv = studentVle[KEY + ["id_site", "date", "sum_click"]].copy()
    sv["date"] = pd.to_numeric(sv["date"], errors="coerce")
    sv = sv[sv["date"] <= C]
    assert sv["date"].max() <= C  # leakage guard

    daily = sv.groupby(KEY + ["date"])["sum_click"].sum().reset_index()
    total = daily.groupby(KEY)["sum_click"].sum().rename("_total")

    parts = []
    # --- recent_windows ---
    def wsum(lo, hi, name):
        m = daily[(daily["date"] >= lo) & (daily["date"] <= hi)]
        return m.groupby(KEY)["sum_click"].sum().rename(name)
    parts += [wsum(C - 6, C, "w1_clicks"), wsum(C - 13, C - 7, "w2_clicks"),
              wsum(C - 20, C - 14, "w3_clicks"), wsum(C - 27, C - 21, "w4_clicks"),
              daily[daily["date"] < 0].groupby(KEY)["sum_click"].sum()
                   .rename("precourse_clicks")]

    # --- trend ---
    parts.append(_grouped_slope(daily))
    lo = int(0.75 * C)
    recent = wsum(lo, C, "_recent")
    tr = pd.concat([total, recent], axis=1).fillna(0.0)
    parts.append((tr["_recent"] / (tr["_total"] + 1)).rename("recent_ratio"))

    # --- recency_decay ---
    g = daily.groupby(KEY)
    last_day = g["date"].max()
    parts.append((C - last_day).rename("days_since_last"))
    decay = daily.assign(dc=daily["sum_click"] * np.exp(-(C - daily["date"]) / 7.0))
    parts.append(decay.groupby(KEY)["dc"].sum().rename("decay_clicks"))

    # --- gaps ---
    ds = daily.sort_values(KEY + ["date"])
    gap = ds.groupby(KEY)["date"].diff()
    gaps = ds.assign(gap=gap).dropna(subset=["gap"]).groupby(KEY)["gap"]
    parts += [gaps.max().rename("max_gap"), gaps.mean().rename("mean_gap")]

    # --- first_activity ---
    first_day = g["date"].min()
    parts += [first_day.rename("first_active_day"),
              (first_day < 0).astype(int).rename("precourse_flag")]

    # --- diversity ---
    svt = sv.merge(vle[["id_site", "code_module", "code_presentation", "activity_type"]],
                   on=["id_site", "code_module", "code_presentation"], how="left")
    tclk = svt.groupby(KEY + ["activity_type"])["sum_click"].sum().reset_index()
    parts.append(tclk.groupby(KEY)["activity_type"].nunique().rename("n_activity_types"))
    tt = tclk.merge(tclk.groupby(KEY)["sum_click"].sum().rename("_t"), on=KEY)
    p = (tt["sum_click"] / tt["_t"]).clip(1e-12)
    parts.append(tt.assign(_e=-p * np.log(p)).groupby(KEY)["_e"].sum()
                   .rename("activity_entropy"))
    parts.append(sv.groupby(KEY)["id_site"].nunique().rename("n_sites"))

    # --- submissions (date_submitted <= C only) ---
    sa = studentAssessment.merge(
        assessments[["id_assessment", "code_module", "code_presentation",
                     "date", "assessment_type"]], on="id_assessment", how="left")
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    sa["date"] = pd.to_numeric(sa["date"], errors="coerce")
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    sa = sa[sa["date_submitted"] <= C]
    assert sa["date_submitted"].max() <= C  # leakage guard

    sa = sa.assign(lead=sa["date"] - sa["date_submitted"])   # deadline is schedule info
    gs = sa.groupby(KEY)
    parts += [gs["lead"].mean().rename("mean_submit_lead"),
              gs["lead"].min().rename("min_submit_lead"),
              sa.assign(_l=(sa["lead"] < 0).astype(int)).groupby(KEY)["_l"].sum()
                .rename("late_submissions"),
              gs.size().rename("submitted_count"),
              gs["date_submitted"].min().rename("first_submit_day"),
              gs["assessment_type"].nunique().rename("n_assess_types_submitted")]

    # --- score trajectory over submitted work only ---
    sc = sa.dropna(subset=["score"]).sort_values(KEY + ["date_submitted"])
    gc = sc.groupby(KEY)["score"]
    parts += [gc.last().rename("last_score"), gc.first().rename("first_score"),
              gc.max().rename("max_score"), gc.min().rename("min_score"),
              (gc.max() - gc.min()).rename("score_range"),
              gc.std().rename("score_std_sub"), gc.size().rename("n_scores")]

    # --- registration (known at registration time) ---
    reg = studentRegistration[KEY + ["date_registration"]].copy()
    reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")
    parts.append(reg.set_index(KEY)["date_registration"].mul(-1)
                    .rename("registration_lead"))

    out = pd.concat(parts, axis=1).reset_index()
    return out


def fill_and_derive(base, cand, cutoff):
    """Merge candidates onto the official cached v3 frame; apply NaN policy;
    add cohort ranks + interactions (computed on <=cutoff values only)."""
    df = base.merge(cand, on=KEY, how="left")
    for c in SENTINEL_DAY_COLS:
        df[c] = df[c].fillna(cutoff + 30)
    zero_cols = [c for g in FEATURE_GROUPS.values() for c in g
                 if c in df.columns and c not in SENTINEL_DAY_COLS
                 and c not in ("rank_clicks", "rank_wa", "rank_active_days",
                                "w1_share", "wa_x_spread", "wa_x_cpd", "active_x_hascw")]
    for c in zero_cols:
        df[c] = df[c].fillna(0)
    df["registration_lead"] = df["registration_lead"].fillna(df["registration_lead"].median())

    # cohort ranks: percentile of the student's <=cutoff behaviour within their
    # (module, presentation) cohort — deployable peer context, no labels used
    coh = df.groupby(["code_module", "code_presentation"])
    df["rank_clicks"] = coh["sum_click"].rank(pct=True)
    df["rank_wa"] = coh["weighted_average"].rank(pct=True)
    df["rank_active_days"] = coh["active_days"].rank(pct=True)

    # share of all <=cutoff clicks that fell in the last 7 days (trend group)
    df["w1_share"] = df["w1_clicks"] / (df["sum_click"] + 1)

    # interactions of existing official features
    df["wa_x_spread"] = df["weighted_average"] * df["study_spread"]
    df["wa_x_cpd"] = df["weighted_average"] * df["clicks_per_day"]
    df["active_x_hascw"] = df["active_days"] * df["has_coursework"]

    allc = [c for g in FEATURE_GROUPS.values() for c in g]
    assert df[allc].isna().sum().sum() == 0
    return df
