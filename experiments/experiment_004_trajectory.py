"""Experiment 004 Phase 3 — student trajectory analysis on the fixed test panel.

Consumes exp004_panel_long.parquet (out-of-sample scores/bands per case per cutoff).
Case = (id_student, code_module, code_presentation). Bands per cutoff: red = top 5%,
amber = 5-20%, green = rest (within-cutoff test ranking). flagged = red|amber.
"""
import json, os
import numpy as np
import pandas as pd

SP = os.environ["SP"]
CUTS = [14, 30, 60, 90, 140]
KEY = ["id_student", "code_module", "code_presentation"]

long = pd.read_parquet(f"{SP}/exp004_panel_long.parquet")
long["case"] = (long["id_student"].astype(str) + "|" + long["code_module"].astype(str)
                + "|" + long["code_presentation"].astype(str))

piv_band = long.pivot_table(index="case", columns="cutoff", values="band", aggfunc="first")
piv_iso = long.pivot_table(index="case", columns="cutoff", values="ISO_ASI", aggfunc="first")
info = long.sort_values("cutoff").groupby("case").agg(
    final_result=("final_result", "first"),
    unreg=("date_unregistration", "first"),
    first_cutoff=("cutoff", "min"), last_cutoff=("cutoff", "max"),
    n_obs=("cutoff", "count"))

FLAG = {"red", "amber"}
out = {"n_cases": int(len(info))}

# restrict trajectory typology to cases observed at c14 (the intervention-eligible cohort)
base = info[info["first_cutoff"] == 14].copy()
bb = piv_band.loc[base.index]

def classify(row):
    seq = [bb.loc[row.name, c] for c in CUTS if pd.notna(bb.loc[row.name, c])]
    flags = [s in FLAG for s in seq]
    if len(seq) <= 2 and row["last_cutoff"] <= 30:
        return "early_exit"                    # withdrew before c60: <=2 observations
    if all(flags):
        return "persistent_high"
    if not any(flags):
        return "persistent_low"
    first, last = flags[0], flags[-1]
    if not first and last:
        # find first flagged cutoff
        fc = [c for c in CUTS if pd.notna(bb.loc[row.name, c]) and bb.loc[row.name, c] in FLAG][0]
        return "late_emerging" if fc >= 90 else "escalating"
    if first and not last:
        return "recovering"
    return "intermittent"

base["traj"] = base.apply(classify, axis=1)
tt = {}
for t, grp in base.groupby("traj"):
    fr = grp["final_result"].value_counts(normalize=True)
    tt[t] = {"n": int(len(grp)),
             "share": float(len(grp) / len(base)),
             "P_withdrawn": float(fr.get("Withdrawn", 0.0)),
             "P_fail": float(fr.get("Fail", 0.0)),
             "P_adverse": float(fr.get("Withdrawn", 0.0) + fr.get("Fail", 0.0)),
             "P_pass_dist": float(fr.get("Pass", 0.0) + fr.get("Distinction", 0.0))}
out["trajectory_categories"] = tt

# ---- first-flag timing for eventually-adverse cases (early vs late identification) ----
adverse = base[base["final_result"].isin(["Withdrawn", "Fail"])]
first_flag = {}
for case in adverse.index:
    fc = [c for c in CUTS if pd.notna(bb.loc[case, c]) and bb.loc[case, c] in FLAG]
    first_flag[case] = fc[0] if fc else None
ff = pd.Series(first_flag)
dist = ff.value_counts(dropna=False).to_dict()
out["first_flag_of_eventual_adverse"] = {
    ("never" if pd.isna(k) else f"c{int(k):03d}"): int(v) for k, v in dist.items()}
out["adverse_flagged_by_c30_share"] = float(ff.isin([14, 30]).mean())
out["adverse_never_flagged_share"] = float(ff.isna().mean())
# same for withdrawn only
wdr = base[base["final_result"] == "Withdrawn"]
ffw = ff.loc[ff.index.isin(wdr.index)]
out["withdrawn_first_flag"] = {
    ("never" if pd.isna(k) else f"c{int(k):03d}"): int(v)
    for k, v in ffw.value_counts(dropna=False).to_dict().items()}

# ---- intervention lead time: first flag -> unregistration day (withdrawn only) ----
lead = []
for case in wdr.index:
    f = ffw.get(case)
    u = wdr.loc[case, "unreg"]
    if pd.notna(f) and pd.notna(u):
        lead.append({"first_flag": int(f), "unreg": float(u), "lead_days": float(u - f)})
ld = pd.DataFrame(lead)
out["lead_time_days"] = {
    "n_withdrawn_flagged_with_unreg": int(len(ld)),
    "median": float(ld["lead_days"].median()),
    "q25": float(ld["lead_days"].quantile(0.25)),
    "q75": float(ld["lead_days"].quantile(0.75)),
    "share_negative": float((ld["lead_days"] < 0).mean()),
    "by_first_flag": {f"c{int(f):03d}": {"n": int(len(g)),
                                          "median": float(g["lead_days"].median())}
                      for f, g in ld.groupby("first_flag")},
}

# ---- band transition matrices between consecutive cutoffs ----
trans = {}
for a, b in zip(CUTS[:-1], CUTS[1:]):
    sub = piv_band[[a, b]].dropna()
    M = pd.crosstab(sub[a], sub[b], normalize="index")
    trans[f"c{a:03d}->c{b:03d}"] = {r: {cc: float(M.loc[r, cc]) for cc in M.columns}
                                    for r in M.index}
    # exit-to-withdrawal rate from each band (case present at a, absent at b)
    present_a = piv_band[a].dropna().index
    gone = [x for x in present_a if pd.isna(piv_band.loc[x, b])]
    ba = piv_band.loc[present_a, a]
    ex = {}
    for bd in ["red", "amber", "green"]:
        in_band = ba[ba == bd].index
        gone_b = [x for x in in_band if x in set(gone)]
        ex[bd] = {"n_band": int(len(in_band)), "exited": int(len(gone_b)),
                  "exit_rate": float(len(gone_b) / max(1, len(in_band)))}
    trans[f"c{a:03d}->c{b:03d}_exits"] = ex
out["band_transitions"] = trans

# ---- risk velocity: delta ISO_ASI between consecutive observations ----
vel = []
for a, b in zip(CUTS[:-1], CUTS[1:]):
    sub = piv_iso[[a, b]].dropna()
    d = (sub[b] - sub[a])
    fr = info.loc[sub.index, "final_result"]
    rapid = d >= 0.15
    vel.append({"step": f"c{a:03d}->c{b:03d}", "n": int(len(sub)),
                "n_rapid_risers": int(rapid.sum()),
                "P_adverse_rapid": float(fr[rapid].isin(["Withdrawn", "Fail"]).mean()) if rapid.sum() else None,
                "P_adverse_others": float(fr[~rapid].isin(["Withdrawn", "Fail"]).mean())})
out["risk_velocity_rapid_rise_ge_0.15"] = vel

json.dump(out, open(f"{SP}/exp004_trajectory.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["trajectory_categories", "adverse_flagged_by_c30_share",
                                      "adverse_never_flagged_share", "lead_time_days"]}, indent=1))
print("saved ->", f"{SP}/exp004_trajectory.json")
