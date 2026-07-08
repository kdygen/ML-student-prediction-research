"""Experiment 004 Phase 5 — capacity-aware intervention policy simulation.

Reference cohort = test-panel cases present in the c14 frame (intervention-eligible from
day 14). Adverse = final_result in {Withdrawn, Fail}. A student can only be reached while
still enrolled (cases leave the frame after unregistration), so waiting for better
predictions shrinks the reachable at-risk pool.

One-shot policies: flag top-K% of the cutoff's test ranking (ASI) at a single cutoff.
Staged policies: split the same total budget across cutoffs; later stages skip
already-flagged cases (top-up).
Value metric: unique truly-adverse reference cases reached (recall of the reference
adverse pool), plus precision of all flags and, for withdrawn, lead time to unregistration.
"""
import json, os
import numpy as np
import pandas as pd

SP = os.environ["SP"]
CUTS = [14, 30, 60, 90, 140]

long = pd.read_parquet(f"{SP}/exp004_panel_long.parquet")
long["case"] = (long["id_student"].astype(str) + "|" + long["code_module"].astype(str)
                + "|" + long["code_presentation"].astype(str))
base = long[long["cutoff"] == 14].set_index("case")
ref_adverse = set(base[base["final_result"].isin(["Withdrawn", "Fail"])].index)
ref_w = set(base[base["final_result"] == "Withdrawn"].index)
N0 = len(base)
print(f"reference cohort: {N0} cases, adverse {len(ref_adverse)} "
      f"({len(ref_adverse)/N0:.3f}), withdrawn {len(ref_w)}")

by_cut = {c: long[long["cutoff"] == c].set_index("case") for c in CUTS}

def flag_topk(c, budget, exclude=frozenset()):
    """Flag up to `budget` cases at cutoff c by ASI rank, skipping excluded; only cases
    from the reference cohort count toward metrics (late registrants are ignored)."""
    d = by_cut[c]
    d = d[d.index.isin(base.index) & ~d.index.isin(exclude)]
    picked = d.sort_values("ASI", ascending=False).head(budget)
    return picked

def eval_flags(flagged: pd.DataFrame):
    idx = set(flagged.index)
    adv_hit = idx & ref_adverse
    w_hit = idx & ref_w
    leads = []
    for case in w_hit:
        u = base.loc[case, "date_unregistration"]
        fc = flagged.loc[case, "cutoff"] if "cutoff" in flagged else None
        if pd.notna(u) and fc is not None:
            leads.append(float(u) - float(fc))
    return {
        "n_flagged": int(len(idx)),
        "precision": float(len(adv_hit) / max(1, len(idx))),
        "adverse_reached": int(len(adv_hit)),
        "recall_of_reference_adverse": float(len(adv_hit) / len(ref_adverse)),
        "withdrawn_reached": int(len(w_hit)),
        "recall_of_reference_withdrawn": float(len(w_hit) / len(ref_w)),
        "median_lead_days_withdrawn": float(np.median(leads)) if leads else None,
    }

out = {"reference": {"n_cases": N0, "n_adverse": len(ref_adverse), "n_withdrawn": len(ref_w)},
       "reachability": {}, "one_shot": {}, "staged": {}}

# reachability of the eventually-adverse pool at each cutoff
for c in CUTS:
    present = set(by_cut[c].index) & set(base.index)
    out["reachability"][f"c{c:03d}"] = {
        "reference_cases_still_enrolled": int(len(present)),
        "adverse_still_enrolled": int(len(present & ref_adverse)),
        "adverse_reachable_share": float(len(present & ref_adverse) / len(ref_adverse)),
        "withdrawn_still_enrolled": int(len(present & ref_w)),
        "withdrawn_reachable_share": float(len(present & ref_w) / len(ref_w)),
    }

# one-shot policies at each cutoff / K
for k in (0.05, 0.10, 0.20):
    budget = int(round(k * N0))
    for c in CUTS:
        f = flag_topk(c, budget)
        f = f.assign(cutoff=c)
        out["one_shot"][f"top{int(k*100)}_at_c{c:03d}"] = eval_flags(f)

# staged policies, total budget = 20% of reference cohort
B = int(round(0.20 * N0))
def staged(splits):  # [(cutoff, fraction_of_B), ...]
    flagged = []
    seen = set()
    for c, fr in splits:
        f = flag_topk(c, int(round(B * fr)), exclude=frozenset(seen))
        f = f.assign(cutoff=c)
        flagged.append(f)
        seen |= set(f.index)
    allf = pd.concat(flagged)
    allf = allf[~allf.index.duplicated()]
    return eval_flags(allf)

out["staged"]["oneshot_c30_20pct"] = staged([(30, 1.0)])
out["staged"]["oneshot_c90_20pct"] = staged([(90, 1.0)])
out["staged"]["oneshot_c140_20pct"] = staged([(140, 1.0)])
out["staged"]["c30_10_c90_10"] = staged([(30, 0.5), (90, 0.5)])
out["staged"]["c14_5_c30_5_c60_5_c90_5"] = staged([(14, .25), (30, .25), (60, .25), (90, .25)])
out["staged"]["c30_10_c60_5_c90_5"] = staged([(30, 0.5), (60, 0.25), (90, 0.25)])

json.dump(out, open(f"{SP}/exp004_policies.json", "w"), indent=1)
print(json.dumps(out["reachability"], indent=1))
print(json.dumps(out["staged"], indent=1))
print("saved ->", f"{SP}/exp004_policies.json")
