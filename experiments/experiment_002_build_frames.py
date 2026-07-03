"""Build augmented frames (official cached v3 frame + all candidate features)
for every cutoff. Written to the scratchpad — the official cache is untouched."""
import os, sys, json
import pandas as pd
sys.path.insert(0, "experiments")
from feature_generation_002 import compute_candidate_features, fill_and_derive, FEATURE_GROUPS

SP = os.environ["SP"]
CUTOFFS = [14, 30, 60, 90, 140]

print("[aug] loading raw tables...", file=sys.stderr)
studentVle = pd.read_csv("data/raw/studentVle.csv", engine="python", on_bad_lines="skip")
vle = pd.read_csv("data/raw/vle.csv")
studentAssessment = pd.read_csv("data/raw/studentAssessment.csv")
assessments = pd.read_csv("data/raw/assessments.csv")
studentRegistration = pd.read_csv("data/raw/studentRegistration.csv")
print("[aug] raw loaded", file=sys.stderr)

for C in CUTOFFS:
    base = pd.read_parquet(f"data/processed/p3/c{C:03d}/mlDataV3.parquet")
    cand = compute_candidate_features(studentVle, vle, studentAssessment,
                                      assessments, studentRegistration, C)
    df = fill_and_derive(base, cand, C)
    out = f"{SP}/aug_c{C:03d}.parquet"
    df.to_parquet(out, index=False)
    allc = [c for g in FEATURE_GROUPS.values() for c in g]
    print(f"[aug] c{C:03d}: {df.shape} (+{len(allc)} candidate cols, NaN=0) -> {out}",
          file=sys.stderr)
print("[aug] DONE", file=sys.stderr)
