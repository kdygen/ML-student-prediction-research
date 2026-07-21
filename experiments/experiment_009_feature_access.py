"""Experiment 009 — feature-access ablation for like-for-like comparison with
Al-azazi & Ghurab (2023), who use demographics + clickstream ONLY (no assessment data).

Arms (identical protocol, models, splits; only the feature set changes):
  A  full 36            current published headline
  B  33 = A - score-derived (rank_wa, score_slope_cw, score_std_cw)   "no grades"
  C  24 = B - all assessment-derived (submissions, leads, completion, focus)
        -> demographics + clickstream only == Al-azazi feature access
Populations: engaged (29,496) and full registered (32,593; matches Al-azazi).
"""
import json, os, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
SP=os.environ["SP"]
REPO="/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
CLS=["Withdrawn","Fail","Pass","Distinction"]

df=pd.read_parquet(SP+"/exp006c_frame_probe.parquet").copy()
sent=df["first_submit_day"]>=(df["course_end"]+29); df.loc[sent,"first_submit_day"]=999
F36=json.load(open(REPO+"/reports/experiment_008_parsimony/parsimony_metrics.json"))["final"]["features"]

SCORE=["rank_wa","score_slope_cw","score_std_cw"]
SUBMISSION=["submitted_count","mean_submit_lead","min_submit_lead","late_submissions",
            "first_submit_day","n_assess_types_submitted","completion_ratio_avail",
            "clicks_per_assessment","assessment_focus"]
ARMS={"A_full_36":F36,
      "B_no_scores_33":[f for f in F36 if f not in SCORE],
      "C_clickstream_demo_24":[f for f in F36 if f not in SCORE+SUBMISSION]}
for k,v in ARMS.items(): print(f"{k}: {len(v)} features")
print("\nArm C features (should be demographics + clickstream only):")
print("  "+", ".join(ARMS["C_clickstream_demo_24"]))

def xgb():
    return XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,random_state=42,objective="multi:softmax",num_class=4,
        eval_metric="mlogloss",n_jobs=-1)

def run(d,feats):
    X,y,g=d[feats],d["target_multi"].values,d["id_student"].values
    A=[];F=[];oof=np.full(len(y),-1)
    for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,groups=g):
        assert len(set(g[a])&set(g[b]))==0
        m=xgb(); m.fit(X.iloc[a],y[a]); p=m.predict(X.iloc[b]); oof[b]=p
        A.append(accuracy_score(y[b],p)); F.append(f1_score(y[b],p,average="macro"))
    r=classification_report(y,oof,target_names=CLS,output_dict=True,zero_division=0)
    return {"n":int(len(d)),"n_features":len(feats),
            "accuracy":float(np.mean(A)),"accuracy_std":float(np.std(A)),
            "macro_f1":float(np.mean(F)),"macro_f1_std":float(np.std(F)),
            "weighted_f1":float(r["weighted avg"]["f1-score"]),
            "per_class_f1":{c:float(r[c]["f1-score"]) for c in CLS},
            "per_class_precision":{c:float(r[c]["precision"]) for c in CLS},
            "per_class_recall":{c:float(r[c]["recall"]) for c in CLS}}

POPS={"engaged_29496":df[~(df["date_unregistration"]<=0)],"full_32593":df}
out={}
for pname,pdata in POPS.items():
    out[pname]={}
    print(f"\n===== {pname} (n={len(pdata)}) =====")
    for aname,feats in ARMS.items():
        r=run(pdata,feats); out[pname][aname]=r
        pc=r["per_class_f1"]
        print(f"  {aname:24s} ({r['n_features']:2d}f)  acc {r['accuracy']:.4f}±{r['accuracy_std']:.4f}  "
              f"macroF1 {r['macro_f1']:.4f}±{r['macro_f1_std']:.4f}  "
              f"| W {pc['Withdrawn']:.3f} F {pc['Fail']:.3f} P {pc['Pass']:.3f} D {pc['Distinction']:.3f}")
json.dump(out,open(REPO+"/reports/experiment_009_feature_access_results.json","w"),indent=1)
print("\nsaved -> reports/experiment_009_feature_access_results.json")
