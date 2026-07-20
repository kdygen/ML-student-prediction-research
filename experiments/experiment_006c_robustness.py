"""Robustness of 006b to the two flagged proxy features.
Variants (all else identical to 006b: same models, same grouped splits, same features):
  V0  006b as published
  V1  fair denominator: completion_ratio_cw -> completion_ratio_avail (deadline<=horizon)
  V2  V1 minus has_vle_activity & has_coursework (never-starter flags)
  V3  V1 on the ENGAGED population only (drop 3,097 who unregistered on/before day 0)
"""
import json, os, numpy as np, pandas as pd, statistics as st
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
SP=os.environ["SP"]
df=pd.read_parquet(SP+"/exp006c_frame_probe.parquet")
meta=json.load(open(SP+"/exp006b_meta.json")); CF=meta["cls_features"]
CLS=["Withdrawn","Fail","Pass","Distinction"]
def xgb(): return XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,
    colsample_bytree=0.8,random_state=42,objective="multi:softmax",num_class=4,
    eval_metric="mlogloss",n_jobs=-1)
def run(d,feats,tag):
    y=d["target_multi"].values; g=d["id_student"].values; X=d[feats]
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42).split(X,y,groups=g))
    assert len(set(g[tr])&set(g[te]))==0
    m=xgb(); m.fit(X.iloc[tr],y[tr]); p=m.predict(X.iloc[te])
    rep=classification_report(y[te],p,output_dict=True,zero_division=0,target_names=CLS)
    reps=[]
    for s in range(5):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=s).split(X,y,groups=g))
        ms=xgb(); ms.fit(X.iloc[a],y[a]); ps=ms.predict(X.iloc[b])
        reps.append((accuracy_score(y[b],ps),f1_score(y[b],ps,average="macro")))
    yw=(y==0).astype(int)
    mb=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,random_state=42,eval_metric="logloss",n_jobs=-1)
    mb.fit(X.iloc[tr],yw[tr]); wauc=roc_auc_score(yw[te],mb.predict_proba(X.iloc[te])[:,1])
    imp=sorted(zip(feats,m.feature_importances_),key=lambda kv:-kv[1])[:5]
    r={"tag":tag,"n":int(len(d)),"acc":float(accuracy_score(y[te],p)),
       "f1":float(f1_score(y[te],p,average="macro")),
       "acc_rep":[float(np.mean([x[0] for x in reps])),float(np.std([x[0] for x in reps]))],
       "f1_rep":[float(np.mean([x[1] for x in reps])),float(np.std([x[1] for x in reps]))],
       "per_class_f1":{c:float(rep[c]["f1-score"]) for c in CLS},
       "withdrawn_binary_auc":float(wauc),"top5":[(f,float(v)) for f,v in imp]}
    print(f"[{tag}] n={r['n']} acc={r['acc']:.4f} F1={r['f1']:.4f} "
          f"(rep {r['acc_rep'][0]:.4f}/{r['f1_rep'][0]:.4f}) Wf1={r['per_class_f1']['Withdrawn']:.3f} "
          f"Ff1={r['per_class_f1']['Fail']:.3f} Wauc={wauc:.4f}",flush=True)
    print(f"      top5: {[(f,round(v,3)) for f,v in r['top5']]}",flush=True)
    return r
CF_fair=[c if c!="completion_ratio_cw" else "completion_ratio_avail" for c in CF]
CF_v2=[c for c in CF_fair if c not in ("has_vle_activity","has_coursework")]
out={}
out["V0_published_006b"]=run(df,CF,"V0 006b published")
out["V1_fair_denominator"]=run(df,CF_fair,"V1 fair denominator")
out["V2_minus_neverstarter_flags"]=run(df,CF_v2,"V2 V1 - never-starter flags")
eng=df[~(df["date_unregistration"]<=0)].copy()
out["V3_engaged_population"]=run(eng,CF_fair,"V3 V1, engaged pop only")
json.dump(out,open(SP+"/exp006c_robustness.json","w"),indent=1)
print("\nsaved -> exp006c_robustness.json")
