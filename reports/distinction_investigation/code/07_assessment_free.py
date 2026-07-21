"""True assessment-free replication of the winning stack, for like-for-like comparison
with Al-azazi & Ghurab (demographics + clickstream only, no assessment table)."""
import json, os, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                             roc_auc_score, average_precision_score)
SP=os.environ["SP"]; REPO="/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
OUT=REPO+"/reports/distinction_investigation"; KEY=["id_student","code_module","code_presentation"]
CLS=["Withdrawn","Fail","Pass","Distinction"]
df=pd.read_parquet(SP+"/exp006c_frame_probe.parquet").copy()
df.loc[df["first_submit_day"]>=(df["course_end"]+29),"first_submit_day"]=999
newf=pd.read_parquet(SP+"/dx_new_features.parquet"); groups=json.load(open(SP+"/dx_feature_groups.json"))
df=df.merge(newf,on=KEY,how="left",validate="one_to_one")
F36=json.load(open(REPO+"/reports/experiment_008_parsimony/parsimony_metrics.json"))["final"]["features"]
SCORE=["rank_wa","score_slope_cw","score_std_cw"]
ASSESS=["submitted_count","mean_submit_lead","min_submit_lead","late_submissions",
        "first_submit_day","n_assess_types_submitted","completion_ratio_avail",
        "clicks_per_assessment","assessment_focus"]
F33=[f for f in F36 if f not in SCORE]
F24=[f for f in F36 if f not in SCORE+ASSESS]          # demographics + clickstream ONLY
H2=groups["H2_depth"]                                   # all clickstream/vle derived
V3=df[~(df["date_unregistration"]<=0)].reset_index(drop=True)
for c in [c for g in groups.values() for c in g]: V3[c]=pd.to_numeric(V3[c],errors="coerce").fillna(0)
mod=pd.get_dummies(V3["code_module"],prefix="mod",drop_first=True).astype(float)
V3=pd.concat([V3,mod],axis=1); MODS=list(mod.columns)
y=V3["target_multi"].values; g_=V3["id_student"].values
print(f"F24 (assessment-free) = {len(F24)} feats; +H2 = {len(F24+H2)}; +module = {len(F24+H2+MODS)}")
assert not any(f in ASSESS+SCORE for f in F24+H2), "assessment leak into arm!"
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(V3[F24],y,groups=g_))
cw={c: len(y)/(4*(y==c).sum()) for c in range(4)}
def xgb4():
    return XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,random_state=42,objective="multi:softmax",num_class=4,
        eval_metric="mlogloss",n_jobs=-1)
def run(feats,tag,weighted=True,tune_tau=False):
    X=V3[feats]; oof=np.full(len(y),-1); taus=[]
    for tr,te in folds:
        assert len(set(g_[tr])&set(g_[te]))==0
        Xtr,ytr,gtr=X.iloc[tr],y[tr],g_[tr]; tau=None
        if tune_tau:
            inner=np.full((len(tr),4),np.nan)
            for a,b in GroupKFold(3).split(Xtr,ytr,groups=gtr):
                mi=xgb4(); mi.fit(Xtr.iloc[a],ytr[a],sample_weight=np.vectorize(cw.get)(ytr[a]))
                inner[b]=mi.predict_proba(Xtr.iloc[b])
            ip=inner.argmax(1); bm=f1_score(ytr,ip,average="macro")
            bt,bd=None,f1_score(ytr==3,ip==3,zero_division=0)
            for t in np.arange(0.20,0.61,0.02):
                cand=ip.copy(); cand[inner[:,3]>t]=3
                d=f1_score(ytr==3,cand==3,zero_division=0)
                if d>bd and f1_score(ytr,cand,average="macro")>=bm-0.005: bt,bd=t,d
            tau=bt
        taus.append(tau)
        m=xgb4()
        m.fit(Xtr,ytr,sample_weight=np.vectorize(cw.get)(ytr) if weighted else None)
        P=m.predict_proba(X.iloc[te]); pred=P.argmax(1)
        if tau is not None: pred[P[:,3]>tau]=3
        oof[te]=pred
    A=[accuracy_score(y[te],oof[te]) for _,te in folds]
    F=[f1_score(y[te],oof[te],average="macro") for _,te in folds]
    Df=[f1_score(y[te]==3,oof[te]==3,zero_division=0) for _,te in folds]
    rep=classification_report(y,oof,target_names=CLS,output_dict=True,zero_division=0)
    r={"arm":tag,"n_features":len(feats),"accuracy":float(np.mean(A)),
       "macro_f1":float(np.mean(F)),"macro_f1_std":float(np.std(F)),
       "D_precision":float(rep["Distinction"]["precision"]),"D_recall":float(rep["Distinction"]["recall"]),
       "D_f1":float(rep["Distinction"]["f1-score"]),"D_f1_per_fold":[float(x) for x in Df],
       "D_f1_std":float(np.std(Df)),"P_f1":float(rep["Pass"]["f1-score"]),
       "F_f1":float(rep["Fail"]["f1-score"]),"W_f1":float(rep["Withdrawn"]["f1-score"]),
       "taus":taus}
    print(f"[{tag:38s}] {len(feats):2d}f acc {r['accuracy']:.4f} F1 {r['macro_f1']:.4f}±{r['macro_f1_std']:.4f} "
          f"| D {r['D_f1']:.3f} (P{r['D_precision']:.2f}/R{r['D_recall']:.2f}) ±{r['D_f1_std']:.3f} "
          f"| W {r['W_f1']:.3f} F {r['F_f1']:.3f} P {r['P_f1']:.3f}",flush=True)
    return r
res=[]
res.append(run(F24,"AF0 assess-free argmax",weighted=False))
res.append(run(F24,"AF1 assess-free weighted"))
res.append(run(F24+H2,"AF2 assess-free +H2 weighted"))
res.append(run(F24+H2+MODS,"AF3 assess-free +H2+mod+tau",tune_tau=True))
# reference: with-assessment winner
res.append(run(F33+H2+MODS,"REF with-assessment winner",tune_tau=True))
# P-vs-D AUC ceiling, assessment-free
pdm=np.isin(y,[2,3]); Xi=V3[pdm].reset_index(drop=True); yb=(y[pdm]==3).astype(int); gb=g_[pdm]
for feats,tag in [(F24,"AF P-vs-D 24"),(F24+H2,"AF P-vs-D 24+H2"),(F33+H2,"REF P-vs-D 33+H2")]:
    au,pr=[],[]
    for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(Xi,yb,groups=gb):
        spw=(yb[a]==0).sum()/(yb[a]==1).sum()
        m=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,
            colsample_bytree=0.8,random_state=42,eval_metric="logloss",scale_pos_weight=spw,n_jobs=-1)
        m.fit(Xi.iloc[a][feats],yb[a]); p=m.predict_proba(Xi.iloc[b][feats])[:,1]
        au.append(roc_auc_score(yb[b],p)); pr.append(average_precision_score(yb[b],p))
    print(f"[{tag:38s}] AUC {np.mean(au):.4f}±{np.std(au):.4f} PR {np.mean(pr):.4f}")
    res.append({"arm":tag,"n_features":len(feats),"auc":float(np.mean(au)),
                "auc_std":float(np.std(au)),"pr_auc":float(np.mean(pr))})
json.dump(res,open(OUT+"/assessment_free_comparison.json","w"),indent=1)
print("\nsaved -> assessment_free_comparison.json")
