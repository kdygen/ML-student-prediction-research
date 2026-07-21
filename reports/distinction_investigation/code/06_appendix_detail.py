"""Appendix detail: exact class weights, per-fold thresholds, importance for the winner."""
import json, os, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
SP=os.environ["SP"]; REPO="/Users/margulankudaibergen/Developer/ML-student prediction/ML-student-prediction-research"
OUT=REPO+"/reports/distinction_investigation"; KEY=["id_student","code_module","code_presentation"]
df=pd.read_parquet(SP+"/exp006c_frame_probe.parquet").copy()
df.loc[df["first_submit_day"]>=(df["course_end"]+29),"first_submit_day"]=999
newf=pd.read_parquet(SP+"/dx_new_features.parquet"); groups=json.load(open(SP+"/dx_feature_groups.json"))
df=df.merge(newf,on=KEY,how="left",validate="one_to_one")
F36=json.load(open(REPO+"/reports/experiment_008_parsimony/parsimony_metrics.json"))["final"]["features"]
F33=[f for f in F36 if f not in ["rank_wa","score_slope_cw","score_std_cw"]]; H2=groups["H2_depth"]
V3=df[~(df["date_unregistration"]<=0)].reset_index(drop=True)
for c in [c for g in groups.values() for c in g]: V3[c]=pd.to_numeric(V3[c],errors="coerce").fillna(0)
mod=pd.get_dummies(V3["code_module"],prefix="mod",drop_first=True).astype(float)
V3=pd.concat([V3,mod],axis=1); MODS=list(mod.columns)
y=V3["target_multi"].values; g_=V3["id_student"].values; BEST=F33+H2+MODS
CLS=["Withdrawn","Fail","Pass","Distinction"]

# --- exact class weights ---
counts={c:int((y==c).sum()) for c in range(4)}
cw={c: len(y)/(4*counts[c]) for c in range(4)}
print("=== EXACT CLASS WEIGHTS  w_c = N / (4 * n_c) ===")
for c in range(4):
    print(f"  {CLS[c]:12s} n={counts[c]:6d}  w={cw[c]:.6f}")
print(f"  N={len(y)}  (weights sum-normalised so sum_c n_c*w_c = N)")

# --- taus from saved final run ---
fin=json.load(open(OUT+"/iteration3_final.json"))["final_stack"]
print("\n=== PER-FOLD D-THRESHOLDS (inner GroupKFold(3) tuned) ===")
print("  taus:", fin["taus"], " | per-fold D F1:", [round(x,3) for x in fin["D_f1_per_fold"]])

# --- final model importance (fit on all data, same config) ---
m=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,
    colsample_bytree=0.8,random_state=42,objective="multi:softmax",num_class=4,
    eval_metric="mlogloss",n_jobs=-1)
m.fit(V3[BEST],y,sample_weight=np.vectorize(cw.get)(y))
imp={}
for t in ["gain","weight","cover"]:
    b=m.get_booster(); s=b.get_score(importance_type=t)
    imp[t]={f: float(s.get(f,0.0)) for f in BEST}
tot=sum(imp["gain"].values()); gain_n={k:v/tot for k,v in imp["gain"].items()}
top=sorted(gain_n.items(),key=lambda kv:-kv[1])[:20]
print("\n=== FINAL MODEL — normalised GAIN importance (top 20) ===")
for i,(f,v) in enumerate(top,1):
    tag=" [H2-revisit]" if f in H2 else (" [module]" if f in MODS else "")
    print(f"  {i:2d}. {f:26s} {v:.4f}{tag}")
print("\n  H2 group total gain share: %.4f"%sum(gain_n[f] for f in H2))
print("  module dummies total gain share: %.4f"%sum(gain_n[f] for f in MODS))

# --- permutation importance for the Distinction class specifically (P-vs-D AUC drop) ---
from sklearn.metrics import roc_auc_score
pdm=np.isin(y,[2,3]); Xp=V3[pdm].reset_index(drop=True); yb=(y[pdm]==3).astype(int); gb=g_[pdm]
tr,te=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(Xp,yb,groups=gb))
spw=(yb[tr]==0).sum()/(yb[tr]==1).sum()
mb=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,
    colsample_bytree=0.8,random_state=42,eval_metric="logloss",scale_pos_weight=spw,n_jobs=-1)
FEAT=F33+H2
mb.fit(Xp.iloc[tr][FEAT],yb[tr])
base_auc=roc_auc_score(yb[te],mb.predict_proba(Xp.iloc[te][FEAT])[:,1])
rng=np.random.default_rng(0); perm={}
Xte=Xp.iloc[te][FEAT].reset_index(drop=True)
for f in FEAT:
    drops=[]
    for r in range(3):
        Xc=Xte.copy(); Xc[f]=rng.permutation(Xc[f].values)
        drops.append(base_auc-roc_auc_score(yb[te],mb.predict_proba(Xc)[:,1]))
    perm[f]=float(np.mean(drops))
pt=sorted(perm.items(),key=lambda kv:-kv[1])[:15]
print(f"\n=== P-vs-D PERMUTATION IMPORTANCE (AUC drop, base AUC {base_auc:.4f}) ===")
for i,(f,v) in enumerate(pt,1):
    tag=" [H2-revisit]" if f in H2 else ""
    print(f"  {i:2d}. {f:26s} {v:+.4f}{tag}")
print("  H2 total AUC drop: %+.4f"%sum(perm[f] for f in H2))
json.dump({"class_weights":{CLS[c]:cw[c] for c in range(4)},"class_counts":{CLS[c]:counts[c] for c in range(4)},
  "taus_per_fold":fin["taus"],"gain_normalised":gain_n,"gain_raw":imp["gain"],
  "cover":imp["cover"],"weight":imp["weight"],
  "pvd_permutation_auc_drop":perm,"pvd_base_auc":float(base_auc),
  "H2_gain_share":float(sum(gain_n[f] for f in H2)),
  "module_gain_share":float(sum(gain_n[f] for f in MODS))},
  open(OUT+"/appendix_importance.json","w"),indent=1)
print("\nsaved -> appendix_importance.json")
