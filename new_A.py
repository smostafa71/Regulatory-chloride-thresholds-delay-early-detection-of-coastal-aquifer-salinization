import numpy as np, pandas as pd, pickle
from scipy import stats
L=np.load('raw/lead_mat.npy'); F=np.load('raw/fp_mat.npy'); TH=list(range(50,245,5))
desc=pd.read_csv('raw/transition_desc.csv')   # prep ordering, aligned with lead_mat rows? check
trans=sorted(pd.read_csv('raw/transition_sites.csv').site)
assert list(desc.site)==list(pd.read_csv('raw/transition_events.csv').site)
ev=pd.read_csv('raw/transition_events.csv')
assert list(ev.site)==trans
j150=TH.index(150)
# ---------- A. threshold ROC and AUC vs EWS
lm=np.nanmedian(L,0); nt=(F==1).sum(0); el=((F==1)|(F==0)).sum(0); fpr=nt/el
for h in [0.0,1.0,2.0]:
    tpr=np.array([(L[:,j]>h).mean() if h>0 else (L[:,j]>0).mean() for j in range(len(TH))])
    x=np.r_[0,fpr[::-1],1]; y=np.r_[0,tpr[::-1],1]; o=np.argsort(x); auc=np.trapezoid(y[o],x[o])
    print(f"lead>{h} yr: TPR@150={tpr[j150]:.2f} FPR@150={fpr[j150]:.3f} | TPR@100={tpr[TH.index(100)]:.2f} FPR@100={fpr[TH.index(100)]:.3f} | ROC AUC={auc:.3f}")
# ---------- B. censoring-aware lead time at 150 (left-censored wells -> lower bounds)
lead=L[:,j150]; cens=desc.started_above150.to_numpy()
def km_median(t,event):
    o=np.argsort(t); t=t[o]; e=event[o]; S=1.0; n=len(t); out=[]
    for i,(ti,ei) in enumerate(zip(t,e)):
        if ei: S*= (1-1/(n-i))
        out.append((ti,S))
    for ti,S in out:
        if S<=0.5: return ti
    return np.nan
print('median lead150 naive all',np.median(lead),' uncensored',np.median(lead[~cens]),'n',(~cens).sum(),' censored lower bounds',np.median(lead[cens]),'n',cens.sum())
print('KM median (left-censored treated as lower bounds):',km_median(lead,~cens))
# bootstrap CI for KM median
rng=np.random.default_rng(3); bs=[]
for _ in range(2000):
    i=rng.integers(0,len(lead),len(lead)); bs.append(km_median(lead[i],~cens[i]))
print('KM median 95% CI',np.nanpercentile(bs,[2.5,97.5]))
# ---------- C. PPV / alarm burden at thresholds (whole record, eligible pool)
nst=F.shape[0]; ntr=L.shape[0]
for th in [100,125,150,175,200]:
    j=TH.index(th); tp=(L[:,j]>0).sum(); tp0=np.isfinite(L[:,j]).sum(); fp=(F[:,j]==1).sum(); fpc=(F[:,j]!=0).sum()
    ppv=tp/(tp+fp); ppv_all=tp0/(tp0+fpc)
    print(f"T={th}: transitions warned in advance (lead>0) {tp}/{ntr} | stable false alarms (>=5y FU) {fp} (+{fpc-fp} censored) of {nst} | PPV(lead>0 vs FP) {ppv:.2f} | PPV(all alarms) {ppv_all:.2f} | alarms per 100 eligible wells {(tp0+fpc)/(ntr+nst)*100:.1f}")
print('prevalence of transitions in eligible pool',ntr/(ntr+nst))
# ---------- J. stricter subsets for lead150 / FP150 (transitions only change lead; FP unchanged)
for lab,mask in [('all',np.ones(ntr,bool)),('median interval<=1 yr',desc.med_int.to_numpy()<=1),('median interval<=0.5 yr',desc.med_int.to_numpy()<=0.5),
                 ('>=10 pre-crossing samples',desc.n_pre.to_numpy()>=10),('>=20 pre-crossing samples',desc.n_pre.to_numpy()>=20),('pre-crossing span>=5 yr',desc.pre_span.to_numpy()>=5),
                 ('not left-censored at 150',~cens),('crossing year>=1990',desc.year250.to_numpy()>=1990),('crossing year>=2000',desc.year250.to_numpy()>=2000)]:
    print(f"{lab:30s} n={mask.sum():3d} median lead150={np.median(lead[mask]):.2f} | lead>=2yr {100*(lead[mask]>=2).mean():.0f}%")
