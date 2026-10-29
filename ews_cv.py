import pandas as pd, numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250'])
stable=pd.read_csv('raw/stable_ids.csv').iloc[:,0].tolist()
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
pre=[G[r.site][G[r.site].date<r.date_250].chloride.to_numpy() for _,r in ev.iterrows()]; pre=[x for x in pre if len(x)>=8]
stab=[G[s].chloride.to_numpy() for s in stable]; lens=np.array([len(v) for v in stab])
def cv(x): m=np.mean(x); return np.std(x)/m if m>0 else np.nan
def sd(x): return np.std(x)
def dind(x,f,detrend):
    n=len(x); w=max(4,n//2); x=np.asarray(x,float)
    if detrend: x=x-gaussian_filter1d(x,sigma=max(1,0.2*n),mode='nearest')
    return f(x[n-w:])-f(x[:w])
def cliffs(a,b): return ((a[:,None]>b[None,:]).sum()-(a[:,None]<b[None,:]).sum())/(len(a)*len(b))
rng=np.random.default_rng(11)
for name,f,detr in [('CV raw',cv,False),('SD detrended (variance)',sd,True)]:
    dT=np.array([dind(x,f,detr) for x in pre]); dT=dT[np.isfinite(dT)]
    ps=[];ds=[]
    for rep in range(500):
        dC=[]
        for x in pre:
            n=len(x); c=np.where(lens>=n)[0]; v=stab[c[rng.integers(len(c))]]; st=rng.integers(0,len(v)-n+1); dC.append(dind(v[st:st+n],f,detr))
        dC=np.array(dC); dC=dC[np.isfinite(dC)]
        ps.append(stats.mannwhitneyu(dT,dC,alternative='greater').pvalue); ds.append(cliffs(dT,dC))
    print(f"{name}: one-sided P median {np.median(ps):.3f} (IQR {np.percentile(ps,25):.3f}-{np.percentile(ps,75):.3f}), frac<0.05 {np.mean(np.array(ps)<0.05):.2f}, Cliff median {np.median(ds):+.3f}")
print('--- log-scale check')
for name,f,detr in [('SD of log(Cl) detrended',sd,True)]:
    preL=[np.log(np.maximum(x,0.5)) for x in pre]; stabL=[np.log(np.maximum(v,0.5)) for v in stab]
    dT=np.array([dind(x,f,detr) for x in preL]); ps=[];ds=[]
    for rep in range(300):
        dC=[]
        for x in preL:
            n=len(x); c=np.where(lens>=n)[0]; v=stabL[c[rng.integers(len(c))]]; st=rng.integers(0,len(v)-n+1); dC.append(dind(v[st:st+n],f,detr))
        dC=np.array(dC); ps.append(stats.mannwhitneyu(dT,dC,alternative='greater').pvalue); ds.append(cliffs(dT,dC))
    print(f"{name}: one-sided P median {np.median(ps):.3f}, frac<0.05 {np.mean(np.array(ps)<0.05):.2f}, Cliff median {np.median(ds):+.3f}")
# correlation of window SD with window mean across all windows in stable wells (heteroscedasticity)
m=[];s=[]
for v in stab[:800]:
    for i in range(0,len(v)-8,8): seg=v[i:i+8]; m.append(np.mean(seg)); s.append(np.std(seg))
print('stable wells: Spearman(window mean, window SD) =',stats.spearmanr(m,s))
