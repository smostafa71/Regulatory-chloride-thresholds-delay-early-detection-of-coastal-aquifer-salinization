import pandas as pd, numpy as np, pickle
from scipy import stats
from scipy.ndimage import gaussian_filter1d
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250'])
stable=pd.read_csv('raw/stable_ids.csv').iloc[:,0].tolist()
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
def ac1(x):
    x=np.asarray(x,float); x=x-x.mean(); d=np.sum(x[:-1]**2)
    return np.sum(x[1:]*x[:-1])/d if d>0 else np.nan
def cv(x): x=np.asarray(x,float); return np.std(x)/np.mean(x) if np.mean(x)>0 else np.nan
def delta(x,detrend=False):
    x=np.asarray(x,float); n=len(x); w=max(4,n//2)
    if detrend: x=x-gaussian_filter1d(x,sigma=max(1,0.2*n),mode='nearest')
    a=[ac1(x[i:i+w]) for i in range(0,n-w+1)]; c=[cv(np.asarray(G0[i:i+w])) if False else np.nan for i in range(0)]
    return a[-1]-a[0] if len(a)>=2 else np.nan, a
def cliffs(a,b):
    a=np.asarray(a); b=np.asarray(b); gt=(a[:,None]>b[None,:]).sum(); lt=(a[:,None]<b[None,:]).sum(); return (gt-lt)/(len(a)*len(b))
pre={}
for _,r in ev.iterrows():
    g=G[r.site]; x=g[g.date<r.date_250].chloride.to_numpy()
    if len(x)>=8: pre[r.site]=x
print('transition wells with >=8 pre-crossing samples:',len(pre))
stab={s:G[s].chloride.to_numpy() for s in stable}
rng=np.random.default_rng(0)
for detrend in [False,True]:
  for ctrl_mode in ['first','random']:
    dT=[]; dC=[]; cT=[]; cC=[]
    for s,x in pre.items():
        d,_=delta(x,detrend); dT.append(d)
        n=len(x); w=max(4,n//2)
        cT.append(cv(x[-w:])-cv(x[:w]))
        cands=[k for k,v in stab.items() if len(v)>=n]
        k=cands[rng.integers(len(cands))]; v=stab[k]
        st=0 if ctrl_mode=='first' else rng.integers(0,len(v)-n+1)
        seg=v[st:st+n]; dc,_=delta(seg,detrend); dC.append(dc); cC.append(cv(seg[-w:])-cv(seg[:w]))
    dT=np.array(dT); dC=np.array(dC); mT=np.isfinite(dT); mC=np.isfinite(dC)
    u=stats.mannwhitneyu(dT[mT],dC[mC],alternative='two-sided'); u1=stats.mannwhitneyu(dT[mT],dC[mC],alternative='greater')
    cTa=np.array(cT); cCa=np.array(cC)
    print(f"detrend={detrend!s:5} ctrl={ctrl_mode:6s} | dAC1 trans median {np.nanmedian(dT):+.3f} ctrl {np.nanmedian(dC):+.3f} | MW two-sided p={u.pvalue:.3f} one-sided p={u1.pvalue:.3f} | Cliff d={cliffs(dT[mT],dC[mC]):+.3f} | dCV p={stats.mannwhitneyu(cTa[np.isfinite(cTa)],cCa[np.isfinite(cCa)]).pvalue:.3f} | dAC1>0 in trans: {100*(dT[mT]>0).mean():.0f}% wilcoxon vs0 p={stats.wilcoxon(dT[mT]).pvalue:.3f}")
