import pandas as pd, numpy as np
from scipy import stats, optimize
from scipy.ndimage import gaussian_filter1d
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250'])
stable=pd.read_csv('raw/stable_ids.csv').iloc[:,0].tolist()
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
def ac1(x):
    x=np.asarray(x,float); x=x-x.mean(); d=np.sum(x[:-1]**2); return np.sum(x[1:]*x[:-1])/d if d>0 else np.nan
def dac1(x,detrend=False):
    x=np.asarray(x,float); n=len(x); w=max(4,n//2)
    if detrend: x=x-gaussian_filter1d(x,sigma=max(1,0.2*n),mode='nearest')
    return ac1(x[n-w:])-ac1(x[:w])
def cliffs(a,b):
    a=np.asarray(a); b=np.asarray(b); return ((a[:,None]>b[None,:]).sum()-(a[:,None]<b[None,:]).sum())/(len(a)*len(b))
pre={}
for _,r in ev.iterrows():
    g=G[r.site]; gg=g[g.date<r.date_250]
    if len(gg)>=8: pre[r.site]=(gg.chloride.to_numpy(), (gg.date-gg.date.min()).dt.days.to_numpy()/365.25)
stab={s:(G[s].chloride.to_numpy(),(G[s].date-G[s].date.min()).dt.days.to_numpy()/365.25) for s in stable}
lens=np.array([len(v[0]) for v in stab.values()]); keys=list(stab)
# ---- 1) repeated random control draws
rng=np.random.default_rng(42)
for detrend in [False,True]:
    dT=np.array([dac1(x,detrend) for x,_ in pre.values()])
    ps1=[]; ps2=[]; ds=[]
    for rep in range(1000):
        dC=[]
        for x,_ in pre.values():
            n=len(x); c=np.where(lens>=n)[0]; k=keys[c[rng.integers(len(c))]]; v=stab[k][0]; st=rng.integers(0,len(v)-n+1)
            dC.append(dac1(v[st:st+n],detrend))
        dC=np.array(dC); m=np.isfinite(dC)
        ps1.append(stats.mannwhitneyu(dT,dC[m],alternative='greater').pvalue); ps2.append(stats.mannwhitneyu(dT,dC[m]).pvalue); ds.append(cliffs(dT,dC[m]))
    ps1=np.array(ps1); ps2=np.array(ps2); ds=np.array(ds)
    print(f"detrend={detrend}: n={len(dT)} | one-sided P median {np.median(ps1):.3f} (IQR {np.percentile(ps1,25):.3f}-{np.percentile(ps1,75):.3f}), frac<0.05={np.mean(ps1<0.05):.2f} | two-sided median {np.median(ps2):.3f} | Cliff d median {np.median(ds):+.3f} (IQR {np.percentile(ds,25):+.3f},{np.percentile(ds,75):+.3f})")
# ---- 2) irregular-sampling AR(1) persistence time tau (TAUEST-type), early vs late half
def tau_fit(t,x):
    x=np.asarray(x,float); t=np.asarray(t,float)
    o=np.argsort(t); t,x=t[o],x[o]; keep=np.r_[True,np.diff(t)>0]; t,x=t[keep],x[keep]
    if len(x)<4 or np.std(x)==0: return np.nan
    x=(x-x.mean())/x.std(); dt=np.diff(t)
    def S(lt):
        a=np.exp(-dt/np.exp(lt)); return np.sum((x[1:]-a*x[:-1])**2)
    r=optimize.minimize_scalar(S,bounds=(np.log(1e-3),np.log(100)),method='bounded'); return np.exp(r.x)
def dtau(x,t,detrend=False):
    n=len(x); h=n//2
    if detrend: x=x-gaussian_filter1d(np.asarray(x,float),sigma=max(1,0.2*n),mode='nearest')
    a=tau_fit(t[:h],x[:h]); b=tau_fit(t[h:],x[h:])
    return np.log(b/a) if (np.isfinite(a) and np.isfinite(b) and a>0 and b>0) else np.nan
for detrend in [False,True]:
    lT=np.array([dtau(x,t,detrend) for x,t in pre.values()]); mT=np.isfinite(lT)
    ps=[]; ds=[]
    for rep in range(300):
        lC=[]
        for x,t in pre.values():
            n=len(x); c=np.where(lens>=n)[0]; k=keys[c[rng.integers(len(c))]]; v,tv=stab[k]; st=rng.integers(0,len(v)-n+1)
            lC.append(dtau(v[st:st+n],tv[st:st+n],detrend))
        lC=np.array(lC); m=np.isfinite(lC)
        ps.append(stats.mannwhitneyu(lT[mT],lC[m],alternative='greater').pvalue); ds.append(cliffs(lT[mT],lC[m]))
    print(f"TAU detrend={detrend}: n={mT.sum()} | median log(tau_late/tau_early) trans={np.median(lT[mT]):+.3f} | within-trans Wilcoxon vs0 p={stats.wilcoxon(lT[mT]).pvalue:.3f} | vs controls one-sided P median {np.median(ps):.3f}, frac<0.05={np.mean(np.array(ps)<0.05):.2f}, Cliff d median {np.median(ds):+.3f}")
np.save('raw/ews_pre_sites.npy',np.array(list(pre)))
