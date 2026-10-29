import pandas as pd, numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250'])
stable=pd.read_csv('raw/stable_ids.csv').iloc[:,0].tolist()
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
T=[]
for _,r in ev.iterrows():
    g=G[r.site]; gg=g[g.date<r.date_250]
    if len(gg)>=8: T.append((gg.date-gg.date.min()).dt.days.to_numpy()/365.25)
C=[(G[s].date-G[s].date.min()).dt.days.to_numpy()/365.25 for s in stable]
Clen=np.array([len(c) for c in C])
def ac1(x):
    x=x-x.mean(); d=np.sum(x[:-1]**2); return np.sum(x[1:]*x[:-1])/d if d>0 else np.nan
def dac1(x):
    n=len(x); w=max(4,n//2); return ac1(x[n-w:])-ac1(x[:w])
def sim_ou(t,lam_fun,rng,obs_sd=0.0):
    x=np.zeros(len(t)); x[0]=rng.normal(0,np.sqrt(1/(2*lam_fun(t[0]))))
    for i in range(1,len(t)):
        dt=max(t[i]-t[i-1],1e-6); lam=lam_fun(0.5*(t[i]+t[i-1]))
        a=np.exp(-lam*dt); x[i]=a*x[i-1]+rng.normal(0,np.sqrt((1-a*a)/(2*lam)))
    return x+rng.normal(0,obs_sd,len(t))
def cliffs(a,b): return ((a[:,None]>b[None,:]).sum()-(a[:,None]<b[None,:]).sum())/(len(a)*len(b))
rng=np.random.default_rng(7)
rows=[]
for lam0 in [1.0,4.0,12.0]:
  for fold in [5,20]:
    for obs in [0.0,0.5]:
      pw=[]; ds=[]
      for rep in range(200):
        dT=[]; dC=[]
        for t in T:
            L=t[-1]-t[0]+1e-9
            lf=lambda s,t0=t[0],L=L: lam0*(1-(1-1/fold)*(s-t0)/L)   # linear decline to lam0/fold at crossing
            x=sim_ou(t,lf,rng,obs*np.sqrt(1/(2*lam0))); dT.append(dac1(x))
            n=len(t); c=np.where(Clen>=n)[0]; tc=C[c[rng.integers(len(c))]]; st=rng.integers(0,len(tc)-n+1); tt=tc[st:st+n]
            xc=sim_ou(tt,lambda s: lam0,rng,obs*np.sqrt(1/(2*lam0))); dC.append(dac1(xc))
        dT=np.array(dT); dC=np.array(dC); m1=np.isfinite(dT); m2=np.isfinite(dC)
        pw.append(stats.mannwhitneyu(dT[m1],dC[m2],alternative='greater').pvalue<0.05); ds.append(cliffs(dT[m1],dC[m2]))
      rows.append(dict(lam0=lam0,memory_yr=1/lam0,fold_factor=fold,obs_noise=obs,power=np.mean(pw),cliff_median=np.median(ds)))
      print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('raw/ews_power.csv',index=False)
