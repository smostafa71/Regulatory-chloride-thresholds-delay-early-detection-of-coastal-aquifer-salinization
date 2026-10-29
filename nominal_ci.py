"""Bootstrap interval of the single nominally significant association (most recent concentration vs
remaining time, relaxed criterion >=6 results over 1.5 yr, n = 41): unadjusted 95% and Bonferroni-adjusted
(1 - 0.05/9 = 99.44%) percentile intervals, 20,000 resamples."""
import numpy as np, pandas as pd
from scipy import stats
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250','date_150'])
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
x=[];y=[]
for _,r in ev.iterrows():
    h=G[r.site]; h=h[h.date<=r.date_150]
    if len(h)<6 or (h.date.max()-h.date.min()).days/365.25<1.5 or h.date.nunique()<3: continue
    rem=(r.date_250-r.date_150).days/365.25
    if rem<=0: continue
    x.append(h.chloride.iloc[-1]); y.append(rem)
x=np.array(x); y=np.array(y); print('n',len(x), stats.spearmanr(x,y))
rng=np.random.default_rng(2026); bs=[]
for _ in range(20000):
    i=rng.integers(0,len(x),len(x)); bs.append(stats.spearmanr(x[i],y[i])[0])
a=0.05/9
print('95%',np.round(np.nanpercentile(bs,[2.5,97.5]),2),' Bonferroni-adjusted',np.round(np.nanpercentile(bs,[100*a/2,100*(1-a/2)]),2))
