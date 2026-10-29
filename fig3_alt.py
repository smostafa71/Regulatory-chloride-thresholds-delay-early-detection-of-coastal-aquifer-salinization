import pandas as pd, numpy as np
from scipy import stats
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250','date_150'])
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
def build(nmin,smin):
    rows=[]
    for _,r in ev.iterrows():
        g=G[r.site]; t150=r.date_150
        if pd.isna(t150): continue
        hist=g[g.date<=t150]; n=len(hist); span=(hist.date.max()-hist.date.min()).days/365.25
        if n<nmin or span<smin or hist.date.nunique()<3: continue
        rem=(r.date_250-t150).days/365.25
        if rem<=0: continue
        t=(hist.date-hist.date.min()).dt.days.to_numpy()/365.25; y=hist.chloride.to_numpy()
        rows.append(dict(curv=np.polyfit(t,y,2)[0],slope=np.polyfit(t,y,1)[0],recent=y[-1],rem=rem))
    return pd.DataFrame(rows)
allp=[]
for nmin,smin in [(10,2.5),(8,2.0),(6,1.5)]:
    P=build(nmin,smin); out=[f">={nmin} samples/{smin} yr: n={len(P)}"]
    for c in ['curv','slope','recent']:
        r,p=stats.spearmanr(P[c],P.rem); allp.append((nmin,c,r,p,P)); out.append(f"{c} rho={r:+.2f} p={p:.3f}")
    print(' | '.join(out))
m=len(allp); print('Bonferroni alpha',0.05/m)
for nmin,c,r,p,P in allp:
    if p<0.05:
        rng=np.random.default_rng(0); bs=[]
        for _ in range(4000):
            i=rng.integers(0,len(P),len(P)); bs.append(stats.spearmanr(P[c].to_numpy()[i],P.rem.to_numpy()[i])[0])
        print('nominal sig:',nmin,c,round(r,2),round(p,3),' Bonferroni sig?',p<0.05/m,' bootstrap 95% CI',np.round(np.nanpercentile(bs,[2.5,97.5]),2))
