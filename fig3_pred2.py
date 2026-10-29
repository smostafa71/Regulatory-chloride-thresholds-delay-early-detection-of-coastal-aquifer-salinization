import pandas as pd, numpy as np, itertools
from scipy import stats
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250','date_150'])
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
def first_single(g,th,after=None):
    x=g.chloride.to_numpy(); i=np.where(x>=th)[0]; return g.date.iloc[i[0]] if len(i) else None
def run(strict,cross,req_pos,below_start,nmin=10,smin=2.5):
    rows=[]
    for _,r in ev.iterrows():
        g=G[r.site]
        t150=r.date_150 if cross=='persistent' else first_single(g,150)
        if t150 is None or pd.isna(t150): continue
        hist=g[g.date<t150] if strict else g[g.date<=t150]
        if len(hist)<3: continue
        if below_start and hist.chloride.iloc[0]>=150: continue
        n=len(hist); span=(hist.date.max()-hist.date.min()).days/365.25
        if n<nmin or span<smin or hist.date.nunique()<3: continue
        rem=(r.date_250-t150).days/365.25
        if req_pos and rem<=0: continue
        t=(hist.date-hist.date.min()).dt.days.to_numpy()/365.25; y=hist.chloride.to_numpy()
        q=np.polyfit(t,y,2)[0]; b,a=np.polyfit(t,y,1)
        tc=(t150-hist.date.min()).days/365.25
        pred=(250-(a+b*tc))/b if b>0 else np.nan          # time from decision moment to reach 250
        pred2=(250-(a+b*t[-1]))/b if b>0 else np.nan
        rows.append(dict(site=r.site,n=n,curv=q,slope=b,recent=y[-1],remaining=rem,pred=pred,pred2=pred2))
    return pd.DataFrame(rows)
for strict,cross,req,bs in itertools.product([True,False],['persistent','single'],[False,True],[False,True]):
    P=run(strict,cross,req,bs)
    if len(P)==0: continue
    s=[f"strict={strict!s:5} cross={cross:10s} rem>0={req!s:5} start<150={bs!s:5} n={len(P):3d}"]
    for c in ['curv','slope','recent']:
        r,p=stats.spearmanr(P[c],P.remaining); s.append(f"{c}:{r:+.2f}({p:.2f})")
    for pc in ['pred','pred2']:
        F=P[(P.slope>0)&np.isfinite(P[pc])]
        e=np.abs(F[pc]-F.remaining); r,p=stats.spearmanr(F[pc],F.remaining)
        s.append(f"|{pc} n={len(F)} MAE={np.median(e):.1f} w1={100*(e<=1).mean():.0f}% rho={r:+.2f}")
    print(' '.join(s))
