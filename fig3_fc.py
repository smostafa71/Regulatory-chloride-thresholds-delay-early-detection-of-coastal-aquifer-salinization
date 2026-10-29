import pandas as pd, numpy as np
from scipy import stats
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250','date_150'])
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
rows=[]
for _,r in ev.iterrows():
    g=G[r.site]; t150=r.date_150
    if pd.isna(t150): continue
    hist=g[g.date<=t150]; n=len(hist); span=(hist.date.max()-hist.date.min()).days/365.25
    if n<10 or span<2.5 or hist.date.nunique()<3: continue
    rem=(r.date_250-t150).days/365.25
    if rem<=0: continue
    t=(hist.date-hist.date.min()).dt.days.to_numpy()/365.25; y=hist.chloride.to_numpy()
    b,a=np.polyfit(t,y,1); q=np.polyfit(t,y,2)
    rec=dict(site=r.site,rem=rem,b=b)
    rec['f_fit_end']=(250-(a+b*t[-1]))/b if b>0 else np.nan
    rec['f_last']=(250-y[-1])/b if b>0 else np.nan
    rec['f_t0']=(250-a)/b - t[-1] if b>0 else np.nan
    rec['f_zero']=(250-a)/b if b>0 else np.nan            # time from record start to 250 (bug-like)
    for k in [3,4,5,6]:
        tl,yl=t[-k:],y[-k:]
        if len(np.unique(tl))>=2:
            bl,al=np.polyfit(tl,yl,1); rec[f'f_loc{k}']=(250-(al+bl*tl[-1]))/bl if bl>0 else np.nan; rec[f'b_loc{k}']=bl
    for w in [1,2,3,5]:
        m=t>=t[-1]-w
        if m.sum()>=2 and len(np.unique(t[m]))>=2:
            bl,al=np.polyfit(t[m],y[m],1); rec[f'f_w{w}']=(250-(al+bl*t[-1]))/bl if bl>0 else np.nan
    # theil-sen
    ts=stats.theilslopes(y,t); rec['f_sen']=(250-(ts[1]+ts[0]*t[-1]))/ts[0] if ts[0]>0 else np.nan
    # quadratic extrapolation
    c=np.poly1d(q); roots=(c-250).r; rr=[x.real for x in roots if abs(x.imag)<1e-9 and x.real>t[-1]]
    rec['f_quad']=min(rr)-t[-1] if rr else np.nan
    rows.append(rec)
P=pd.DataFrame(rows); print('n',len(P))
for c in [c for c in P.columns if c.startswith('f_')]:
    F=P[np.isfinite(P[c])&(P[c]>=0)] if True else P
    F2=P[np.isfinite(P[c])]
    for lab,FF in [('all finite',F2),('nonneg',F)]:
        if len(FF)<5: continue
        e=np.abs(FF[c]-FF.rem); r,p=stats.spearmanr(FF[c],FF.rem)
        print(f"{c:10s} {lab:10s} n={len(FF):2d} MAE={np.median(e):5.2f} w1={100*(e<=1).mean():3.0f}% rho={r:+.2f} p={p:.2f}")
P.to_csv('raw/fig3_forecasts.csv',index=False)
