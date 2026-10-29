import pandas as pd, numpy as np, pickle
from scipy import stats
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250','date_150'])
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
def yrs(dt,t0): return (dt-t0).dt.days.to_numpy()/365.25
rows=[]
for _,r in ev.iterrows():
    g=G[r.site]; pre=g[g.date<r.date_250]
    n=len(pre); span=(pre.date.max()-pre.date.min()).days/365.25 if n>1 else 0
    rec=dict(site=r.site,n_pre=n,span_pre=span,baseline=r.baseline_chloride)
    if n>=3 and span>0 and len(np.unique(pre.date))>=3:
        t=yrs(pre.date,pre.date.min()); y=pre.chloride.to_numpy()
        q=np.polyfit(t,y,2); l=np.polyfit(t,y,1)
        rec.update(quad=q[0],curv2=2*q[0])
        # F-test quad vs linear
        rss2=np.sum((y-np.polyval(q,t))**2); rss1=np.sum((y-np.polyval(l,t))**2)
        if n>3 and rss2>0: F=(rss1-rss2)/(rss2/(n-3)); rec['F_p']=1-stats.f.cdf(F,1,n-3)
        # sampling density first vs second half of pre-transition time span
        mid=span/2; rec['n_first']=(t<mid).sum(); rec['n_second']=(t>=mid).sum()
    rows.append(rec)
D=pd.DataFrame(rows)
for nmin,smin in [(12,3),(12,3.0)]:
    sel=D[(D.n_pre>=nmin)&(D.span_pre>=smin)]
    print('n with >=%d pre samples & span>=%.1f yr:'%(nmin,smin),len(sel))
sel=D[(D.n_pre>=12)&(D.span_pre>=3)].copy()
def summ(x,lab):
    w=stats.wilcoxon(x) if len(x)>0 else None
    print(f"{lab:10s} n={len(x):3d} pos%={100*(x>0).mean():.0f} median={np.median(x):+.3f} Wilcoxon p={w.pvalue:.4f}")
for col in ['quad','curv2']:
    print('---',col)
    summ(sel[col],'all')
    for lo,hi,lab in [(-1,50,'<50'),(50,150,'50-150'),(150,1e9,'>150')]:
        summ(sel[(sel.baseline>=lo)&(sel.baseline<hi)][col] if lo>=0 else sel[sel.baseline<50][col],lab)
# grouping with <=/>= boundaries
print('group sizes (baseline<50, 50<=b<=150, >150):',(sel.baseline<50).sum(),((sel.baseline>=50)&(sel.baseline<=150)).sum(),(sel.baseline>150).sum())
print('F-test quad better p<0.05:',(sel.F_p<0.05).sum(),'of',len(sel))
w=stats.wilcoxon(sel.n_first,sel.n_second); print('density first vs second half Wilcoxon p',w.pvalue, ' MannWhitney', stats.mannwhitneyu(sel.n_first,sel.n_second).pvalue)
D.to_csv('raw/fig3_curv.csv',index=False)
