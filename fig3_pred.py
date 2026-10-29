import pandas as pd, numpy as np
from scipy import stats
D=pd.read_csv('raw/fig3_curv.csv'); sel=D[(D.n_pre>=12)&(D.span_pre>=3)]
for lab,x in [('50-150 incl',sel[(sel.baseline>=50)&(sel.baseline<=150)].quad),('>150',sel[sel.baseline>150].quad)]:
    print(f"{lab:12s} n={len(x)} pos%={100*(x>0).mean():.0f} med={np.median(x):+.3f} p={stats.wilcoxon(x).pvalue:.4f}")
# ---------- predictability
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250','date_150'])
G={s:g.sort_values('date') for s,g in nat.groupby('site')}
def first_cross_single(g,th):
    i=np.where(g.chloride.to_numpy()>=th)[0]; return g.date.iloc[i[0]] if len(i) else None
def analyse(nmin,smin,cross='persistent',slope_win=None):
    rows=[]
    for _,r in ev.iterrows():
        g=G[r.site]
        t150=r.date_150 if cross=='persistent' else first_cross_single(g,150)
        if t150 is None or pd.isna(t150): continue
        hist=g[g.date<=t150]
        n=len(hist); span=(hist.date.max()-hist.date.min()).days/365.25
        if n<nmin or span<smin or hist.date.nunique()<3: continue
        t=(hist.date-hist.date.min()).dt.days.to_numpy()/365.25; y=hist.chloride.to_numpy()
        q=np.polyfit(t,y,2)[0]
        if slope_win is None: lt,ly=t,y
        else: m=t>=t[-1]-slope_win; lt,ly=t[m],y[m]
        b,a=np.polyfit(lt,ly,1) if len(np.unique(lt))>=2 else (np.nan,np.nan)
        b_full,a_full=np.polyfit(t,y,1)
        remaining=(r.date_250-t150).days/365.25
        # linear extrapolation to 250 from full-history linear fit
        pred=(250-(a_full+b_full*t[-1]))/b_full if b_full>0 else np.nan
        rows.append(dict(site=r.site,n=n,span=span,curv=q,slope=b,slope_full=b_full,recent=y[-1],remaining=remaining,pred=pred))
    return pd.DataFrame(rows)
for cross in ['persistent','single']:
  for sw in [None,2.0,3.0]:
    P=analyse(10,2.5,cross,sw)
    out=[f"{cross:10s} slopewin={sw}: n={len(P)}"]
    for c in ['curv','slope','recent']:
        s=P.dropna(subset=[c]); r,p=stats.spearmanr(s[c],s.remaining); out.append(f"{c} rho={r:+.2f} p={p:.2f}")
    F=P[(P.slope_full>0)&np.isfinite(P.pred)]
    if len(F):
        err=np.abs(F.pred-F.remaining); r,p=stats.spearmanr(F.pred,F.remaining)
        out.append(f"| forecast n={len(F)} MAE_med={np.median(err):.2f} within1={100*(err<=1).mean():.0f}% rho_pred={r:+.2f}")
    print(' '.join(out))
P=analyse(10,2.5,'persistent',None); P.to_csv('raw/fig3_pred.csv',index=False)
