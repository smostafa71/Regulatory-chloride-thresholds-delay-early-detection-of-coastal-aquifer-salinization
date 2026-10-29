import pandas as pd, numpy as np, glob
from scipy import stats
W=pd.concat([pd.read_csv(f) for f in glob.glob('raw/wl/*.csv')],ignore_index=True)
W['date']=pd.to_datetime(W.date,errors='coerce',utc=True).dt.tz_localize(None); W=W.dropna(subset=['date','wl'])
W['param']=W.param.astype(int); W['wl']=W.wl.astype(float)
keep=[]
for s,g in W.groupby('site'):
    for p in [72019,62610,62611]:
        if (g.param==p).any(): keep.append(g[g.param==p]); break
W=pd.concat(keep); cnt=W.groupby('site').size(); Wv=W[W.site.isin(cnt[cnt>=5].index)].copy()
Wv.to_csv('raw/waterlevels_transition.csv',index=False)
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250','date_150'])
med=Wv.groupby('site').wl.median().rename('wl_med').reset_index()
d=ev.merge(med,on='site',how='inner')
r,p=stats.spearmanr(d.lead_150.astype(float).to_numpy(),d.wl_med.astype(float).to_numpy()); print('rho(lead150, median WL)',round(r,3),round(p,4),'n',len(d))
print('param used:',Wv.groupby('site').param.first().value_counts().to_dict())
def test(kind,win):
    diffs=[]
    for _,r in ev.iterrows():
        g=Wv[Wv.site==r.site]
        if not len(g): continue
        sign=1 if g.param.iloc[0]==72019 else -1
        t=r.date_250; dw=pd.Timedelta(days=365.25*win)
        pre=g[(g.date<t)&(g.date>=t-dw)]
        if kind=='pre_vs_earlier': other=g[g.date<t-dw]; val=lambda: pre.wl.median()-other.wl.median()
        elif kind=='pre_vs_prev_window': other=g[(g.date<t-dw)&(g.date>=t-2*dw)]; val=lambda: pre.wl.median()-other.wl.median()
        elif kind=='post_vs_pre': other=g[(g.date>=t)&(g.date<t+dw)]; val=lambda: other.wl.median()-pre.wl.median()
        if len(pre) and len(other): diffs.append(sign*val())
    diffs=np.array(diffs,float)
    return len(diffs),np.median(diffs),stats.wilcoxon(diffs).pvalue
for kind in ['pre_vs_earlier','pre_vs_prev_window','post_vs_pre']:
    for win in [1,2,3,5]:
        n,m,p=test(kind,win); print(f"{kind:20s} {win}y n={n:3d} median diff={m:+7.2f} ft  Wilcoxon p={p:.3f}")
