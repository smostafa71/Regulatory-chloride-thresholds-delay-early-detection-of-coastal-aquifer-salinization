import pandas as pd, numpy as np, pickle, requests, io
from scipy import stats
V1=pickle.load(open('raw/V1.pkl','rb')); arr=V1['arr']; lab=V1['res']['labels']
m=pd.read_csv('raw/meta_pool.csv',low_memory=False).set_index('site')
pool=[s for s in lab.index if lab[s]!='short']
rows=[]
for s in pool:
    cl,dt,st=arr[s]; iv=np.diff(dt).astype(float)/365.25
    rows.append(dict(site=s,label=lab[s],state=st,n=len(cl),span=(dt[-1]-dt[0]).astype(int)/365.25,med_int=np.median(iv) if len(iv) else np.nan,first_year=int(str(dt[0])[:4]),last_year=int(str(dt[-1])[:4])))
P=pd.DataFrame(rows).set_index('site').join(m[['depth_ft','coast_km_marine','lithology']] if 'lithology' in m else m[['depth_ft','coast_km_marine']])
P.to_csv('raw/pool_desc.csv')
grp={'transition':'Transition (n=145)','fresh_or_noisy':'Fresh/noisy','always_saline':'Always saline'}
st_ids=set(pd.read_csv('raw/stable_ids.csv').iloc[:,0])
P['grp']=np.where(P.label=='transition','T',np.where(P.index.isin(st_ids),'S','O'))
def q(x): x=x.dropna(); return f"{x.median():.1f} [{x.quantile(.25):.1f}–{x.quantile(.75):.1f}] (n={len(x)})"
for c in ['n','span','med_int','depth_ft','coast_km_marine','first_year']:
    a=P[P.grp=='T'][c].dropna(); b=P[P.grp=='S'][c].dropna(); p=stats.mannwhitneyu(a,b).pvalue
    print(f"{c:16s} T {q(P[P.grp=='T'][c])} | S {q(P[P.grp=='S'][c])} | always-saline {q(P[P.label=='always_saline'][c])} | MW p={p:.3g}")
ct=pd.crosstab(P.state,P.grp); print(ct.T.to_string())
# ---- D. watch list: wells whose most recent sample is in [150,250) and recent (>=2015)
nat=V1['nat']
last=nat.sort_values('date').groupby('site').tail(2)
g=last.groupby('site').agg(last_date=('date','max'),last_cl=('chloride','last'),prev_cl=('chloride','first'),n2=('chloride','size'),state=('state','first'))
ever250={s for s,(cl,dt,st) in arr.items() if (cl>=250).any()}
for since in ['2010-01-01','2015-01-01','2020-01-01']:
    w=g[(g.last_date>=since)&(g.last_cl>=150)&(g.last_cl<250)]
    w2=w[(w.n2==2)&(w.prev_cl>=150)]        # two most recent samples >=150 (persistent)
    w3=w2[~w2.index.isin(ever250)]          # never yet >=250
    print(f"since {since[:4]}: last sample 150-250: {len(w)} | last two >=150: {len(w2)} | and never >=250: {len(w3)} | states {w3.state.value_counts().to_dict()}")
    tot=g[g.last_date>=since]; print('   wells sampled since',since[:4],':',len(tot))
w3.to_csv('raw/watchlist_2015.csv') if False else None
w=g[(g.last_date>='2015-01-01')&(g.last_cl>=150)&(g.last_cl<250)]; w2=w[(w.n2==2)&(w.prev_cl>=150)]; w3=w2[~w2.index.isin(ever250)]
w3.to_csv('raw/watchlist_2015.csv')
