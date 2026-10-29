import pandas as pd, numpy as np
from scipy import stats
d=pd.read_csv('raw/trans_full.csv')     # V1 events (prep ordering lead_150) merged with meta
L=np.load('raw/lead_mat.npy'); TH=list(range(50,245,5)); F=np.load('raw/fp_mat.npy')
assert list(d.site)==sorted(d.site)
d['lead150']=L[:,TH.index(150)]
nat=d.national_aquifer_name.fillna(d.wqp_aquifer).fillna('')
loc=d.aquifer_name.fillna(d.wqp_local_aquifer).fillna('')
km=d.coast_km_marine
floridan=nat.str.contains('Floridan')|loc.str.contains('Floridan|Avon Park|Suwannee|Tampa Limestone',regex=True)
gulfsand=(d.state=='LA')&nat.str.contains('Coastal lowlands')
mech=np.where(km>100,'M4',np.where(floridan,'M2',np.where(gulfsand,'M3',np.where(km<=15,'M1','M4'))))
d['mech']=mech
names={'M1':'Lateral seawater/estuarine intrusion (unconfined–semi-confined coastal aquifers)',
       'M2':'Lateral intrusion and upconing in the confined karst Floridan aquifer system',
       'M3':'Pumping-induced encroachment of relict saline water (Gulf Coast confined sands)',
       'M4':'Inland or non-marine sources (>15 km from shore outside M2–M3, or >100 km)'}
d['mech_name']=d.mech.map(names)
def lith(n,l,s):
    t=(n+' '+l).lower()
    if any(k in t for k in ['biscayne','floridan','avon park','suwannee','tampa limestone','edwards']): return 'Karst carbonate'
    if 'intermediate' in t or 'hawthorn' in t or 'tamiami' in t: return 'Mixed carbonate–clastic'
    if any(k in t for k in ['saprolite','crystalline','mudstone']): return 'Fractured / consolidated'
    if t.strip()=='' : return 'Unknown'
    return 'Unconsolidated clastic'
d['lithology']=[lith(a,b,s) for a,b,s in zip(nat,loc,d.state)]
# SC Floridan has nat from WQP
print(d.groupby('mech').agg(n=('site','size'),states=('state',lambda x: ','.join(sorted(set(x)))),km_med=('coast_km_marine','median'),lead_med=('lead150','median'),p2=('lead150',lambda x:(x>=2).mean())).round(2).to_string())
print(stats.kruskal(*[g.lead150 for _,g in d.groupby('mech')]))
print(d.groupby('lithology').agg(n=('site','size'),lead_med=('lead150','median'),p2=('lead150',lambda x:(x>=2).mean()),depth=('depth_ft','median')).round(2).to_string())
print(stats.kruskal(*[g.lead150 for k,g in d.groupby('lithology') if k!='Unknown' and len(g)>=3]))
d.to_csv('raw/trans_classified.csv',index=False)
# ---- coastal-distance subsets with matching stable-control subsets (FP)
m=pd.read_csv('raw/meta_pool.csv',low_memory=False).set_index('site')
st_ids=pd.read_csv('raw/stable_ids.csv').iloc[:,0].tolist()
km_st=m.reindex(st_ids).coast_km_marine.to_numpy()
j=TH.index(150); j1=TH.index(100); j2=TH.index(200)
def fp_of(mask,jj):
    Fj=F[mask,jj]; return (Fj==1).sum()/((Fj==1)|(Fj==0)).sum()
print('\nstable controls with coords',np.isfinite(km_st).sum(),'of',len(st_ids))
for lab,tm,sm in [('all',np.ones(len(d),bool),np.ones(len(st_ids),bool)),
                  ('<=10 km',km.to_numpy()<=10,km_st<=10),('<=25 km',km.to_numpy()<=25,km_st<=25),('<=50 km',km.to_numpy()<=50,km_st<=50),
                  ('excluding M4',d.mech.to_numpy()!='M4',np.ones(len(st_ids),bool)),('M1 only',d.mech.to_numpy()=='M1',km_st<=15)]:
    print(f"{lab:14s} transitions n={tm.sum():3d} lead150 med={np.median(d.lead150[tm]):.2f} p2={100*(d.lead150[tm]>=2).mean():.0f}% | stable n={sm.sum():4d} FP150={100*fp_of(sm,j):.1f}% FP100={100*fp_of(sm,j1):.1f}% FP200={100*fp_of(sm,j2):.1f}% | lead100={np.median(L[tm,j1]):.2f} lead200={np.median(L[tm,j2]):.2f}")
