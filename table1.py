import pandas as pd, numpy as np
d=pd.read_csv('raw/trans_classified.csv')
L=np.load('raw/lead_mat.npy'); TH=list(range(50,245,5)); desc=pd.read_csv('raw/transition_desc.csv')
d['lead100']=L[:,TH.index(100)]; d['lead200']=L[:,TH.index(200)]
d=d.merge(desc[['site','n','n_pre','span','pre_span','med_int','first','started_above150','year250','first_year']],on='site',how='left')
nat=d.national_aquifer_name.fillna(d.wqp_aquifer).fillna(''); loc=d.aquifer_name.fillna(d.wqp_local_aquifer).fillna('')
cf=d.county_fips.astype(int)
def cluster(i):
    r=d.iloc[i]; n=nat.iloc[i]; l=loc.iloc[i]; st=r.state; c=cf.iloc[i]; km=r.coast_km_marine; m=r.mech
    if m=='M4': return '12 Inland or non-marine sources'
    if m=='M3': return '7 Gulf Coast confined sands, Louisiana'
    if m=='M2':
        if st in ('GA','SC'): return '5 Floridan, coastal Georgia–South Carolina' if km<20 else '6 Floridan, inland (deep zones)'
        if c==95: return '4 Floridan, east-central Florida (Cocoa well field)'
        if km>50: return '6 Floridan, inland (deep zones)'
        return '3 Floridan, Florida Gulf and Atlantic coasts'
    # M1
    if st=='FL' and c in (86,11,99,85): return '1 Biscayne and surficial aquifers, SE Florida'
    if st=='FL': return '2 Surficial and intermediate aquifers, SW Florida'
    if st=='TX': return '8 Chicot aquifer, Texas Gulf Coast'
    if st in ('MD','NJ','DE','NY'): return '9 Atlantic Coastal Plain and Long Island'
    if st=='MA': return '10 Glacial outwash, Cape Cod and Plymouth'
    if st=='CA': return '11 California coastal basins'
    if st=='GA': return '9b Surficial aquifer, coastal Georgia'
    return '?'
d['system']=[cluster(i) for i in range(len(d))]
T=d.groupby('system').agg(n=('site','size'),mech=('mech',lambda x: ','.join(sorted(set(x)))),states=('state',lambda x: ', '.join(sorted(set(x)))),
    depth_med=('depth_ft','median'),depth_lo=('depth_ft',lambda x:x.quantile(.25)),depth_hi=('depth_ft',lambda x:x.quantile(.75)),
    km_med=('coast_km_marine','median'),lead_med=('lead150','median'),p2=('lead150',lambda x:100*(x>=2).mean()),base_med=('baseline_chloride','median'))
print(T.round(1).to_string())
d.to_csv('raw/trans_inventory_full.csv',index=False)
out=d[['site','state','county_fips','lat','lon','name','national_aquifer_name','aquifer_name','wqp_local_aquifer','aqfr_type_cd','depth_ft','coast_km_marine','system','mech','lithology',
       'n','span','pre_span','med_int','baseline_chloride','first','started_above150','date_250','lead100','lead150','lead200']].copy()
out.columns=['site_id','state','county_fips','lat','lon','station_name','national_aquifer','local_aquifer_nwis','local_aquifer_wqp','aquifer_type_nwis','well_depth_ft','dist_marine_shoreline_km','hydrogeologic_system','mechanism_class','lithology',
       'n_samples','record_years','pre_crossing_years','median_sampling_interval_yr','baseline_chloride_mgL','first_sample_mgL','left_censored_at_150','date_sustained_250','lead_100_yr','lead_150_yr','lead_200_yr']
# lead time at 150 mg/L from the archived per-well event table of the original analysis (used for leave-one-state-out)
o=pd.read_csv('raw/transition_events_original.csv',parse_dates=['date_150','date_250']).set_index('site')
out['lead_150_yr_event_table']=out.site_id.map(((o.date_250-o.date_150).dt.days/365.25).clip(lower=0))
out.round(3).to_csv('figs/TableS10_transition_inventory.csv',index=False)
