import json, pickle, numpy as np, pandas as pd, math
from shapely.geometry import shape, Point, box
from shapely.ops import transform
from shapely.strtree import STRtree
import pyproj
V1=pickle.load(open('raw/V1.pkl','rb')); arr=V1['arr']; res=V1['res']; trans=res['trans']
lab=res['labels']; pool=sorted([s for s in lab.index if lab[s]!='short'])
wq=pd.read_csv('raw/wqp_station_pool.csv',low_memory=False).rename(columns={'MonitoringLocationIdentifier':'site'})
nw=pd.read_csv('raw/nwis_site_expanded_pool.csv',dtype=str)
na=pd.read_csv('raw/national_aquifer_codes.csv').rename(columns={'id':'nat_aqfr_cd'})
ac=pd.read_csv('raw/aquifer_codes.csv').rename(columns={'id':'aqfr_cd'})
m=wq[['site','OrganizationIdentifier','MonitoringLocationName','LatitudeMeasure','LongitudeMeasure','StateCode','CountyCode','AquiferName','LocalAqfrName','AquiferTypeName','WellDepthMeasure/MeasureValue','WellDepthMeasure/MeasureUnitCode','WellHoleDepthMeasure/MeasureValue','ConstructionDateText','VerticalMeasure/MeasureValue']].copy()
m.columns=['site','org','name','lat','lon','state_fips','county_fips','wqp_aquifer','wqp_local_aquifer','wqp_aqtype','wqp_depth','wqp_depth_unit','wqp_hole','wqp_constr','wqp_alt']
m=m.drop_duplicates('site')
nwk=nw[['site','nat_aqfr_cd','aqfr_cd','aqfr_type_cd','well_depth_va','hole_depth_va','construction_dt','site_tp_cd','alt_va','station_nm']].drop_duplicates('site')
m=m.merge(nwk,on='site',how='left').merge(na,on='nat_aqfr_cd',how='left').merge(ac,on='aqfr_cd',how='left')
for c in ['well_depth_va','hole_depth_va','wqp_depth','wqp_hole','alt_va']: m[c]=pd.to_numeric(m[c],errors='coerce')
# depth in ft: NWIS well depth, else hole depth, else WQP depth (convert m->ft)
wqd=np.where(m.wqp_depth_unit.fillna('').str.lower().isin(['m','meters']), m.wqp_depth*3.28084, m.wqp_depth)
m['depth_ft_nwis']=m.well_depth_va.combine_first(m.hole_depth_va)   # as in submitted pipeline
m['depth_ft']=m.depth_ft_nwis.combine_first(pd.Series(wqd,index=m.index))
m['state']=[arr[s][2] if s in arr else None for s in m.site]
m['label']=[lab.get(s) for s in m.site]
# ---------- original coast_km (global_land_mask, as notebook cell 58) for USGS transition wells only
from global_land_mask import globe
def ckm(lat,lon):
    if pd.isna(lat) or pd.isna(lon): return np.nan
    if not globe.is_land(lat,lon): return 0.0
    for r in range(2,200,2):
        dd=r/111.0
        for a in range(0,360,10):
            la=lat+dd*math.cos(math.radians(a)); lo=lon+dd*math.sin(math.radians(a))/max(0.2,math.cos(math.radians(lat)))
            if not globe.is_land(la,lo): return float(r)
    return np.nan
# NWIS coords were used in submission
nwc=nw[['site','dec_lat_va','dec_long_va']].drop_duplicates('site')
nwc['dec_lat_va']=pd.to_numeric(nwc.dec_lat_va,errors='coerce'); nwc['dec_long_va']=pd.to_numeric(nwc.dec_long_va,errors='coerce')
m=m.merge(nwc,on='site',how='left')
m['coast_km_glm']=[ckm(a,b) if s in set(trans) and s.startswith('USGS-') else np.nan for s,a,b in zip(m.site,m.dec_lat_va,m.dec_long_va)]
# ---------- marine shoreline distance (Natural Earth 10m coastline, geodesic via local AEQD)
gj=json.load(open('gis/ne_10m_coastline.geojson'))
geoms=[shape(f['geometry']) for f in gj['features']]
geoms=[g for g in geoms if g.bounds[0]<-60 and g.bounds[2]>-130 and g.bounds[1]<55 and g.bounds[3]>20]
tree=STRtree(geoms)
def marine_km(lat,lon):
    if pd.isna(lat) or pd.isna(lon): return np.nan
    for pad in [1.0,3.0,6.0]:
        cand=tree.query(box(lon-pad,lat-pad,lon+pad,lat+pad))
        if len(cand): break
    aeqd=pyproj.Transformer.from_crs('EPSG:4326',pyproj.CRS(proj='aeqd',lat_0=lat,lon_0=lon,units='m'),always_xy=True).transform
    d=min(transform(aeqd,geoms[i]).distance(Point(0,0)) for i in cand)
    return d/1000.0
m['coast_km_marine']=[marine_km(a,b) for a,b in zip(m.lat,m.lon)]
m.to_csv('raw/meta_pool.csv',index=False)
T=m[m.site.isin(trans)]
print('transition meta rows',len(T),' coords',T.lat.notna().sum(),' depth (nwis)',T.depth_ft_nwis.notna().sum(),' depth any',T.depth_ft.notna().sum())
print('coast_km_glm valid',T.coast_km_glm.notna().sum(),' median',T.coast_km_glm.median())
print('marine km median',T.coast_km_marine.median()); print(T.coast_km_marine.describe())
print(T[['coast_km_glm','coast_km_marine']].corr(method='spearman'))
