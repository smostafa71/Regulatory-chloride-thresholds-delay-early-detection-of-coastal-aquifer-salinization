import requests, io, pandas as pd, numpy as np, pickle, time
V1=pickle.load(open('raw/V1.pkl','rb')); arr=V1['arr']; res=V1['res']
trans=res['trans']
lab=res['labels']
pool=sorted([s for s in lab.index if lab[s]!='short'])   # 2,139 wells with >=20 samples
print('pool',len(pool))
S=requests.Session(); S.headers.update({'User-Agent':'academic-research/1.0'})
# --- NWIS expanded site info for USGS wells in pool
usgs=[s.replace('USGS-','') for s in pool if s.startswith('USGS-')]
def read_rdb(t):
    lines=[l for l in t.splitlines() if not l.startswith('#') and l.strip()]
    df=pd.read_csv(io.StringIO('\n'.join(lines)),sep='\t',dtype=str)
    return df.iloc[1:].reset_index(drop=True)
parts=[]
for i in range(0,len(usgs),80):
    ch=usgs[i:i+80]
    for k in range(4):
        r=S.get('https://waterservices.usgs.gov/nwis/site/',params={'format':'rdb','sites':','.join(ch),'siteOutput':'expanded','siteStatus':'all'},timeout=180)
        if r.status_code==200 and 'agency_cd' in r.text: parts.append(read_rdb(r.text)); break
        time.sleep(3)
nw=pd.concat(parts,ignore_index=True); nw['site']='USGS-'+nw.site_no
nw.to_csv('raw/nwis_site_expanded_pool.csv',index=False)
print('NWIS rows',len(nw),'cols',len(nw.columns))
# --- WQP station for all pool sites (coords, aquifer, well depth, construction)
wq=[]
for i in range(0,len(pool),150):
    ch=pool[i:i+150]
    data={'siteid':ch,'mimeType':'csv'}
    for k in range(4):
        r=S.post('https://www.waterqualitydata.us/data/Station/search',data=data,timeout=300)
        if r.status_code==200: wq.append(pd.read_csv(io.StringIO(r.text),low_memory=False)); break
        print('retry',r.status_code); time.sleep(3)
wq=pd.concat(wq,ignore_index=True); wq.to_csv('raw/wqp_station_pool.csv',index=False)
print('WQP station rows',len(wq)); print(list(wq.columns))
