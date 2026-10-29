import requests, pandas as pd, time, os, json
trans=pd.read_csv('raw/transition_sites.csv'); usgs=[s for s in trans.site if s.startswith('USGS-')]
S=requests.Session(); S.headers.update({'User-Agent':'academic-research/1.0','Accept':'application/json'})
BASE='https://api.waterdata.usgs.gov/ogcapi/v0/collections/field-measurements/items'
GW=['72019','62610','62611']
os.makedirs('raw/wl',exist_ok=True)
for i,sid in enumerate(usgs):
    f=f'raw/wl/{sid}.csv'
    if os.path.exists(f): continue
    feats=[]; off=0
    while True:
        for k in range(4):
            try:
                r=S.get(BASE,params={'f':'json','monitoring_location_id':sid,'limit':10000,'offset':off},timeout=120)
                if r.status_code==200: js=r.json(); break
                time.sleep(5*(k+1))
            except Exception: time.sleep(5*(k+1))
        else: js={'features':[]}
        fs=js.get('features',[]); feats+=fs
        if len(fs)<10000: break
        off+=10000
    rows=[]
    for ft in feats:
        p=ft.get('properties',{}); pc=str(p.get('parameter_code',''))
        if pc in GW:
            try: rows.append(dict(site=sid,date=p.get('time'),wl=float(p.get('value')),param=pc))
            except: pass
    pd.DataFrame(rows,columns=['site','date','wl','param']).to_csv(f,index=False)
    time.sleep(0.3)
    if i%20==0: print(i,flush=True)
print('done')
