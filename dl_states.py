import requests, zipfile, io, os, time, pandas as pd
states={"VA":"51","MD":"24","DE":"10","NJ":"34","CT":"09","RI":"44","MA":"25","MS":"28","AL":"01"}
os.makedirs('raw',exist_ok=True)
for st,f in states.items():
    dest=f'raw/raw_{st}.csv'
    if os.path.exists(dest): print(st,'cached'); continue
    p={"statecode":f"US:{f}","characteristicName":"Chloride","startDateLo":"01-01-1980","siteType":"Well",
       "mimeType":"csv","zip":"yes","dataProfile":"resultPhysChem"}
    for k in range(5):
        try:
            r=requests.get("https://www.waterqualitydata.us/data/Result/search",params=p,timeout=900)
            if r.status_code!=200: raise RuntimeError(r.status_code)
            raw=r.content
            if raw[:2]==b'PK':
                zf=zipfile.ZipFile(io.BytesIO(raw)); nm=[n for n in zf.namelist() if n.endswith('.csv')][0]
                df=pd.read_csv(zf.open(nm),low_memory=False)
            else: df=pd.read_csv(io.BytesIO(raw),low_memory=False)
            df.to_csv(dest,index=False); print(st,len(df),'rows',flush=True); break
        except Exception as e:
            print(st,'fail',k,e,flush=True); time.sleep(5*(k+1))
