import pandas as pd, numpy as np, glob
files={}
for f in glob.glob('raw/raw_*.csv'): files[f.split('raw_')[1][:2]]=f
expected={'FL':(97793,8858),'GA':(13063,1628),'SC':(2239,863),'NC':(3046,1092),'TX':(10410,2858),
          'LA':(12011,2814),'CA':(31217,9741),'NY':(15801,4332),'VA':(2442,1396),'MD':(6754,2259),'DE':(2447,884),
          'NJ':(8677,3578),'CT':(1029,454),'RI':(329,169),'MA':(6626,1523),'MS':(3337,1968),'AL':(960,573)}
cols={'MonitoringLocationIdentifier':'site','ActivityStartDate':'date','ResultMeasureValue':'chloride',
      'ResultMeasure/MeasureUnitCode':'unit','ResultStatusIdentifier':'status','ResultSampleFractionText':'fraction',
      'ActivityTypeCode':'acttype','ResultDetectionConditionText':'detcond','USGSPCode':'pcode','ProviderName':'provider',
      'LastUpdated':'lastupd','ActivityLocation/LatitudeMeasure':'alat','ActivityLocation/LongitudeMeasure':'alon',
      'MeasureQualifierCode':'qual','ResultValueTypeName':'valtype','SampleAquifer':'sample_aquifer',
      'ActivityDepthHeightMeasure/MeasureValue':'act_depth','ActivityDepthHeightMeasure/MeasureUnitCode':'act_depth_unit'}
parts=[]
for st,f in sorted(files.items()):
    raw=pd.read_csv(f,low_memory=False)
    d=raw[list(cols)].rename(columns=cols)
    d['date']=pd.to_datetime(d['date'],errors='coerce')
    d['chloride']=pd.to_numeric(d['chloride'],errors='coerce')
    d=d.dropna(subset=['site','date','chloride'])
    d=d[(d.chloride>=0)&(d.chloride<50000)]
    full=d.drop_duplicates(subset=['site','date','chloride']).sort_values(['site','date'])
    e=expected[st]
    # if fresh download has more rows, test date cutoffs to find snapshot
    print(f"{st}: {len(full)} rows {full.site.nunique()} sites | expected {e} | max date {full.date.max().date()} | lastupd max {str(full.lastupd.max())[:10]}")
    full['state']=st
    parts.append(full)
df=pd.concat(parts,ignore_index=True)
df.to_parquet('raw/national_rebuilt_all.parquet')
print('TOTAL',len(df),df.site.nunique())
