import pandas as pd, numpy as np, pickle
allw=pickle.load(open('raw/allw.pkl','rb'))
ev=pd.read_csv('raw/transition_events.csv',parse_dates=['date_250','date_150','first_date','last_date'])
rows=[]
for _,r in ev.iterrows():
    cl,dt=allw[r.site]; t250=np.datetime64(r.date_250.date())
    pre=dt<t250; n_pre=pre.sum()
    span=(dt[-1]-dt[0]).astype(int)/365.25
    pre_span=(t250-dt[0]).astype(int)/365.25
    post=(dt[-1]-t250).astype(int)/365.25
    first=cl[0]
    # single exceedances >=250 before the sustained crossing
    n_pre_exc=(cl[pre]>=250).sum()
    # first persistent 150 crossing index
    t150=np.datetime64(r.date_150.date()) if pd.notna(r.date_150) else None
    i150=np.where(dt==t150)[0][0] if t150 is not None else None
    started_above150 = (i150==0)
    intervals=np.diff(dt).astype(float)/365.25
    rows.append(dict(site=r.site,state=r.state,n=len(cl),n_pre=n_pre,span=span,pre_span=pre_span,post=post,
        first=first,baseline=r.baseline_chloride,n_pre_exc=n_pre_exc,started_above150=started_above150,
        i150=i150,med_int=np.median(intervals),mean_int=intervals.mean(),cv_int=intervals.std()/intervals.mean(),
        max_gap=intervals.max(),lead150=r.lead_150,year250=r.date_250.year,first_year=dt[0].astype('datetime64[Y]').astype(int)+1970))
D=pd.DataFrame(rows); D.to_csv('raw/transition_desc.csv',index=False)
q=lambda s: f"min {s.min():.2f} | q10 {s.quantile(.1):.2f} | q25 {s.quantile(.25):.2f} | med {s.median():.2f} | q75 {s.quantile(.75):.2f} | q90 {s.quantile(.9):.2f} | max {s.max():.2f}"
for c in ['n','n_pre','span','pre_span','post','med_int','mean_int','cv_int','max_gap','baseline','first','lead150','year250','first_year']:
    print(f"{c:12s}", q(D[c]))
print('wells with any single >=250 before sustained crossing:',(D.n_pre_exc>0).sum())
print('wells whose first persistent 150 crossing is at first sample (left-censored):',D.started_above150.sum())
print('first sample >=150:',(D['first']>=150).sum(),' baseline>=150:',(D.baseline>=150).sum())
print('lead150 == 0:',(D.lead150==0).sum(),' lead150>=2:',(D.lead150>=2).sum(), ' frac', (D.lead150>=2).mean())
print('n_pre <8:',(D.n_pre<8).sum(), ' n_pre<10', (D.n_pre<10).sum())
print('median lead150 among NOT left-censored:',D[~D.started_above150].lead150.median(), 'n',(~D.started_above150).sum())
print('median lead150 among left-censored:',D[D.started_above150].lead150.median())
print('median interval <=365 d:',(D.med_int<=1).sum(), ' <=0.5 yr', (D.med_int<=0.5).sum())
