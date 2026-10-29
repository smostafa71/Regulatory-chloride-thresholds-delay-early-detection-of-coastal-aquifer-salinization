import pandas as pd, numpy as np, json
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
trans=pd.read_csv('raw/transition_sites.csv')
tset=sorted(trans.site)   # deterministic order
state_of=nat.groupby('site')['state'].first()
T250,PERSIST,FOLLOWUP=250.0,2,5.0
def prep(df):
    out={}
    for s,g in df.sort_values('date').groupby('site'):
        out[s]=(g['chloride'].to_numpy(), g['date'].to_numpy().astype('datetime64[D]'))
    return out
allw=prep(nat)
import pickle; pickle.dump(allw,open('raw/allw.pkl','wb'))
def pcross(cl,dt,th,persist=PERSIST):
    run=0
    for i in range(len(cl)):
        if cl[i]>=th:
            run+=1
            if run>=persist: return dt[i-persist+1]
        else: run=0
    return None
# ---- cell 51: events
rows=[]
for s in tset:
    cl,dt=allw[s]; n=len(cl)
    t250=pcross(cl,dt,T250)
    rec={'site':s,'state':state_of[s],'n_samples':n,'date_250':t250,'baseline_chloride':float(np.median(cl[:max(3,n//4)])),
         'first_date':dt[0],'last_date':dt[-1]}
    d=np.diff(dt).astype(float); rec['median_interval_days']=float(np.median(d))
    for w in [100,125,150,175,200]: rec[f'date_{w}']=pcross(cl,dt,w)
    rows.append(rec)
ev=pd.DataFrame(rows); ev.to_csv('raw/transition_events.csv',index=False)
print('events',len(ev),'median baseline',ev.baseline_chloride.median(),'median interval days',ev.median_interval_days.median())
# ---- stable controls
stable=[]; stable_ids=[]
for s in allw:
    if s in set(tset) or len(allw[s][0])<20: continue
    cl,dt=allw[s]
    if pcross(cl,dt,T250) is not None: continue
    if np.median(cl)<T250: stable.append((cl,dt)); stable_ids.append(s)
print('stable controls',len(stable))
pd.Series(stable_ids).to_csv('raw/stable_ids.csv',index=False)
THRESHOLDS=list(range(50,245,5))
trans_arr=[]
for s in tset:
    cl,dt=allw[s]; t=pcross(cl,dt,T250); trans_arr.append((s,cl,dt,t))
lead_mat=np.full((len(trans_arr),len(THRESHOLDS)),np.nan)
for i,(s,cl,dt,t) in enumerate(trans_arr):
    for j,th in enumerate(THRESHOLDS):
        tw=pcross(cl,dt,th)
        if tw is not None and tw<=t: lead_mat[i,j]=(t-tw).astype(int)/365.25
fp_mat=np.zeros((len(stable),len(THRESHOLDS)),dtype=np.int8)
for i,(cl,dt) in enumerate(stable):
    for j,th in enumerate(THRESHOLDS):
        tw=pcross(cl,dt,th)
        if tw is None: fp_mat[i,j]=0
        elif (dt[-1]-tw).astype(int)/365.25>=FOLLOWUP: fp_mat[i,j]=1
        else: fp_mat[i,j]=-1
np.save('raw/lead_mat.npy',lead_mat); np.save('raw/fp_mat.npy',fp_mat)
def curve_from(L,F):
    lm=np.nanmedian(L,axis=0); nt=(F==1).sum(0); el=((F==1)|(F==0)).sum(0)
    return lm, np.where(el>0,nt/el,np.nan)
lm,fp=curve_from(lead_mat,fp_mat)
raw_fp=(fp_mat!=0).sum(0)/len(stable)
cdf=pd.DataFrame({'thresh':THRESHOLDS,'lead_med':lm,'lead_mean':np.nanmean(lead_mat,0),'n_wells':(~np.isnan(lead_mat)).sum(0),
   'fp_corr':fp,'fp_raw':raw_fp,'n_trueFP':(fp_mat==1).sum(0),'n_cens':(fp_mat==-1).sum(0)})
cdf.to_csv('raw/threshold_curve.csv',index=False)
print(cdf[cdf.thresh.isin([50,100,125,150,175,200,240])].round(3).to_string(index=False))
# regional
lead150=lead_mat[:,THRESHOLDS.index(150)]
ev['lead_150']=lead150
for st_,g in ev.groupby('state'):
    v=g.lead_150.dropna()
    if len(v)>=3: print(st_,len(v),round(v.median(),2))
ev.to_csv('raw/transition_events.csv',index=False)
