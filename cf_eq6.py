import pickle, numpy as np, pandas as pd, sys
sys.path.insert(0,'.'); import core
trans_traj,stable=pickle.load(open('raw/cf_inputs.pkl','rb'))   # exact submitted inputs (140 / 1380)
stable=[(s,t,v) for s,t,v in stable]
POL=[("annual_250",365,250),("annual_150",365,150),("semiannual_150",182,150),("quarterly_150",91,150),("semiannual_100",182,100),("quarterly_100",91,100),("annual_100",365,100)]
def run(mode,persist=1):
    out=[]; perwell={}
    for name,period,th in POL:
        leads_all=[]; det=0; leads_det=[]
        for s,(t,v,ref) in trans_traj.items():
            if t[-1]-t[0]<period: continue
            ph_leads=[]
            for ph in np.linspace(0,period,8,endpoint=False):
                sched=np.arange(t[0]+ph,t[-1],period)
                if len(sched)<2: continue
                sv=np.interp(sched,t,v)
                d=core.interp_cross(sched,sv,th) if mode=='orig' else core.first_sample_cross(sched,sv,th,persist)
                if d is not None: ph_leads.append((ref-d)/365.25)
            if not ph_leads: continue
            Lw=np.mean(ph_leads); perwell[(name,s)]=Lw
            if mode=='orig':
                if Lw>=-0.5: leads_det.append(max(Lw,0))
            else:
                if Lw>0: leads_det.append(Lw)          # alarm strictly before the 250 reference
        fp=0
        for s,t,v in stable:
            if t[-1]-t[0]<period: continue
            ds=[]
            for ph in np.linspace(0,period,8,endpoint=False):
                sched=np.arange(t[0]+ph,t[-1],period)
                if len(sched)<2: continue
                sv=np.interp(sched,t,v)
                d=core.interp_cross(sched,sv,th) if mode=='orig' else core.first_sample_cross(sched,sv,th,persist)
                if d is not None: ds.append(d)
            if ds and (t[-1]-np.mean(ds))/365.25>=5: fp+=1
        L=np.array(leads_det)
        out.append(dict(policy=name,n_det=len(L),lead_med=np.median(L) if len(L) else np.nan,p2=(L>=2).mean() if len(L) else np.nan,fp=fp/len(stable)))
    return pd.DataFrame(out),perwell
for mode,ps in [('orig',1),('eq6',1),('eq6',2)]:
    df,pw=run(mode,ps); print(f'\n=== mode={mode} persist={ps} (of {len(trans_traj)} wells with reference)'); print(df.round(3).to_string(index=False))
    pickle.dump(pw,open(f'raw/cf_perwell_{mode}_{ps}.pkl','wb'))
