import pandas as pd, numpy as np
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride','state']]
trans=sorted(pd.read_csv('raw/transition_sites.csv').site)
T250=250.0; N_PHASES=8; MIN_FOLLOWUP=5.0
G={s:g for s,g in nat.groupby('site')}
def traj(site):
    g=G[site].sort_values('date')
    t=g['date'].map(pd.Timestamp.toordinal).to_numpy(float); v=g['chloride'].to_numpy(float)
    if len(np.unique(t))<len(t):
        d=pd.DataFrame({'t':t,'v':v}).groupby('t',as_index=False)['v'].mean(); t,v=d.t.to_numpy(),d.v.to_numpy()
    return t,v
def interp_cross(t,v,th):
    for i in range(1,len(v)):
        if v[i]>=th:
            if v[i-1]>=th: return t[i]
            frac=(th-v[i-1])/(v[i]-v[i-1]) if v[i]!=v[i-1] else 0
            return t[i-1]+frac*(t[i]-t[i-1])
    return None
trans_traj={}; miss=[]
for s in trans:
    t,v=traj(s)
    if len(t)<4: miss.append((s,'short')); continue
    ref=interp_cross(t,v,T250)
    if ref is not None: trans_traj[s]=(t,v,ref)
    else: miss.append((s,'noref'))
print('with ref',len(trans_traj),'missing',miss)
counts=nat.groupby('site').size()
stable=[]
for s in counts[counts>=20].index:
    if s in set(trans): continue
    t,v=traj(s)
    if interp_cross(t,v,T250) is not None: continue
    if np.median(v)<T250: stable.append((s,t,v))
print('stable',len(stable))
import pickle; pickle.dump((trans_traj,stable),open('raw/cf_inputs.pkl','wb'))
def detect_interp(t,v,period,th,n_phases=N_PHASES):
    if t[-1]-t[0]<period: return None
    dates=[]
    for ph in np.linspace(0,period,n_phases,endpoint=False):
        sched=np.arange(t[0]+ph,t[-1],period)
        if len(sched)<2: continue
        sv=np.interp(sched,t,v); d=interp_cross(sched,sv,th)
        if d is not None: dates.append(d)
    return np.mean(dates) if dates else None
POL=[("annual_250",365,250),("annual_150",365,150),("semiannual_150",182,150),("quarterly_150",91,150),("semiannual_100",182,100),("quarterly_100",91,100)]
for name,period,th in POL:
    leads=[]
    for s,(t,v,ref) in trans_traj.items():
        d=detect_interp(t,v,period,th)
        if d is not None:
            L=(ref-d)/365.25
            if L>=-0.5: leads.append(max(L,0))
    leads=np.array(leads); fp=0
    for s,t,v in stable:
        d=detect_interp(t,v,period,th)
        if d is not None and (t[-1]-d)/365.25>=MIN_FOLLOWUP: fp+=1
    print(f"{name:15s} n_det {len(leads):3d} med {np.median(leads):.2f} p2 {(leads>=2).mean():.2f} FP {fp/len(stable):.3f}")
