import pandas as pd, numpy as np, pickle
allw=pickle.load(open('raw/allw.pkl','rb'))
ev=pd.read_csv('raw/transition_events.csv',parse_dates=['date_250','date_150'])
grid=np.arange(-12,0.01,0.5)
M=np.full((len(ev),len(grid)),np.nan)
for i,r in ev.iterrows():
    cl,dt=allw[r.site]; t=(dt-np.datetime64(r.date_250.date())).astype(int)/365.25
    d=pd.DataFrame({'t':t,'v':cl}).groupby('t',as_index=False).v.median()
    t,v=d.t.to_numpy(),d.v.to_numpy()
    ok=(grid>=t.min())&(grid<=t.max())
    M[i,ok]=np.interp(grid[ok],t,v)
B=pd.DataFrame({'yr':grid,'n':np.isfinite(M).sum(0),'p10':np.nanpercentile(M,10,0),'p25':np.nanpercentile(M,25,0),'p50':np.nanpercentile(M,50,0),'p75':np.nanpercentile(M,75,0),'p90':np.nanpercentile(M,90,0)})
print(B.round(0).to_string(index=False))
np.save('raw/fig2_M.npy',M); B.to_csv('raw/fig2_band.csv',index=False)
# fraction of wells below 150 at each lag (among wells with data)
print('fraction of wells <150 at lag:',{g:round(float(np.nanmean(M[:,k]<150)),2) for k,g in enumerate(grid) if g in (-10,-6,-4,-2,-1,0)})
