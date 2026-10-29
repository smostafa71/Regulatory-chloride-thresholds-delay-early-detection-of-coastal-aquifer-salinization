"""Knee bootstrap (coarse operational grid) and leave-one-state-out, computed from the
per-well event table exactly as in the submitted analysis (raw/transition_events_original.csv
= the archived transition_events.csv of the original pipeline)."""
import numpy as np, pandas as pd
te=pd.read_csv('raw/transition_events_original.csv')
for c in ['date_100','date_125','date_150','date_175','date_200','date_250']: te[c]=pd.to_datetime(te[c],errors='coerce')
te['lead150']=((te['date_250']-te['date_150']).dt.days/365.25).clip(lower=0)
print('full-sample median lead at 150:', round(te.lead150.median(),3))
# leave-one-state-out: pooled median with each state's wells removed (states with >=4 transitions)
big=te.state.value_counts(); states=[s for s in big.index if big[s]>=4]
loso={s: round(te[te.state!=s].lead150.median(),2) for s in states}
print('leave-one-state-out:', loso, 'range', min(loso.values()), max(loso.values()))
# frontier elbow on the operational grid 100-200 mg/L (25 mg/L steps), bootstrap over wells
tc=pd.read_csv('raw/threshold_curve_original.csv').sort_values('thresh'); x=tc.thresh.values.astype(float); fp=tc.false_positive.values*100
grid=np.array([100,125,150,175,200]); fpg=np.interp(grid,x,fp)
L={T:((te['date_250']-te[f'date_{T}']).dt.days/365.25).clip(lower=0).values for T in grid}
def elbow(idx):
    med=np.array([np.nanmedian(L[T][idx]) for T in grid])
    xn=(fpg-fpg.min())/(fpg.max()-fpg.min()); yn=(med-med.min())/(med.max()-med.min())
    chord=yn[0]+(yn[-1]-yn[0])*(xn-xn[0])/(xn[-1]-xn[0])
    return grid[np.argmax(yn-chord)]
rng=np.random.default_rng(0); n=len(te)
els=np.array([elbow(rng.integers(0,n,n)) for _ in range(2000)])
v,c=np.unique(els,return_counts=True)
print('bootstrap elbow (2,000 resamples): median',np.median(els),'90% interval',np.percentile(els,[5,95]),'shares',dict(zip(v.tolist(),(100*c/len(els)).round(0).tolist())))
