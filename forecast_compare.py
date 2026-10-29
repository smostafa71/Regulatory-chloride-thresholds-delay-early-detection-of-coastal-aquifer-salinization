"""Forecast accuracy against the naive forecast (Table 2): paired bootstrap of the difference in
median absolute error (4,000 resamples) and Wilcoxon signed-rank tests on paired absolute errors."""
import numpy as np, pandas as pd
from scipy import stats
P=pd.read_csv('raw/fig3_forecasts.csv'); rem=P.rem.to_numpy()
FC=[c for c in P.columns if c.startswith('f_')]; P[FC]=P[FC].clip(lower=0)   # a remaining-time forecast below zero (latest value already >250) counts as zero
P['naive']=[np.median(np.delete(rem,i)) for i in range(len(rem))]
rng=np.random.default_rng(11)
rows=[]
for k,lab in [('f_loc4','local linear, last 4'),('f_w2','local linear, last 2 yr'),('f_quad','quadratic'),('f_last','linear rate from latest value'),
              ('f_fit_end','fitted line, full history'),('f_sen','Theil-Sen'),('f_loc3','local linear, last 3'),('f_loc5','local linear, last 5'),('f_loc6','local linear, last 6')]:
    m=np.isfinite(P[k]).to_numpy(); e=np.abs(P[k][m]-P.rem[m]).to_numpy(); en=np.abs(P.naive[m]-P.rem[m]).to_numpy()
    d=[np.median(e[i])-np.median(en[i]) for i in (rng.integers(0,m.sum(),m.sum()) for _ in range(4000))]
    lo,hi=np.percentile(d,[2.5,97.5]); p=stats.wilcoxon(e,en).pvalue
    rows.append((lab,m.sum(),round(np.median(e),2),round(np.median(e)-np.median(en),2),round(lo,2),round(hi,2),round(p,4)))
print(pd.DataFrame(rows,columns=['forecast','n','MAE','diff_vs_naive','ci_lo','ci_hi','wilcoxon_p']).to_string(index=False))
