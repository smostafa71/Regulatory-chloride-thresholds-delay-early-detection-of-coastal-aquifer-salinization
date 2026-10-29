"""Adjusted covariate models and partial rank correlation (Table S6, adjusted rows).
Ordinary least squares of log(lead + 0.25 yr) at 150 mg/L (per-well event table) on baseline chloride,
median sampling interval, NWIS well depth, raster distance to coast and state group (FL, GA, LA, CA, other)."""
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
o = pd.read_csv('raw/transition_events_original.csv', parse_dates=['date_150', 'date_250']).set_index('site')
t = pd.read_csv('raw/trans_classified.csv').set_index('site')
d = pd.DataFrame(index=o.index)
d['lead'] = ((o.date_250 - o.date_150).dt.days / 365.25).clip(lower=0)
d['y'] = np.log(d.lead + 0.25); d['base'] = o.baseline_chloride; d['intv'] = o.median_interval_days / 365.25
d['lintv'] = np.log1p(o.median_interval_days);   # log(1 + interval in days); eight wells have a median interval of 0 days
d['depth'] = t.depth_ft_nwis.reindex(d.index); d['km'] = t.coast_km_glm.reindex(d.index)
st = t.state.reindex(d.index); d['sg'] = st.where(st.isin(['FL', 'GA', 'LA', 'CA']), 'other')
models = [('Model 1: baseline + interval + depth + distance + state', 'y ~ base + intv + depth + km + C(sg)'),
          ('Model 2: without state', 'y ~ base + intv + depth + km'),
          ('Model 3: log interval', 'y ~ base + lintv + depth + km + C(sg)'),
          ('Model 4: without depth', 'y ~ base + intv + km + C(sg)'),
          ('Model 5: baseline + distance only', 'y ~ base + km')]
rows = []
for lab, f in models:
    vars_ = [v for v in ['base', 'intv', 'lintv', 'depth', 'km'] if v in f]
    dd = d.dropna(subset=['y'] + vars_)
    m = smf.ols(f, data=dd).fit()
    r = dict(model=lab, n=len(dd), km_coef=m.params['km'], km_p=m.pvalues['km'])
    if 'depth' in f: r.update(depth_coef=m.params['depth'], depth_p=m.pvalues['depth'])
    rows.append(r)
R = pd.DataFrame(rows); pd.set_option('display.float_format', lambda v: f'{v:.5g}'); print(R.to_string(index=False))
# partial Spearman correlation of depth with lead, controlling for baseline and interval (rank residuals)
dd = d.dropna(subset=['depth']).copy()
rk = dd[['lead', 'depth', 'base', 'intv']].rank()
X = np.c_[np.ones(len(rk)), rk.base, rk.intv]
res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
r, p = stats.pearsonr(res(rk.lead.to_numpy()), res(rk.depth.to_numpy()))
# P from the t distribution with n - 2 - k degrees of freedom (k = 2 controls)
n = len(rk); tt = r * np.sqrt((n - 4) / (1 - r * r)); p2 = 2 * stats.t.sf(abs(tt), n - 4)
print(f'partial Spearman rho(depth, lead | baseline, interval) = {r:.3f}, P = {p2:.3f}, n = {n}')
R.to_csv('raw/covariate_models.csv', index=False)
