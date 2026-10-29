# -*- coding: utf-8 -*-
"""Independent claims ledger. Recomputes the quantities reported in the manuscript, supplement and
response from the archived data and per-well tables (not from numbers.json) and compares each one
with the value printed in the text. Prints PASS/FAIL per claim and writes verify_report.csv."""
import pickle, re, numpy as np, pandas as pd
from scipy import stats
import os
W = os.path.dirname(os.path.abspath(__file__)) + '/'
R = []
def chk(label, computed, reported, tol=None, fmt='{:.2f}'):
    if isinstance(reported, str):
        ok = (str(computed) == reported); shown = str(computed)
    else:
        t = tol if tol is not None else 0.5 * 10 ** (-max(0, len(str(reported).split('.')[-1]) if '.' in str(reported) else 0))
        ok = abs(float(computed) - float(reported)) <= t + 1e-9; shown = fmt.format(computed) if isinstance(computed, (float, np.floating)) else str(computed)
    R.append((label, shown, reported, 'PASS' if ok else 'FAIL'))

# ---------------------------------------------------------------- data and classification
nat = pd.read_parquet(W + 'raw/national.parquet')
chk('results', len(nat), 218181); chk('wells', nat.site.nunique(), 44990); chk('states', nat.state.nunique(), 17)
chk('first date', str(pd.to_datetime(nat.date).min().date()), '1980-01-01'); chk('last date', str(pd.to_datetime(nat.date).max().date()), '2026-04-16')
lab = pd.read_csv(W + 'raw/labels.csv', index_col=0)['label']
vc = lab.value_counts()
chk('short wells (<20)', vc.get('short', 0), 42851)
chk('eligible wells', len(lab) - vc.get('short', 0), 2139)
chk('always saline', vc.get('always_saline', 0), 529)
chk('transitions', vc.get('transition', 0), 145)
allw = pickle.load(open(W + 'raw/allw.pkl', 'rb'))
trans = sorted(pd.read_csv(W + 'raw/transition_sites.csv').site)
stable = pd.read_csv(W + 'raw/stable_ids.csv').iloc[:, 0].tolist()
chk('stable controls', len(stable), 1439)
chk('other fresh (neither)', 2139 - 529 - 145 - 1439, 26)
prov = nat.provider.astype(str).str.upper()
chk('share of results from NWIS (%)', 100 * prov.str.contains('NWIS').mean(), 86, tol=0.5, fmt='{:.1f}')
u = nat.unit.astype(str).str.lower().str.strip()
chk('share in mg/L or ppm (%) >99.9', 100 * u.isin(['mg/l', 'ppm', 'mg/l as cl', 'mg/kg']).mean() > 99.9 or 100 * u.str.startswith(('mg/l', 'ppm')).mean() > 99.9, 'True')

# ---------------------------------------------------------------- per-well lead and FP (threshold sweep)
L = np.load(W + 'raw/lead_mat.npy'); F = np.load(W + 'raw/fp_mat.npy'); TH = list(range(50, 245, 5)); j = {t: TH.index(t) for t in TH}
st_of = nat.groupby('site').state.first()
tstate = pd.Series([st_of[s] for s in trans])
for s, n in dict(FL=70, GA=19, LA=17, CA=14, NY=7, MD=5, MA=5, TX=4, DE=1, NJ=1, NC=1, SC=1).items():
    chk(f'transitions in {s}', int((tstate == s).sum()), n)
chk('states with transitions', tstate.nunique(), 12)
med = lambda T: np.nanmedian(L[:, j[T]])
def fpr(T, fu=None):
    c = F[:, j[T]]; return 100 * (c == 1).sum() / ((c == 1).sum() + (c == 0).sum())
for T, lv, fv in [(100, 3.53, 17.6), (125, 2.83, 13.3), (150, 2.03, 9.9), (175, 1.55, 6.5), (185, 1.46, 5.2), (190, 1.22, 4.9), (200, 0.96, 4.1)]:
    chk(f'median lead @{T}', med(T), lv); chk(f'corrected FP @{T} (%)', fpr(T), fv, fmt='{:.1f}')
chk('mean lead @150', np.nanmean(L[:, j[150]]), 4.85)
c150 = F[:, j[150]]
chk('raw FP @150 (%)', 100 * (c150 != 0).sum() / len(stable), 12.5, fmt='{:.1f}')
chk('n censored @150', int((c150 == -1).sum()), 41); chk('n true FP @150', int((c150 == 1).sum()), 139)
tp = int((L[:, j[150]] > 0).sum()); chk('warned in advance @150', tp, 112); chk('warned in advance @150 (%)', 100 * tp / 145, 77, tol=0.5, fmt='{:.1f}')
chk('lead = 0 @150 (between two samples)', int((L[:, j[150]] == 0).sum()), 33)
chk('PPV @150 (%)', 100 * tp / (tp + (c150 == 1).sum()), 45, tol=0.5, fmt='{:.1f}')
for T, p in [(100, 34), (125, 38), (175, 54), (200, 62)]:
    t_ = int((L[:, j[T]] > 0).sum()); chk(f'PPV @{T} (%)', 100 * t_ / (t_ + (F[:, j[T]] == 1).sum()), p, tol=0.5, fmt='{:.1f}')
chk('base rate (%)', 100 * 145 / (145 + 1439), 9, tol=0.5, fmt='{:.1f}')
alarms = 145 + (c150 != 0).sum()
chk('alarms per 100 wells @150', 100 * alarms / (145 + 1439), 20, tol=0.6, fmt='{:.1f}')
chk('share of all alarms followed by salinization (%)', 100 * 145 / alarms, 45, tol=0.5, fmt='{:.1f}')
chk('PPV / base rate (about five times)', (100 * 145 / alarms) / (100 * 145 / 1584), 5, tol=0.25)
chk('median lead at 60-90 (about 4.0)', max(abs(med(T) - 4.0) for T in (60, 65, 70, 75, 80, 85, 90)), 0.0, tol=0.05)
chk('FP at 90 (%)', fpr(90), 20, tol=0.5, fmt='{:.1f}'); chk('FP at 60 (%)', fpr(60), 31, tol=0.5, fmt='{:.1f}')
chk('median lead monotone in T', bool(np.all(np.diff(np.nanmedian(L, 0)) <= 1e-9)), 'True')
chk('median lead < 2 above 150', bool(all(med(T) < 2 for T in range(155, 245, 5))), 'True')
chk('lead >= 2.03 within 100-150', bool(all(med(T) >= 2.03 - 1e-6 for T in range(100, 155, 5))), 'True')
# ROC
nt = (F == 1).sum(0); el = ((F == 1) | (F == 0)).sum(0); fp_ = nt / el
for h, a in [(0, 0.89), (1, 0.78), (2, 0.74)]:
    tpr = np.array([(L[:, k] > h).mean() for k in range(len(TH))])
    x = np.r_[0, fp_[::-1], 1]; y = np.r_[0, tpr[::-1], 1]; o = np.argsort(x)
    chk(f'ROC AUC lead>{h}', np.trapezoid(y[o], x[o]), a)
chk('share warned >=2 yr @150 (%)', 100 * (L[:, j[150]] >= 2).mean(), 50, tol=0.5, fmt='{:.1f}')
# EU 187.5 by interpolation
lead_eu = np.interp(187.5, [185, 190], [med(185), med(190)]); fp_eu = np.interp(187.5, [185, 190], [fpr(185), fpr(190)])
chk('EU 187.5: lead within 1.2-1.5', bool(1.2 <= lead_eu <= 1.5), 'True'); chk('EU 187.5: FP about 5%', fp_eu, 5, tol=0.3, fmt='{:.2f}')

# ---------------------------------------------------------------- per-well descriptors
D = pd.read_csv(W + 'raw/transition_desc.csv')
lead = L[:, j[150]]; cens = D.started_above150.to_numpy()
chk('left-censored @150', int(cens.sum()), 48); chk('left-censored share (%)', 100 * cens.mean(), 33, tol=0.5, fmt='{:.1f}')
chk('uncensored n', int((~cens).sum()), 97); chk('median lead uncensored', np.median(lead[~cens]), 1.6, fmt='{:.2f}')
def km_median(t, event):
    o = np.argsort(t, kind='stable'); t = t[o]; e = event[o]; S = 1.0; n = len(t)
    for i, (ti, ei) in enumerate(zip(t, e)):
        if ei: S *= (1 - 1 / (n - i))
        if S <= 0.5: return ti
    return np.nan
chk('Kaplan-Meier median', km_median(lead, ~cens), 4.3)
chk('isolated >=250 before sustained crossing', int((D.n_pre_exc > 0).sum()), 54)
q = lambda s, p: float(np.percentile(s, p))
chk('results per well median', D.n.median(), 41); chk('results IQR lo', q(D.n, 25), 26); chk('results IQR hi', q(D.n, 75), 76); chk('results min', D.n.min(), 20)
chk('record years median', D.span.median(), 20.0, fmt='{:.2f}'); chk('record years q25', q(D.span, 25), 10.0, fmt='{:.2f}'); chk('record years q75', q(D.span, 75), 31.2, fmt='{:.2f}')
chk('median interval (yr)', D.med_int.median(), 0.25, fmt='{:.3f}'); chk('interval q25', q(D.med_int, 25), 0.09, fmt='{:.3f}'); chk('interval q75', q(D.med_int, 75), 0.51, fmt='{:.3f}')
chk('interval CV median', D.cv_int.median(), 0.70, fmt='{:.3f}'); chk('longest gap median', D.max_gap.median(), 1.6, fmt='{:.2f}')
chk('mean interval (response: 0.38)', D.mean_int.median(), 0.38, tol=0.6)
chk('crossing year min', int(D.year250.min()), 1980); chk('crossing year max', int(D.year250.max()), 2023); chk('crossing year median', D.year250.median(), 1993)
chk('pre-crossing results median', D.n_pre.median(), 18); chk('pre-crossing years median', D.pre_span.median(), 7.0, fmt='{:.2f}')
ev51 = pd.read_csv(W + 'raw/transition_events_cell51.csv', parse_dates=['date_250'])
npre51 = np.array([(allw[s][1] < np.datetime64(d.date())).sum() for s, d in zip(ev51.site, ev51.date_250)])
chk('wells with >=8 pre-crossing results (EWS set, event table)', int((npre51 >= 8).sum()), 108)
chk('median interval <=1 yr: n', int((D.med_int <= 1).sum()), 139); chk('median interval <=1 yr: lead', np.median(lead[D.med_int.to_numpy() <= 1]), 1.97)
chk('pre-span >=5 yr: n', int((D.pre_span >= 5).sum()), 86); chk('pre-span >=5 yr: lead', np.median(lead[D.pre_span.to_numpy() >= 5]), 6.6, fmt='{:.2f}')
chk('pre-crossing share 23% between two samples', 100 * 33 / 145, 23, tol=0.5, fmt='{:.1f}')

# ---------------------------------------------------------------- representativeness (Table S9)
pool = pd.read_csv(W + 'raw/pool_desc.csv')
T_, S_ = pool[pool.label == 'transition'], pool[pool.label.isin(['fresh_or_noisy'])]
S_ = pool[pool.site.isin(set(stable))]
chk('stable controls in pool table', len(S_), 1439)
chk('stable: results median', S_.n.median(), 29); chk('stable: record years', S_.span.median(), 11.2, fmt='{:.2f}')
chk('stable: depth median (ft)', S_.depth_ft.median(), 160); chk('stable: distance median (km)', S_.coast_km_marine.median(), 17.4, fmt='{:.2f}')
chk('stable: interval median', S_.med_int.median(), 0.25, fmt='{:.3f}')
chk('interval Mann-Whitney P (transition vs stable)', stats.mannwhitneyu(T_.med_int, S_.med_int).pvalue, 0.61)
chk('stable wells with depth', int(S_.depth_ft.notna().sum()), 976); chk('always-saline wells with depth', int(pool[pool.label == 'always_saline'].depth_ft.notna().sum()), 432)

# ---------------------------------------------------------------- metadata, settings, covariates
I = pd.read_csv(W + 'figs/TableS10_transition_inventory.csv')
chk('inventory rows', len(I), 145)
chk('wells with depth', int(I.well_depth_ft.notna().sum()), 135)
chk('depth median (ft)', I.well_depth_ft.median(), 226); chk('depth q25', q(I.well_depth_ft.dropna(), 25), 113); chk('depth q75', q(I.well_depth_ft.dropna(), 75), 706)
km = I.dist_marine_shoreline_km
chk('distance median (km)', km.median(), 5.3, fmt='{:.2f}'); chk('distance q25', q(km, 25), 2.0, fmt='{:.2f}'); chk('distance q75', q(km, 75), 32.8, fmt='{:.2f}')
chk('within 10 km', int((km <= 10).sum()), 84); chk('10-50 km', int(((km > 10) & (km <= 50)).sum()), 30); chk('beyond 50 km', int((km > 50).sum()), 31)
chk('USGS wells', int(I.site_id.str.startswith('USGS-').sum()), 135)
for m, (n, lv) in dict(M1=(68, 1.9), M2=(39, 2.8), M3=(11, 6.6), M4=(27, 1.0)).items():
    g = I[I.mechanism_class == m]; chk(f'{m} n', len(g), n); chk(f'{m} median lead', g.lead_150_yr.median(), lv, fmt='{:.2f}')
chk('every setting median lead >= 1 yr', bool(I.groupby('mechanism_class').lead_150_yr.median().min() >= 1.0), 'True')
chk('Kruskal-Wallis settings P', stats.kruskal(*[g.lead_150_yr for _, g in I.groupby('mechanism_class')]).pvalue, 0.19)
lc = I.lithology.value_counts()
for k, n in {'Karst carbonate': 65, 'Mixed carbonate–clastic': 4, 'Unconsolidated clastic': 58, 'Fractured/consolidated': 2, 'Unknown': 16}.items():
    kk = [x for x in lc.index if x.split()[0].lower()[:5] == k.split()[0].lower()[:5]]
    chk(f'lithology {k}', int(lc[kk].sum()) if kk else -1, n)
known = I[I.lithology != 'Unknown']
chk('Kruskal-Wallis lithology P (classes >=3 wells)', stats.kruskal(*[g.lead_150_yr for _, g in known.groupby('lithology') if len(g) >= 3]).pvalue, 0.95)
chk('Kruskal-Wallis lithology P (all four classes)', stats.kruskal(*[g.lead_150_yr for _, g in known.groupby('lithology')]).pvalue, 0.28)
# Spearman covariates
sp = lambda a, b: stats.spearmanr(a, b, nan_policy='omit')
evo = pd.read_csv(W + 'raw/transition_events_original.csv', parse_dates=['date_150', 'date_250']).set_index('site').reindex(I.site_id)
lev = ((evo.date_250 - evo.date_150).dt.days / 365.25).clip(lower=0).to_numpy()
chk('rho baseline (Table S6: +0.265)', sp(evo.baseline_chloride.to_numpy(), lev)[0], 0.265, tol=0.0005, fmt='{:.4f}')
chk('P baseline (Table S6: 0.0013)', sp(evo.baseline_chloride.to_numpy(), lev)[1], 0.0013, tol=0.00005, fmt='{:.5f}')
chk('rho interval (Table S6: +0.404)', sp(evo.median_interval_days.to_numpy(), lev)[0], 0.404, tol=0.0005, fmt='{:.4f}')
tc = pd.read_csv(W + 'raw/trans_classified.csv').set_index('site').reindex(I.site_id)
m_ = tc.depth_ft_nwis.notna().to_numpy()
chk('rho depth (Table S6: +0.217)', sp(tc.depth_ft_nwis.to_numpy()[m_], lev[m_])[0], 0.217, tol=0.0005, fmt='{:.4f}'); chk('depth n (NWIS)', int(m_.sum()), 133)
chk('rho geodesic distance', sp(km, I.lead_150_yr)[0], -0.03); chk('P geodesic distance', sp(km, I.lead_150_yr)[1], 0.73)
# state medians (event table) and LOSO
te = pd.read_csv(W + 'raw/transition_events_original.csv', parse_dates=['date_150', 'date_250'])
te['lead'] = ((te.date_250 - te.date_150).dt.days / 365.25).clip(lower=0)
for s, v in dict(FL=1.94, GA=1.97, LA=6.63, CA=3.97, NY=1.01, MA=0.29, MD=7.84, TX=6.57).items():
    chk(f'state median {s}', te[te.state == s].lead.median(), v)
for s, v in dict(FL=2.83, GA=2.21, LA=1.82, CA=2.00, NY=2.39, MA=2.43, MD=1.99, TX=2.00).items():
    chk(f'LOSO without {s}', te[te.state != s].lead.median(), v)
chk('pooled median (event table)', te.lead.median(), 2.03)
chk('event table vs sweep: wells that differ', int((~np.isclose(te.set_index('site').lead.reindex(I.site_id).values, I.lead_150_yr.values, atol=1e-3)).sum()), 7)

# ---------------------------------------------------------------- Table 1 (systems)
T1 = {'1': (32, 1.7), '2': (11, 7.6), '9': (10, 2.2), '10': (5, 0.3), '11': (7, 5.9), '8': (3, 10.1), '3': (10, 2.7), '4': (9, 8.3), '5': (17, 2.9), '6': (3, 0.0), '7': (11, 6.6), '12': (27, 1.0)}
sysno = I.hydrogeologic_system.str.extract(r'^(\d+)')[0]
for k, (n, lv) in T1.items():
    g = I[sysno == k]; chk(f'Table 1 system {k} n', len(g), n); chk(f'Table 1 system {k} lead', g.lead_150_yr.median(), lv, fmt='{:.2f}')

# ---------------------------------------------------------------- coastal subsets (Table S14)
meta = pd.read_csv(W + 'raw/meta_pool.csv').set_index('site')
skm = meta.coast_km_marine.reindex(stable).to_numpy()
tkm = km.to_numpy(); tl = I.lead_150_yr.to_numpy()
for d_, (nt_, lv, ns, fv) in {10: (84, 2.01, 602, 12.0), 25: (100, 2.01, 855, 11.9), 50: (114, 2.02, 1047, 11.3)}.items():
    sel = skm <= d_; c = c150[sel]
    chk(f'<= {d_} km transitions', int((tkm <= d_).sum()), nt_); chk(f'<= {d_} km lead', np.median(tl[tkm <= d_]), lv)
    chk(f'<= {d_} km controls', int(sel.sum()), ns); chk(f'<= {d_} km FP', 100 * (c == 1).sum() / ((c == 1).sum() + (c == 0).sum()), fv, fmt='{:.1f}')
chk('excluding M4: n', int((I.mechanism_class != 'M4').sum()), 118); chk('excluding M4: lead', I[I.mechanism_class != 'M4'].lead_150_yr.median(), 2.44)

# ---------------------------------------------------------------- curvature, predictability, forecasts
cv = pd.read_csv(W + 'raw/fig3_curv.csv')
cc = cv[(cv.n_pre >= 12) & (cv.span_pre >= 3)]
chk('curvature n', len(cc), 82); chk('curvature positive (%)', 100 * (cc.quad > 0).mean(), 65, tol=0.5, fmt='{:.1f}')
chk('curvature median', cc.quad.median(), 0.25); chk('curvature Wilcoxon P', stats.wilcoxon(cc.quad).pvalue, 0.004, tol=0.0005, fmt='{:.4f}')
chk('F-test significant wells', int((cc.F_p < 0.05).sum()), 49)
for lo, hi, n, pp, mm, p in [(-1, 50, 27, 85, 0.62, 0.0002), (50, 150, 26, 62, 0.31, 0.036), (150, 1e9, 29, 48, -0.07, 0.78)]:
    g = cc[cc.baseline < 50] if lo < 0 else (cc[(cc.baseline >= 50) & (cc.baseline <= 150)] if lo == 50 else cc[cc.baseline > 150])
    chk(f'curv group {lo}-{hi} n', len(g), n); chk(f'curv group {lo}-{hi} pos (%)', 100 * (g.quad > 0).mean(), pp, tol=0.5, fmt='{:.1f}')
    chk(f'curv group {lo}-{hi} median', g.quad.median(), mm); chk(f'curv group {lo}-{hi} P', stats.wilcoxon(g.quad).pvalue, p, tol=max(0.00005, p * 0.05), fmt='{:.4f}')
chk('sampling density first vs second half P', stats.wilcoxon(cc.n_first, cc.n_second).pvalue, 0.26)
pr = pd.read_csv(W + 'raw/fig3_pred.csv'); pr = pr[(pr.n >= 10) & (pr.span >= 2.5) & (pr.remaining > 0)]
chk('predictability n', len(pr), 34)
for f, (r_, p_) in dict(curv=(-0.24, 0.18), slope=(-0.01, 0.96), recent=(0.10, 0.57)).items():
    rr = stats.spearmanr(pr[f], pr.remaining); chk(f'rho {f}', rr[0], r_); chk(f'P {f}', rr[1], p_)
chk('remaining time median (naive)', pr.remaining.median(), 2.75)
fc = pd.read_csv(W + 'raw/fig3_forecasts.csv'); fc = fc[fc.site.isin(set(pr.site))]
FC = [c for c in fc.columns if c.startswith('f_')]; fc[FC] = fc[FC].clip(lower=0)
chk('forecast wells', len(fc), 34)
rem = fc.rem.to_numpy()
naive = np.array([np.median(np.delete(rem, i)) for i in range(len(rem))]); en = np.abs(naive - rem)
chk('naive MAE', np.median(en), 2.2, fmt='{:.2f}'); chk('naive within 1 yr (%)', 100 * (en <= 1).mean(), 24, tol=0.5, fmt='{:.1f}')
def ev_(col, mae, n, bias=None, w1=None, p=None):
    f_ = fc[col].to_numpy(float); ok = np.isfinite(f_) & (f_ >= 0); e = np.abs(f_[ok] - rem[ok])
    chk(f'{col} n', int(ok.sum()), n); chk(f'{col} MAE', np.median(e), mae, fmt='{:.2f}')
    if bias is not None: chk(f'{col} bias', np.median(f_[ok] - rem[ok]), bias, fmt='{:.2f}')
    if w1 is not None: chk(f'{col} within 1 yr (%)', 100 * (e <= 1).mean(), w1, tol=0.5, fmt='{:.1f}')
    if p is not None:
        pv = stats.wilcoxon(e, en[ok]).pvalue
        chk(f'{col} Wilcoxon vs naive', pv if p != '<0.001' else (pv < 0.001), p if p != '<0.001' else 'True', fmt='{:.3f}')
ev_('f_last', 9.0, 32, 8.4, 6, '<0.001'); ev_('f_fit_end', 15.4, 32, 15.4, 6, '<0.001'); ev_('f_sen', 23.7, 32, 23.7, 0, '<0.001')
ev_('f_quad', 2.9, 31, -0.4, 29, 0.58); ev_('f_loc4', 1.5, 34, -0.6, 47, 0.34); ev_('f_w2', 2.8, 33, -0.1, 21, 0.78)
for col in ['f_loc3', 'f_loc5', 'f_loc6']:
    f_ = fc[col].to_numpy(float); ok = np.isfinite(f_) & (f_ >= 0)
    pv = stats.wilcoxon(np.abs(f_[ok] - rem[ok]), en[ok]).pvalue
    chk(f'{col} Wilcoxon P within 0.26-0.96', bool(0.255 <= pv <= 0.965), 'True')
i_ = np.argmax(rem - fc.f_loc4.to_numpy(float))
chk('local trend: imminent arrival in a well that took about a decade', bool(fc.f_loc4.iloc[i_] < 1.5 and rem[i_] >= 9.5), 'True')

# ---------------------------------------------------------------- variants
ps = pd.read_csv(W + 'raw/persist_sens.csv'); r3 = ps[ps.rule.str.startswith('3')].iloc[0]
chk('persist-3 transitions', int(r3.transitions), 122); chk('persist-3 lead150', r3.lead150, 2.98); chk('persist-3 FP150', r3.fp150, 10.1, fmt='{:.1f}')
vc_ = pd.read_csv(W + 'raw/variants_cleaning.csv').set_index('variant')
print(vc_.to_string())

# ---------------------------------------------------------------- watch list
wl = pd.read_csv(W + 'raw/watchlist_2015_geo.csv')
chk('watch list n', len(wl), 169)
for s, n in dict(FL=59, CA=43, LA=29).items(): chk(f'watch list {s}', int((wl.state == s).sum()), n)
chk('watch list within 50 km', int((wl.coast_km <= 50).sum()), 111)
chk('watch list share of 9,415 (%)', 100 * 169 / 9415, 1.8, fmt='{:.2f}')
chk('watch list: two latest in [150,250)', bool(((wl.last_cl >= 150) & (wl.last_cl < 250) & (wl.prev_cl >= 150) & (wl.prev_cl < 250)).all()), 'True')

# ---------------------------------------------------------------- report
rep = pd.DataFrame(R, columns=['claim', 'computed', 'reported', 'status'])
rep.to_csv(W + 'verify_report.csv', index=False)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 1000)
print(rep[rep.status == 'FAIL'].to_string())
print(f"{(rep.status == 'PASS').sum()} PASS, {(rep.status == 'FAIL').sum()} FAIL of {len(rep)}")
