"""Reproduction + corrected pipeline for HYDROL84325.
V1 = exactly the submitted notebook logic (cells 45, 51, 52/65, 53, 63).
V2 = the Methods text as written (QA filtering + median per well-date).
"""
import glob, os, pickle
import numpy as np, pandas as pd

W = os.path.dirname(os.path.abspath(__file__)) + '/'
A = W + 'raw/'
ORDER = ["FL","GA","SC","NC","VA","MD","DE","NJ","NY","CT","RI","MA","TX","LA","MS","AL","CA"]
COLS = {'MonitoringLocationIdentifier':'site','ActivityStartDate':'date','ResultMeasureValue':'chloride',
        'ResultMeasure/MeasureUnitCode':'unit','ResultStatusIdentifier':'status','ActivityTypeCode':'acttype',
        'ActivityMediaName':'media','ActivityMediaSubdivisionName':'mediasub','ResultValueTypeName':'valtype',
        'ResultDetectionConditionText':'detcond','ProviderName':'provider','ResultSampleFractionText':'fraction'}
QC_EXCLUDE = {'Quality Control Field Sample Equipment Rinsate Blank','Quality Control Sample-Reference Material',
              'Quality Control Sample-Field Blank','Quality Control Sample-Reference Sample','Quality Control Sample-Trip Blank',
              'Quality Control Sample-Field Spike','Quality Control Sample-Lab Spike','Quality Control Sample-Lab Blank',
              'Quality Control Sample-Blind','Quality Control Sample-Spike Solution','Quality Control Sample-Lab Matrix Spike',
              'Quality Control Sample-Lab Matrix Spike Duplicate','Quality Control Sample-Measurement Precision Sample',
              'Quality Control Sample-Other','Quality Control Sample-Inter-lab Split','Quality Control Sample-Lab Control Sample/Blank Spike'}

def raw_files():
    f = {}
    for p in glob.glob(W + 'raw/raw_*.csv'): f[p.split('raw_')[1][:2]] = p
    return f

def load_raw_all(cache=W + 'raw/raw_all.pkl'):
    if os.path.exists(cache): return pd.read_pickle(cache)
    parts = []
    for st, p in raw_files().items():
        d = pd.read_csv(p, low_memory=False, usecols=list(COLS)).rename(columns=COLS)
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d['chloride'] = pd.to_numeric(d['chloride'], errors='coerce')
        d = d.dropna(subset=['site', 'date', 'chloride'])
        d = d[(d.chloride >= 0) & (d.chloride < 50000)]
        if st == 'VA': d = d[d.date <= '2025-12-31']   # snapshot of submitted analysis
        d['state'] = st
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df['o'] = df.state.map({s: i for i, s in enumerate(ORDER)})
    df = df.sort_values(['o', 'site', 'date'], kind='stable').drop(columns='o').reset_index(drop=True)
    df.to_pickle(cache)
    return df

def build(version='V1', drop_prelim=True, raw=None):
    raw = load_raw_all() if raw is None else raw
    if version == 'V1':
        d = raw.drop_duplicates(subset=['site', 'date', 'chloride'])
        return d[['site', 'date', 'chloride', 'state']].reset_index(drop=True)
    d = raw.copy()
    d = d[(d.media == 'Water') & (d.mediasub.isna() | d.mediasub.isin(['Groundwater', 'Ground Water']))]
    d = d[~d.acttype.isin(QC_EXCLUDE)]
    u = d.unit.fillna('mg/l').str.lower()
    d.loc[u == 'ug/l', 'chloride'] = d.loc[u == 'ug/l', 'chloride'] / 1000.0
    d = d[u.isin(['mg/l', 'ppm', 'ug/l'])]
    if drop_prelim: d = d[d.status.fillna('') != 'Preliminary']
    d = d.drop_duplicates(subset=['site', 'date', 'chloride'])
    d = (d.groupby(['site', 'date'], as_index=False)
           .agg(chloride=('chloride', 'median'), state=('state', 'first')))
    d['o'] = d.state.map({s: i for i, s in enumerate(ORDER)})
    d = d.sort_values(['o', 'site', 'date'], kind='stable').drop(columns='o').reset_index(drop=True)
    return d

def arrays(nat):
    out = {}
    for s, g in nat.groupby('site', sort=False):
        g = g.sort_values('date', kind='stable')
        out[s] = (g['chloride'].to_numpy(float), g['date'].to_numpy().astype('datetime64[D]'), g['state'].iloc[0])
    return out

# ---------------- classification (cell 45) ----------------
def classify_arr(cl, thresh=250.0, min_n=20, persist=2):
    n = len(cl)
    if n < min_n: return 'short'
    init = np.median(cl[:max(3, n // 3)])
    later = cl[n // 3:]; run = 0; ok = False
    for x in later:
        if x >= thresh:
            run += 1
            if run >= persist: ok = True; break
        else: run = 0
    if init < thresh and ok: return 'transition'
    if init >= thresh: return 'always_saline'
    return 'fresh_or_noisy'

def pcross(cl, dt, th, persist=2, start=0):
    run = 0
    for i in range(start, len(cl)):
        if cl[i] >= th:
            run += 1
            if run >= persist: return dt[i - persist + 1]
        else: run = 0
    return None

def pcross_idx(cl, th, persist=2, start=0):
    run = 0
    for i in range(start, len(cl)):
        if cl[i] >= th:
            run += 1
            if run >= persist: return i - persist + 1
        else: run = 0
    return None

def run_core(arr, persist=2, followup=5.0, thresholds=tuple(range(50, 245, 5)), min_n=20):
    labels = {s: classify_arr(v[0], min_n=min_n, persist=persist) for s, v in arr.items()}
    trans = sorted([s for s, l in labels.items() if l == 'transition'])
    tset = set(trans)
    stable = []
    for s, (cl, dt, st) in arr.items():
        if s in tset or len(cl) < min_n: continue
        if pcross(cl, dt, 250.0, persist) is not None: continue
        if np.median(cl) < 250.0: stable.append(s)
    TH = list(thresholds)
    L = np.full((len(trans), len(TH)), np.nan); T250 = []
    for i, s in enumerate(trans):
        cl, dt, _ = arr[s]; t = pcross(cl, dt, 250.0, persist); T250.append(t)
        for j, th in enumerate(TH):
            tw = pcross(cl, dt, th, persist)
            if tw is not None and tw <= t: L[i, j] = (t - tw).astype(int) / 365.25
    F = np.zeros((len(stable), len(TH)), dtype=np.int8)
    for i, s in enumerate(stable):
        cl, dt, _ = arr[s]
        for j, th in enumerate(TH):
            tw = pcross(cl, dt, th, persist)
            if tw is None: F[i, j] = 0
            elif (dt[-1] - tw).astype(int) / 365.25 >= followup: F[i, j] = 1
            else: F[i, j] = -1
    return dict(labels=pd.Series(labels), trans=trans, stable=stable, TH=TH, L=L, F=F, T250=T250)

def curve(L, F):
    lm = np.nanmedian(L, axis=0); nt = (F == 1).sum(0); el = ((F == 1) | (F == 0)).sum(0)
    return lm, np.where(el > 0, nt / el, np.nan)

def knee_of(lm, fp, ths):
    ths = np.asarray(ths); m = ~(np.isnan(lm) | np.isnan(fp)); x, y, th = fp[m], lm[m], ths[m]
    xn = (x - x.min()) / (np.ptp(x) + 1e-9); yn = (y - y.min()) / (np.ptp(y) + 1e-9)
    d = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0])
    return th[np.argmax(d)]

def knee_boot(L, F, TH, n=2000, seed=0):
    rng = np.random.default_rng(seed); ks = []
    for _ in range(n):
        ti = rng.integers(0, len(L), len(L)); si = rng.integers(0, len(F), len(F))
        a, b = curve(L[ti], F[si]); ks.append(knee_of(a, b, TH))
    ks = np.array(ks)
    return np.median(ks), np.percentile(ks, 5), np.percentile(ks, 95)

def fp_table(res, arr, thresholds=(100, 125, 150, 175, 200), followup=5.0, persist=2):
    TH = res['TH']; rows = []
    for w in thresholds:
        j = TH.index(w); col = res['L'][:, j]; Fj = res['F'][:, j]
        n_true = int((Fj == 1).sum()); n_cens = int((Fj == -1).sum()); n_alarm = n_true + n_cens
        rows.append(dict(threshold=w, lead_median=np.nanmedian(col), lead_mean=np.nanmean(col),
                         fp_raw_pct=100 * n_alarm / len(res['stable']), n_censored=n_cens, n_true_fp=n_true,
                         fp_corrected_pct=100 * n_true / (len(res['stable']) - n_cens)))
    return pd.DataFrame(rows)

# ---------------- counterfactual (cell 63) ----------------
def traj_from_arr(cl, dt):
    t = dt.astype('datetime64[D]').astype(np.int64).astype(float) + 719163.0  # to proleptic ordinal
    v = cl.astype(float)
    if len(np.unique(t)) < len(t):
        d = pd.DataFrame({'t': t, 'v': v}).groupby('t', as_index=False)['v'].mean(); t, v = d.t.to_numpy(), d.v.to_numpy()
    return t, v

def interp_cross(t, v, th):
    for i in range(1, len(v)):
        if v[i] >= th:
            if v[i - 1] >= th: return t[i]
            frac = (th - v[i - 1]) / (v[i] - v[i - 1]) if v[i] != v[i - 1] else 0
            return t[i - 1] + frac * (t[i] - t[i - 1])
    return None

def first_sample_cross(sched, sv, th, persist=1):
    run = 0
    for i in range(len(sv)):
        if sv[i] >= th:
            run += 1
            if run >= persist: return sched[i - persist + 1]
        else: run = 0
    return None

def cf_inputs(arr, trans, min_n=20):
    tset = set(trans); tt = {}
    for s in trans:
        t, v = traj_from_arr(arr[s][0], arr[s][1])
        if len(t) < 4: continue
        ref = interp_cross(t, v, 250.0)
        if ref is not None: tt[s] = (t, v, ref)
    stable = []
    for s, (cl, dt, st) in arr.items():
        if s in tset or len(cl) < min_n: continue
        t, v = traj_from_arr(cl, dt)
        if interp_cross(t, v, 250.0) is not None: continue
        if np.median(v) < 250.0: stable.append((s, t, v))
    return tt, stable

def detect(t, v, period, th, mode='orig', n_phases=8, persist=1, rng=None):
    """mode='orig': submitted code (interpolated crossing between scheduled samples, mean over 8 phases).
       mode='eq6' : Eq.(6) as written: first scheduled sample >= T (alarm date), returns list per phase."""
    if t[-1] - t[0] < period: return None
    phases = np.linspace(0, period, n_phases, endpoint=False)
    out = []
    for ph in phases:
        sched = np.arange(t[0] + ph, t[-1], period)
        if len(sched) < 2: continue
        sv = np.interp(sched, t, v)
        d = interp_cross(sched, sv, th) if mode == 'orig' else first_sample_cross(sched, sv, th, persist)
        if d is not None: out.append(d)
    if not out: return None
    return np.mean(out) if mode == 'orig' else out

def cf_policy(tt, stable, period, th, mode='orig', persist=1, followup=5.0, late_tol=0.5):
    leads = []; per_well = {}
    for s, (t, v, ref) in tt.items():
        d = detect(t, v, period, th, mode, persist=persist)
        if d is None: continue
        if mode == 'orig':
            Lw = (ref - d) / 365.25
            if Lw >= -late_tol: leads.append(max(Lw, 0)); per_well[s] = max(Lw, 0)
        else:
            Ls = [(ref - x) / 365.25 for x in d]
            Lm = float(np.mean(Ls))
            per_well[s] = Lm
            if Lm >= -late_tol: leads.append(max(Lm, 0))
    leads = np.array(leads); fp = 0
    for s, t, v in stable:
        d = detect(t, v, period, th, mode, persist=persist)
        if d is None: continue
        dd = d if mode == 'orig' else np.mean(d)
        if (t[-1] - dd) / 365.25 >= followup: fp += 1
    return dict(n_det=len(leads), lead_med=float(np.median(leads)) if len(leads) else np.nan,
                p2=float((leads >= 2).mean()) if len(leads) else np.nan, fp=fp / len(stable), per_well=per_well)
