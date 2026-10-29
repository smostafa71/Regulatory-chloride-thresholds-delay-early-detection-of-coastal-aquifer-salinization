"""Sensitivity of the censoring-corrected false-positive rate to the follow-up window (3, 5, 7 years).
Uses the per-well series and stable controls written by step2_core.py (Supplementary Note 1)."""
import pickle, numpy as np, pandas as pd
allw = pickle.load(open('raw/allw.pkl', 'rb'))
stable = pd.read_csv('raw/stable_ids.csv').iloc[:, 0].tolist()
def pcross(cl, dt, th, persist=2):
    run = 0
    for i in range(len(cl)):
        if cl[i] >= th:
            run += 1
            if run >= persist: return dt[i - persist + 1]
        else: run = 0
    return None
rows = []
for fu in (3, 5, 7):
    for th in (100, 125, 150, 175, 200):
        nt = nc = 0
        for s in stable:
            cl, dt = allw[s]; tw = pcross(cl, dt, th)
            if tw is None: continue
            if (dt[-1] - tw).astype(int) / 365.25 >= fu: nt += 1
            else: nc += 1
        rows.append(dict(followup_yr=fu, threshold=th, n_true_fp=nt, n_censored=nc, fp_corrected_pct=round(100 * nt / (len(stable) - nc), 1)))
R = pd.DataFrame(rows); R.to_csv('raw/fp_followup_window.csv', index=False)
print(R.pivot(index='threshold', columns='followup_yr', values='fp_corrected_pct'))
