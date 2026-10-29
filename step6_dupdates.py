import pandas as pd, numpy as np, pickle
nat=pd.read_parquet('raw/national.parquet')
trans=sorted(pd.read_csv('raw/transition_sites.csv').site)
k=nat.groupby(['site','date']).size()
print('national rows',len(nat),' unique site-dates',len(k),' rows on duplicated site-dates',int(k[k>1].sum()), f"({k[k>1].sum()/len(nat)*100:.1f}%)")
T=nat[nat.site.isin(trans)]
kt=T.groupby(['site','date']).size()
print('transition rows',len(T),' unique site-dates',len(kt),' rows in dup dates',int(kt[kt>1].sum()))
per=T.groupby('site').agg(n=('chloride','size'),nd=('date','nunique'))
print('transition wells with any duplicate date',(per.n>per.nd).sum(),'| wells with <20 unique dates',(per.nd<20).sum(),'| <10 unique dates',(per.nd<10).sum())
# does the persistent 250 crossing use two same-day samples?
allw=pickle.load(open('raw/allw.pkl','rb'))
def pcross_idx(cl,dt,th,persist=2):
    run=0
    for i in range(len(cl)):
        if cl[i]>=th:
            run+=1
            if run>=persist: return i-persist+1
        else: run=0
    return None
same=0
for s in trans:
    cl,dt=allw[s]; i=pcross_idx(cl,dt,250)
    if dt[i]==dt[i+1]: same+=1
print('transitions whose sustained-250 pair is two samples on the SAME day:',same)
# Method-as-described: median per site-date, then classify
med=nat.groupby(['site','date'],as_index=False).agg(chloride=('chloride','median'),state=('state','first'))
print('after daily median: rows',len(med),'sites',med.site.nunique())
THRESH,MIN_N,MIN_PERSIST=250.0,20,2
def classify(cl):
    n=len(cl)
    if n<MIN_N: return 'short'
    init=np.median(cl[:max(3,n//3)]); later=cl[n//3:]; p=0; ok=False
    for x in later:
        if x>=THRESH:
            p+=1
            if p>=MIN_PERSIST: ok=True;break
        else: p=0
    if init<THRESH and ok: return 'transition'
    if init>=THRESH: return 'always_saline'
    return 'fresh_or_noisy'
lab={}
for s,g in med.groupby('site'): lab[s]=classify(g.sort_values('date').chloride.to_numpy())
lab=pd.Series(lab); print(lab.value_counts())
newT=set(lab[lab=='transition'].index); oldT=set(trans)
print('overlap',len(newT&oldT),' only old',len(oldT-newT),' only new',len(newT-oldT))
med.to_parquet('raw/national_dailymedian.parquet'); lab.to_csv('raw/labels_dailymedian.csv')
