import pandas as pd, numpy as np
df=pd.read_parquet('raw/national_rebuilt_all.parquet')
df=df[~((df.state=='VA')&(df.date>'2025-12-31'))]
order=["FL","GA","SC","NC","VA","MD","DE","NJ","NY","CT","RI","MA","TX","LA","MS","AL","CA"]
df['o']=df.state.map({s:i for i,s in enumerate(order)})
df=df.sort_values(['o','site','date']).drop(columns='o').reset_index(drop=True)
df.to_parquet('raw/national.parquet')
nat=df[['site','date','chloride','state']]
nat.to_csv('raw/chloride_national.csv',index=False)
print('rows',len(nat),'sites',nat.site.nunique(),'dates',nat.date.min().date(),nat.date.max().date())
THRESH, MIN_N, MIN_PERSIST = 250.0, 20, 2
def classify(g):
    g = g.sort_values("date"); n = len(g)
    if n < MIN_N: return "short"
    cl = g["chloride"].to_numpy()
    init = np.median(cl[:max(3, n//3)])
    later = cl[n//3:]; persist = 0; ok = False
    for x in later:
        if x >= THRESH:
            persist += 1
            if persist >= MIN_PERSIST: ok = True; break
        else: persist = 0
    if init < THRESH and ok: return "transition"
    if init >= THRESH:       return "always_saline"
    return "fresh_or_noisy"
labels={}; meta={}
for site,g in nat.groupby('site'):
    labels[site]=classify(g); meta[site]=g.state.iloc[0]
lab=pd.Series(labels); st=pd.Series(meta)
print(lab.value_counts())
tr=lab[lab=='transition'].index
print(st.loc[tr].value_counts())
pd.DataFrame({'site':tr,'state':st.loc[tr].values}).to_csv('raw/transition_sites.csv',index=False)
lab.rename('label').to_csv('raw/labels.csv')
