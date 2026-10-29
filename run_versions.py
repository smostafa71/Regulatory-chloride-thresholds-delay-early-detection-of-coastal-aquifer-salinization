import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0,'.'); import core
raw=core.load_raw_all()
print('raw rows',len(raw))
out={}
for ver in ['V1','V2']:
    nat=core.build(ver,raw=raw)
    arr=core.arrays(nat)
    res=core.run_core(arr)
    lab=res['labels'].value_counts()
    ft=core.fp_table(res,arr)
    lm,fp=core.curve(res['L'],res['F'])
    kn=core.knee_of(lm,fp,res['TH'])
    kb=core.knee_boot(res['L'],res['F'],res['TH'],n=2000)
    states=pd.Series([arr[s][2] for s in res['trans']]).value_counts()
    print(f"\n===== {ver}: rows {len(nat)} sites {nat.site.nunique()} | dates {nat.date.min().date()}..{nat.date.max().date()}")
    print(lab.to_dict()); print('stable',len(res['stable'])); print(states.to_dict())
    print(ft.round(2).to_string(index=False))
    print('knee',kn,'boot median/90%',kb)
    pickle.dump(dict(nat=nat,arr=arr,res=res),open(f'raw/{ver}.pkl','wb'))
