import pandas as pd, numpy as np
L=np.load('raw/lead_mat.npy'); F=np.load('raw/fp_mat.npy')
TH=np.array(list(range(50,245,5)))
ev=pd.read_csv('raw/transition_events.csv')
def curve_from(L,F):
    lm=np.nanmedian(L,axis=0); nt=(F==1).sum(0); el=((F==1)|(F==0)).sum(0)
    return lm, np.where(el>0,nt/el,np.nan)
def knee_of(lm,fp,ths):
    m=~(np.isnan(lm)|np.isnan(fp)); x,y,th=fp[m],lm[m],ths[m]
    xn=(x-x.min())/(np.ptp(x)+1e-9); yn=(y-y.min())/(np.ptp(y)+1e-9)
    d=np.abs((yn[-1]-yn[0])*xn-(xn[-1]-xn[0])*yn+xn[-1]*yn[0]-yn[-1]*xn[0])
    return th[np.argmax(d)]
lm,fp=curve_from(L,F); print('knee full',knee_of(lm,fp,TH))
for seed in [0,1,2]:
    rng=np.random.default_rng(seed); ks=[]
    for _ in range(2000):
        ti=rng.integers(0,len(L),len(L)); si=rng.integers(0,len(F),len(F))
        a,b=curve_from(L[ti],F[si]); ks.append(knee_of(a,b,TH))
    ks=np.array(ks); print('seed',seed,'2000 boot: median',np.median(ks),'90%',np.percentile(ks,5),np.percentile(ks,95), ' 95%',np.percentile(ks,2.5),np.percentile(ks,97.5))
# knee restricted to 100-240 range? or with thresholds 100..200
for lo,hi in [(100,240),(75,225),(100,200)]:
    m=(TH>=lo)&(TH<=hi)
    print('knee range',lo,hi,knee_of(lm[m],fp[m],TH[m]))
# LOSO
j=list(TH).index(150)
full=np.nanmedian(L[:,j]); print('full median 150',full)
for st in ev.state.value_counts().index:
    keep=(ev.state!=st).to_numpy()
    print('LOSO',st,(~keep).sum(),round(np.nanmedian(L[keep,j]),2))
