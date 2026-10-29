import numpy as np, pandas as pd, pickle, matplotlib.pyplot as plt
from figstyle import *
style()
allw=pickle.load(open('raw/allw.pkl','rb'))
ev=pd.read_csv('raw/transition_events.csv',parse_dates=['date_250'])
desc=pd.read_csv('raw/transition_desc.csv'); L=np.load('raw/lead_mat.npy'); TH=list(range(50,245,5))
lead=L[:,TH.index(150)]; cens=desc.started_above150.to_numpy()
B=pd.read_csv('raw/fig2_band.csv')
fig=plt.figure(figsize=(7.2,5.4))
# a funnel
a=fig.add_axes([0.08,0.60,0.36,0.33])
labs=['Monitored wells','≥20 samples\n(eligible)','Transitions']; vals=[44990,2139,145]
a.barh([2,1,0],vals,color=[NAVY,BLUE,ORANGE],height=0.6)
for y,v in zip([2,1,0],vals): a.text(v*1.25,y,f'{v:,}',va='center',fontsize=8,fontweight='bold',color=INK)
a.set_xscale('log'); a.set_xlim(60,4e5); a.set_yticks([2,1,0]); a.set_yticklabels(labs); a.set_xlabel('Number of wells (log scale)')
a.text(200,0.47,'eligible: 145 transitions · 1,439 stable controls\n529 always saline · 26 other',fontsize=6.0,color=INK2,va='center')
panel(a,'a','From 44,990 wells to 145 transitions')
# b trajectories + band
b=fig.add_axes([0.56,0.60,0.42,0.33])
for i,r in ev.iterrows():
    cl,dt=allw[r.site]; t=(dt-np.datetime64(r.date_250.date())).astype(int)/365.25
    m=(t>=-12)&(t<=0.05); b.plot(t[m],cl[m],color=MUTED,lw=0.3,alpha=0.35,zorder=1)
b.fill_between(B.yr,B.p10,B.p90,color=BLUE,alpha=0.15,lw=0,label='10th–90th percentile',zorder=2)
b.fill_between(B.yr,B.p25,B.p75,color=BLUE,alpha=0.30,lw=0,label='25th–75th percentile',zorder=2)
b.plot(B.yr,B.p50,color=NAVY,lw=2,label='Median across wells',zorder=3)
b.axhline(250,color=ORANGE,lw=1.2); b.axhline(150,color=NAVY,lw=1,ls='--')
b.text(-5.2,262,'250 mg L$^{-1}$ (regulatory)',fontsize=6.3,color=INK2); b.text(-11.8,156,'150 mg L$^{-1}$ (warning)',fontsize=6.3,color=INK2)
b.set_xlim(-12,0.3); b.set_ylim(0,320); b.set_xlabel('Years before first sustained 250 mg L$^{-1}$ crossing'); b.set_ylabel('Chloride (mg L$^{-1}$)')
b.legend(loc='upper left',bbox_to_anchor=(0.0,1.0),fontsize=6.0,frameon=True,facecolor='white',edgecolor='none',framealpha=0.9)
panel(b,'b','Chloride rises gradually through 150 mg L$^{-1}$')
# c ECDF of lead + KM
c=fig.add_axes([0.08,0.09,0.42,0.36])
def ecdf(x): x=np.sort(x); return np.r_[0,x], np.r_[0,np.arange(1,len(x)+1)/len(x)]
x,y=ecdf(lead[~cens]); c.step(x,y,where='post',color=BLUE,lw=1.6,label=f'Crossed 150 within record (n={(~cens).sum()}): median {np.median(lead[~cens]):.1f} yr')
x,y=ecdf(lead); c.step(x,y,where='post',color=NAVY,lw=1.6,ls='--',label=f'All wells, censored values as observed (n=145): median {np.median(lead):.1f} yr')
# KM (censored = lower bounds)
o=np.argsort(lead); t=lead[o]; e=~cens[o]; S=1.0; n=len(t); kx=[0]; ky=[0]
for i,(ti,ei) in enumerate(zip(t,e)):
    if ei: S*=(1-1/(n-i)); kx.append(ti); ky.append(1-S)
c.step(kx+[30],ky+[ky[-1]],where='post',color=ORANGE,lw=1.6,label='Kaplan–Meier, left-censored as lower bounds: median 4.3 yr')
c.axhline(0.5,color=GRID,lw=0.8,zorder=0); c.set_xlim(0,25); c.set_ylim(0,1.0)
c.set_xlabel('Lead time at 150 mg L$^{-1}$ (years)'); c.set_ylabel('Cumulative fraction of wells')
c.legend(loc='lower right',fontsize=6)
panel(c,'c','Per-well lead time and left-censoring')
# d: lead time vs pre-crossing record length (data availability)
d=fig.add_axes([0.60,0.09,0.38,0.36])
ps=desc.pre_span.to_numpy()
d.scatter(ps[~cens],lead[~cens],s=12,c=BLUE,ec='white',lw=0.4,label='crossed 150 within record')
d.scatter(ps[cens],lead[cens],s=14,c=ORANGE,marker='^',ec='white',lw=0.4,label='crossed 150 at the first samples (lower bound)')
d.plot([0,40],[0,40],color=MUTED,lw=0.8,ls=':',label='lead = record length before crossing')
d.set_xlim(0,40); d.set_ylim(0,30); d.set_xlabel('Length of record before the 250 crossing (years)'); d.set_ylabel('Lead time at 150 mg L$^{-1}$ (years)')
d.legend(loc='upper left',fontsize=6)
panel(d,'d','Lead time is bounded by the observed record')
save(fig,'Fig3_inventory')
