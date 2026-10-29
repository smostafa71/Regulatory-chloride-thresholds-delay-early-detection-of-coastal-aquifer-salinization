import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from figstyle import *
style()
nat=pd.read_parquet('raw/national.parquet')[['site','date','chloride']]
ev=pd.read_csv('raw/transition_events_cell51.csv',parse_dates=['date_250','date_150'])
D=pd.read_csv('raw/fig3_curv.csv'); sel=D[(D.n_pre>=12)&(D.span_pre>=3)]
P=pd.read_csv('raw/fig3_forecasts.csv')
FC=[c for c in P.columns if c.startswith('f_')]; P[FC]=P[FC].clip(lower=0)   # a remaining-time forecast below zero (latest value already >250) counts as zero
fig=plt.figure(figsize=(7.2,5.6))
# a representative well
site='USGS-390218074560901'; r=ev[ev.site==site].iloc[0]; g=nat[(nat.site==site)].sort_values('date'); pre=g[g.date<r.date_250]
t=(pre.date-pre.date.min()).dt.days/365.25; y=pre.chloride
a=fig.add_axes([0.08,0.61,0.38,0.32])
a.scatter(t,y,s=14,c=BLUE,ec='white',lw=0.4,zorder=3,label='observed (pre-crossing)')
tt=np.linspace(0,t.max(),200); q=np.polyfit(t,y,2); l=np.polyfit(t,y,1)
a.plot(tt,np.polyval(q,tt),color=ORANGE,lw=1.8,label=f'quadratic (curvature {q[0]:+.2f} mg L$^{{-1}}$ yr$^{{-2}}$)')
a.plot(tt,np.polyval(l,tt),color=GREY,lw=1.2,ls='--',label='linear')
a.set_xlabel('Years of monitoring'); a.set_ylabel('Chloride (mg L$^{-1}$)'); a.legend(fontsize=6,loc='upper left')
panel(a,'a','Accelerating rise (Cape May, NJ; Cohansey Sand)')
# b curvature by baseline
b=fig.add_axes([0.58,0.61,0.40,0.32])
groups=[('<50',sel[sel.baseline<50].quad),('50–150',sel[(sel.baseline>=50)&(sel.baseline<=150)].quad),('>150',sel[sel.baseline>150].quad)]
cols=[BLUE,AQUA,GREY]
bp=b.boxplot([g for _,g in groups],widths=0.5,patch_artist=True,showfliers=False,medianprops=dict(color=INK,lw=1.2),whiskerprops=dict(color=INK2,lw=0.7),capprops=dict(color=INK2,lw=0.7))
for patch,c in zip(bp['boxes'],cols): patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(INK2)
rng=np.random.default_rng(1); labs_b=[]
for i,(_,gv) in enumerate(groups):
    b.scatter(i+1+rng.uniform(-0.12,0.12,len(gv)),gv,s=6,c=INK2,alpha=0.5,zorder=3,ec='none')
    p=stats.wilcoxon(gv).pvalue; star='***' if p<0.001 else ('*' if p<0.05 else 'n.s.')
    labs_b.append(f"{groups[i][0]}\nn={len(gv)}, {100*(gv>0).mean():.0f}% > 0\n{star}")
b.axhline(0,color=INK2,lw=0.7); b.set_ylim(-2.2,4.0); b.set_xticks([1,2,3]); b.set_xticklabels(labs_b,fontsize=6.6)
b.set_xlabel('Baseline chloride (mg L$^{-1}$)'); b.set_ylabel('Curvature (mg L$^{-1}$ yr$^{-2}$)')
panel(b,'b','Acceleration is strongest in the freshest wells (n = 82)')
# c predicted vs actual
c=fig.add_axes([0.08,0.07,0.38,0.34])
F1=P[np.isfinite(P.f_last)]; F2=P[np.isfinite(P.f_loc4)]
c.scatter(F1.rem,F1.f_last,s=16,c=ORANGE,marker='s',ec='white',lw=0.4,label=f'Linear trend from latest value (n={len(F1)})',zorder=3)
c.scatter(F2.rem,F2.f_loc4,s=16,c=BLUE,marker='o',ec='white',lw=0.4,label=f'Linear, last 4 samples (n={len(F2)})',zorder=3)
c.plot([0.1,200],[0.1,200],color=INK2,lw=0.8,ls='--',label='1:1')
c.set_xscale('log'); c.set_yscale('log'); c.set_xlim(0.2,30); c.set_ylim(0.2,900)
c.set_xlabel('Actual remaining time to 250 mg L$^{-1}$ (yr)'); c.set_ylabel('Forecast remaining time (yr)'); c.legend(fontsize=6,loc='upper left',frameon=True,facecolor='white',edgecolor='none',framealpha=0.9)
panel(c,'c','Forecasts at the 150 mg L$^{-1}$ decision point (n = 34)')
# d MAE by method with bootstrap CI vs naive
d=fig.add_axes([0.62,0.07,0.36,0.34])
rem=P.rem.to_numpy(); loo=np.array([np.median(np.delete(rem,i)) for i in range(len(rem))]); P['naive']=loo
methods=[('naive','Naive: median of\nother wells'),('f_loc4','Local linear\n(last 4 samples)'),('f_w2','Local linear\n(last 2 yr)'),('f_quad','Quadratic,\nfull record'),('f_last','Linear slope,\nfrom last value'),('f_fit_end','Linear,\nfull record'),('f_sen','Theil–Sen,\nfull record')]
rng=np.random.default_rng(5); rows=[]
for k,lab in methods:
    m=np.isfinite(P[k]); e=np.abs(P[k][m]-P.rem[m]).to_numpy()
    bs=[np.median(e[rng.integers(0,len(e),len(e))]) for _ in range(3000)]
    rows.append((lab,np.median(e),*np.percentile(bs,[2.5,97.5]),m.sum()))
yy=np.arange(len(rows))[::-1]
for (lab,med,lo,hi,n),yv,(k,_) in zip(rows,yy,methods):
    col=GREY if k=='naive' else (ORANGE if k in ('f_fit_end','f_sen','f_last') else BLUE)
    d.plot([lo,hi],[yv,yv],color=col,lw=2,solid_capstyle='round'); d.scatter([med],[yv],s=28,c=col,zorder=3,ec='white',lw=0.6)
    d.text(max(hi,med)*1.15,yv,f'{med:.1f} yr',va='center',fontsize=6.3,color=INK)
nv=rows[0]; d.axvspan(nv[2],nv[3],color=GREY,alpha=0.15,lw=0)
d.set_xscale('log'); d.set_xlim(0.3,90); d.set_yticks(yy); d.set_yticklabels([r[0] for r in rows],fontsize=6.2)
d.set_xlabel('Median absolute error (yr; 95% bootstrap CI)')
panel(d,'d','No trend forecast significantly beats a naive baseline')
save(fig,'Fig4_shape_forecast')
pd.DataFrame(rows,columns=['method','MAE','lo','hi','n']).to_csv('raw/forecast_table.csv',index=False)
print(pd.DataFrame(rows,columns=['method','MAE','lo','hi','n']).round(2))
