import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from figstyle import *
style()
# ---------- S1: EWS evidence
pw=pd.read_csv('raw/ews_power.csv')
L=np.load('raw/lead_mat.npy'); F=np.load('raw/fp_mat.npy'); TH=list(range(50,245,5))
nt=(F==1).sum(0); el=((F==1)|(F==0)).sum(0); fpr=nt/el
fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.0)); fig.subplots_adjust(wspace=0.32,left=0.17,right=0.98,bottom=0.17,top=0.86)
p0=pw[pw.obs_noise==0].reset_index(drop=True); p1=pw[pw.obs_noise>0].reset_index(drop=True)
labs=[f"memory {1/r.lam0:.2g} yr, ×{r.fold_factor:.0f} slowing" for r in p0.itertuples()]
y=np.arange(len(p0))[::-1]
a.axvspan((1+0.086)/2,(1+0.140)/2,color=ORANGE,alpha=0.22,lw=0); a.text(0.557,-0.75,'observed\nΔAC1',fontsize=6,color=ORANGE,ha='center',va='bottom')
a.axvline(0.5,color=INK2,lw=0.7,ls=':')
a.scatter((1+p0.cliff_median)/2,y,s=30,c=BLUE,zorder=3,label='no measurement noise')
a.scatter((1+p1.cliff_median)/2,y,s=30,facecolors='white',edgecolors=NAVY,lw=1.2,zorder=3,label='with measurement noise')
for yi,r0,r1 in zip(y,p0.itertuples(),p1.itertuples()):
    a.text(max((1+r0.cliff_median)/2,(1+r1.cliff_median)/2)+0.006,yi,f"power {100*r0.power:.0f}% / {100*r1.power:.0f}%",va='center',fontsize=5.8,color=INK)
a.set_yticks(y); a.set_yticklabels(labs,fontsize=6.2); a.set_xlim(0.49,0.72); a.set_xlabel('AUC = (1 + Cliff δ)/2 of ΔAC1')
a.legend(fontsize=5.8,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(0.5,1.0)); a.set_ylim(-0.9,len(p0)+0.25)
panel(a,'a','Known fold, sampled at the real wells\' times')
for h,c,ls in [(0,BLUE,'-'),(1,NAVY,'--'),(2,AQUA,'-.')]:
    tpr=np.array([(L[:,j]>h).mean() for j in range(len(TH))])
    xx=np.r_[0,fpr[::-1],1]; yy=np.r_[0,tpr[::-1],1]; o=np.argsort(xx); auc=np.trapezoid(yy[o],xx[o])
    b.plot(xx[o],yy[o],color=c,ls=ls,lw=1.6,label=f'Threshold rule, warning >{h} yr ahead (AUC {auc:.2f})')
b.plot([0,1],[0,1],color=INK2,lw=0.7,ls=':')
b.scatter([0.5],[0.5+0.11],s=0); b.text(0.55,0.40,'ΔAC1 (critical slowing down):\nAUC ≈ 0.54–0.57;\nbest-case fold ≤ 0.64',fontsize=6.3,color=ORANGE)
b.set_xlabel('False-positive rate (stable wells)'); b.set_ylabel('Fraction of transitions warned'); b.set_xlim(0,1); b.set_ylim(0,1); b.legend(fontsize=5.8,loc='lower right')
panel(b,'b','Discrimination: threshold rule vs dynamical precursor')
save(fig,'FigS1_ews_power_roc')
# ---------- S2: well characteristics
P=pd.read_csv('raw/pool_desc.csv'); st=set(pd.read_csv('raw/stable_ids.csv').iloc[:,0])
P['grp']=np.where(P.label=='transition','Transition (145)',np.where(P.site.isin(st),'Stable control (1,439)',np.where(P.label=='always_saline','Always saline (529)','other')))
vars_=[('n','Samples per well',True),('span','Record length (yr)',False),('med_int','Sampling interval (yr)',True),('depth_ft','Well depth (ft)',True),('coast_km_marine','Distance to shore (km)',True)]
fig,axs=plt.subplots(1,5,figsize=(7.2,2.5)); fig.subplots_adjust(wspace=0.45,left=0.07,right=0.99,bottom=0.17,top=0.74)
cols={'Transition (145)':ORANGE,'Stable control (1,439)':BLUE,'Always saline (529)':GREY}
for ax,(v,t,lg) in zip(axs,vars_):
    for g,c in cols.items():
        x=np.sort(P[P.grp==g][v].dropna().to_numpy()); x=x[x>0] if lg else x
        ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=c,lw=1.4,label=g)
    if lg:
        ax.set_xscale('log'); from matplotlib.ticker import LogLocator,NullFormatter; ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(t,fontsize=7,loc='left'); ax.set_ylim(0,1)
axs[0].set_ylabel('Cumulative fraction'); h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=3,fontsize=6.5,bbox_to_anchor=(0.5,1.0))
save(fig,'FigS2_well_characteristics')
print('ok')
