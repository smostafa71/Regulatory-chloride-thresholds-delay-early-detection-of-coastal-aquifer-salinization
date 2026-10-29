import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figstyle import *
style()
fig=plt.figure(figsize=(7.2,6.4)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
def box(x,y,w,h,text,fc='#ffffff',ec=INK2,fs=6.6,bold_first=True,lw=0.7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=1.2',fc=fc,ec=ec,lw=lw))
    lines=text.split('\n')
    if bold_first:
        ax.text(x+w/2,y+h-1.4,lines[0],ha='center',va='top',fontsize=fs+0.6,fontweight='bold',color=INK)
        ax.text(x+w/2,y+h-4.2,'\n'.join(lines[1:]),ha='center',va='top',fontsize=fs,color=INK,linespacing=1.25)
    else: ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,linespacing=1.25)
def arrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color=INK2,lw=0.8,shrinkA=0,shrinkB=0,mutation_scale=8))
# Stage 1-2
box(3,91,94,7.5,"1  Data assembly (Water Quality Portal / NWIS, 17 coastal states, 1980–2026)\n218,181 chloride results · 44,990 wells · exact duplicates removed (99.9% reported in mg L$^{-1}$) · NWIS site metadata (depth, aquifer) · NWIS/OGC water levels (100 wells)",fc='#eef4fb',ec=BLUE)
arrow(50,91,50,88.6)
box(3,74,94,14.3,"2  Well classification (Section 2.3, Eqs. 1–2)\n≥20 samples → 2,139 eligible wells; baseline = median of the first third of the record\nTransition (n = 145): baseline < 250 and ≥2 consecutive samples ≥ 250 later in the record\nStable control (n = 1,439): never two consecutive samples ≥ 250, median < 250  ·  Always saline (n = 529): excluded\nAny level T (50–240 mg L$^{-1}$) is crossed under the same two-consecutive rule; lead time L(T) = t$_{250}$ − t$_{T}$",fc='#eef4fb',ec=BLUE)
for x in [14.5,38.5,62.5,86.5]: arrow(50,74,x,71.1)
cols=[("A  Dynamical precursors","Question: is salinization preceded\nby critical slowing down?\nMethod: rolling AC1, CV, memory\ntime τ (irregular-sampling AR(1));\ntransition vs control windows;\npower test at real sample times"),
      ("B  Trajectory shape &\nforecastability","Question: does the rise carry\ninformation on arrival time?\nMethod: quadratic curvature by\nbaseline; forecasts at the 150\ncrossing (linear, Theil–Sen, local,\nquadratic) vs naive baseline"),
      ("C  Threshold trade-off","Question: how much lead time does\na lower trigger buy, at what\nfalse-alarm cost?\nMethod: sweep T = 50–240 mg L$^{-1}$;\ncensoring-corrected FPR (Eq. 5),\nPPV, ROC, elbow bootstrap"),
      ("D  Counterfactual\nmonitoring","Question: is lead time a threshold\neffect or a sampling effect?\nMethod: interpolate each well;\nfixed 250 reference; resample\nannual/semi-annual/quarterly\nat T = 250, 150, 100 (Eq. 6)")]
xs=[3,27,51,75]
for (t,b),x in zip(cols,xs):
    ax.add_patch(FancyBboxPatch((x,51.5),22,19,boxstyle='round,pad=0.25,rounding_size=1.2',fc='#ffffff',ec=INK2,lw=0.7))
    ax.text(x+11,70.0,t,ha='center',va='top',fontsize=7.1,fontweight='bold',color=INK,linespacing=1.1)
    ax.text(x+11,64.6,b,ha='center',va='top',fontsize=6.3,color=INK,linespacing=1.25)
prods=[("Product","ΔAC1, ΔCV, Δlog τ per well;\nMann–Whitney, Cliff's δ ⇒ AUC;\nachievable δ under a known fold"),
       ("Product","curvature (mg L$^{-1}$ yr$^{-2}$) by\nbaseline group; forecast error\nand bias vs naive median"),
       ("Product","median lead time and FPR per T;\ndecision frontier; warning band;\nPPV and alarms per 100 wells"),
       ("Product","median lead time (depth) and\nnumber detected (coverage)\nper policy")]
mean=[("Physical meaning","loss of resilience (slower\nrecovery = higher memory)\nbefore a fold-type transition"),
      ("Physical meaning","steepening signal of an\napproaching saline front vs\nmixing-zone (linear) rise"),
      ("Physical meaning","a decision rule, not a\ngeochemical boundary"),
      ("Physical meaning","separates the two policy\nlevers: trigger level vs\nsampling effort")]
res=["→ Results 3.2: precursor too weak\nfor well-level warning (AUC ≈ 0.56)","→ Results 3.3: real acceleration,\nno forecast beats naive baseline","→ Results 3.4: 100–150 mg L$^{-1}$ band;\n≈2 yr lead at FPR < 10%","→ Results 3.5: threshold sets lead,\nfrequency sets coverage"]
for i,x in enumerate(xs):
    arrow(x+11,51.5,x+11,49.6)
    box(x,39.5,22,9.6,prods[i][0]+'\n'+prods[i][1],fc='#fbf6f2',ec=ORANGE,fs=6.1)
    arrow(x+11,39.5,x+11,37.6)
    box(x,27.5,22,9.6,mean[i][0]+'\n'+mean[i][1],fc='#f1f8f4',ec=AQUA,fs=6.1)
    ax.text(x+11,25.6,res[i],ha='center',va='top',fontsize=6.2,color=INK2,style='italic',linespacing=1.2)
box(3,2,45,14.5,"Robustness (Results 3.7, Supplementary)\ncensoring correction of FPR · left-censoring (Kaplan–Meier) ·\nleave-one-state-out · coastal-only subsets (≤10/25/50 km) ·\npersistence rule (2 vs 3 samples) · stricter QA · strict alarm-date\ncounterfactual · data-availability subsets",fc='#f6f6f4',ec=GREY,fs=6.1)
box(52,2,45,14.5,"Context (Section 2.1, Fig. 1, Table 1)\nhydrogeologic setting and salinization mechanism per well\n(M1–M4); aquifer lithology; distance to marine shoreline;\nwell depth; representativeness vs stable wells;\ncurrent 150-tier watch list (Discussion)",fc='#f6f6f4',ec=GREY,fs=6.1)
save(fig,'Fig2_workflow')
