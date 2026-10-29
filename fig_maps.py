import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from shapely.geometry import shape
from figstyle import *
style()
gj=json.load(open('gis/ne_10m_admin_1_states_provinces_lakes.geojson'))
STUDY={'AL','CA','CT','DE','FL','GA','LA','MA','MD','MS','NC','NJ','NY','RI','SC','TX','VA'}
feats=[(f['properties'].get('postal'),shape(f['geometry'])) for f in gj['features'] if f['properties'].get('adm0_a3')=='USA' and f['properties'].get('postal') not in ('AK','HI')]
def draw_states(ax):
    for pc,g in feats:
        polys=[g] if g.geom_type=='Polygon' else list(g.geoms)
        for p in polys:
            x,y=p.exterior.xy
            ax.fill(x,y,color='#dbe9ec' if pc in STUDY else '#f1efe9',ec='#b9b6ad',lw=0.35,zorder=1)
d=pd.read_csv('raw/trans_classified.csv')
def setup(ax,xl,yl):
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect(1/np.cos(np.deg2rad(np.mean(yl)))); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color('#b9b6ad'); s.set_linewidth(0.5)
    ax.set_facecolor('#f7fafb')
def plot_wells(ax,ms=26):
    for k,v in MECH.items():
        s=d[d.mech==k]; ax.scatter(s.lon,s.lat,s=ms,c=v['c'],marker=v['m'],ec='white',lw=0.6,zorder=3,label=f"{v['lab']} (n={len(s)})")
fig=plt.figure(figsize=(7.2,5.9))
ax=fig.add_axes([0.0,0.43,0.66,0.55]); draw_states(ax); setup(ax,(-124.8,-66.5),(24.3,49.5)); plot_wells(ax,22)
ax.plot([-88,-78.6,-78.6,-88,-88],[24.4,24.4,32.6,32.6,24.4],color=INK2,lw=0.6,ls='--',zorder=4)
panel(ax,'a','All 145 transition wells by salinization setting')
ax2=fig.add_axes([0.665,0.43,0.335,0.55]); draw_states(ax2); setup(ax2,(-88,-78.6),(24.4,32.6)); plot_wells(ax2,26)
panel(ax2,'b','Florida, Georgia and South Carolina')
for txt,(x,y) in {'Biscayne\n(SE Florida)':(-79.25,25.6),'Brunswick':(-79.85,31.15),'Cocoa\nwell field':(-79.75,28.55),'Collier/Lee\n(SW Florida)':(-83.1,26.2),'Tampa Bay':(-83.75,27.75),'Jacksonville':(-79.95,30.3),'Palm Beach':(-79.15,27.05)}.items():
    ax2.text(x,y,txt,fontsize=6,color=INK2,ha='center',va='center',zorder=5)
h,l=ax.get_legend_handles_labels()
fig.legend(h,l,loc='upper left',bbox_to_anchor=(0.02,0.485),ncol=2,fontsize=6.8,handletextpad=0.3,columnspacing=1.2)
# stacked bars by state
ax3=fig.add_axes([0.08,0.06,0.90,0.31])
order=d.state.value_counts().index.tolist()
bottom=np.zeros(len(order))
for k,v in MECH.items():
    cnt=np.array([((d.state==s)&(d.mech==k)).sum() for s in order])
    ax3.bar(range(len(order)),cnt,bottom=bottom,color=v['c'],width=0.7,ec='white',lw=0.8); bottom+=cnt
for i,s in enumerate(order):
    ax3.text(i,bottom[i]+1.2,str(int(bottom[i])),ha='center',fontsize=7,color=INK)
ax3.set_xticks(range(len(order))); ax3.set_xticklabels([f"{s}\n{np.median(d[d.state==s].lead150):.1f} yr" for s in order]); ax3.set_ylabel('Transition wells'); ax3.set_ylim(0,80); ax3.set_xlim(-0.6,len(order)-0.4)
ax3.yaxis.grid(True,color=GRID,lw=0.5); ax3.set_axisbelow(True)
panel(ax3,'c','Transition wells by state and setting (state median lead time at 150 mg L$^{-1}$ under each label)')
save(fig,'Fig1_map_settings')
# ---------- Fig S3 watch list
w=pd.read_csv('raw/watchlist_2015_geo.csv')
fig=plt.figure(figsize=(7.2,4.0)); ax=fig.add_axes([0,0,1,0.92]); draw_states(ax); setup(ax,(-124.8,-66.5),(24.3,49.5))
ax.scatter(d.lon,d.lat,s=10,c=MUTED,marker='o',ec='none',zorder=2,label=f'Historical transition wells (n={len(d)})')
for lab,m,c,mk in [('≤10 km from shore',w.coast_km<=10,BLUE,'o'),('10–50 km',(w.coast_km>10)&(w.coast_km<=50),ORANGE,'s'),('>50 km',w.coast_km>50,AQUA,'^')]:
    ax.scatter(w[m].lon,w[m].lat,s=24,c=c,marker=mk,ec='white',lw=0.6,zorder=3,label=f'Current 150-tier watch list, {lab} (n={m.sum()})')
ax.legend(loc='lower left',fontsize=6.8)
ax.set_title('Wells sampled since 2015 whose two most recent samples lie in 150–250 mg L$^{-1}$ and that have never reached 250 mg L$^{-1}$ (n=169)',fontsize=8,loc='left')
save(fig,'FigS3_watchlist_map')
print('ok')
