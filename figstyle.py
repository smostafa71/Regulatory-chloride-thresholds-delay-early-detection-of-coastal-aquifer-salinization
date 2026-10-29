import matplotlib as mpl, matplotlib.pyplot as plt
BLUE='#2a78d6'; ORANGE='#eb6834'; AQUA='#1baf7a'; GREY='#8c8b86'; INK='#0b0b0b'; INK2='#52514e'; MUTED='#9a9993'; GRID='#e6e5e1'
NAVY='#184f95'
MECH={'M1':dict(c=BLUE,m='o',lab='M1 Lateral seawater/estuarine intrusion'),
      'M2':dict(c=ORANGE,m='s',lab='M2 Floridan karst: lateral intrusion & upconing'),
      'M3':dict(c=AQUA,m='^',lab='M3 Relict saline water, Gulf Coast confined sands'),
      'M4':dict(c=GREY,m='D',lab='M4 Inland / non-marine or uncertain source')}
def style():
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'legend.frameon':False,'legend.fontsize':7,
        'savefig.dpi':300,'figure.dpi':150,'axes.grid':False,'lines.linewidth':1.6})
def panel(ax,letter,title=None,x=None,y=None):
    ax.set_title(r'$\bf{'+letter+'}$   '+(title or ''),loc='left',fontsize=8.5,color=INK,pad=5)
def save(fig,name):
    fig.savefig(f'figs/{name}.png',bbox_inches='tight',dpi=300); fig.savefig(f'figs/{name}.pdf',bbox_inches='tight')
