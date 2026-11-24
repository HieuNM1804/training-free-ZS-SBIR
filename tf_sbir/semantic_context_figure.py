"""Plot source-disjoint category decision ceilings and equal-patch full-gallery controls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def run():
    root=Path('/content/sbir/results')
    folder=root/'semantic-context-union'
    splits=['sketchy_1','sketchy_2']
    data=[json.loads((folder/(s+'.json')).read_text()) for s in splits]
    limits=[json.loads((root/'semantic-dependence-limits'/(s+'.json')).read_text())['assessment'] for s in splits]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={'width_ratios':[.9,1.5]})
    red,blue,light='#b45438','#234e70','#b5cfdd'
    for i,row in enumerate(limits):
        fixed=row['unfixable_errors']
        total=row['errors_independent']
        prop=100*fixed/total
        ax.barh(i,prop,height=.42,color=blue)
        ax.barh(i,100-prop,left=prop,height=.42,color=light)
        ax.text(3,i,f'{fixed} / {total} cannot reverse',color='white',va='center',size=10)
        ax.text(50,i+.31,f'{total-fixed} potentially reversible',ha='center',size=9,color='#36454e')
    ax.set(xlim=(0,100),ylim=(1.7,-.6),xlabel='Share of independent-model errors (%)')
    ax.set_yticks([0,1],['Ext-1','Ext-2'])
    ax.set_title('A   Which pair errors can dependence fix?',loc='left',fontsize=11,weight='bold',pad=12)
    conditions=['separate','raw_additive','joint_union','full']
    for i,(row,color,label) in enumerate(zip(data,[blue,red],['Ext-1  mAP@all','Ext-2  mAP@200'])):
        yy=[row['metrics'][k+'/words'][0] for k in conditions]
        bx.plot(range(4),yy,marker=['o','s'][i],color=color,label=label,linewidth=1.8)
        for j,y in enumerate(yy):
            bx.annotate(f'{y:.1f}',(j,y),xytext=(0,8 if i==0 else -14),textcoords='offset points',ha='center',size=9,color=color)
    bx.set_xticks(range(4),['Separate\nstrips','Additive\nprediction','Joint read,\nsame patches','Complete\nsketch'])
    bx.set_ylabel('Full-gallery mAP (%)')
    bx.margins(x=.11,y=.27)
    bx.yaxis.set_major_locator(MaxNLocator(nbins=5))
    bx.set_title('B   What changes when CLIP reads the parts jointly?',loc='left',fontsize=11,weight='bold',pad=12)
    bx.legend(frameon=False,loc='lower right',fontsize=8)
    for a in (ax,bx):
        a.spines[['top','right']].set_visible(False)
        a.tick_params(length=3,color='#81909a',labelsize=9)
    bx.grid(axis='y',alpha=.16)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(folder/'semantic-context-and-limits.pdf',bbox_inches='tight')
    fig.savefig(folder/'semantic-context-and-limits.png',dpi=175,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    run()
