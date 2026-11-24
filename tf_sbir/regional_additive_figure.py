"""Plot the paired effect of the regional intervention, using class-weighted estimates throughout."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    folder=Path('/content/sbir/results/regional-additive-full')
    report=json.loads((folder/'paired-effects.json').read_text())
    names={'sketchy_1':'Sketchy Ext-1','sketchy_2':'Sketchy Ext-2','tuberlin':'TU-Berlin'}
    fig,ax=plt.subplots(figsize=(7.2,3.5),layout='constrained')
    ax.axvline(0,color='#818181',linewidth=1,linestyle='--')
    for mode,shift,color,marker,label in [
            ('with_gcwe',.12,'#175f85','o','With GCWE'),
            ('without_gcwe',-.12,'#ba562b','s','Without GCWE')]:
        for i,(key,split_name) in enumerate(names.items()):
            stat=report[key]['all'][mode]['class_ap']
            center=stat['mean_across_classes']
            lo,hi=stat['bootstrap95']
            y=2-i+shift
            ax.hlines(y,lo,hi,color=color,linewidth=1.7)
            ax.scatter(center,y,color=color,marker=marker,s=34,zorder=3,label=label if i==0 else None)
    ax.set_yticks([2,1,0],list(names.values()))
    ax.set_ylim(-.65,2.53)
    ax.set_xlabel('Change in AP, averaged equally across classes (percentage points)')
    ax.set_title('Additive versus original region matching on the full gallery',loc='left')
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x',color='#dddddd',linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0)
    ax.legend(loc='lower left',ncol=2,frameon=False,fontsize=9)
    fig.savefig(folder/'paired-class-effects.pdf',bbox_inches='tight')
    fig.savefig(folder/'paired-class-effects.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    run()
