"""Plot paired source identification on the new photos, with original and randomized coordinates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    root=Path('/content/sbir/results/regional-layout-forecast')
    reports=json.loads((root/'conditional-information.json').read_text())['assessment']
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.0),sharex=True,sharey=True,layout='constrained')
    methods=[('registered','Original coordinates','#0a6655'),('uniform_orientation','Orientation-averaged','#6d7682'),
             ('inferred','Inferred orientation','#447aa3')]
    for col,split in enumerate(('sketchy_1','sketchy_2')):
        ax=axes[col]
        data=reports[split+'/same']['conditional']['global_wrong']
        ax.axvline(50,color='#9fa2a5',lw=1,ls='--')
        for i,(key,_,color) in enumerate(methods):
            row=data[key]
            ci=[100*x for x in row['pair_bootstrap95']]
            x=100*row['mean']
            ax.plot(ci,[i,i],color=color,lw=2)
            ax.scatter(x,i,color=color,zorder=3,s=34)
            ax.annotate(f'{x:.1f}%',xy=(x,i),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(f'Sketchy Ext-{col+1} · {data["registered"]["pairs"]} photo pairs',fontsize=10)
        ax.set_xlim(40,83)
        ax.set_ylim(2.5,-.65)
        ax.set_xlabel('Correct source assignments (%)')
        ax.grid(axis='x',color='#eeeeee')
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,labelsize=9)
    axes[0].set_yticks(range(3),[v[1] for v in methods])
    fig.suptitle('Same-class assignments missed by the global semantic score',fontsize=12)
    fig.savefig(root/'conditional-information.pdf')
    fig.savefig(root/'conditional-information.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    run()
