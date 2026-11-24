"""Separate query-weighted transfer effects, class heterogeneity and historic selection IDs."""
from pathlib import Path

import numpy as np
import torch

from .concept_benchmark import atomic_json


PRIMARY='same_partition'
CONTROLS=('original','semantic_regional','wrong_coordinates','individual_matches','uncentered_energy')


def effects(primary,comparison,labels):
    delta=(100*(primary.double()-comparison.double())).numpy()
    classes,inverse=np.unique(labels.numpy(),return_inverse=True)
    counts=np.bincount(inverse)
    totals=np.stack([np.bincount(inverse,weights=delta[:,j]) for j in range(2)],axis=1)
    means=totals/counts[:,None]
    rng=np.random.default_rng(42)
    draws=rng.integers(0,len(classes),size=(10000,len(classes)))
    micro=totals[draws].sum(1)/counts[draws].sum(1)[:,None]
    macro=means[draws].mean(1)
    per_class={str(k):dict(count=int(n),mean_points=v.tolist()) for k,n,v in zip(classes,counts,means)}
    had_hit=comparison[:,1].numpy()>0
    has_hit=primary[:,1].numpy()>0
    transitions={}
    for name,mask in [('gained',~had_hit & has_hit),('lost',had_hit & ~has_hit),
                      ('retained',had_hit & has_hit),('absent',~had_hit & ~has_hit)]:
        transitions[name]=dict(queries=int(mask.sum()),contribution_to_map=float(delta[mask,0].sum()/len(labels)))
    assert abs(sum(x['contribution_to_map'] for x in transitions.values())-delta[:,0].mean())<1e-8
    return dict(points=delta.mean(0).tolist(),micro95=np.quantile(micro,[.025,.975],axis=0).T.tolist(),
        macro_points=means.mean(0).tolist(),macro95=np.quantile(macro,[.025,.975],axis=0).T.tolist(),
        improved_classes=(means>0).sum(0).tolist(),classes=len(classes),per_class=per_class,
        query_wins=(delta>0).sum(0).tolist(),query_losses=(delta<0).sum(0).tolist(),
        query_ties=(delta==0).sum(0).tolist(),hit_at_precision_cutoff=transitions)


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Native GCWE','No GCWE','Scrambled coordinates','Unbinned matches','Raw offset votes']
    titles=['Sketchy Ext-1','Sketchy Ext-2','TU-Berlin']
    fig,axes=plt.subplots(1,3,figsize=(10,3.3),sharey=True,layout='constrained')
    for axis,(split,pack),title in zip(axes,report.items(),titles):
        axis.axvline(0,color='.55',lw=.8)
        for i,control in enumerate(CONTROLS):
            r=pack['all']['contrasts'][control]
            ci=r['micro95'][0]
            color='#18577b' if control=='original' else '#686f77'
            axis.plot(ci,[i,i],color=color,lw=1.5)
            axis.plot(r['points'][0],i,'o',color=color,ms=5)
        axis.set_title(title,fontsize=11)
        axis.set_xlabel('mAP difference (points)',fontsize=9)
        axis.grid(axis='x',color='.92',lw=.6)
        axis.spines[['top','right','left']].set_visible(False)
        axis.tick_params(labelsize=8)
    axes[0].set_yticks(range(len(names)),names)
    axes[0].invert_yaxis()
    fig.suptitle('GCWE with displacement concentration minus each control',fontsize=12)
    fig.savefig(output/'displacement-comparisons.pdf')
    fig.savefig(output/'displacement-comparisons.png',dpi=175)
    plt.close(fig)


def run():
    root=Path('/content/sbir/results')
    output=root/'gcwe-displacement-full'
    selected=[]
    for split in ('sketchy_1','sketchy_2'):
        grouped=[]
        for family in ('gcwe-displacement-consensus','gcwe-displacement-concentration'):
            rec=torch.load(root/family/(split+'.pt'),weights_only=True,mmap=True)
            grouped.append(rec['qi'])
        within=torch.cat(grouped)
        assert len(within)==3072 and len(within.unique())==len(within)
        selected.append(within)
    selection=torch.cat(selected).unique()
    report={}
    for split in ('sketchy_1','sketchy_2','tuberlin'):
        rec=torch.load(output/(split+'.pt'),weights_only=True,mmap=True)
        ref=torch.load(root/'regional-additive-full'/(split+'.pt'),weights_only=True,mmap=True)
        assert all(torch.equal(rec[key],ref[key]) for key in ('qi','pi','labels'))
        rows=rec['rows']
        assert set(rows)=={PRIMARY,*CONTROLS}
        groups={'all':torch.ones(len(rec['qi']),dtype=torch.bool)}
        if split!='tuberlin':
            groups['outside_either_selection']=~torch.isin(rec['qi'],selection)
        data={}
        for name,mask in groups.items():
            metrics={k:(100*v[mask].double().mean(0)).tolist() for k,v in rows.items()}
            contrasts={k:effects(rows[PRIMARY][mask],rows[k][mask],rec['labels'][mask]) for k in CONTROLS}
            data[name]=dict(queries=int(mask.sum()),metrics=metrics,contrasts=contrasts)
        report[split]=data
    atomic_json(output/'paired-effects.json',report)
    plot(report,output)
    print({k:v['all']['contrasts']['original']['points'] for k,v in report.items()},flush=True)


if __name__=='__main__':
    run()
