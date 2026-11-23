"""Locate same-budget candidate omissions in the frozen cross-domain retrieval channels."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .benchmark import merge_inputs
from .data import SKETCHY_SPLITS,UNSEEN_CLASSES
from .gallery_region_panels import CONFIG,Index
from .standardized_context_audit import audit_inputs


def interleaved(a,b):
    # A diagnostic candidate set of exactly k distinct images. Its purpose is
    # to measure availability, not to tune a final ranking or select on TU.
    rows=[]
    for x,y in zip(a.tolist(),b.tolist(),strict=True):
        sequence=dict.fromkeys(v for pair in zip(x,y,strict=True) for v in pair)
        rows.append(list(sequence)[:len(x)])
    return torch.tensor(rows)


def first_positive(hits):
    return torch.where(hits.any(1),hits.int().argmax(1)+1,hits.size(1)+1)


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,encoded,_,q,p,flipped,src,tgt,scale=audit_inputs(w,all_sketches=True)
    assert torch.equal(encoded,torch.arange(len(q)))
    data_root=json.loads((w/'ready.json').read_text())['root']
    query_files,_=merge_inputs(data_root,SKETCHY_SPLITS,'sketch')
    photo_files,_=merge_inputs(data_root,SKETCHY_SPLITS,'photo')
    result={}
    for s,(qi,labels) in qs.items():
        pi,classes=gs[s]
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(z@tgt.T)).softmax(1) for z in g.split(256)])
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        lists={k:[] for k in ('native','direct_words','visual')}
        for ii in qi.split(128):
            v=q[ii].cuda()
            f=flipped[ii].float().cuda()
            source=((scale*(v@src.T)).softmax(1)+(scale*(f@src.T)).softmax(1))/2
            word=source@bank.T
            scores={'native':graph.score(CONFIG,word),'direct_words':word,'visual':((v+f)/2)@g.T}
            for k,score in scores.items():lists[k].append(score.topk(200).indices.cpu())
        lists={k:torch.cat(rows) for k,rows in lists.items()}
        for k in ('direct_words','visual'):
            lists['native_and_'+k]=interleaved(lists['native'],lists[k])
        hits={k:classes[v]==labels[:,None] for k,v in lists.items()}
        report={'queries':len(qi),'gallery':len(g),'budgets':{}}
        for budget in (1,16,64,200):
            masks={k:h[:,:budget].any(1) for k,h in hits.items()}
            native=masks['native']
            by={}
            for k,m in masks.items():
                by[k]={'queries_with_positive':int(m.sum()),
                       'recover_native_miss':int((~native&m).sum()),'lose_native_hit':int((native&~m).sum()),
                       'mean_correct_photos_in_budget':float(hits[k][:,:budget].sum(1).double().mean())}
            report['budgets'][budget]=by
        native=hits['native'].any(1)
        visual=hits['visual'].any(1)
        direct=hits['direct_words'].any(1)
        miss=~native
        report['native_top200_misses']={'total':int(miss.sum()),'visual_only':int((miss&visual&~direct).sum()),
            'direct_only':int((miss&direct&~visual).sum()),'both':int((miss&direct&visual).sum()),
            'neither':int((miss&~visual&~direct).sum())}
        # Characterize the retrieved classes only after all five lists are final.
        counts=torch.zeros(len(labels),len(UNSEEN_CLASSES[s])).scatter_add_(
            1,classes[lists['native']],torch.ones(len(labels),200))
        mode_fraction=counts.max(1).values/200
        report['examples']={}
        for key,mask in [('native_miss_visual_hit',miss&visual),('native_miss_all_miss',miss&~visual&~direct),
                         ('native_hit_visual_miss',native&~visual)]:
            ids=mask.nonzero().flatten()
            if not len(ids):continue
            chosen=ids[mode_fraction[ids].argsort(stable=True)[len(ids)//2]]
            report['examples'][key]={'count':len(ids),'mean_modal_class_fraction_in_native_head':float(mode_fraction[ids].mean()),
                'median_example':{'query':str(Path(query_files[qi[chosen]]).relative_to(data_root)),
                    'query_class':UNSEEN_CLASSES[s][labels[chosen]],
                    'results':{k:{'first':str(Path(photo_files[pi[v[chosen,0]]]).relative_to(data_root)),
                                 'first_relevant_rank_or_201':int(first_positive(hits[k][chosen:chosen+1])[0])} for k,v in lists.items()}}}
        report['native_misses_by_class']={UNSEEN_CLASSES[s][i]:int((miss&(labels==i)).sum()) for i in labels.unique()}
        result[s]=report
        print(s,report['native_top200_misses'],flush=True)
        del graph,bank
    out=w/'results/unseen-candidate-coverage'
    out.mkdir(exist_ok=True,parents=True)
    (out/'sketchy.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    run()
