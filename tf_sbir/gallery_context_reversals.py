"""Compare fixed sketch/photo identities after two independent native Sketchy runs."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_policy import outputs
from .data import UNSEEN_CLASSES
from .gallery_panel_resume import region_views, semantic_inputs
from .gallery_region_panels import CONFIG, Index
from .positive_foil import base_parts, STRONG


def coordinates(global_ids, common):
    lookup={v:i for i,v in enumerate(global_ids.tolist())}
    return torch.tensor([lookup[v] for v in common.tolist()])


def consensus_examples(first,second,t1,t2,truth,query_files,image_files,classes,photo_classes):
    examples=[]
    for c in sorted(set(classes)):
        for was_correct in [True,False]:
            which=[j for j,v in enumerate(classes) if v==c and
                   bool(truth[j,t1[j]])==was_correct and bool(truth[j,t2[j]])!=was_correct]
            if not which:continue
            idx=torch.tensor(which)
            a,b=t1[idx],t2[idx]
            gaps=(first[idx,a]-first[idx,b]).abs()+(second[idx,a]-second[idx,b]).abs()
            j=int(idx[gaps.argsort()[len(idx)//2]])
            examples.append({'query':str(query_files[j].parent.name+'/'+query_files[j].name),
                             'correct_class':c,'selected_ext1':str(image_files[t1[j]].name),
                             'selected_ext2':str(image_files[t2[j]].name),
                             'class_ext1':photo_classes[t1[j]],'class_ext2':photo_classes[t2[j]],
                             'same_pair_ext1_margin':float(first[j,t1[j]]-first[j,t2[j]]),
                             'same_pair_ext2_margin':float(second[j,t1[j]]-second[j,t2[j]])})
    return examples


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    cfg,paths,qs,photo_paths,gs,q,p,src,tgt,scale=semantic_inputs(w,'sketchy')
    ss=('sketchy_1','sketchy_2')
    def common_ids(groups):
        return torch.tensor(sorted(set(groups[ss[0]][0].tolist())&set(groups[ss[1]][0].tolist())))
    shared_queries,shared_photos=common_ids(qs),common_ids(gs)
    cache={}
    qnames={}
    pnames={}
    for s in ss:
        qi,yl=qs[s]
        pi,gl=gs[s]
        where_q=coordinates(qi,shared_queries)
        where_p=coordinates(pi,shared_photos)
        # Construct every graph on the complete native gallery. Only store common
        # columns after scoring. No features, scores, or ranks from the other
        # split enter the retrieval pipeline.
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        views,_=region_views(w,s,cfg,paths,qi,q)
        folder,tree,regional=base_parts(w,s)
        scores={}
        for begin in range(0,len(qi),192):
            rows=where_q[(where_q>=begin)&(where_q<begin+192)]
            if not len(rows):continue
            pack=torch.load(folder/'full'/f'{begin:06d}.pt',mmap=True,weights_only=True)
            for ix in rows.split(48):
                out_id=torch.searchsorted(shared_queries,qi[ix])
                assert torch.equal(shared_queries[out_id],qi[ix])
                offset=ix-begin
                assert torch.equal(pack['ids'][offset],ix)
                reg=regional[ix].cuda()
                _,old,_=next(outputs(tree,pack['densities'][offset].cuda(),reg,reg,[STRONG]))
                a=(scale*(q[qi[ix]].cuda()@src.T)).softmax(1)
                b=(scale*(views['flipped'][ix].float().cuda()@src.T)).softmax(1)
                evidence=((a+b)/2)@bank.T
                log=evidence.clamp_min(1e-30).log()
                diff=graph.score(CONFIG,evidence).clamp_min(1e-30).log()
                geom=graph.score(CONFIG,log)
                single=graph.score(CONFIG,a@bank.T).clamp_min(1e-30).log()
                new=reg+.65*(diff-single)
                strong=new+(old-reg)*new.std(1,keepdim=True)/reg.std(1,keepdim=True)
                for name,val in [('direct',log),('diffused',diff),('geometric',geom),('strong',strong)]:
                    if name not in scores:scores[name]=torch.full((len(shared_queries),len(shared_photos)),torch.nan)
                    scores[name][out_id]=val[:,where_p].cpu()
        assert all(v.isfinite().all() for v in scores.values())
        cache[s]=scores
        qnames[s]=[UNSEEN_CLASSES[s][j] for j in yl[where_q].tolist()]
        pnames[s]=[UNSEEN_CLASSES[s][j] for j in gl[where_p].tolist()]
        del graph,bank
    assert qnames[ss[0]]==qnames[ss[1]] and pnames[ss[0]]==pnames[ss[1]]
    categories=sorted(set(qnames[ss[0]]))
    labels=torch.tensor([categories.index(v) for v in qnames[ss[0]]])
    gallery=torch.tensor([categories.index(v) for v in pnames[ss[0]]])
    match=labels[:,None]==gallery[None,:]
    result={'queries':len(labels),'shared_photos':len(gallery),'classes':categories,
            'native_gallery_sizes':[len(gs[s][0]) for s in ss],'scores':{}}
    direct=cache[ss[0]]['direct']
    directs=[cache[s]['direct'] for s in ss]
    result['max_direct_logscore_difference']=float((directs[0]-directs[1]).abs().max())
    # Fixed diagnostic pairs. Select boundaries in the direct score on common
    # photos, then check both native retrievals on exactly the same pairs.
    order=direct.topk(min(200,len(gallery))).indices
    hits=match.gather(1,order)
    boundary=hits[:,:-1]^hits[:,1:]
    left,right=order[:,:-1],order[:,1:]
    polarity=hits[:,:-1].float()*2-1
    row=torch.arange(len(labels))
    for name in cache[ss[0]]:
        first,second=[cache[s][name] for s in ss]
        t1,t2=first.argmax(1),second.argmax(1)
        h1,h2=match[row,t1],match[row,t2]
        gaps=[(v.gather(1,left)-v.gather(1,right))*polarity for v in (first,second)]
        def tally(mask):
            return {'pairs':int(mask.sum()),'rank_reversed':int(((gaps[0]*gaps[1]<0)&mask).sum()),
                    'ext1_correct_ext2_wrong':int(((gaps[0]>0)&(gaps[1]<0)&mask).sum()),
                    'ext2_correct_ext1_wrong':int(((gaps[1]>0)&(gaps[0]<0)&mask).sum())}
        result['scores'][name]={
            'top_common_correct':[int(h1.sum()),int(h2.sum())],
            'different_top_photo':int((t1!=t2).sum()),
            'different_top_class':int((gallery[t1]!=gallery[t2]).sum()),
            'top_correctness_changed':int((h1^h2).sum()),
            'fixed_direct_boundaries':tally(boundary),
            'per_class':{c:tally(boundary&(labels==i)[:,None]) for i,c in enumerate(categories)}}
        if name in ('diffused','strong'):
            result['scores'][name]['examples']=consensus_examples(first,second,t1,t2,match,
                  [Path(paths[j]) for j in shared_queries],[Path(photo_paths[j]) for j in shared_photos],
                  qnames[ss[0]],pnames[ss[0]])
    out=w/'results/gallery-context-reversals'
    out.mkdir(exist_ok=True,parents=True)
    ii,jj=boundary.nonzero(as_tuple=True)
    torch.save({'queries':shared_queries,'photos':shared_photos,'rows':ii,'left':left[ii,jj],
                'right':right[ii,jj],'orientation':polarity[ii,jj]},
               out/'fixed-boundaries.pt')
    (out/'sketchy.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='scores'}),flush=True)

if __name__=='__main__':
    run()
