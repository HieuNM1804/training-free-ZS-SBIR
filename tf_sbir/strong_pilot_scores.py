"""Reconstruct the published two-view regional/hierarchical score on specified query IDs."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .branch_view_inputs import prepare_gallery
from .branch_view_policy import outputs
from .branch_view_study import PrefixReplay
from .cross_view_region_cache import two_pose_regions
from .cross_view_regions import CandidateRegions
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images, configure_image_worker, normal_transform
from .encoding import load_frozen_clip,save_tensor
from .gallery_panel_resume import semantic_inputs
from .gallery_region_panels import CONFIG,Index
from .positive_foil import STRONG,raw_ranks
from .query_partition_tree import Hierarchy
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index
from .retrieval_evidence import load_kd


def location(w,s,rows):
    uid=hashlib.sha256(rows.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    return w/'cache/frozen-strong-query-subsets'/f'{s}-{uid}.pt'


@torch.inference_mode()
def run(w,ds,full):
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=semantic_inputs(w,ds)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    if ds=='tuberlin' and not full:raise ValueError('TU requires all 30 unseen classes')
    for s,(qi,y) in qs.items():
        rows=torch.arange(len(qi)) if full else torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        pi,gl=gs[s]
        path=location(w,s,rows)
        if path.is_file():
            old=torch.load(path,mmap=True,weights_only=True)
            if not torch.equal(old['qi'],qi[rows]) or not torch.equal(old['pi'],pi):raise ValueError('Different data')
            continue
        g=p[pi].cuda()
        gallery_paths=[str(pp[i]) for i in pi.tolist()]
        stamp=digest({'gallery':gallery_paths,'model':CHECKPOINT,'source':'regional-64/RGB16'})
        words,region_index=index(w,s,g,tgt,scale,stamp[:16])
        pack=prepare_gallery(w,s,g,tgt,scale,gallery_paths)
        tree=Hierarchy([v.cuda() for v in pack['ids']],[v.cuda() for v in pack['parents']],pack['sizes'])
        count=tree.sizes[0]
        banks=pack['banks'].cuda().flatten(0,1).T.contiguous()
        prior=torch.bincount(tree.ids[0],minlength=count).float()/len(g)
        del pack
        torch.set_float32_matmul_precision('highest')
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        photos=torch.cat([(scale*(v@tgt.T)).softmax(1) for v in g.split(256)])
        model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
        paths=[qp[i] for i in qi[rows].tolist()]
        loader=DataLoader(Images(paths,normal_transform(),'rgb'),batch_size=48,
            num_workers=min(3,len(os.sched_getaffinity(0))),pin_memory=True,worker_init_fn=configure_image_worker)
        scores=[]
        native=[]
        details=[]
        replay=PrefixReplay(kd,gl,s)
        start=0
        for pixels in loader:
            sl=slice(start,start+len(pixels))
            full_q=q[qi[rows[sl]]].cuda()
            mirrored,regions=two_pose_regions(model.visual,pixels.cuda(non_blocking=True))
            poses=torch.stack([full_q,mirrored.float()],1)
            post=(scale*(poses@src.T)).softmax(2)
            original=graph.score(CONFIG,post[:,0]@photos.T)
            lex=graph.score(CONFIG,post.mean(1)@photos.T).clamp_min(1e-30).log()
            reader=CandidateRegions(full_q,mirrored,regions[:,:,0],regions[:,:,1],src,words,region_index,
                                    src.new_empty((0,len(src))),original)
            reader.prepare(scale)
            reg=.65*reader.first+.35*reader.read('self',1.,'gallery')
            b=len(full_q)
            x=torch.cat([poses[:,None],regions.float()],1).reshape(-1,512)
            density=[]
            for part in x.split(448):
                value=((scale*(part@src.T)).softmax(1)@banks).reshape(-1,3,count).clamp_min(1e-32)
                density.append(value/(value*prior).sum(2,keepdim=True))
            density=torch.cat(density).reshape(b,14,2,3,count)
            _,old,_=next(outputs(tree,density,reg,reg,[STRONG]))
            reg_new=reg+.65*(lex-original.clamp_min(1e-30).log())
            complete=reg_new+(old-reg)*reg_new.std(1,keepdim=True)/reg.std(1,keepdim=True)
            scores.append(complete.cpu())
            native.append(lex.cpu())
            rank,quick=raw_ranks(complete,y[rows[sl]],gl.cuda(),s)
            official=replay.chunk((rank.cpu(),y[rows[sl]]))
            assert (official-quick).abs().max()<2e-6
            details.append(official)
            start+=b
        assert start==len(rows)
        record={'score':torch.cat(scores),'native':torch.cat(native),'qi':qi[rows],'pi':pi,'kd':torch.cat(details),'rows':rows}
        save_tensor(record,path)
        print(s,'frozen strong',len(rows),(100*record['kd'].double().mean(0)).tolist(),flush=True)
        del graph,photos,model,banks,region_index,tree,record,scores,native

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['sketchy','tuberlin'],default='sketchy')
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    run(Path('/content/sbir'),args.dataset,args.full)
