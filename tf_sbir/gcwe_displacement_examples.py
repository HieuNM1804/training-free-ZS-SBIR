"""Explain a gained and lost retrieval hit with exact contributions from frozen public words."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from .geometry_word_flat import (CHECKPOINT, CONFIG, GLOBAL_WEIGHT, REGIONAL_WEIGHT, Index, atomic_json,
    audit_inputs, digest, distance_agreement, geometry_cache, index, load_frozen_clip, log_score,
    make_word_bank, rank_density, regional_readout, revision, reweight, semantic_inputs)
from .gcwe_displacement_consensus import concentration_controls,offset_geometry
from .regional_scale_position_contrast import complete_windows
from .semantic import read_words
import torch
from torch.nn import functional as F


def representative(record,excluded,split):
    native=record['rows']['original']
    new=record['rows']['same_partition']
    delta=new[:,0]-native[:,0]
    usable=torch.ones_like(delta,dtype=torch.bool) if split=='tuberlin' else ~torch.isin(record['qi'],excluded)
    selected=[]
    for mask in ((native[:,1]==0)&(new[:,1]>0)&(delta>0),(native[:,1]>0)&(new[:,1]==0)&(delta<0)):
        ids=(usable&mask).nonzero().flatten()
        order=delta[ids].abs().argsort(stable=True)
        selected.append(ids[order[len(ids)//2]])
    return torch.stack(selected)


def diffusion_rows(graph,indices):
    # Transpose the same ordered linear operations used on the score in Index.path.
    kernel=graph.kernels[0]
    e=torch.zeros((len(graph.d.p),len(indices)),device=indices.device)
    e[indices,torch.arange(len(indices),device=indices.device)]=1.
    for _ in range(CONFIG.iterations):
        e=torch.sparse.mm(kernel.reverse,e)
    e=(e.T @ kernel.status).T.contiguous()
    for _ in range(CONFIG.iterations):
        e=torch.sparse.mm(kernel.local,e)
    weights=e.T.contiguous()
    return weights/weights.sum(1,keepdim=True)


def posterior_pair(tokens,blank,patches,scale,dist,words,bank,nominees,fold,priors):
    geometry=offset_geometry(tokens.device)
    def verify(x):
        logits=scale*(x[:,None]@patches.transpose(-1,-2)).float()
        c=(logits.softmax(-1)+logits.softmax(-2))/98
        return (distance_agreement(c,dist,dist),concentration_controls(c,geometry)['same_partition'])
    real,null=verify(tokens),verify(blank)
    evidence=bank[nominees]
    updated=[[],[]]
    for t in range(2):
        eligible=fold[nominees]!=bool(t)
        den=eligible.float()[:,None].bmm(evidence).squeeze(1)+priors[1-t]
        for j in range(2):
            mass=rank_density(real[j]-null[j],eligible,8)
            num=mass[:,None].bmm(evidence).squeeze(1)+priors[1-t]
            updated[j].append(reweight(words,num/den.clamp_min(1e-30)))
    return [torch.stack(v,1) for v in updated]


def word_contributions(graph,bank,fold,revised,candidates,expected):
    influence=diffusion_rows(graph,candidates)
    # E[k,f,w] is the weighted word content of gallery fold f flowing into candidate k.
    contents=torch.stack([(influence*(fold==bool(t)).float()) @ bank for t in range(2)],1)
    pieces=[(contents*v[None]).sum(1).double() for v in revised]
    totals=torch.stack([v.sum(1) for v in pieces])
    if not torch.allclose(totals,expected.double(),rtol=2e-4,atol=1e-9):
        raise ValueError(f'Transposed diffusion failed to reconstruct the selected targets: {totals}, {expected}')
    change=totals[1]-totals[0]
    ratio=change/totals[0]
    factor=torch.where(ratio.abs()<1e-10,1./totals[0],torch.log1p(ratio)/change)
    contribution=GLOBAL_WEIGHT*(pieces[1]-pieces[0])*factor[:,None]
    return contribution[0]-contribution[1]


@torch.inference_mode()
def run(dataset):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    w=Path('/content/sbir')
    root=w/'results/gcwe-displacement-full'
    old_ids=[torch.load(w/'results'/family/(s+'.pt'),weights_only=True)['qi']
        for family in ('gcwe-displacement-consensus','gcwe-displacement-concentration') for s in ('sketchy_1','sketchy_2')]
    excluded=torch.cat(old_ids).unique()
    inputs=semantic_inputs(w,dataset)
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    vocab=read_words(cfg.vocabulary)
    views=complete_windows(w,dataset,inputs)
    _,_,ordered,_,gq,photo,flip,gsrc,gtgt,gscale=audit_inputs(w,all_sketches=True,dataset=dataset)
    assert torch.equal(ordered,torch.arange(len(qp)))
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    coord=torch.cartesian_prod(torch.arange(7),torch.arange(7)).float().cuda()
    dist=torch.cdist(coord,coord)
    output={}
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        rec=torch.load(root/(split+'.pt'),mmap=True,weights_only=True)
        assert torch.equal(qi,rec['qi']) and torch.equal(pi,rec['pi']) and torch.equal(labels,rec['labels'])
        ix=representative(rec,excluded,split)
        ids=qi[ix]
        gallery=p[pi].cuda()
        gg=F.normalize(photo[pi].cuda())
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        geo_graph=Index(SimpleNamespace(p=gg),rgb_only=True)
        stamp=digest({'gallery':[str(pp[i]) for i in pi.tolist()],'model':CHECKPOINT,'source':'regional-64/RGB16'})
        heads,compact_bank=index(w,split,gallery,tgt,scale,stamp[:16])
        bank=make_word_bank(gallery,tgt,scale,256)
        gb=make_word_bank(gg,gtgt,gscale,160)
        near,compact,gp,sp,flat=geometry_cache(w,split,gg,gq,flip,qi,pi,qp,pp,vision)
        parts=torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).cuda().tensor_split(2)
        fold=torch.zeros(len(pi),device='cuda',dtype=torch.bool)
        fold[parts[1]]=True
        priors=[gb[part].mean(0) for part in parts]
        poses=torch.stack((q[ids].cuda(),views['flipped'][ids].float().cuda()),1)
        crops=views['regions'][ids].cuda()
        post=(scale*(poses @ src.T)).softmax(2).mean(1)
        raw=post @ bank.T
        semantic=log_score(graph.score(CONFIG,raw))
        regional=regional_readout(poses,crops,src,scale,heads,compact_bank,raw)
        words=((gscale*(gq[ids].cuda()@gsrc.T)).softmax(1)+(gscale*(flip[ids].float().cuda()@gsrc.T)).softmax(1))/2
        direct=log_score(geo_graph.score(CONFIG,words @ gb.T))
        updates=posterior_pair(sp[ix].cuda(),flat[ix].cuda(),gp[compact[ix].cuda()],gscale,dist,words,gb,near[ix],fold,priors)
        diffused=[]
        for upd in updates:
            values=torch.where(fold[None],upd[:,1] @ gb.T,upd[:,0] @ gb.T)
            diffused.append(geo_graph.score(CONFIG,values))
        scores=[GLOBAL_WEIGHT*semantic+REGIONAL_WEIGHT*regional+GLOBAL_WEIGHT*(log_score(v)-direct) for v in diffused]
        order=[s.argsort(1,descending=True) for s in scores]
        records=[]
        for row,direction in enumerate(('gained','lost')):
            classes=gl.cuda()
            hit=[classes[o[row]]==labels[ix[row]].item() for o in order]
            cutoff=200 if split=='sketchy_2' else 100
            official=[rec['rows'][name][ix[row]] for name in ('original','same_partition')]
            actual=[float(v[:cutoff].float().mean()) for v in hit]
            if max(abs(x-float(obs[1])) for x,obs in zip(actual,official))>1e-6:
                raise ValueError('Selected case does not replay its official precision')
            which=1 if direction=='gained' else 0
            relevant=order[which][row,hit[which].nonzero()[0,0]]
            rival=order[1-which][row,0]
            candidates=torch.stack([relevant,rival])
            expected=torch.stack([v[row,candidates] for v in diffused])
            attribution=word_contributions(geo_graph,gb,fold,[v[row] for v in updates],candidates,expected)
            margins=[float(s[row,relevant]-s[row,rival]) for s in scores]
            residual=abs(float(attribution.sum())-(margins[1]-margins[0]))
            if residual>1e-3:
                raise ValueError('Word-level margin decomposition failed')
            def terms(largest):
                ixw=attribution.topk(5,largest=largest).indices.tolist()
                return [[vocab[i],float(attribution[i])] for i in ixw]
            photo_ids=pi[candidates.cpu()].tolist()
            records.append(dict(direction=direction,query_id=int(ids[row]),query_path=str(qp[ids[row]])[-180:],
                photo_paths=[str(pp[i])[-180:] for i in photo_ids],photo_ids=photo_ids,
                original_metrics=(100*official[0].double()).tolist(),new_metrics=(100*official[1].double()).tolist(),
                best_relevant_rank=[int(v.nonzero()[0,0])+1 for v in hit],
                selected_pair_ranks=[[(o[row]==i).nonzero().item()+1 for i in candidates] for o in order],
                relevant_minus_rival_log_scores=margins,verification_error=residual,
                positive_word_contributions=terms(True),negative_word_contributions=terms(False)))
        output[split]=records
        del graph,geo_graph,gp,bank,gb
    atomic_json(root/('word-cases-'+dataset+'.json'),dict(commit=revision(),cases=output))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',choices=['sketchy','tuberlin'])
    run(parser.parse_args().dataset)
