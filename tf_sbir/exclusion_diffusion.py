"""Restrict gallery propagation using reproducible negative concept association.

All graph construction is gallery-only. Use four disjoint reference panels and
ignore a panel for an edge whenever it contains either endpoint.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_policy import outputs
from .branch_view_study import PrefixReplay
from .gallery_concept_exclusion import event_bank
from .gallery_panel_resume import region_views, semantic_inputs
from .gallery_region_panels import CONFIG, Index
from .positive_foil import base_parts, raw_ranks, STRONG
from .retrieval_evidence import load_kd


def exclusion_mask(g, terms, vocabulary):
    n=len(g)
    device=g.device
    top=terms[:,:8]
    shares=torch.zeros(vocabulary,n,device=device,dtype=torch.bool)
    shares[top,torch.arange(n,device=device)[:,None]]=True
    panels=torch.randperm(n,generator=torch.Generator().manual_seed(7736))[:4096].cuda().tensor_split(4)
    veto=torch.ones(n,n,device=device,dtype=torch.bool)
    for panel in panels:
        local=g[panel]@g[panel].T
        local.fill_diagonal_(-torch.inf)
        near=local.topk(8).indices
        events=event_bank(top,terms[panel],vocabulary)
        neighbors=torch.cat([v[:,near].mean(2) for v in events.split(80)])
        f=events.float().sum(1)
        h=neighbors.float().sum(1)
        ef=events.float()
        inside=torch.zeros(n,device=device,dtype=torch.bool)
        inside[panel]=True
        for start in range(0,n,128):
            sl=slice(start,start+128)
            # Accumulate the exact count in fp32 to avoid classifying rounding
            # differences around zero as exclusions.
            x=events[sl].float()
            joint=torch.mm(x,ef.T)
            success=joint*len(panel)<f[sl,None]*f[None]
            joint=torch.mm(neighbors[sl].float(),ef.T)
            success &= joint*len(panel)<h[sl,None]*f[None]
            veto[sl] &= success|inside[sl,None]|inside[None]
    for start in range(0,n,128):
        sl=slice(start,start+128)
        veto[sl] &= ~shares[top[sl]].any(1)
    # The local-to-neighbor test is directional; both endpoints must agree.
    return veto & veto.T


def kernels(kernel, g, ids, mask):
    n=len(g)
    variants={k:copy.copy(kernel) for k in ('excluded','mass_control','local_only','dense_only')}
    diagnostics={}
    def attenuate(weight,cut):
        # Shared 0.5-nat bins measured from each row's maximum affinity.
        bins=((weight.amax(1,keepdim=True).clamp_min(1e-30).log()-
               weight.clamp_min(1e-30).log())*2).long().clamp(0,127)
        total=weight.new_zeros(len(weight),128).scatter_add_(1,bins,weight)
        lost=torch.zeros_like(total).scatter_add_(1,bins,weight*cut)
        control=weight*(1-(lost/total.clamp_min(1e-30)).gather(1,bins))
        kept=weight.masked_fill(cut,0)
        def normalized(v):
            mass=v.sum(1,keepdim=True)
            return torch.where(mass>1e-12,v/mass.clamp_min(1e-30),weight)
        return normalized(kept),normalized(control)
    csr=kernel.local
    rr=torch.arange(n,device=g.device).repeat_interleave(csr.crow_indices().diff().long())
    cc=csr.col_indices().long()
    assert torch.all(csr.crow_indices().diff()==ids.size(1))
    k=ids.size(1)
    cut=mask[rr,cc].reshape(n,k)
    weight=csr.values().reshape(n,k)
    diagnostics['local_lost_mass']=float((weight*cut).sum()/n)
    kept,control=attenuate(weight,cut)
    for name,v in [('excluded',kept),('local_only',kept),('mass_control',control)]:
        mat=torch.sparse_coo_tensor(torch.stack([rr,cc]),v.flatten(),(n,n)).coalesce()
        variants[name].local=mat.to_sparse_csr()
        variants[name].reverse=mat.transpose(0,1).coalesce().to_sparse_csr()
    dense=kernel.status
    blocked=torch.empty_like(dense)
    sham=torch.empty_like(dense)
    mass=dense.new_zeros(())
    for j in range(0,n,128):
        sl=slice(j,j+128);w=dense[sl];m=mask[sl]
        blocked[sl],sham[sl]=attenuate(w,m)
        mass+=(w*m).sum()
    diagnostics['dense_lost_mass']=float(mass/n)
    variants['excluded'].status=blocked
    variants['dense_only'].status=blocked
    variants['mass_control'].status=sham
    return variants,diagnostics


@torch.inference_mode()
def run(workspace,ds,official):
    cfg,paths,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(workspace,ds)
    out={}
    kd,_=load_kd(workspace)
    for s,(qi,y) in qs.items():
        pi,gl=gs[s];cl=gl.cuda();g=F.normalize(p[pi].cuda())
        readings=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
        terms=readings.topk(32).indices
        torch.set_float32_matmul_precision('highest')
        veto=exclusion_mask(g,terms,len(src))
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        graph.prepare(CONFIG)
        native=graph.kernels[0]
        changes,diagnostics=kernels(native,g,graph.neighbors[0][0][:,:CONFIG.neighbors],veto)
        wrong=veto.new_zeros((),dtype=torch.long)
        for j in range(0,len(g),128):
            wrong+=(veto[j:j+128]&(cl[j:j+128,None]!=cl)).sum()
        diagnostics['all_pairs_vetoed']=int(veto.sum())
        diagnostics['fraction_different_class_among_vetoed']=float(wrong/veto.sum().clamp_min(1))
        print(s,'graph',json.dumps(diagnostics),flush=True)
        del veto
        views,_=region_views(workspace,s,cfg,paths,qi,q)
        folder,tree,regional=base_parts(workspace,s)
        replayer=PrefixReplay(kd,gl,s)
        per_query={}
        max_error=0.
        for start in range(0,len(qi),48):
            sl=slice(start,start+48);offset=start%192
            if offset==0:pack=torch.load(folder/'full'/f'{start:06d}.pt',mmap=True,weights_only=True)
            sub=slice(offset,offset+len(y[sl]))
            assert torch.equal(pack['ids'][sub],torch.arange(start,start+len(y[sl])))
            reg=regional[sl].cuda()
            _,strong,_=next(outputs(tree,pack['densities'][sub].cuda(),reg,reg,[STRONG]))
            cell=(strong-reg)/reg.std(1,keepdim=True)
            a=(scale*(q[qi[sl]].cuda()@src.T)).softmax(1)
            b=(scale*(views['flipped'][sl].float().cuda()@src.T)).softmax(1)
            evidence=(a+b)/2@readings.T
            graph.kernels=[native]
            old=graph.score(CONFIG,a@readings.T).clamp_min(1e-30).log()
            for name,variant in {'baseline':native,**changes}.items():
                graph.kernels=[variant]
                diffused=graph.score(CONFIG,evidence).clamp_min(1e-30).log()
                region=reg+.65*(diffused-old)
                ranks,vals=raw_ranks(region+region.std(1,keepdim=True)*cell,y[sl],cl,s)
                k=len(ranks) if official else 1
                exact=replayer.chunk((ranks[:k].cpu(),y[sl][:k]))
                max_error=max(max_error,float((exact-vals[:k]).abs().max()))
                assert max_error<2e-6
                per_query.setdefault(name,[]).append(exact if official else vals)
        z={k:torch.cat(v).double() for k,v in per_query.items()}
        out[s]={'graph':diagnostics,'queries':len(y),'gallery':len(gl),'official_all':official,
                'max_checked_error':max_error,'metrics':{k:(v.mean(0)*100).tolist() for k,v in z.items()}}
        for key in ['baseline','mass_control']:
            d=(z['excluded']-z[key])*100
            dm=torch.stack([d[y==l].mean(0) for l in y.unique()])
            boot=torch.randint(len(dm),(10000,len(dm)),generator=torch.Generator().manual_seed(817))
            out[s]['excluded minus '+key]={'class_mean':dm.mean(0).tolist(),'classes_up':(dm>0).sum(0).tolist(),
                'class_ci95':dm[boot].mean(1).quantile(torch.tensor([.025,.975],dtype=dm.dtype),dim=0).tolist()}
        print(s,json.dumps(out[s]),flush=True)
        del graph,changes,native,variant,readings
    dest=workspace/'results/no-label-exclusion-diffusion';dest.mkdir(exist_ok=True,parents=True)
    (dest/(ds+('-official' if official else '')+'.json')).write_text(json.dumps(out,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['sketchy','tuberlin'],default='sketchy')
    parser.add_argument('--official',action='store_true')
    args=parser.parse_args()
    run(Path('/content/sbir'),args.dataset,args.official)

