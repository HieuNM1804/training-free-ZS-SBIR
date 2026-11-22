"""Measure whether the directions induced by one partial drawing anticipate variation among other human drawings of the same verified subject."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .drawing_evidence_subsets import FAMILY,PARTITIONS,subset_bank
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .persistent_evidence_study import lattice
from .recoverable_patch_study import locate
from .retrieval_diagnosis import stratified_sample
from .semantic_neuron_gallery import FAMILY as ORIGINAL


def walsh(device):
    m=torch.arange(64,device=device)
    bits=torch.stack([(m>>i)&1 for i in range(6)],1)
    signs=1-2*((bits@bits.T)%2)
    h=signs.float()/8.
    if (h@h.T-torch.eye(64,device=device)).abs().amax()>1e-7:raise ValueError("Invalid orthogonal Boolean decomposition")
    return h,bits.sum(1)


def subspaces(values,h,order,sets,perms):
    z=values.float()
    for k,idx in sets.items():
        # Remove the mean response to retaining k groups, identically for correct and permuted membership.
        z[:,idx]=z[:,idx]-z[:,idx].mean(1,keepdim=True)
    out={}
    for control,perm in (("actual",None),("shuffled_A",perms[0]),("shuffled_B",perms[1])):
        coef=torch.einsum("ts,bsd->btd",h,z if perm is None else z[:,perm])
        for k in (1,2,3):
            features=coef[:,order==k]
            gram=features@features.transpose(1,2)
            val,vec=torch.linalg.eigh(gram)
            for rank in (2,4):
                vals=val[:,-rank:]
                q=vec[:,:,-rank:].transpose(1,2)@features
                # Unrealized directions do not receive credit for explaining drawing variation.
                vecnorm=vals.clamp_min(1e-14).rsqrt()*((vals>val[:,-1:]*1e-7)&(vals>1e-10))
                out[f"{control}/{k}/{rank}"]=q*vecnorm[:,:,None]
    return out


def fixed_targets(raw,full_q,panels,qids):
    # Summed scatter over every pair of other drawings equals the centered sum below.
    # Pad each individual centered panel with zero; vary neither its trace nor the weight of its image.
    same,other,n=[],[],[]
    for i in qids:
        a,b=panels[i]
        if len(a)<2:raise ValueError("A single drawing has no independent within-subject scatter")
        if len(a)!=len(b):raise ValueError("Size-matched alternate object is missing")
        n.append(len(a))
        for ids,out in ((a,same),(b,other)):
            v=raw[full_q[ids]].float()
            v=v-v.mean(0)
            out.append(v/v.norm().clamp_min(1e-12))
    limit=max(n)
    def stack(lst):
        t=torch.zeros(len(lst),limit,512)
        for j,z in enumerate(lst):t[j,:len(z)]=z
        return t
    return stack(same),stack(other),torch.tensor(n)


def estimate(q,v):
    return (q@v.transpose(1,2)).square().sum((1,2))


def ci(v,labels):
    lab=labels.unique(return_inverse=True)[1]
    count=lab.bincount().double()
    sums=torch.zeros_like(count).index_add_(0,lab,v.double())
    ix=torch.randint(len(count),(3000,len(count)),generator=torch.Generator().manual_seed(27449))
    boot=sums[ix].sum(1)/count[ix].sum(1)
    return [float(z) for z in boot.quantile(torch.tensor([.025,.975],dtype=torch.double))]


@torch.inference_mode()
def run(w):
    inp=semantic_inputs(w,"sketchy")
    _,_,qs,_,_,raw,*_=inp
    torch.backends.cuda.matmul.allow_tf32=False
    data=subset_bank(w,"sketchy","pilot-query")
    sets,perms=lattice("cuda")
    h,order=walsh("cuda")
    references=torch.load(w/"results"/ORIGINAL/"choice-repeatability-full.pt",mmap=True,weights_only=True)
    report,stored={},{}
    for s,(qi,y) in qs.items():
        pilot=stratified_sample(y,36,63241)
        related={i:(a,b) for i,a,b,_ in references[s]["panels"] if len(a)>=2}
        ids=torch.tensor([i for i in pilot.tolist() if i in related])
        loc=locate(data["ids"],qi[ids])
        same,different,count=fixed_targets(raw,qi,related,ids.tolist())
        rank_items=torch.load(w/"results"/ORIGINAL/(s+".pt"),mmap=True,weights_only=True)["metrics"]["regional"][:,0]
        # Compare the very same target differences to a different, equally difficult probe of the same class.
        alt=[]
        for q in ids.tolist():
            siblings=related[q][0]
            blocked=torch.isin(ids,torch.tensor([q,*siblings]))
            candidates=torch.where((y[ids]==y[q])&~blocked)[0]
            k=candidates[(rank_items[ids[candidates]]-rank_items[q]).abs().argmin()]
            alt.append(int(k))
        alt=torch.tensor(alt)
        values={}
        for gp,part in enumerate(PARTITIONS):
            assembled={}
            for start in range(0,len(ids),64):
                ix=loc[start:start+64]
                basis=subspaces(data["subsets"][ix,gp].cuda(),h,order,sets,perms)
                for key,bd in basis.items():assembled.setdefault(key,[]).append(bd.cpu())
            for name,boxes in assembled.items():
                q=torch.cat(boxes)
                columns=[]
                for start in range(0,len(ids),64):
                    sl=slice(start,start+64)
                    own=q[sl].cuda()
                    different_probe=q[alt[sl]].cuda()
                    a,b=same[sl].cuda(),different[sl].cuda()
                    columns.append(torch.stack((estimate(own,a),estimate(own,b),estimate(different_probe,a),
                                                estimate(different_probe,b)),1).cpu())
                values[part+"/"+name]=torch.cat(columns)
        summary={}
        for k,x in values.items():
            own=x[:,0]-x[:,1]
            ref=x[:,0]-x[:,2]
            actual_diff=own-(x[:,2]-x[:,3])
            gp,mode,deg,rank=k.split("/")
            perm=.5*(values[f"{gp}/shuffled_A/{deg}/{rank}"]+values[f"{gp}/shuffled_B/{deg}/{rank}"])
            crossed_null=(x[:,0]-x[:,1])-(perm[:,0]-perm[:,1])
            summary[k]={"mean_trace_captured":x.mean(0).tolist(),"correct_object_advantage":float(own.mean()),
                        "correct_probe_advantage":float(ref.mean()),"crossed_probe_target_advantage":float(actual_diff.mean()),
                        "actual_minus_shuffle_object_specificity":float(crossed_null.mean()),"actual_minus_shuffle_CI95":ci(crossed_null,y[ids])}
        report[s]={"queries":len(ids),"source_images":len({tuple(sorted([int(i),*related[int(i)][0]])) for i in ids}),
                   "mean_additional_drawings":float(count.float().mean()),"subspaces":summary}
        stored[s]={"query_ids":ids,"labels":y[ids],"number_other_drawings":count,"alternate_query":alt,"trace":values}
        print(s,len(ids),[(k,round(v["correct_probe_advantage"],4),round(v["actual_minus_shuffle_object_specificity"],4))
                          for k,v in summary.items() if k.startswith("grid/actual") and k.endswith("/4")],flush=True)
    p=w/"results"/FAMILY/"human-drawing-variance.pt"
    save_tensor(stored,p)
    atomic_json(p.with_suffix(".json"),{"revision":revision(),"splits":report,"per_query_sha256":checksum(p),
                "columns":["own_probe_same_object","own_probe_other_object","alternate_probe_same_object","alternate_probe_other_object"],
                "design":"Every other verified drawing of the same photographed object provides only retrospective variation. Compare an alternative subject and probe of matched class, original difficulty, and number of drawings. Fit no model, include neither probe sketch in the evaluated same-subject scatter."})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(cli.parse_args().workspace)
