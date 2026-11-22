"""Check what drives cross-domain contextual agreement on exact source photos, equally ranked same-class alternatives and real retrieval errors."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import revision
from .drawing_evidence_subsets import PARTITIONS,subset_bank
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .paired_fragment_area_control import read_area
from .paired_fragment_inputs import FAMILY
from .recoverable_patch_study import prepare
from .response_geometry_instance_audit import examples
from .semantic import read_words
from .whole_fragment_likelihood import catalog,prob,target_tables


def mechanism(q1,q0,p1,p0):
    # In this small matched panel compute the full word-level covariance decomposition in double precision.
    q1,q0=q1[:,None].double(),q0[:,None].double()
    p1,p0=p1.double(),p0.double()
    def dot(a,b):return (a*b).sum(-1).clamp_min(1e-100)
    a,b,c,d=dot(q1,p1),dot(q1,p0),dot(q0,p1),dot(q0,p0)
    interaction=(a.log()-c.log())-(b.log()-d.log())
    pi=q0*p0/d[:,:,None]
    uq=(q1/q0.clamp_min(1e-100))*(d/b)[:,:,None]-1
    up=(p1/p0.clamp_min(1e-100))*(d/c)[:,:,None]-1
    term=pi*uq*up
    pos=(uq>=0)&(up>=0)
    neg=(uq<0)&(up<0)
    co=(term*pos).sum(-1)
    down=(term*neg).sum(-1)
    mixed=(term*(~(pos|neg))).sum(-1)
    err=float(((1+co+down+mixed)-interaction.exp()).abs().div(interaction.exp().clamp_min(1e-15)).max())
    if err>2e-6:raise ValueError("Word-level components do not conserve the four-state likelihood ratio")
    return {"four_way":interaction,"whole":a.log(),"fragments":d.log(),"source":a.log()-c.log(),
            "target":a.log()-b.log(),"coemergence":co,"joint_suppression":down,"opposite_signs":mixed,
            "share_positive_terms_from_coemergence":co/(co+down).clamp_min(1e-15)},term,err


def wins(x):
    return ((x>0).double()+.5*(x==0)).mean().item()


def row(v,y):
    cls=y.unique(return_inverse=True)[1];n=cls.bincount().double()
    win=(v>0).double()+.5*(v==0)
    sums=torch.zeros_like(n).index_add_(0,cls,win)
    ix=torch.randint(len(n),(2500,len(n)),generator=torch.Generator().manual_seed(9147))
    ci=(sums[ix].sum(1)/n[ix].sum(1)).quantile(torch.tensor([.025,.975],dtype=torch.float64))
    return {"n":len(v),"fraction_prefers_relevant":wins(v),"CI95_class":ci.tolist(),"median_margin":float(v.median())}


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    cfg,qp,qs,pp,gs,q,*_=inputs
    src,tgt,scale=inputs[-3:]
    torch.backends.cuda.matmul.allow_tf32=False
    d=subset_bank(w,"sketchy","pilot-query")
    root=w/"results"/FAMILY
    options=read(root/"pilot.json")
    primary=options["main"].split("/")
    n=int(primary[0]);left,right,graph=primary[1:4]
    words=read_words(cfg.vocabulary)
    report,captured={},{}
    for s in qs:
        p=prepare(w,s,inputs,d)
        qi,y=qs[s];pi,g=gs[s]
        cat=catalog(w,s,inputs)
        axes=src[cat["order"][:n]]
        bank=target_tables(w,s,"sketchy",inputs,cat,n)
        qid=qi[p["ids"]]
        q1=prob(q[qid].cuda(),axes,scale)
        def partial(query_part):
            if query_part=="area_shuffle":v=read_area(w,"sketchy","pilot-query",qid).cuda()
            else:v=d["subsets"][p["positions"]][:,PARTITIONS.index(query_part),2**torch.arange(6)].cuda()
            return prob(v,axes,scale).mean(1)
        q0=partial(left)
        subj,triple,rank=examples(qp,pp,qi,pi,y,g,p["ids"],p["base"])
        inverse=torch.full((int(pi.max())+1,),-1,dtype=torch.int64)
        inverse[pi]=torch.arange(len(pi))
        triple=inverse[triple]
        base=p["base"];yl=p["y"]
        relevant=(g[None]==yl[:,None])
        best_pos=base.masked_fill(~relevant,-torch.inf).argmax(1)
        best_neg=base.masked_fill(relevant,-torch.inf).argmax(1)
        errors=torch.where(g[base.argmax(1)]!=yl)[0]
        tests={"matched_source":(subj,triple,((0,1,"source_vs_same_class"),(0,2,"source_vs_wrong"),(1,2,"other_relevant_vs_wrong"))),
               "top1_errors":(errors,torch.stack((best_pos[errors],best_neg[errors]),1),((0,1,"best_relevant_vs_top_wrong"),))}
        summaries,raw={},{}
        example_pairs=[]
        for kind,(sel,idx,comparisons) in tests.items():
            desc=defaultdict(list);maxerr=0.
            for off in range(0,len(sel),32):
                sl=slice(off,off+32)
                use=sel[sl];pids=idx[sl].cuda()
                b1=bank["whole",graph][pids];b0=bank[right,graph][pids]
                metrics,contribution,err=mechanism(q1[use],q0[use],b1,b0)
                other,_,e2=mechanism(q1[use],q0[use],b1,bank[right,graph][cat["permutation"].cuda()[pids]])
                maxerr=max(maxerr,err,e2)
                metrics["shuffled_within_gallery_cluster"]=other["four_way"]
                metrics["true_minus_shuffled"]=metrics["four_way"]-other["four_way"]
                for j,k,name in comparisons:
                    for tag,value in metrics.items():
                        desc[name+"/"+tag].append((value[:,j]-value[:,k]).float().cpu())
                if kind=="top1_errors" and off==0:
                    for j in range(min(3,len(use))):
                        ids=contribution[j,0].abs().topk(5).indices.cpu()
                        example_pairs.append({"query":str(qp[int(qid[use[j]])]).split("/sketch/")[-1],"positive_photo":str(pp[int(pi[pids[j,0]])]).split("/photo/")[-1],
                                         "negative_photo":str(pp[int(pi[pids[j,1]])]).split("/photo/")[-1],
                                         "strongest_signed_terms_on_positive":[[str(words[int(cat["order"][k])]),float(contribution[j,0,k])] for k in ids]})
            tables={k:torch.cat(v) for k,v in desc.items()}
            summaries[kind]={"queries":len(sel),"comparisons":{k:row(a,yl[sel]) for k,a in tables.items()},
                             "relative_covariance_replay_error":maxerr}
            raw[kind]={"sketch_ids":qid[sel],"gallery_ids":pi[idx],"labels":yl[sel],"margins":tables}
        report[s]={"primary":options["main"],"groups":summaries,"examples":example_pairs,"median_rank_distance_to_same_class":float((rank[:,0]-rank[:,1]).abs().float().median())}
        captured[s]=raw
        print(s,"matched photo and category margin",len(subj),len(errors),flush=True)
        del bank,p
    path=root/"matched-objects-and-words.pt"
    save_tensor(captured,path)
    atomic_json(path.with_suffix(".json"),{"revision":revision(),"splits":report,"per_query_sha256":checksum(path),
                "scope":"The exact source photo and benchmark categories are used only retrospectively to label pairs. Co-emergence and co-suppression are relative to the fragment-conditioned measure of the particular candidate."})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
