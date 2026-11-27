"""Test native write profiles on real pairs matched in endpoint cosine and two text responses."""

import argparse
import json

import open_clip
import torch
from torch.nn.functional import normalize

from .concept_benchmark import checksum,atomic_json
from .description_block_choices import NativeReads,pair_axes,order_key
from .description_block_results import Population
from .description_choice_blocks import inputs,PILOT,OUT
from .encoding import load_frozen_clip,save_tensor
from .native_block_probe import SEED,deterministic
from .wordnet_renaming_pilot import SPLITS
from .wordnet_renaming_text import Sense


def ranks(s):
    values,order=s.sort(dim=1,stable=True)
    pos=torch.arange(s.shape[1],device=s.device).expand_as(order)
    change=values[:,1:]!=values[:,:-1]
    edge=torch.ones(len(s),1,dtype=torch.bool,device=s.device)
    lo=pos.masked_fill(~torch.cat((edge,change),1),0).cummax(1).values
    hi=pos.masked_fill(~torch.cat((change,edge),1),s.shape[1]-1).flip((1,)).cummin(1).values.flip((1,))
    return torch.empty_like(order).scatter_(1,order,lo+hi)


def matched(covariates,positive,eligible):
    # Each rank is integer-valued. Order candidates lexicographically by maximum and total error.
    dis=(covariates-covariates.gather(1,positive[:,None,None].expand(-1,1,covariates.shape[2]))).abs()
    n=covariates.shape[1]
    score=dis.amax(2)*(6*n+1)+dis.sum(2)
    score.masked_fill_(~eligible,torch.iinfo(score.dtype).max)
    if not eligible.any(1).all():raise ValueError("Empty control bank")
    negative=score.argmin(1)
    discrepancy=dis[torch.arange(len(score),device=score.device),negative].amax(1).double()/(2*(n-1))
    return negative,discrepancy


def centered_writes(prefix,direction,mean_write):
    values=torch.einsum("bnd,bd->bn",prefix[:,1:]-prefix[:,:-1]-mean_write,direction)
    return normalize(values-values.mean(1,keepdim=True),dim=1)


def outcome(delta):
    return (delta>0).double()+.5*(delta==0).double()


def comparison(q,p,n,p_perm,n_perm):
    return {"aligned":(q*(p-n)).sum(1),"sorted":(q.sort(1).values*(p.sort(1).values-n.sort(1).values)).sum(1),
            "independent_permutation":(q*(p.gather(1,p_perm)-n.gather(1,n_perm))).sum(1)}


def summarize_bracket(rows, labels, source_ids):
    sides = ("cosine_below", "cosine_above")
    low, high = (rows[k] for k in sides)
    keep = low["available"] & high["available"]
    if not keep.any() or not torch.equal(low["pos"], high["pos"]):
        raise ValueError("Invalid two-sided panel")
    if not ((low["native_cosine_margin"][keep] > 0) & (high["native_cosine_margin"][keep] < 0)).all():
        raise ValueError("Cosine does not bracket every retained positive")
    if not (low["neg"][keep] != high["neg"][keep]).all():
        raise ValueError("A negative occurs on both sides")
    pop = Population(labels[keep], source_ids[keep])
    keys = [k for k in low if "/" in k or k == "native_cosine_margin"]
    side_scores = {side: {k: outcome(rows[side][k][keep]) for k in keys} for side in sides}
    means = {k: (side_scores[sides[0]][k] + side_scores[sides[1]][k]) / 2 for k in keys}
    if not (means["native_cosine_margin"] == 0.5).all():
        raise ValueError("Native accuracy must equal half for each retained sketch")
    text = means["sketch/text/aligned"]
    differences = {k: pop.effect(text - v) for k, v in means.items() if k != "sketch/text/aligned"}
    quantiles = torch.tensor([.5, .9, .99, 1.], dtype=torch.float64)
    side_report = {}
    for side, scores in side_scores.items():
        t = rows[side]
        side_report[side] = {
            "accuracy": {k: pop.effect(v) for k, v in scores.items()},
            "abs_cosine_gap_median": t["native_cosine_margin"][keep].abs().median().item(),
            "rank_discrepancy_percentiles": (100 * torch.quantile(t["rank_mismatch"][keep], quantiles)).tolist(),
        }
    return {
        "eligible": int(keep.sum()), "total": len(keep), "sources": len(pop.counts), "classes": len(pop.classes),
        "missing_below": int((~low["available"]).sum()), "missing_above": int((~high["available"]).sum()),
        "class_counts": {str(c.item()): [int((keep & (labels == c)).sum()), int((labels == c).sum())]
                         for c in labels.unique()},
        "sides": side_report, "paired_accuracy": {k: pop.effect(v) for k, v in means.items()},
        "aligned_text_minus_comparator": differences,
    }


@torch.inference_mode()
def run(name,revision,fixed_revision,native_revision,cosine_bracket=False):
    meta,index,source_all=inputs()
    fixed_path=OUT/f"{name}-fixed.pt"
    old=json.loads(fixed_path.with_suffix(".json").read_bytes())
    if old["revision"]!=fixed_revision or checksum(fixed_path)!=old["sha256"]:raise ValueError("Unverified fixed panels")
    f=torch.load(fixed_path,mmap=True,weights_only=True)
    qi,labels=index["sketch"]["views"][name]
    subjects,own=source_all[qi].unique(return_inverse=True)
    classes=torch.empty(len(subjects),dtype=labels.dtype)
    classes[own]=labels
    if not torch.equal(f["query_ids"],qi) or not torch.equal(f["source_ids"],subjects[own]) or not torch.equal(classes[own],labels):
        raise ValueError("Population mismatch")
    model=load_frozen_clip(meta["checkpoint"],torch.device("cuda"))
    senses=[Sense(**s) for s in json.loads((PILOT/"inventory.json").read_bytes())["senses"]]
    texts=pair_axes(model,open_clip.get_tokenizer("ViT-B-32"),senses,f["pair"])[f["peer"]]
    text_norm=texts.norm(dim=1)
    valid=text_norm>1e-9
    # A genuinely identical pair of captions contributes ties. Preserve its query and measure its frequency.
    texts=normalize(texts,dim=1)
    del model
    random=normalize(torch.randn(len(qi),512,generator=torch.Generator().manual_seed(SEED)),dim=1).cuda()
    depths=torch.arange(13,device="cuda")
    sk=NativeReads("sketch",native_revision)
    ph=NativeReads("photo",native_revision)
    refs,_,endpoints=ph.read(subjects,depths)
    mean_write=refs.mean(0).diff(dim=0)
    c=classes.cuda()
    perm=torch.stack([torch.randperm(12,generator=torch.Generator().manual_seed(int.from_bytes(
        order_key("profile-permutation",name,i)[:8],"little")%(2**63))) for i in subjects.tolist()]).cuda()
    panels=(("cosine_below","cosine_above") if cosine_bracket
            else ("different_source_right_class","exact_source_within_class"))
    rows={k:{} for k in panels}
    for start in range(0,len(qi),128):
        sl=slice(start,start+128)
        qpath,_,qend=sk.read(qi[sl],depths)
        cos=qend@endpoints.T
        d,r=texts[sl],random[sl]
        axes={"text":d,"isotropic":r}
        score1=d@endpoints.T
        score2=r@endpoints.T
        cov=torch.stack((ranks(cos),ranks(score1),ranks(score2)),2)
        truth=labels[sl].cuda()
        source=own[sl].cuda()
        pos_other=own[f["peer"][sl]].cuda()
        for kind in panels:
            if kind!="exact_source_within_class":
                pos=pos_other
                eligible=c[None,:]!=truth[:,None]
            else:
                pos=source
                eligible=(c[None,:]==truth[:,None]) & (torch.arange(len(c),device=c.device)[None,:]!=pos[:,None])
            if cosine_bracket:
                positive_cosine=cos.gather(1,pos[:,None])
                side=(cos<positive_cosine if kind=="cosine_below" else cos>positive_cosine)
                possible=eligible & side
                available=possible.any(1)
                # Keep IDs for every original query. The missing side's placeholder is never scored.
                eligible=torch.where(available[:,None],possible,eligible)
            neg,dist=matched(cov,pos,eligible)
            ri=torch.arange(len(pos),device=c.device)
            batch={"pos":subjects[pos.cpu()],"neg":subjects[neg.cpu()],"rank_mismatch":dist,
                   "native_cosine_margin":cos[ri,pos]-cos[ri,neg],
                   "text_endpoint_difference":score1[ri,pos]-score1[ri,neg],
                   "random_endpoint_difference":score2[ri,pos]-score2[ri,neg]}
            if cosine_bracket:
                batch["available"]=available
            for a,axis in axes.items():
                positive=centered_writes(refs[pos],axis,mean_write)
                negative=centered_writes(refs[neg],axis,mean_write)
                for role,path in (("sketch",qpath),("source",refs[source])):
                    if kind=="exact_source_within_class" and role=="source":continue
                    q=centered_writes(path,axis,mean_write)
                    if not torch.isfinite(q).all():raise ValueError("Nonfinite profile")
                    found=comparison(q,positive,negative,perm[pos],perm[neg])
                    for k,v in found.items():batch[f"{role}/{a}/{k}"]=v
            for k,v in batch.items():rows[kind].setdefault(k,[]).append(v.cpu())
    rows={panel:{k:torch.cat(parts) for k,parts in t.items()} for panel,t in rows.items()}
    pop=Population(labels,subjects[own])
    reports={}
    if cosine_bracket:
        reports["cosine_bracket"]=summarize_bracket(rows,labels,subjects[own])
    for panel,t in (() if cosine_bracket else rows.items()):
        acc={k:outcome(v) for k,v in t.items() if "/" in k or k=="native_cosine_margin"}
        base=acc["sketch/text/aligned"]
        effects={}
        for k,v in acc.items():
            effects[k]=pop.effect(v)
            if k!="sketch/text/aligned":effects["aligned_text_minus_"+k]=pop.effect(base-v)
        effects["aligned_text_minus_chance"]=pop.effect(base-.5)
        reports[panel]={"accuracy_and_differences":effects,
                "rank_discrepancy_percentiles":(100*torch.quantile(t["rank_mismatch"],torch.tensor([.5,.9,.99,1.],dtype=torch.float64))).tolist(),
                "median_abs_matched_cosine_difference":t["native_cosine_margin"].abs().median().item(),
                "median_abs_matched_text_difference":t["text_endpoint_difference"].abs().median().item(),
                "median_abs_matched_isotropic_difference":t["random_endpoint_difference"].abs().median().item()}
    prefix="bracket" if cosine_bracket else "profile"
    dest=OUT/f"{prefix}-{name}.pt"
    save_tensor({"rows":rows,"query_ids":qi,"source_ids":subjects[own],"classes":labels,"text_norm":text_norm.cpu()},dest)
    atomic_json(dest.with_suffix(".json"),{"revision":revision,"fixed_revision":fixed_revision,"native_revision":native_revision,
                "split":name,"queries":len(qi),"sources":len(subjects),"seed":SEED,"identical_definitions":int((~valid).sum()),
                "reports":reports,"raw_sha256":checksum(dest),"device":torch.cuda.get_device_name()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--fixed-revision",required=True)
    p.add_argument("--native-revision",required=True)
    p.add_argument("--cosine-bracket",action="store_true",help="Balance native cosine exactly for each eligible sketch.")
    a=p.parse_args()
    deterministic()
    run(a.split,a.revision,a.fixed_revision,a.native_revision,a.cosine_bracket)
