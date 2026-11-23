"""Post-ranking diagnosis of name availability and the covariance that changes a real correct/incorrect photo margin."""
import argparse
from collections import Counter
from pathlib import Path
import re

import torch

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import revision
from .data import UNSEEN_CLASSES
from .gallery_panel_resume import semantic_inputs
from .query_partition_study import locked
from .retrieval_diagnosis import stratified_sample
from .semantic import read_words
from .shared_private_names import FAMILY,path
from .shared_private_score import Row


def spelling(s):return re.sub("[^a-z0-9]","",s.casefold())


def term_indices(words,classes):
    d={spelling(n):i for i,n in enumerate(words)}
    return torch.tensor([d.get(spelling(n),-1) for n in classes],device="cuda")


def namerank(scores,true,eligible,index):
    # Oracle diagnostic: restrict a previously formed full-public-word reading to the eligible class spellings.
    safe=true.clamp_min(0);direct=scores.gather(1,safe[:,None])
    whole=(1+(scores>direct).sum(1));forced=1+(scores[:,index[index>=0]]>direct).sum(1)
    return {"eligible":int(eligible.sum()),"top1_among_listed_names":int(((forced==1)&eligible).sum()),
            "top32_in_public_dictionary":int(((whole<=32)&eligible).sum()),"median_rank_in_public_dictionary":float(whole[eligible].float().median())}


def abbreviate(s):return "/".join(Path(s).parts[-3:])


@torch.inference_mode()
def run(w,ds,phase):
    if ds=="tuberlin" and not (w/"results"/FAMILY/"freeze.json").is_file():raise ValueError("No TU diagnosis before a locked retrieval method")
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=semantic_inputs(w,ds)
    root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    words=read_words(cfg.vocabulary)
    torch.set_float32_matmul_precision("highest")
    for s,(qi,labels) in qs.items():
        cache=path(w,s);geo=torch.load(cache/"geometry.pt",mmap=True,weights_only=True)
        obs=torch.load(cache/"photo-words.pt",mmap=True,weights_only=True)
        cat=term_indices(words,UNSEEN_CLASSES[s])
        pi,gl=gs[s]
        ix=stratified_sample(labels,36,63241) if phase=="pilot" else torch.arange(len(qi))
        y=labels[ix];true=cat[y.cuda()];eligible=true>=0
        x=q[qi[ix]].cuda()
        raw=scale*(x@src.T);lk=raw.softmax(1)
        prior=obs.mean(0).cuda().clamp_min(1e-32)
        root_logits=raw+prior.log()[None]
        detected={"whole":namerank(raw,true,eligible,cat),"gallery_root":namerank(root_logits,true,eligible,cat)}
        base=torch.load(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"),mmap=True,weights_only=True)
        best=[];wrong=[]
        for off in range(0,len(ix),96):
            scores=base[ix[off:off+96]].cuda()
            positive=y[off:off+96,None].cuda()==gl.cuda()[None]
            best.append(scores.masked_fill(~positive,-torch.inf).argmax(1).cpu())
            wrong.append(scores.argmax(1).cpu())
        pos=torch.cat(best);top=torch.cat(wrong)
        bad=gl[top]!=y;otherword=cat[gl[top].cuda()]
        check=bad.cuda()&eligible&(otherword>=0)
        rr=torch.arange(len(ix),device=x.device);actual=raw[rr,true.clamp_min(0)];rival=raw[rr,otherword.clamp_min(0)]
        part={"raw_name_prefers_true_to_wrong":int((check&(actual>rival)).sum()),"eligible_original_errors":int(check.sum()),
              "regional_original_top1_errors":int(bad.sum()),
              "weighted_name_prefers_true_to_wrong":int((check&(root_logits[rr,true.clamp_min(0)]>root_logits[rr,otherword.clamp_min(0)])).sum())}
        # Score only after pair IDs are recorded, using the locked or already prespecified pilot reading.
        if phase=="full":
            selected=Row.parse(locked(root/"freeze.json")["config"]["primary"])
        elif (root/"shortlist.json").is_file():
            selected=Row.parse(locked(root/"shortlist.json")["config"]["picked"]["tree/joint"])
        else:
            selected=Row("tree-r75-p25",2,1.,.6,"joint",1.)
        pri=[z.cuda().double() for z in geo["priors"]]
        peers=[torch.load(cache/f"words-{h}-{selected.model}-{selected.depth}.pt",weights_only=True,mmap=True).cuda().double() for h in (0,1)]
        grp=geo["groups"][selected.depth]
        fold=torch.empty(len(pi),dtype=torch.int64)
        for h,ids in enumerate(geo["folds"]):fold[ids]=h
        # For queries whose first result was correct, the best negative is still useful for diagnostic margins.
        badid=top.clone()
        for off in range(0,len(ix),96):
            sl=slice(off,off+96);b=base[ix[sl]].cuda()
            bneg=b.masked_fill(gl.cuda()[None]==y[sl,None].cuda(),-torch.inf).argmax(1).cpu()
            badid[sl]=bneg
        wh=[];edges=[];names=[];maxerr=0.;factors=[]
        for off in range(0,len(ix),48):
            sl=slice(off,off+48);a=lk[sl].double()
            both=[]
            for ids in (pos[sl],badid[sl]):
                z=obs[ids].cuda().double()
                g=grp[ids].cuda();h=fold[ids];rs=[torch.where((h.cuda()==0)[:,None],peers[0][g],peers[1][g]),
                       torch.where((h.cuda()==0)[:,None],pri[0][None],pri[1][None])]
                ppred,baseline=rs
                ratio=(z/baseline.clamp_min(1e-32)).clamp_min(1e-30).pow(selected.power)
                norm=(ppred*ratio).sum(1)
                prob=selected.alpha*norm/(1-selected.alpha+selected.alpha*norm)
                aphoto=(a*z).sum(1);amean=(a*ppred).sum(1)
                ajoint=(a*ppred*ratio).sum(1)/norm
                scalar=aphoto+prob*(amean-aphoto);specific=aphoto+prob*(ajoint-aphoto)
                term=prob[:,None]*ppred*(a-amean[:,None])*(ratio/norm[:,None]-1.)
                maxerr=max(maxerr,float((term.sum(1)-(specific-scalar)).abs().amax()))
                gain=specific.log()-scalar.log()
                both.append(torch.stack((gain,scalar.log(),specific.log(),prob,norm),1).cpu())
                if len(both)==1:
                    topterms=term.argmax(1).tolist()
                    names.extend([words[k] for k in topterms])
            wh.append(both[0]);edges.append(both[1])
        left=torch.cat(wh);right=torch.cat(edges);effect=left[:,0]-right[:,0]
        base_gap=left[:,1]-right[:,1];after_gap=left[:,2]-right[:,2]
        if maxerr>1e-8:raise ValueError("Name covariance does not explain score difference")
        examples=[]
        for label,mask in (("rescued",bad&(base_gap<0)&(after_gap>0)),("lost",bad&(base_gap>0)&(after_gap<0))):
            ids=torch.where(mask)[0].tolist()
            if ids:
                i=sorted(ids,key=lambda k:(float(effect[k]),k))[len(ids)//2]
                examples.append({"outcome":label,"query":abbreviate(qp[int(qi[ix[i]])]),"true_photo":abbreviate(pp[int(pi[pos[i]])]),
                    "wrong_photo":abbreviate(pp[int(pi[badid[i]])]),"supporting_public_term_for_true_photo":names[i],
                    "log_gap_scalar":float(base_gap[i]),"log_gap_verified":float(after_gap[i])})
        part.update({"verified_margin_increases_on_regional_errors":int((bad&(effect>0)).sum()),
                     "scalar_has_positive_above_negative_on_errors":int((bad&(base_gap>0)).sum()),
                     "verified_has_positive_above_negative_on_errors":int((bad&(after_gap>0)).sum()),
                     "true_candidate_mean_borrowing":float(left[bad,3].mean()),"wrong_candidate_mean_borrowing":float(right[bad,3].mean())})
        atomic_json(root/(phase+"-mechanism-"+s+".json"),{"code":revision(),"setting":selected.label(),"queries":len(ix),"eligible_class_spellings":int((cat>=0).sum()),
             "oracle_name_diagnosis":detected,"actual_first_error":part,"covariance_identity_error":maxerr,"examples":examples,
             "preselected_before_inspecting_labels":True,"usage":"class labels and spellings are consulted only in this post-ranking diagnostic"})
        print(s,detected,part,maxerr,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--phase",choices=("pilot","full"),default="pilot")
    a=p.parse_args();run(a.workspace,a.dataset,a.phase)
