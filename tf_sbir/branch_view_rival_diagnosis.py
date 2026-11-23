"""Measure whether the same observed sketch contains a pose-stable view discriminating a mistaken retrieval from a relevant one."""
import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .branch_view_inputs import FAMILY,literal_word_ids
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .data import UNSEEN_CLASSES
from .gallery_panel_resume import semantic_inputs,region_views
from .retrieval_diagnosis import stratified_sample
from .semantic import read_words


def decisions(delta):
    # Views use no class information when selected: reversing the two names leaves the selection identical.
    orig,flip=delta.unbind(-1);avg=(orig+flip)*.5
    rr=torch.arange(len(orig),device=orig.device)
    both=torch.minimum(orig,flip)
    extreme=(flip[rr,orig.abs().argmax(1)]+orig[rr,flip.abs().argmax(1)])*.5
    return {"whole":orig[:,0],"whole_two_poses":avg[:,0],"average_all":avg.mean(1),"most_decisive":extreme,
            "any_view_both_poses":both.amax(1),"any_additional_both_poses":both[:,1:].amax(1),
            "small_both_poses":both[:,1:10].amax(1),"large_both_poses":both[:,10:].amax(1)}


def summarize(d,eligible,wrong):
    full=d["whole"]>0;bad=eligible&wrong;good=eligible&~wrong
    base_error=bad&~full;base_ok=good&full
    r={"all_eligible":int(eligible.sum()),"retrieval_errors":int(bad.sum()),"retrieval_correct":int(good.sum()),
       "original_pair_lost_on_errors":int(base_error.sum()),"original_pair_won_on_correct":int(base_ok.sum()),"decisions":{}}
    for k,v in d.items():
        wins=v>0
        r["decisions"][k]={"wins_on_errors":int((bad&wins).sum()),"new_positive_on_original_errors":int((base_error&wins).sum()),
                           "loss_on_originally_correct_pairs":int((base_ok&~wins).sum()),"win_rate":float(wins[eligible].float().mean())}
    return r


@torch.inference_mode()
def run(w):
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=semantic_inputs(w,"sketchy")
    words=read_words(cfg.vocabulary)
    root=w/"results"/FAMILY;root.mkdir(parents=True,exist_ok=True)
    for s,(qi,yall) in qs.items():
        pi,gl=gs[s];ix=stratified_sample(yall,36,63241);y=yall[ix]
        names=literal_word_ids(words,UNSEEN_CLASSES[s])
        ids=torch.tensor([v[0] if len(v)==1 else -1 for v in names],device=src.device)
        region,_=region_views(w,s,cfg,qp,qi,q)
        full_score=torch.load(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"),mmap=True,weights_only=True)[ix].cuda()
        gt=gl.cuda();yy=y.cuda();mask=gt[None]==yy[:,None]
        neg=full_score.masked_fill(mask,-torch.inf).argmax(1);pos=full_score.masked_fill(~mask,-torch.inf).argmax(1)
        wrong=gt[full_score.argmax(1)]!=yy
        wyes=ids[yy];wno=ids[gt[neg]];eligible=(wyes>=0)&(wno>=0)
        tree=torch.load(w/"cache"/FAMILY/s/"gallery.pt",mmap=True,weights_only=True)
        levels=[x.cuda() for x in tree["ids"]]
        a=defaultdict(list)
        for i in range(0,len(ix),96):
            sl=slice(i,i+96);j=ix[sl]
            full=torch.stack((q[qi[j]].cuda(),region["flipped"][j].float().cuda()),1)[:,None]
            rv=torch.cat((full,region["regions"][j].float().cuda()),1)
            direction=src[wyes[sl].clamp_min(0)]-src[wno[sl].clamp_min(0)]
            dd=scale*(rv*direction[:,None,None]).sum(3)
            for k,v in decisions(dd).items():a[k].append(v.cpu())
        literal=summarize({k:torch.cat(v) for k,v in a.items()},eligible.cpu(),wrong.cpu())
        g=defaultdict(lambda:defaultdict(list));allrows=[]
        for f in sorted((w/"cache"/FAMILY/s/"pilot").glob("[0-9]*.pt")):
            data=torch.load(f,mmap=True,weights_only=True);den=data["densities"][:,:,:,0].cuda();start=len(allrows)
            for k in data["ids"].tolist():allrows.append(k)
            if not torch.equal(torch.tensor(allrows),ix[:len(allrows)]):raise ValueError("Pilot indices differ")
            for j in (0,1,2):
                if j:
                    up=tree["parents"][j-1].cuda();counts=torch.bincount(levels[j-1],minlength=tree["sizes"][j-1]).float()
                    sizes=torch.bincount(levels[j],minlength=tree["sizes"][j]).float()
                    merged=den.new_zeros((*den.shape[:3],len(sizes))).index_add_(3,up,den*(counts/sizes[up]))
                    den=merged
                sl=slice(start,len(allrows));r=torch.arange(len(den),device=den.device);lr=den.clamp_min(1e-30).log().permute(0,3,1,2)
                dif=lr[r,levels[j][pos[sl]]]-lr[r,levels[j][neg[sl]]]
                for k,v in decisions(dif).items():g[j][k].append(v.cpu())
        groups={str(j+1):summarize({k:torch.cat(v) for k,v in vals.items()},(levels[j][pos]!=levels[j][neg]).cpu(),wrong.cpu()) for j,vals in g.items()}
        atomic_json(root/("rival-diagnosis-"+s+".json"),{"code":revision(),"queries":len(ix),"gallery":len(gl),"classes":len(yy.unique()),
             "public_literal_name_pair":literal,"gallery_pair":groups,"pair_selection":"highest positive and negative in existing full regional ranking; labels only reveal errors after scores",
             "oracle":"any_view_both_poses is an upper bound only; never used to assign scores"})
        print(s,literal,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(ap.parse_args().workspace)
