"""Cache rival-group scores and measure same-domain calibration on held-out photographs."""
import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .query_fragment_reference_inputs import dataset
from .retrieval_diagnosis import stratified_sample
from .window_extreme_inputs import FAMILY,check_scope
from .window_extreme_calibration import Calibration,assemble_level,SPECS


@torch.inference_mode()
def run(w,ds,stage):
    check_scope(w,ds)
    if ds=="tuberlin" and stage=="pilot":raise ValueError("No TU development")
    _,_,qs,_,gs,_,_=dataset(w,ds)
    get=lambda f:torch.load(f,mmap=True,weights_only=True)
    for s,(qi,y) in qs.items():
        root=w/"cache"/FAMILY/s;read(root/"ready.json");geo=get(root/"structure.pt")
        folder=root/("groups-"+stage);folder.mkdir(exist_ok=True)
        if (folder/"done.json").exists():continue
        ix=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(y))
        queries=get(root/"sketches.pt")[ix].cuda()
        gl=gs[s][1].cuda();gold=y[ix].cuda()
        regional=get(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"))
        pos=[];neg=[];errors=[]
        for first in range(0,len(ix),192):
            ii=ix[first:first+192];yy=gold[first:first+len(ii)]
            raw=regional[ii].cuda();ok=yy[:,None]==gl[None]
            pos.append(raw.masked_fill(~ok,-torch.inf).argmax(1));neg.append(raw.masked_fill(ok,-torch.inf).argmax(1))
            errors.append(gl[raw.argmax(1)]!=yy)
        pos=torch.cat(pos);neg=torch.cat(neg);miss=torch.cat(errors)
        g=geo["groups"].cuda();up=geo["upper"].cuda();folds=[p.cuda() for p in geo["parts"]]
        for level in (2,3):
            qv,label=assemble_level(queries,g,up,level);k=qv.size(-1)
            prior=torch.bincount(label,minlength=k).float();prior/=prior.sum()
            for domain in ("photo","source"):
                file=folder/f"{domain}-{level}.pt"
                if file.exists():continue
                configs={}
                controls=[];corner=defaultdict(lambda:[0,0])
                observed={k:torch.zeros(len(ix),2) for k in SPECS}
                raw=get(root/("photos-"+domain+".pt")).cuda()
                pv,_=assemble_level(raw,g,up,level);del raw
                for hold in range(3):
                    for donor in (i for i in range(3) if i!=hold):
                        bank=3-hold-donor
                        model=Calibration(pv[:,:,:,bank],label,folds[donor])
                        controls.append({"held":hold,"null":donor,"index":bank,"min_null_sample":min(model.counts),"median":float(torch.tensor(model.counts).float().median())})
                        for j in range(0,len(ix),128):
                            values,logs=model.score(qv[j:j+128,:,:,bank],prior)
                            # These diagnostic IDs are revealed after the complete rival matrices are computed.
                            n=len(next(iter(values.values())));local=torch.arange(n,device=g.device)
                            pnode=label[pos[j:j+n]];nnode=label[neg[j:j+n]]
                            for name,z in logs.items():
                                observed[name][j:j+n]+=torch.stack((z[local,pnode,nnode],z[local,nnode,pnode]),1).cpu()/6
                            for name,v in values.items():
                                if name not in configs:configs[name]=torch.zeros(len(ix),3,k)
                                configs[name][j:j+len(v),hold]+=.5*v.cpu()
                        # Check the empirical tail for genuinely held-out gallery images, conditioned on their unsigned visual group, before using any class labels.
                        sel=folds[hold][torch.randperm(len(folds[hold]),generator=torch.Generator().manual_seed(48741))[:min(160,len(folds[hold]))].to(g.device)]
                        for ids in sel.split(80):
                            _,logs=model.score(pv[ids,:,:,bank],prior)
                            these=label[ids];mask=torch.arange(k,device=g.device)[None]!=these[:,None]
                            for name in ("whole","gain","gain-unconditional","gain-mixed","gain-pooled"):
                                p=logs[name][torch.arange(len(ids),device=g.device),these].exp()
                                corner[name][0]+=int(((p<=.05)&mask).sum());corner[name][1]+=int(mask.sum())
                    print(s,stage,level,domain,"held",hold,flush=True)
                valid=miss.cpu()&(label[pos]!=label[neg]).cpu()
                save_tensor({"ids":ix,"labels":label.cpu(),"parts":geo["parts"],"scores":configs,
                             "diagnostic_pair":observed,"eligible_original_errors":valid},file)
                atomic_json(file.with_suffix(".json"),{"code":revision(),"group_count":k,"folds":controls,
                            "original_errors_in_different_groups":int(valid.sum()),
                            "favors_relevant_on_original_errors":{k:int(((z[:,0]>z[:,1])&valid).sum()) for k,z in observed.items()},
                            "held_photo_tail_at_005":{k:{"below":z[0],"comparisons":z[1],"fraction":z[0]/z[1]} for k,z in corner.items()}})
                del configs,pv
        atomic_json(folder/"done.json",{"code":revision(),"queries":len(ix),"levels":[2,3],"domains":["photo","source"]})
        del queries


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy");p.add_argument("--stage",choices=("pilot","full"),default="pilot")
    a=p.parse_args();run(a.workspace,a.dataset,a.stage)
