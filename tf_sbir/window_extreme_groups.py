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
    _,_,qs,_,_,_,_=dataset(w,ds)
    get=lambda f:torch.load(f,mmap=True,weights_only=True)
    for s,(qi,y) in qs.items():
        root=w/"cache"/FAMILY/s;read(root/"ready.json");geo=get(root/"structure.pt")
        folder=root/("groups-"+stage);folder.mkdir(exist_ok=True)
        if (folder/"done.json").exists():continue
        ix=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(y))
        queries=get(root/"sketches.pt")[ix].cuda()
        g=geo["groups"].cuda();up=geo["upper"].cuda();folds=[p.cuda() for p in geo["parts"]]
        for level in (2,3):
            qv,label=assemble_level(queries,g,up,level);k=qv.size(-1)
            prior=torch.bincount(label,minlength=k).float();prior/=prior.sum()
            for domain in ("photo","source"):
                file=folder/f"{domain}-{level}.pt"
                if file.exists():continue
                configs={}
                controls=[];corner=defaultdict(lambda:[0,0])
                raw=get(root/("photos-"+domain+".pt")).cuda()
                pv,_=assemble_level(raw,g,up,level);del raw
                for hold in range(3):
                    for donor in (i for i in range(3) if i!=hold):
                        bank=3-hold-donor
                        model=Calibration(pv[:,:,:,bank],label,folds[donor])
                        controls.append({"held":hold,"null":donor,"index":bank,"min_null_sample":min(model.counts),"median":float(torch.tensor(model.counts).float().median())})
                        for j in range(0,len(ix),128):
                            values,_=model.score(qv[j:j+128,:,:,bank],prior)
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
                save_tensor({"ids":ix,"labels":label.cpu(),"parts":geo["parts"],"scores":configs},file)
                atomic_json(file.with_suffix(".json"),{"code":revision(),"group_count":k,"folds":controls,
                            "held_photo_tail_at_005":{k:{"below":z[0],"comparisons":z[1],"fraction":z[0]/z[1]} for k,z in corner.items()}})
                del configs,pv
        atomic_json(folder/"done.json",{"code":revision(),"queries":len(ix),"levels":[2,3],"domains":["photo","source"]})
        del queries


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy");p.add_argument("--stage",choices=("pilot","full"),default="pilot")
    a=p.parse_args();run(a.workspace,a.dataset,a.stage)
