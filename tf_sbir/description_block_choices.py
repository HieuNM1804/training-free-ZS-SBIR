"""Score the same target descriptions at each native CLIP block on full unseen Sketchy."""

import argparse
from collections import defaultdict
import hashlib
import json

import open_clip
import torch
from torch.nn.functional import normalize

from .concept_benchmark import atomic_json,checksum
from .description_choice_blocks import OUT,PARTS,PILOT,inputs
from .encoding import load_frozen_clip,save_tensor
from .native_block_probe import SEED,deterministic
from .wordnet_renaming_pilot import SPLITS,nominate
from .wordnet_renaming_text import Sense,anonymize,describe,encode


DEPTHS=(4,8,12)


def order_key(kind,split,i):
    return hashlib.sha256(f"{SEED}/{kind}/{split}/{i}".encode()).digest()


def controls(split,qi,labels,sources):
    peer=torch.empty_like(qi)
    redraw=torch.empty_like(qi)
    other=torch.empty_like(qi)
    classes=sorted(labels.unique().tolist(),key=lambda c:order_key("classes",split,c))
    ordered={}
    for cls in classes:
        subjects=defaultdict(list)
        for offset in torch.where(labels==cls)[0].tolist():
            subjects[int(sources[offset])].append(offset)
        groups=[]
        for subject in sorted(subjects,key=lambda s:order_key("subject",split,s)):
            ids=torch.tensor(sorted(subjects[subject],key=lambda i:order_key("sketch",split,int(qi[i]))))
            redraw[ids]=ids.roll(1)
            groups.append(ids)
        sequence=torch.cat(groups)
        step=max(map(len,groups))
        if 2*step>len(sequence):
            raise ValueError("No derangement at the specified offset")
        peer[sequence]=sequence.roll(step)
        ordered[cls]=sequence
    for i,cls in enumerate(classes):
        target=ordered[classes[(i+1)%len(classes)]]
        idx=ordered[cls]
        other[idx]=target[torch.arange(len(idx))*len(target)//len(idx)]
    if (not torch.equal(peer.sort().values,torch.arange(len(qi))) or (sources[peer]==sources).any()
        or not torch.equal(labels[peer],labels) or (labels[other]==labels).any()
        or not torch.equal(sources[redraw],sources)):
        raise ValueError("Control assignment failed")
    _,group,counts=sources.unique(return_inverse=True,return_counts=True)
    if not torch.equal(qi[redraw]==qi,counts[group]==1):raise ValueError("Missing redraw flag is inconsistent")
    return peer,redraw,other


def pilot_rows(name,qi,gi):
    parts=[]
    for i in (0,1):
        p=PILOT/f"{name}-{i}.pt"
        m=json.loads(p.with_suffix(".json").read_bytes())
        if m["tensor_sha256"]!=checksum(p) or m["revision"]!="6fa17be0cdb20611110879dec3dafc1b31f0f798":
            raise ValueError("Unverified original pilot")
        parts.append(torch.load(p,mmap=True,weights_only=True))
    order=torch.cat([r["positions"] for r in parts]).argsort()
    def merge(key):
        return torch.cat([r[key] for r in parts])[order]
    if not torch.equal(merge("query_ids"),qi) or any(not torch.equal(p["gallery_ids"],gi) for p in parts):
        raise ValueError("Unseen set order differs from archived KD metrics")
    return {k:merge(k) for k in ("pair","candidate_metrics","margins","classes")}


def canonical_utility(name,senses):
    parts=[]
    for i in (0,1):
        p=PILOT/"ceiling"/f"{name}-{i}.pt"
        m=json.loads(p.with_suffix(".json").read_bytes())
        if m["revision"]!="50904116bfa71ad76e6ae6c65f5fec5a235d713d" or checksum(p)!=m["sha256"]:
            raise ValueError("Unverified full-menu class AP")
        parts.append(torch.load(p,mmap=True,weights_only=True))
    if parts[0]["strings"]!=parts[1]["strings"]:raise ValueError("Unequal sentence banks")
    lookup={s:i for i,s in enumerate(parts[0]["strings"])}
    ids=torch.cat([p["indices"] for p in parts])
    table=torch.cat([p["values"] for p in parts])[ids.argsort()]
    return table[[lookup[describe("photo",s.anonymous)] for s in senses]].contiguous()


class NativeReads:
    def __init__(self,mode,revision):
        blobs=[]
        for i in range(PARTS):
            p=OUT/f"native-{mode}-{i}.pt"
            m=json.loads(p.with_suffix(".json").read_bytes())
            if checksum(p)!=m["sha256"] or m["revision"]!=revision:raise ValueError("Unverified native features")
            blobs.append(torch.load(p,mmap=True,weights_only=True))
        ids=torch.cat([p["ids"] for p in blobs])
        if len(ids.unique())!=len(ids):raise ValueError("Duplicate image ID")
        self.lookup=torch.full((int(ids.max())+1,),-1,dtype=torch.long)
        self.lookup[ids]=torch.arange(len(ids))
        self.bias=blobs[0]["bias"].cuda()
        if any(not torch.equal(blobs[0]["bias"],p["bias"]) for p in blobs):raise ValueError("Unequal model weights")
        self.views={}
        for view in (("rgb","mirror") if mode=="sketch" else ("rgb",)):
            self.views[view]={k:torch.cat([p[view][k] for p in blobs]).cuda()
                              for k in ("prefix","native","shift","norm")}

    def read(self,image_ids,depths,combined=False):
        ids=self.lookup[image_ids]
        if (ids<0).any():raise ValueError("Missing image")
        ids=ids.cuda()
        panels=[]
        for view in (("rgb","mirror") if combined else ("rgb",)):
            b=self.views[view]
            y=b["prefix"][ids[:,None],depths]
            shift=b["shift"][ids[:,None],depths,None]
            norm=b["norm"][ids[:,None],depths,None]
            local=(y+shift*self.bias)/norm
            panels.append((y,local,b["native"][ids]))
        if not combined:return panels[0]
        (p1,l1,v1),(p2,l2,v2)=panels
        length=(v1+v2).norm(dim=1,keepdim=True)
        return (p1+p2)/length[:,:,None],normalize(l1+l2,dim=2),(v1+v2)/length


def pair_axes(model,tok,senses,pairs):
    distinct,inverse=pairs.unique(dim=0,return_inverse=True)
    captions=[]
    for ai,bi in distinct.tolist():
        a,b=senses[ai],senses[bi]
        aliases=tuple(sorted(set(a.aliases)|set(b.aliases)))
        ga,gb=anonymize(a.definition,aliases),anonymize(b.definition,aliases)
        if not ga or not gb:ga,gb=a.anonymous,b.anonymous
        captions.extend((describe("drawing",ga),describe("drawing",gb)))
    texts=encode(model,tok,captions).reshape(len(distinct),2,-1)
    return (texts[:,0]-texts[:,1])[inverse].contiguous()


def fixed_matches(reads,ids,combined,directions):
    depths=torch.arange(13,device="cuda")
    led,loc,end=[],[],[]
    for start in range(0,len(ids),256):
        sl=slice(start,start+256)
        y,l,v=reads.read(ids[sl],depths,combined)
        d=directions[sl]
        led.append((y*d[:,None,:]).sum(2).cpu())
        loc.append((l[:,DEPTHS,:]*d[:,None,:]).sum(2).cpu())
        end.append((v*d).sum(1).cpu())
    return {"ledger":torch.cat(led),"local":torch.cat(loc),"native":torch.cat(end)}


def full_choices(reads,ids,combined,texts,heads):
    depths=torch.tensor(DEPTHS,device="cuda")
    picks=[]
    for part in ids.split(256):
        ledger,local,native=reads.read(part,depths,combined)
        vectors=torch.cat((ledger,local,native[:,None,:]),dim=1)
        result=nominate(vectors.flatten(0,1),texts,heads).reshape(len(part),vectors.shape[1],2)
        picks.append(result.to(device="cpu",dtype=torch.int16))
    return torch.cat(picks)


@torch.inference_mode()
def run(name,phase,revision,native_revision):
    meta,index,global_sources=inputs()
    qi,labels=index["sketch"]["views"][name]
    gi,_=index["photo"]["views"][name]
    pilot=pilot_rows(name,qi,gi)
    if not torch.equal(labels,pilot["classes"]):raise ValueError("Query label mismatch")
    source=global_sources[qi]
    peer,redraw,other=controls(name,qi,labels,source)
    base={"query_ids":qi,"gallery_ids":gi,"source_ids":source,"peer":peer,"redraw":redraw,"other":other,
          "classes":labels,"pair":pilot["pair"],"depths":list(DEPTHS)}
    model=load_frozen_clip(meta["checkpoint"],torch.device("cuda"))
    tok=open_clip.get_tokenizer("ViT-B-32")
    inventory=json.loads((PILOT/"inventory.json").read_bytes())
    senses=[Sense(**s) for s in inventory["senses"]]
    if len(senses)!=1135:raise ValueError("Unexpected WordNet bank")
    sketch=NativeReads("sketch",native_revision)
    photo=NativeReads("photo",native_revision)
    checks={}
    if phase=="fixed":
        d=pair_axes(model,tok,senses,pilot["pair"])
        base["candidate_metrics"]=pilot["candidate_metrics"]
        for panel,axis in (("self",d),("independent",d[peer])):
            scores={}
            for key,reader,images,pooled in (("sketch",sketch,qi,False),("sketch_two",sketch,qi,True),
                ("redraw",sketch,qi[redraw],False),("source",photo,source,False),
                ("same_class_photo",photo,source[peer],False),("other_class_photo",photo,source[other],False)):
                scores[key]=fixed_matches(reader,images,pooled,axis)
            base[panel]=scores
        replay=base["self"]["sketch_two"]["native"]
        orig=pilot["margins"][:,1]
        checks["archived_margin_max_abs"]=(replay-orig).abs().max().item()
        checks["archived_choice_agreement"]=((replay<0)==(orig<0)).double().mean().item()
        if checks["archived_margin_max_abs"]>.002:
            raise ValueError(f"Original pilot decision not reproduced {checks}")
        base["pooled_self_native_reread"]=replay
        base["self"]["sketch_two"]["native"]=orig
    else:
        texts=encode(model,tok,[describe("drawing",s.anonymous) for s in senses])
        heads=torch.tensor([s.head for s in senses],device="cuda")
        unique,back=source.unique(return_inverse=True)
        base["sketch"]=full_choices(sketch,qi,False,texts,heads)
        base["sketch_two"]=full_choices(sketch,qi,True,texts,heads)
        base["source"]=full_choices(photo,unique,False,texts,heads)[back]
        base["class_utility"]=canonical_utility(name,senses)
        checks["replay_of_both_original_nominations"]=(base["sketch_two"][:,-1].long()==pilot["pair"]).all(1).double().mean().item()
        # Check regrets against all allowed alternatives, then use the exact archived endpoint in reported curves.
        regrets=[]
        for start in range(0,len(qi),256):
            sl=slice(start,start+256)
            _,_,v=sketch.read(qi[sl],torch.tensor([12],device="cuda"),True)
            scores=v @ texts.T
            ids=pilot["pair"][sl].cuda()
            best=scores.max(1).values-scores.gather(1,ids[:,:1]).squeeze(1)
            eligible=heads[None,:]!=heads[ids[:,0],None]
            second=scores.masked_fill(~eligible,-torch.inf).max(1).values-scores.gather(1,ids[:,1:]).squeeze(1)
            regrets.append(torch.stack((best,second),dim=1).cpu())
        regret=torch.cat(regrets)
        checks["max_native_regret_for_original_two"]=regret.amax(0).tolist()
        if regret.max()>.002:raise ValueError(f"Archived alternatives missed by a material cosine margin {checks}")
        base["pooled_native_reread"]=base["sketch_two"][:,-1].clone()
        base["sketch_two"][:,-1]=pilot["pair"].short()
    dest=OUT/f"{name}-{phase}.pt"
    save_tensor(base,dest)
    atomic_json(dest.with_suffix(".json"),{"revision":revision,"native_revision":native_revision,
                "split":name,"phase":phase,"queries":len(qi),"gallery":len(gi),"subjects":len(source.unique()),
                "classes":len(labels.unique()),"checks":checks,"sha256":checksum(dest),"seed":SEED,
                "device":torch.cuda.get_device_name()})
    print(name,phase,checks,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--phase",choices=("fixed","free"),required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--native-revision",required=True)
    a=p.parse_args()
    deterministic()
    run(a.split,a.phase,a.revision,a.native_revision)
