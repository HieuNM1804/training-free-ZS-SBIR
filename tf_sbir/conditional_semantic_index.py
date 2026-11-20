"""A CLIP conditional mean contrasted against the uniform gallery distribution."""

import argparse
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib
from pathlib import Path

import open_clip
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import SPLIT_HASHES,digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .metrics import cosine_batches
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta,tensor
from .semantic import read_words
from .transfer_benchmark import write_result


PRIMARY="bare-witness"
MODES=("native","bare-direct","bare-conditional","bare-witness","named-witness","rgb-witness",
       "shuffled-witness","reference-half-a","reference-half-b")
SIZES={"sketchy_1":(14179,17100,25),"sketchy_2":(12694,12552,21),"tuberlin":(2130,27262,30)}


def protocol(folder):
    src=Path(__file__).resolve().parents[1]/"docs/CONDITIONAL_INDEX_PROTOCOL_VI.md"
    contents={"revision":revision(),"source_protocol_sha256":checksum(src),"checkpoint":CHECKPOINT,"primary":PRIMARY,
              "controls":list(MODES),"parameters_from_benchmark":0,"image_reference":"every gallery image",
              "temperature":"logit_scale.exp() from OpenAI ViT-B/32","external_vocabulary":20931}
    file=folder/"protocol.json"
    if file.exists():
        lock=read(file); sha=lock.pop("sha256")
        if digest(lock)!=sha or any(lock[k]!=v for k,v in contents.items()):
            raise ValueError("Inconsistent preregistration")
        return lock|{"sha256":sha}
    if any(folder.glob("*/*.json")):
        raise ValueError("Scores preceded protocol")
    contents["utc"]=datetime.now(timezone.utc).isoformat()
    atomic_json(file,contents|{"sha256":digest(contents)})
    return read(file)


def bare_words(args,words):
    signature=hashlib.sha256("\0".join(words).encode()).hexdigest()
    file=args.cache/f"literal-clip-words-{CHECKPOINT[:16]}-{signature[:16]}.pt"
    if not file.exists():
        model=load_frozen_clip(args.clip_checkpoint,torch.device("cuda"))
        tokenizer=open_clip.get_tokenizer("ViT-B-32")
        bank=[]
        for start in range(0,len(words),384):
            tokens=tokenizer(words[start:start+384]).cuda()
            bank.append(F.normalize(model.encode_text(tokens).float()).cpu())
        save_tensor({"bank":torch.cat(bank),"scale":float(model.logit_scale.exp()),"words_sha256":signature},file)
    record=torch.load(file,map_location="cpu",weights_only=True)
    if record["words_sha256"]!=signature:
        raise ValueError("Changed public vocabulary")
    return record["bank"].cuda(),record["scale"]


def keys(photos,source,target,scale):
    result=[]
    for block in photos.split(384):
        mass=((block@target.T)*scale).softmax(1)
        result.append(F.normalize(mass@source))
    return torch.cat(result)


@dataclass
class Gallery:
    photos: torch.Tensor
    literal: torch.Tensor
    named: torch.Tensor
    scale: float

    def __post_init__(self):
        self.ids=torch.arange(len(self.photos),device=self.photos.device)
        self.shuffled=torch.randperm(len(self.photos),generator=torch.Generator().manual_seed(184)).to(self.photos.device)
        halves=torch.randperm(len(self.photos),generator=torch.Generator().manual_seed(711)).tensor_split(2)
        self.halves=[a.to(self.photos.device) for a in halves]

    def vectors(self,q,mode):
        if mode=="native":
            return q,self.photos
        if mode=="bare-direct":
            return q,self.literal
        base=self.named if mode=="named-witness" else self.literal
        if mode=="rgb-witness":
            base=self.photos
        ids=self.halves[0] if mode=="reference-half-a" else self.ids
        if mode=="reference-half-b":
            ids=self.halves[1]
        code=base[self.shuffled] if mode=="shuffled-witness" else base
        code=code[ids]
        values=self.photos[ids]
        query=[]
        mean=values.mean(0)
        for block in q.split(256):
            weights=((block@code.T)*self.scale).softmax(1)
            conditional=weights@values
            if mode!="bare-conditional":
                conditional=conditional-mean
            query.append(F.normalize(conditional))
        return torch.cat(query),self.photos


def analyze_weights(q,g,ql,pl):
    sample=stratified_sample(ql,32,8669)
    labels=ql[sample].cuda()
    relevant=labels[:,None]==pl.cuda()[None]
    observed={}
    for name,key in (("rgb",g.photos),("bare_words",g.literal),("named_prompt",g.named)):
        mass=((q[sample]@key.T)*g.scale).softmax(1)
        correct=(mass*relevant).sum(1)
        entropy=-(mass*mass.clamp_min(1e-35).log()).sum(1)
        observed[name]={"same_class_mass":float(correct.mean()),"median_effective_photos":float(entropy.exp().median()),
                        "uniform_same_class_mass":float(relevant.float().mean())}
    return observed


@torch.inference_mode()
def run(w,stage):
    root=w/"results/conditional-semantic-index"
    locked=protocol(root)
    if stage=="tuberlin" and read(root/"sketchy-complete.json")["freeze_sha256"]!=locked["sha256"]:
        raise ValueError("Missing source-domain evaluation")
    args,qp,queries,pp,photos,rgb_q,rgb_p=global_inputs(w,"tuberlin" if stage=="tuberlin" else "sketchy")
    words=read_words(args.vocabulary)
    if len(words)!=20931:
        raise ValueError("Unexpected public dictionary")
    bare,scale=bare_words(args,words)
    named_q,named_p=(v.cuda() for v in load_words(args,words))
    kd,kd_hash=load_kd(w)
    output={"freeze_sha256":locked["sha256"],"literal_terms":len(words),"scale":scale,"kd_source_sha256":kd_hash,"splits":{}}
    for s,(qi,ql) in queries.items():
        pi,pl=photos[s]; k=200 if s=="sketchy_2" else 100
        if (len(ql),len(pl),len(ql.unique()))!=SIZES[s] or len(pl.unique())!=SIZES[s][2] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:
            raise ValueError("Wrong evaluation data")
        pilot=stage=="pilot"
        if pilot:
            ids=stratified_sample(ql,32,9001); qi,ql=qi[ids],ql[ids]
        q,p=rgb_q[qi].cuda(),rgb_p[pi].cuda()
        gallery=Gallery(p,keys(p,bare,bare,scale),keys(p,named_q,named_p,scale),scale)
        folder=root/("heldout" if stage=="tuberlin" else "sketchy")
        all_details,tops,results={},{},{}
        for mode in MODES:
            query,bank=gallery.vectors(q,mode)
            if pilot:
                ids=(query@bank.T).topk(k).indices
                correct=pl.cuda()[ids]==ql.cuda()[:,None]
                results[mode]={"precision":float(correct.float().mean()),"top1":float(correct[:,0].float().mean())}
                continue
            meta={"revision":locked["revision"],"backbone":"OpenAI CLIP ViT-B/32","gpu":torch.cuda.get_device_name(),
                  "checkpoint_sha256":CHECKPOINT,"split_sha256":SPLIT_HASHES[s],"classes":len(UNSEEN_CLASSES[s]),
                  "freeze_sha256":locked["sha256"],"exploratory_unseen":False,
                  "external_vocabulary":0 if mode in ("native","rgb-witness") else len(words)}
            args.output=folder; args.metric_workers=5 if s!="sketchy_2" else 3
            def rows():
                for block in cosine_batches(query,bank,4):
                    yield from block.cpu()
            write_result(args,s,mode,rows(),ql,pl,meta,{"readout":mode,"scale":scale})
            record=read(folder/f"{s}-{mode}.json")
            details=tensor(folder/"per-query"/f"{s}-{mode}.pt")
            if details.shape!=(len(q),2) or max(abs(float(details[:,i].mean())-record[m]) for i,m in enumerate(("mAP","precision")))>1e-6:
                raise ValueError("Invalid per-query output")
            all_details[mode]=details
            results[mode]={k:record[k] for k in ("precision","mAP")}
            tops[mode]=torch.cat([(block@bank.T).argsort(dim=1,descending=True,stable=True)[:,:10].cpu() for block in query.split(128)])
            if mode==PRIMARY:
                ka,kp,mk,pk=kd(query,bank,ql,pl,s)
                results[mode]["original_KD"]={"mAP":float(ka),"precision":float(kp),"map_k":mk,"p_k":pk,
                    "max_abs_difference":max(abs(float(ka)-record["mAP"]),abs(float(kp)-record["precision"]))}
                single,_=gallery.vectors(q[:1],mode)
                results[mode]["independent_single_vs_batch_embedding_max"]=float((single-query[:1]).abs().max())
        if pilot:
            output["splits"][s]={"queries":len(q),"gallery":len(p),"scores":results}
            print(s,results,flush=True)
            continue
        contrasts={name:paired_delta(all_details[PRIMARY],value,ql) for name,value in all_details.items() if name!=PRIMARY}
        changes={}
        for name,t in tops.items():
            bad=pl[t]!=ql[:,None]
            hist=t[bad].bincount(minlength=len(p))
            changes[name]={"correct_top1":int((~bad[:,0]).sum()),"wrong_top10_share_top1pct":float(hist.topk(max(1,round(len(p)*.01))).values.sum()/hist.sum().clamp_min(1))}
        new=(pl[tops[PRIMARY][:,0]]==ql); old=(pl[tops["native"][:,0]]==ql)
        examples={}
        for tag,mask in (("rescued",new&~old),("regressed",~new&old),("still_wrong",~new&~old)):
            ids=mask.nonzero().flatten()
            rng=torch.Generator().manual_seed(8502)
            ids=ids[torch.randperm(len(ids),generator=rng)[:4]]
            examples[tag]=[{"sketch":"/".join(Path(qp[int(qi[j])]).parts[-3:]),
                            "native":"/".join(Path(pp[int(pi[tops["native"][j,0]])]).parts[-3:]),
                            "selected":"/".join(Path(pp[int(pi[tops[PRIMARY][j,0]])]).parts[-3:])} for j in ids]
        output["splits"][s]={"queries":len(q),"gallery":len(p),"classes":len(UNSEEN_CLASSES[s]),"scores":results,
            "top10":changes,"rescued":int((new&~old).sum()),"regressed":int((~new&old).sum()),"contrasts":contrasts,
            "conditional_mass":analyze_weights(q,gallery,ql,pl),"examples":examples,
            "per_class_AP_P":{n:{key:all_details[key][ql==i].mean(0).tolist() for key in MODES}
                              for i,n in enumerate(UNSEEN_CLASSES[s])}}
        save_tensor(tops,folder/f"{s}-top10.pt")
        print(s,results,flush=True)
    atomic_json(root/f"{stage}-complete.json",output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--stage",choices=("pilot","sketchy","tuberlin"),required=True)
    a=parser.parse_args()
    run(a.workspace,a.stage)
