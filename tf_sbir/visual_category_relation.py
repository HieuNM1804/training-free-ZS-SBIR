"""Read a category relation from four counterbalanced boards without naming either class."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.nn.functional import interpolate, normalize

from .description_choice_blocks import inputs
from .kaggle_unseen_features import DEFAULT_CACHE
from .native_block_probe import SEED, deterministic
from .photo_name_interaction import BOARD, OUT as NAME_INDEX, PARTS, Encoder, loader, ordered_shards, record, verified
from .photo_name_evaluation import queries
from .wordnet_renaming_pilot import SPLITS
from .wordnet_renaming_text import encode


OUT=DEFAULT_CACHE.parent/"results/visual-category-relation"
PROMPTS=("two objects of the same kind.","two objects of different kinds.")
HALF=BOARD//2


def design(qi,labels,gi,classes,source,names):
    rng=np.random.default_rng(SEED)
    all_classes=classes.unique().tolist()
    pictures={c:gi[classes==c].numpy() for c in all_classes}
    by_name={c:{n:gi[(classes==c)&(names==n)].numpy() for n in names[classes==c].unique().tolist()} for c in all_classes}
    sketches={}
    for q,c,s in zip(qi.tolist(),labels.tolist(),source[qi].tolist(),strict=True):
        sketches.setdefault(c,{}).setdefault(s,[]).append(q)
    rows=[]
    for c1,c2 in combinations(all_classes,2):
        common=sorted(set(by_name[c1])&set(by_name[c2]))
        for _ in range(4):
            easy=[int(rng.choice(pictures[c])) for c in (c1,c2)]
            hard=[]
            if common:
                noun=rng.choice(common)
                hard=[int(rng.choice(by_name[c][noun])) for c in (c1,c2)]
            query=[]
            for i,c in enumerate((c1,c2)):
                exc={easy[i]}
                if hard:exc.add(hard[i])
                s=int(rng.choice([s for s in sketches[c] if s not in exc]))
                query.append(int(rng.choice(sketches[c][s])))
            for arm,photos in enumerate((easy,hard)):
                if photos:rows.append((arm,c1,c2,*query,*photos))
    return torch.tensor(rows,dtype=torch.long)


def thumbnails(paths,ids,base,workers):
    images=[]
    for pixels in loader(paths,ids,base,256,workers):
        small=interpolate(pixels.cuda(non_blocking=True).float(),size=(HALF,HALF),mode="bilinear",
                          align_corners=False,antialias=True)
        images.append(small.round().clamp_(0,255).byte())
    return torch.cat(images)


def boards(left,right):
    n=len(left) if left is not None else len(right)
    x=torch.ones((n,3,BOARD,BOARD),device="cuda")
    band=slice(HALF//2,HALF//2+HALF)
    if left is not None:x[:,:,band,:HALF]=left.float()/255
    if right is not None:x[:,:,band,HALF:]=right.float()/255
    return x


def read_raw(encoder,pixels):
    return encoder.model.encode_image(((pixels-encoder.mean)/encoder.std).to(encoder.visual.conv1.weight.dtype)).float()


def accuracy(delta):
    return (delta>0).double()+.5*(delta==0).double()


def effect(x,identity,total_classes):
    pairs,group=identity.unique(dim=0,return_inverse=True)
    values=torch.stack([x[group==i].double().mean() for i in range(len(pairs))]).numpy()
    subs=np.random.default_rng(SEED).multinomial(total_classes,np.full(total_classes,1/total_classes),size=5000)
    w=subs[:,pairs[:,0].numpy()]*subs[:,pairs[:,1].numpy()]
    w=w[w.sum(1)>0]
    bounds=np.quantile((w@values)/w.sum(1),(.025,.975))
    return {"trials":len(x),"class_pairs":len(pairs),"trial_percent":100*x.double().mean().item(),
            "class_pair_percent":100*float(values.mean()),"interval95":(100*bounds).tolist()}


def summarize(t,pred,total_classes):
    out={}
    identity=t[:,1:3]
    for kind in (0,1):
        take=t[:,0]==kind
        variants={"all":take,"native_wrong":take&(pred["native"]<0)}
        result={}
        for label,mask in variants.items():
            if not mask.any():continue
            win={k:accuracy(v[mask]) for k,v in pred.items()}
            estimates={k:effect(v,identity[mask],total_classes) for k,v in win.items()}
            estimates["joint_minus_chance"]=effect(win["joint"]-.5,identity[mask],total_classes)
            for k in ("additive","native"):
                estimates["joint_minus_"+k]=effect(win["joint"]-win[k],identity[mask],total_classes)
            result[label]=estimates
        out[("ordinary","same_photo_noun")[kind]]=result
    return out


@torch.inference_mode()
def run(split,part,revision,names_revision,native_revision):
    meta,idx,source=inputs()
    qi,labels=idx["sketch"]["views"][split]
    gi,classes=idx["photo"]["views"][split]
    pics=ordered_shards([NAME_INDEX/f"names-{i}.pt" for i in range(PARTS)],names_revision,torch.arange(len(idx["photo"]["paths"])))
    all_rows=design(qi,labels,gi,classes,source,pics["name"][gi])
    positions=torch.arange(part,len(all_rows),2)
    t=all_rows[positions]
    qids,qmap=t[:,3:5].unique(return_inverse=True)
    pids,pmap=t[:,5:7].unique(return_inverse=True)
    base=Path(meta["datasets"]["sketchy"]["root"])
    q=thumbnails(idx["sketch"]["paths"],qids,base,8)
    p=thumbnails(idx["photo"]["paths"],pids,base,8)
    enc=Encoder(meta["checkpoint"])
    enc.close()
    texts=encode(enc.model,open_clip.get_tokenizer("ViT-B-32"),list(PROMPTS))
    axis=(texts[0]-texts[1]).double()
    blank=read_raw(enc,torch.ones((1,3,BOARD,BOARD),device="cuda")).double()
    qorig=queries(qids,native_revision)
    porig=pics["native"][pids].cuda()
    native=((qorig[qmap[:,0]]-qorig[qmap[:,1]])*(porig[pmap[:,0]]-porig[pmap[:,1]])).sum(1).cpu()
    predictions={"native":native}
    for swapped in (False,True):
        def build(first,second):
            return boards(second,first) if swapped else boards(first,second)
        qa=torch.cat([read_raw(enc,build(v,None)) for v in q.split(256)]).double()
        pb=torch.cat([read_raw(enc,build(None,v)) for v in p.split(256)]).double()
        values={"joint":[],"raw":[],"additive":[]}
        max_err=0.
        for start in range(0,len(t),80):
            qs=qmap[start:start+80][:,[0,0,1,1]].flatten().cuda()
            ps=pmap[start:start+80][:,[0,1,0,1]].flatten().cuda()
            composed=read_raw(enc,build(q[qs],p[ps])).double()
            separate=qa[qs]+pb[ps]-blank
            def contrast(z):
                y=(z@axis).reshape(-1,4)
                return (y[:,0]-y[:,1])+(y[:,3]-y[:,2])
            max_err=max(max_err,float(contrast(separate).abs().max()))
            for k,v in (("joint",normalize(composed,dim=1)),("raw",composed),("additive",normalize(separate,dim=1))):
                values[k].append(contrast(v).cpu())
        if max_err>1e-8:raise ValueError("Independent scalar effects did not cancel")
        orientation="photo_left" if swapped else "sketch_left"
        for k,parts in values.items():predictions[k+"_"+orientation]=torch.cat(parts)
    for k in ("joint","raw","additive"):
        predictions[k]=.5*(predictions[k+"_photo_left"]+predictions[k+"_sketch_left"])
    record(OUT/f"{split}-{part}.pt",{"positions":positions,"trials":t,"predictions":predictions},revision,
           {"trials":len(t),"part":part,"split":split,"prompts":list(PROMPTS),"native_revision":native_revision,
            "name_revision":names_revision,"maximum_additive_error_last_orientation":max_err})


def collect(split,revision):
    _,idx,_=inputs()
    packs=[verified(OUT/f"{split}-{i}.pt",revision) for i in range(2)]
    pos=torch.cat([x["positions"] for x in packs]);order=pos.argsort()
    if not torch.equal(pos[order],torch.arange(len(pos))):raise ValueError("Incomplete assignment trials")
    trial=torch.cat([x["trials"] for x in packs])[order]
    prediction={k:torch.cat([x["predictions"][k] for x in packs])[order] for k in packs[0]["predictions"]}
    classes=len(idx["sketch"]["views"][split][1].unique())
    record(OUT/f"summary-{split}.pt",{"trials":trial,"margins":prediction},revision,
           {"split":split,"classes":classes,"results":summarize(trial,prediction,classes),"prompts":list(PROMPTS)})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",required=True,choices=SPLITS)
    p.add_argument("--part",type=int,choices=(0,1))
    p.add_argument("--revision",required=True)
    p.add_argument("--names-revision",required=True)
    p.add_argument("--native-revision",required=True)
    a=p.parse_args()
    deterministic()
    if a.part is None:collect(a.split,a.revision)
    else:run(a.split,a.part,a.revision,a.names_revision,a.native_revision)
