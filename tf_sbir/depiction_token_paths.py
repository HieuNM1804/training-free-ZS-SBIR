"""Exchange the same noun's contextual states between two depictions in frozen CLIP text."""

import argparse
from pathlib import Path

import open_clip
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT, runtime
from .encoding import load_frozen_clip, save_tensor
from .foundation import DEFAULT_DOMAIN_TEMPLATES, load_words
from .semantic import read_words


FAMILY = "depiction-token-paths"
LAYERS = (1, 3, 6, 9, 11)


def advance(transformer, x, mask, start, stop):
    batch_first = transformer.batch_first
    hidden = x if batch_first else x.transpose(0,1).contiguous()
    for i in range(start,stop):
        hidden = transformer.resblocks[i](hidden,attn_mask=mask)
    return hidden if batch_first else hidden.transpose(0,1)


def end_readout(model, hidden, eot):
    normalized = model.ln_final(hidden)
    row = normalized[torch.arange(len(hidden),device=hidden.device),eot]
    if isinstance(model.text_projection,torch.nn.Linear):
        return F.normalize(model.text_projection(row).float())
    return F.normalize((row@model.text_projection).float())


def tokenize_pair(tokenizer, word_list, source_prompt, target_prompt):
    s = tokenizer([source_prompt.format(w) for w in word_list])
    t = tokenizer([target_prompt.format(w) for w in word_list])
    prefix = target_prompt.split("{}")[0].strip()
    other_prefix = source_prompt.split("{}")[0].strip()
    start = int(tokenizer([prefix])[0].argmax())
    if int(tokenizer([other_prefix])[0].argmax()) != start:
        raise ValueError("The paired public prompts put the name at different positions")
    eot = t.argmax(1)
    if not torch.equal(eot,s.argmax(1)) or not torch.equal(s[:,start:],t[:,start:]):
        raise ValueError("The public name tokens are not aligned")
    length=int(eot.max())+1
    positions=torch.arange(length)
    # Include all actual BPE tokens of each public label. A standalone trailing
    # full stop belongs to the caption suffix. Merged punctuation stays in the name.
    dot=tokenizer(["."])[0,1]
    has_stop=t.gather(1,(eot-1)[:,None])[:,0]==dot
    end=eot-has_stop.long()
    if (end<=start).any():
        raise ValueError("Empty or truncated public name")
    noun=(positions[None]>=start)&(positions[None]<end[:,None])
    return s[:,:length].cuda(),t[:,:length].cuda(),noun.cuda(),eot.cuda()


def hybrid_bank(model, tokenized, cuts, check=False):
    src,tgt,noun,eot=tokenized
    n=len(src)
    tokens=torch.cat((tgt,src))
    depth=len(model.transformer.resblocks)
    if depth!=12 or model.text_pool_type!="argmax":
        raise ValueError("This cache requires the OpenAI ViT-B/32 causal text tower")
    dtype=model.transformer.get_cast_dtype()
    hidden=model.token_embedding(tokens).to(dtype)+model.positional_embedding[:tokens.size(1)].to(dtype)
    mask=model.attn_mask[:tokens.size(1),:tokens.size(1)]
    states={}
    cursor=0
    for l in sorted(set((*cuts,depth))):
        hidden=advance(model.transformer,hidden,mask,cursor,l)
        states[l]=hidden
        cursor=l
    ordinary=end_readout(model,states[depth],torch.cat((eot,eot)))
    base,source=ordinary.chunk(2)
    outputs={}
    for l in cuts:
        ht,hs=states[l].chunk(2)
        name_only=torch.where(noun[:,:,None],hs,ht)
        context_only=torch.where(noun[:,:,None],ht,hs)
        both=advance(model.transformer,torch.cat((name_only,context_only)),mask,l,depth)
        name,context=end_readout(model,both,torch.cat((eot,eot))).chunk(2)
        outputs[l]=torch.stack((base,name,context,source),1)
    error=0.
    if check:
        full=F.pad(tokens,(0,model.context_length-tokens.size(1)))
        reference=F.normalize(model.encode_text(full).float())
        error=float((1-(ordinary*reference).sum(1)).abs().max())
        if error>.002:
            raise ValueError(f"Trimming causal padding changed native readout {error}")
    return outputs,error


def decompose(values):
    tt,only_noun,only_context,ss=values.unbind(1)
    first=only_noun-tt
    second=ss-only_context
    noun=(first+second)*.5
    rest=((only_context-tt)+(ss-only_noun))*.5
    closure=float(((ss-tt)-noun-rest).abs().max())
    return torch.stack((noun,rest,first,second-first),1),closure


@torch.inference_mode()
def run(w,batch):
    cfg=runtime(w,"sketchy")
    names=read_words(cfg.vocabulary)
    if len(names)!=20931:
        raise ValueError("Changed public dictionary")
    meta={"checkpoint":CHECKPOINT,"vocabulary":checksum(cfg.vocabulary),"layers":list(LAYERS),
          "templates":DEFAULT_DOMAIN_TEMPLATES,"partition":"entire noun vs complement; average the two exchange orders v1"}
    signature=digest(meta)
    folder=w/"cache"/FAMILY
    file=folder/"paired.pt"
    desc=file.with_suffix(".json")
    if file.is_file() and desc.is_file() and read(desc)["signature"]==signature:
        print("existing complete paired-token index",flush=True)
        return
    reference=load_words(cfg,names)
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
    tokenizer=open_clip.get_tokenizer("ViT-B-32")
    cache={l:torch.empty((len(names),4,512),dtype=torch.float16) for l in LAYERS}
    checks={"trimmed_max_1_minus_cos":0.,"endpoints_max_1_minus_cos":0.,"factorial_closure":0.,
            "noun_null_at_input_or_final":0.,"worst_storage_rms":0.}
    stats={l:torch.zeros(3,dtype=torch.double) for l in LAYERS}
    for start in range(0,len(names),batch):
        part=names[start:start+batch]
        probe=start==0
        cuts=(0,*LAYERS,12) if probe else LAYERS
        pooled={}
        for st,tt in zip(*DEFAULT_DOMAIN_TEMPLATES,strict=True):
            measurements,error=hybrid_bank(model,tokenize_pair(tokenizer,part,st,tt),cuts,check=probe or start%(batch*40)==0)
            checks["trimmed_max_1_minus_cos"]=max(checks["trimmed_max_1_minus_cos"],error)
            for l,out in measurements.items():
                if l in pooled: pooled[l]+=out
                else: pooled[l]=out
        for l,total in pooled.items():
            coalitions=F.normalize(total,dim=-1)
            p,closure=decompose(coalitions)
            checks["factorial_closure"]=max(checks["factorial_closure"],closure)
            if l in (0,12):
                checks["noun_null_at_input_or_final"]=max(checks["noun_null_at_input_or_final"],float(p[:,0].norm(dim=-1).max()))
                continue
            cache[l][start:start+len(part)]=p.half().cpu()
            checks["worst_storage_rms"]=max(checks["worst_storage_rms"],float((p-p.half().float()).square().mean().sqrt()))
            if l==LAYERS[0]:
                s,t=(z[start:start+len(part)].cuda() for z in reference)
                err=torch.stack(((1-(s*coalitions[:,3]).sum(1)).abs().amax(),(1-(t*coalitions[:,0]).sum(1)).abs().amax())).amax()
                checks["endpoints_max_1_minus_cos"]=max(checks["endpoints_max_1_minus_cos"],float(err))
            lengths=p.norm(dim=-1)
            stats[l]+=torch.stack((lengths[:,0].sum(),lengths[:,1].sum(),lengths[:,3].sum())).double().cpu()
        if checks["endpoints_max_1_minus_cos"]>.002 or checks["noun_null_at_input_or_final"]>.003 or checks["factorial_closure"]>1e-5:
            raise ValueError(f"Invalid paired text edits {checks}")
        if start%(32*batch)==0:
            print("names",start,len(names),checks,flush=True)
    save_tensor(cache,file)
    atomic_json(desc,meta|{"signature":signature,"names":len(names),"checks":checks,
                           "per_layer_mean_norm_noun_rest_interaction":{str(l):(v/len(names)).tolist() for l,v in stats.items()}})
    print("finished",file.stat().st_size,checks,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--batch",type=int,default=192)
    a=parser.parse_args()
    run(a.workspace,a.batch)
