"""Factor the CLIP text readout of a contextual noun into attention mass and conditional message."""

import argparse
import math
from pathlib import Path

import open_clip
import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_token_paths import FAMILY as TOKENS, advance, hybrid_bank, tokenize_pair
from .encoding import load_frozen_clip, save_tensor
from .foundation import DEFAULT_DOMAIN_TEMPLATES
from .semantic import read_words


FAMILY = "noun-attention-transfer"


def read_message(model, block, eot_state, outside, inside, mass):
    message = outside + mass[:, :, None]*(inside-outside)
    to_residual = block.attn.out_proj(message.flatten(1).to(eot_state.dtype))
    x = eot_state + block.ls_1(to_residual)
    x = x + block.ls_2(block.mlp(block.ln_2(x)))
    x = model.ln_final(x)
    if isinstance(model.text_projection,torch.nn.Linear):
        x = model.text_projection(x)
    else:
        x = x @ model.text_projection
    return F.normalize(x.float())


def summarise(logits, values, allowed):
    masked = logits.masked_fill(~allowed[:,None],-torch.inf)
    weight = masked.softmax(-1)
    summary = (weight[:,:,:,None]*values).sum(2)
    return masked.logsumexp(-1), summary


def trace(model, tokens, check=False):
    s,t,noun,eot = tokens
    batch = len(t)
    pair = torch.cat((t,s))
    dtype = model.transformer.get_cast_dtype()
    x = model.token_embedding(pair).to(dtype)+model.positional_embedding[:pair.size(1)].to(dtype)
    mask = model.attn_mask[:pair.size(1),:pair.size(1)]
    x = advance(model.transformer,x,mask,0,11)
    block = model.transformer.resblocks[11]
    native_t,native_s = x.chunk(2)
    attention = block.attn
    h = attention.num_heads
    normalized = block.ln_1(x)
    q,k,v=F.linear(normalized,attention.in_proj_weight,attention.in_proj_bias).chunk(3,dim=-1)
    q,k,v=(a.unflatten(-1,(h,-1)).transpose(1,2).float().contiguous() for a in (q,k,v))
    pair_eot=torch.cat((eot,eot))
    q=q[torch.arange(2*batch,device=q.device),:,pair_eot,:]
    qt,qs=q.chunk(2)
    kt,ks=k.chunk(2)
    vt,vs=v.chunk(2)
    outside=(torch.arange(t.size(1),device=t.device)[None]<=eot[:,None]) & ~noun
    rows=torch.arange(batch,device=t.device)
    joint={}
    native_mass=[]
    for context,(query,state) in enumerate(((qt,native_t),(qs,native_s))):
        lt=(kt*query[:,:,None]).sum(-1)/math.sqrt(query.size(-1))
        ls=(ks*query[:,:,None]).sum(-1)/math.sqrt(query.size(-1))
        zout,vo=summarise(lt if context==0 else ls,vt if context==0 else vs,outside)
        zt,nv_t=summarise(lt,vt,noun)
        zs,nv_s=summarise(ls,vs,noun)
        d=zs-zt
        center=(zs+zt)*.5-zout
        native_mass.append(torch.stack(((zt-zout).sigmoid(),(zs-zout).sigmoid()),-1))
        for kind,contr in (("observed",d),("common",d.mean(1,keepdim=True)),("head_swap",d.roll(1,1))):
            mt=(center-.5*contr).sigmoid()
            ms=(center+.5*contr).sigmoid()
            subset=[]
            # First index encodes attention mass, second conditional noun message.
            for mass,nv in ((mt,nv_t),(mt,nv_s),(ms,nv_t),(ms,nv_s)):
                subset.append(read_message(model,block,state[rows,eot],vo,nv,mass))
            joint[context,kind]=torch.stack(subset,1)
    error=0.
    if check:
        expected=hybrid_bank(model,tokens,(11,))[0][11]
        # Compare all four true endpoints to the ordinary token-swap intervention.
        got=torch.stack((joint[0,"observed"][:,0],joint[0,"observed"][:,3],
                         joint[1,"observed"][:,0],joint[1,"observed"][:,3]),1)
        error=float((1-(got*expected).sum(-1)).abs().max())
        if error>.001:
            raise ValueError(f"Final EOT reconstruction disagreed with the actual block {error}")
    return joint,torch.stack(native_mass,1),error


def two_factors(four):
    low_old,low_new,high_old,high_new=four.unbind(1)
    mass=((high_old-low_old)+(high_new-low_new))*.5
    message=((low_new-low_old)+(high_new-high_old))*.5
    return mass,message


@torch.inference_mode()
def run(w,batch):
    cfg=runtime(w,"sketchy")
    names=read_words(cfg.vocabulary)
    if len(names)!=20931:
        raise ValueError("Vocabulary changed")
    parent=w/"cache"/TOKENS/"paired.pt"
    parent_meta=read(parent.with_suffix(".json"))
    reference=torch.load(parent,mmap=True,weights_only=True)[11][:,0]
    meta={"checkpoint":CHECKPOINT,"vocabulary":checksum(cfg.vocabulary),"parent":parent_meta["signature"],
          "partition":"final block EOT per-head name mass versus conditional name message, two backgrounds",
          "methods":["mass","message","common-head mass","rotated-head mass","whole noun","source-background mass","source-background message"]}
    sig=digest(meta)
    file=w/"cache"/FAMILY/"interventions.pt"
    if file.is_file() and read(file.with_suffix(".json"))["signature"]==sig:
        return
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
    tok=open_clip.get_tokenizer("ViT-B-32")
    all_paths=torch.empty((len(names),len(meta["methods"]),512),dtype=torch.float16)
    stats=torch.zeros(3,dtype=torch.float64)
    squares=torch.zeros(2,dtype=torch.float64)
    worst=torch.zeros(2,device="cuda")
    for left in range(0,len(names),batch):
        words=names[left:left+batch]
        pooled={}
        for sk,ph in zip(*DEFAULT_DOMAIN_TEMPLATES,strict=True):
            vals,probs,error=trace(model,tokenize_pair(tok,words,sk,ph),check=left%(24*batch)==0)
            worst[0]=max(float(worst[0]),error)
            if left==0:
                a,b=probs.unbind(-1)
                if not torch.isfinite(probs).all() or torch.any(a<0) or torch.any(b>1):
                    raise ValueError("Invalid noun attention probabilities")
            stats+=torch.stack((probs[...,0].mean((1,2)).sum(),probs[...,1].mean((1,2)).sum(),
                                (probs[...,1]-probs[...,0]).abs().mean((1,2)).sum())).double().cpu()/2
            for key,tensor in vals.items():
                pooled[key]=pooled.get(key,0)+tensor
        factors={k:two_factors(F.normalize(v,dim=-1)) for k,v in pooled.items()}
        mass=(factors[0,"observed"][0]+factors[1,"observed"][0])*.5
        message=(factors[0,"observed"][1]+factors[1,"observed"][1])*.5
        whole=(F.normalize(pooled[0,"observed"],dim=-1)[:,3]-F.normalize(pooled[0,"observed"],dim=-1)[:,0]
               +F.normalize(pooled[1,"observed"],dim=-1)[:,3]-F.normalize(pooled[1,"observed"],dim=-1)[:,0])*.5
        common=(factors[0,"common"][0]+factors[1,"common"][0])*.5
        swapped=(factors[0,"head_swap"][0]+factors[1,"head_swap"][0])*.5
        block=torch.stack((mass,message,common,swapped,whole,*factors[1,"observed"]),1)
        all_paths[left:left+len(words)]=block.half().cpu()
        expected=reference[left:left+len(words)].cuda().float()
        squares+=torch.stack(((whole-expected).square().sum(),expected.square().sum())).double().cpu()
        closure=float((whole-mass-message).abs().amax())
        worst[1]=max(float(worst[1]),closure)
        if closure>2e-6:
            raise ValueError("The proposed factors failed to reproduce the full noun effect")
        if left%(24*batch)==0:
            print("factorial name cache",left,len(names),"cos/closure",worst.tolist(),flush=True)
    rms=float((squares[0]/squares[1]).sqrt())
    if rms>.09:
        raise ValueError(f"New readout changed the original contextual noun effect by {rms}")
    save_tensor(all_paths,file)
    atomic_json(file.with_suffix(".json"),meta|{"signature":sig,"rows":len(names),"worst_cos_or_closure":worst.tolist(),
                "relative_rms_to_archived_noun_path":rms,"mean_target_mass_source_mass_absolute_change":(stats/len(names)).tolist()})
    print("stored name attention paths",file.stat().st_size,"relative RMS",rms,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--batch",type=int,default=192)
    a=p.parse_args()
    run(a.workspace,a.batch)
