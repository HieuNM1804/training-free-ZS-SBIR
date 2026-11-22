"""Locate the creation of a sign-independent CLIP response by recomputing the frozen tail from counterfactual midpoints."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .data import Images, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .recoverable_patch_information import PatchInformation
from .recoverable_patch_inputs import restore


FAMILY="patch-response-mediation"
TYPES=("hidden","shuffled","isotropic_matched","isotropic_uniform")
CUTS=(0,2,4,6,8,10,12)


def read_block(v,block,z):
    # Keep the native token axis expected by this checkpoint, independently of installed OpenCLIP conventions.
    if v.transformer.batch_first:
        return block(z)
    return block(z.transpose(0,1).contiguous()).transpose(0,1)


def stem(v,native):
    tokens=native.flatten(2).transpose(1,2).to(v.conv1.weight.dtype)
    cls=v.class_embedding.to(tokens).expand(len(tokens),1,-1)
    state=torch.cat((cls,tokens),1)+v.positional_embedding.to(tokens)
    return v.ln_pre(v.patch_dropout(state))


def project(v,state):
    x,_=v._pool(state)
    return F.normalize((x@v.proj).float())


def rand_dir(native):
    generator=torch.Generator(device="cpu").manual_seed(232843)
    direction=torch.randn((1,768,7,7),generator=generator).to(native)
    return F.normalize(direction,dim=1)


def directions(reader,px,native):
    _,d,_,_,_=reader.parts(px)
    iso=rand_dir(native)
    perpatch=d["hidden"].norm(dim=1,keepdim=True)
    # A distinct, image-only control uses native patch magnitude and a scalar from the earlier diagnostic.
    uniform=.005*native.flatten(1).norm(dim=1,keepdim=True)[:,:,None,None]/7
    return (d["hidden"],d["rotated_hidden"],iso*perpatch,iso*uniform)


def even_odd(x):
    native=x[0:1]
    pos=x[1::2]
    neg=x[2::2]
    even=.5*(pos.float()+neg.float())-native.float()
    odd=.5*(pos.float()-neg.float())
    return even,odd


def layer_stats(state):
    even,odd=even_odd(state)
    p=even[:,:,1:].square().sum(-1)
    q=odd[:,:,1:].square().sum(-1)
    c=even[:,:,0].square().sum(-1)
    d=odd[:,:,0].square().sum(-1)
    den=(p.sum(-1)+q.sum(-1)).clamp_min(1e-20)
    native=state[0,:,1:].float().norm(dim=-1)
    top=native.argmax(-1).view(1,-1,1).expand(len(p),-1,-1)
    shape=even[:,:,1:].flatten(2)
    return torch.stack((p.sum(-1)/den,c/(c+d).clamp_min(1e-20),p.max(-1).values/p.sum(-1).clamp_min(1e-20),
                        (p/p.sum(-1,keepdim=True).clamp_min(1e-20)).gather(-1,top).squeeze(-1),
                        F.cosine_similarity(shape,shape[:1].expand_as(shape),dim=-1),
                        c.sqrt(),d.sqrt()),-1)


def trace(v,reader,px):
    n=v.conv1(px.to(v.conv1.weight.dtype)).float()
    delta=directions(reader,px,n)
    originals=torch.stack([n,*[n+sign*2*d for d in delta for sign in (1,-1)]])
    b=len(px)
    group=len(originals)
    live=stem(v,originals.flatten(0,1)).reshape(group,b,50,768)
    studies=[layer_stats(live)]
    names=[]
    def append_cut(state,l):
        even,_=even_odd(state[:group])
        midpoint=state[:1].float()+even
        names.extend((m,l) for m in TYPES)
        return torch.cat((state,midpoint.to(state.dtype)))
    live=append_cut(live,0)
    for l,block in enumerate(v.transformer.resblocks,1):
        all_states=live.flatten(0,1)
        # The most expensive final block has many counterfactual streams, all sharing one frozen set of weights.
        out=torch.cat([read_block(v,block,part.contiguous()) for part in all_states.split(192)])
        live=out.reshape(-1,b,50,768)
        studies.append(layer_stats(live[:group]))
        if l in CUTS:live=append_cut(live,l)
    f=project(v,live.flatten(0,1)).reshape(-1,b,512)
    base=f[0]
    exact=F.normalize(v(px.to(v.conv1.weight.dtype)).float())
    native_gap=(base-exact).norm(dim=1)
    even,odd=even_odd(f[:group])
    projection={}
    for (mode,l),emb in zip(names,f[group:],strict=True):
        d=emb-base
        e=even[TYPES.index(mode)]
        mag=e.square().sum(1).clamp_min(1e-20)
        projection[f"{mode}/{l}"]=torch.stack(((d*e).sum(1)/mag,F.cosine_similarity(d,e,dim=1),
                                              d.norm(dim=1),e.norm(dim=1),(d-e).norm(dim=1)),1)
    endpoint=torch.stack((even.norm(dim=-1),odd.norm(dim=-1),
                          F.cosine_similarity(even,even[:1].expand_as(even),dim=-1)),dim=-1)
    return torch.stack(studies,2).permute(1,0,2,3),projection,endpoint.transpose(0,1),native_gap


@torch.inference_mode()
def run(w,count):
    cfg,qp,_,pp,*_=semantic_inputs(w,"sketchy")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    reader=PatchInformation(v)
    folder=w/"results"/FAMILY
    report,data={},{}
    rng=torch.Generator().manual_seed(90185)
    for ds,stage,paths in (("sketch","pilot-query",qp),("photo","diagnostic-photo",pp)):
        features,_=restore(w,"sketchy",stage)
        pool=features["ids"]
        ids=pool[torch.randperm(len(pool),generator=rng)[:count]].sort().values
        del features
        dataset=Images([paths[int(i)] for i in ids],normal_transform(),"rgb")
        dl=DataLoader(dataset,batch_size=12,num_workers=cfg.workers,worker_init_fn=configure_image_worker,pin_memory=True)
        observations,terms,terminal,replay=[],defaultdict(list),[],[]
        for i,pixels in enumerate(dl):
            x,a,finish,check=trace(v,reader,pixels.cuda(non_blocking=True))
            observations.append(x.cpu());terminal.append(finish.cpu());replay.append(check.cpu())
            for k,t in a.items():terms[k].append(t.cpu())
            if i%8==0:print(ds,i*12,len(ids),flush=True)
        layer=torch.cat(observations)
        endpoint=torch.cat(terminal)
        checks=torch.cat(replay)
        cuts={k:torch.cat(t) for k,t in terms.items()}
        if not torch.isfinite(layer).all() or float(checks.max())>.012:raise ValueError("Independent native feature path changed")
        report[ds]={"images":len(ids),"native_L2_max":float(checks.max()),"native_L2_median":float(checks.median()),
                    "layers":{m:layer[:,i].median(0).values.tolist() for i,m in enumerate(TYPES)},
                    "cuts":{k:c.median(0).values.tolist() for k,c in cuts.items()},
                    "endpoint":{m:endpoint[:,i].median(0).values.tolist() for i,m in enumerate(TYPES)}}
        data[ds]={"image_ids":ids,"layers":layer,"cuts":cuts,"endpoint":endpoint,"native_L2":checks}
    path=folder/"midpoint-traces.pt"
    save_tensor(data,path)
    atomic_json(path.with_suffix(".json"),{"revision":revision(),"per_query_sha256":checksum(path),
               "seed":90185,"cuts":CUTS,"directions":TYPES,"amplitude":2,"domains":report,
               "layer_fields":["patch_even_energy_share","cls_even_energy_share","largest_patch_share_of_even_energy",
                               "largest_native_token_share_of_even_energy","patch_even_cosine_to_true_direction","CLS_even_norm","CLS_odd_norm"],
               "cut_fields":["projection_on_terminal_even_response","cosine_to_terminal","actual_replayed_norm","terminal_norm","unexplained_norm"],
               "endpoint_fields":["even_norm","odd_norm","even_cosine_to_true_direction"],
               "GPU":torch.cuda.get_device_name(),"scope":"mechanism diagnosis only, no recruitment of other queries at inference"})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--count",type=int,default=240)
    args=cli.parse_args()
    run(args.workspace,args.count)
