"""Build a target-only table of early neural responses paired with later semantic descriptors from the same frozen image patches."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import CLIP_STD, Images, normal_transform, configure_image_worker
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .patch_activation_source import allocate
from .patch_mediation_inputs import attention, mlp
from .patch_response_mediation import read_block


FAMILY="semantic-patch-neuron-replacement"
LAYERS=(2,3)
GROUPS=("curvature","sensor","native","matched","sensor_matched","random")
DISCOVERY,BANK,HOLDOUT=192,768,144
SEED=40483
MAX_UNITS=32


def move_pixels(x,dy,dx):
    h,w=x.shape[-2:]
    b=max(abs(dx),abs(dy))
    y=F.pad(x,(b,b,b,b),mode="reflect")
    return y[:,:,b-dy:b-dy+h,b-dx:b-dx+w]


def select_ids(absolute):
    g=torch.Generator().manual_seed(SEED)
    n=DISCOVERY+BANK+HOLDOUT
    ids=absolute[torch.randperm(len(absolute),generator=g)[:n]]
    return ids[:DISCOVERY],ids[DISCOVERY:DISCOVERY+BANK],ids[-HOLDOUT:]


def batches(files,ids,workers,bsize=32):
    ds=Images([files[int(i)] for i in ids],normal_transform(),"rgb")
    return DataLoader(ds,batch_size=bsize,num_workers=workers,pin_memory=True,worker_init_fn=configure_image_worker)


def discover(v,loader):
    raw={l:[] for l in LAYERS}
    noisy={l:[] for l in LAYERS}
    mag={l:[] for l in LAYERS}
    axes=((0,2),(2,0))
    noise_generator=torch.Generator().manual_seed(73163)
    for px in loader:
        px=px.cuda(non_blocking=True)
        b=len(px)
        # Two independent realizations of a tiny achromatic sensor perturbation in physical RGB units.
        noise=torch.randn((2,b,1,224,224),generator=noise_generator).to(px)*(2/255)
        noise=noise/px.new_tensor(CLIP_STD)[None,None,:,None,None]
        views=torch.cat([*[move_pixels(px,sign*y,sign*x) for y,x in axes for sign in (1,-1)],
                         *[px+sign*n for n in noise for sign in (1,-1)]])
        h=v._embeds(views.to(v.conv1.weight.dtype))
        for l,block in enumerate(v.transformer.resblocks[:max(LAYERS)],1):
            y=attention(v,block,h)
            if l in LAYERS:
                p=y.reshape(8,b,50,768)[:,:,1:]
                measures=[allocate(block,p[2*j],p[2*j+1]) for j in range(4)]
                # Normalize each sample and direction before combining the two spatial axes.
                mass=[a.abs()/a.abs().sum(1,keepdim=True).clamp_min(1e-12) for a,_,_ in measures]
                raw[l].append(torch.stack(mass[:2]).cpu())
                noisy[l].append(torch.stack(mass[2:]).cpu())
                mag[l].append(torch.stack([m for _,_,m in measures[:2]]).cpu())
            h=mlp(block,y)
    selected,audit={},{}
    rng=torch.Generator().manual_seed(52671)
    for l in LAYERS:
        faces=torch.cat(raw[l],1)
        sensor=torch.cat(noisy[l],1)
        strength=faces.mean((0,1))
        sensor_strength=sensor.mean((0,1))
        activities=torch.cat(mag[l],1).mean((0,1))
        order=strength.argsort(descending=True,stable=True)
        large=activities.argsort(descending=True,stable=True)
        sensor_order=sensor_strength.argsort(descending=True,stable=True)
        def matched_units(ranking):
            blocked=torch.zeros_like(activities,dtype=torch.bool)
            blocked[ranking[:MAX_UNITS]]=True
            matches=[]
            for j in ranking[:MAX_UNITS]:
                distances=(activities.clamp_min(1e-24).log()-activities[j].clamp_min(1e-24).log()).abs()
                k=distances.masked_fill(blocked,torch.inf).argmin()
                blocked[k]=True;matches.append(k)
            return torch.stack(matches)
        choices={"curvature":order[:MAX_UNITS],"sensor":sensor_order[:MAX_UNITS],"native":large[:MAX_UNITS],
                 "matched":matched_units(order),"sensor_matched":matched_units(sensor_order),
                 "random":torch.randperm(len(order),generator=rng)[:MAX_UNITS]}
        selected[l]=choices
        audit[l]={"units":{k:z.tolist() for k,z in choices.items()},"split_and_axis_recurrence":{}}
        for k in (4,8,32):
            a=faces[:,::2].mean((0,1)).topk(k).indices
            z=faces[:,1::2].mean((0,1)).topk(k).indices
            ax=faces.mean(1).topk(k).indices
            audit[l]["split_and_axis_recurrence"][k]={"alternating_photos_overlap":int(torch.isin(a,z).sum()),
                "horizontal_vertical_overlap":int(torch.isin(ax[0],ax[1]).sum()),
                "sensor_independent_noise_overlap":int(torch.isin(sensor[0].mean(0).topk(k).indices,sensor[1].mean(0).topk(k).indices).sum()),
                "sensor_translation_overlap":int(torch.isin(sensor_order[:k],order[:k]).sum()),
                "sensor_discovery_share":float(sensor_strength[sensor_order[:k]].sum()),
                "fraction_native_activity_matched":float(activities[choices["matched"][:k]].sum()/activities[order[:k]].sum()),
                "discovery_share":float(strength[order[:k]].sum())}
    return selected,audit


def native(v,pixels,units):
    x=pixels.to(v.conv1.weight.dtype)
    h=v._embeds(x)
    values,early,states={},{},{}
    for l,block in enumerate(v.transformer.resblocks,1):
        if l in LAYERS:
            post=attention(v,block,h)
            normalized=block.ln_2(post)
            fc,act,last=tuple(block.mlp.children())
            a=act(fc(normalized))
            for group,ids in units[l].items():
                values[l,group]=a[:,1:,ids]
            early[l]=F.normalize(normalized[:,1:].float(),dim=-1).half()
            h=post+block.ls_2(last(a))
            states[l]=(post,h)
        else:
            h=read_block(v,block,h)
    late=F.normalize((v.ln_post(h)@v.proj).float(),dim=-1)
    return {"late":late[:,1:].half(),"scene":late[:,0],"early":early,"values":values,"states":states}


def nearest(rows,table,n=32):
    keys=rows.reshape(-1,rows.size(-1)).half()
    table=table.reshape(-1,table.size(-1))
    out=[]
    for q in keys.split(384):
        out.append((q@table.T).topk(n,dim=1).indices)
    return torch.cat(out).reshape(*rows.shape[:-1],n)


def read_donors(source,bank,perm):
    # Every value in the table is an actual activation of a real target photo.
    late=nearest(source["late"],bank["late"])
    scene=nearest(source["scene"][:,None].half(),bank["scene"])
    predicted={}
    for l in LAYERS:
        early=nearest(source["early"][l],bank["early"][l])
        for group in GROUPS:
            values=bank["values"][l,group]
            flat=values.flatten(0,1)
            def sample(index,k):
                return flat[index[:,:,:k]].float().mean(2)
            cur={"late4":sample(late,4),"late32":sample(late,32),"early":sample(early,32),
                 "shuffled":sample(perm[late],32),"mean":flat.float().mean(0)[None,None],
                 "zero":flat.new_zeros(1,1,flat.size(-1)).float()}
            # Same token grid position, borrowed from globally similar photo donors.
            cur["scene"]=values[scene[:,0,:32]].float().mean(1)
            predicted[l,group]=cur
    return predicted


def to_gpu(bank):
    return {k:({i:t.cuda() for i,t in v.items()} if isinstance(v,dict) else v.cuda()) for k,v in bank.items()}


@torch.inference_mode()
def index(w,ds):
    cfg,_,qs,pp,gs,*_=semantic_inputs(w,ds)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    folder=w/"cache"/FAMILY
    folder.mkdir(parents=True,exist_ok=True)
    for s,(_,g) in gs.items():
        pi=gs[s][0]
        start,donor,test=select_ids(pi)
        sign=digest({"model":CHECKPOINT,"gallery":list(map(str,[pp[int(i)] for i in pi])),"sampling":[SEED,DISCOVERY,BANK,HOLDOUT],
                     "probe":"opposite translations and independent small achromatic sensor observations; normalized per image/direction v2"})
        file=folder/(s+"-memory-v2.pt")
        record=file.with_suffix(".json")
        if file.is_file() and record.is_file():
            if read(record)["signature"]!=sign:raise ValueError("Mismatch in target gallery")
            continue
        units,measured=discover(v,batches(pp,start,cfg.workers,16))
        gpu_units={l:{k:i.cuda() for k,i in u.items()} for l,u in units.items()}
        rows=[]
        for x in batches(pp,donor,cfg.workers):
            z=native(v,x.cuda(non_blocking=True),gpu_units)
            rows.append({k:({i:t.cpu() for i,t in b.items()} if isinstance(b,dict) else b.cpu())
                         for k,b in z.items() if k!="states"})
        table={k:({i:torch.cat([r[k][i] for r in rows]) for i in first} if isinstance(first,dict)
                  else torch.cat([r[k] for r in rows])) for k,first in rows[0].items()}
        save_tensor({"discovery":start,"donor":donor,"holdout":test,"units":units,"table":table},file)
        atomic_json(record,{"signature":sign,"split":s,"gallery_size":len(pi),"recurrence":measured,"candidates":len(donor)*49})
        print("indexed",s,measured[2]["split_and_axis_recurrence"][8],measured[3]["split_and_axis_recurrence"][8],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    a=cli.parse_args()
    index(a.workspace,a.dataset)
