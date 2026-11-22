"""Intervene at frozen attention and MLP boundaries and keep enough information to compare scores before and after each boundary."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .patch_response_mediation import FAMILY, project, read_block, stem
from .recoverable_patch_information import PatchInformation
from .recoverable_patch_inputs import make_index


TYPES=("hidden","shuffled","uniform_a","uniform_b")
CUTS=("input","a1","m1","a2","m2","a3","m3","a4","m4")
ARMS=[(m,t) for m in TYPES for t in (*CUTS,"positive","negative")]
SEEDS=(232843,346673)


def random_axes(native):
    axes=[]
    for seed in SEEDS:
        rng=torch.Generator().manual_seed(seed)
        noise=torch.randn((1,768,7,7),generator=rng).to(native)
        axes.append(F.normalize(noise,dim=1))
    return axes


def attention(v,block,x):
    state=x if v.transformer.batch_first else x.transpose(0,1).contiguous()
    query=block.ln_1(state)
    y=state+block.ls_1(block.attention(q_x=query,k_x=None,v_x=None,attn_mask=None))
    return y if v.transformer.batch_first else y.transpose(0,1)


def mlp(block,x):
    return x+block.ls_2(block.mlp(block.ln_2(x)))


def grouped(block,states):
    shape=states.shape
    y=torch.cat([block(chunk.contiguous()) for chunk in states.flatten(0,1).split(192)])
    return y.reshape(shape)


def extract(v,reader,pixels):
    t=v.conv1(pixels.to(v.conv1.weight.dtype)).float()
    _,d,_,_,_=reader.parts(pixels)
    magnitude=.005*t.flatten(1).norm(dim=1)[:,None,None,None]/7
    perturb=(d["hidden"],d["rotated_hidden"],*[direction*magnitude for direction in random_axes(t)])
    b=len(pixels)
    x=stem(v,torch.cat([t,*[t+sg*2*e for e in perturb for sg in (1,-1)],t])).reshape(10,b,50,768)
    indexes={}
    def insert(state,tag):
        mid=.5*(state[1:9:2].float()+state[2:9:2].float())
        for j,mode in enumerate(TYPES):indexes[mode,tag]=len(state)+j
        return torch.cat((state,mid.to(state.dtype)))
    x=insert(x,"input")
    for i,block in enumerate(v.transformer.resblocks,1):
        if i<=4:
            x=grouped(lambda h:attention(v,block,h),x)
            x=insert(x,f"a{i}")
            x=grouped(lambda h:mlp(block,h),x)
            x=insert(x,f"m{i}")
        else:
            x=grouped(lambda h:read_block(v,block,h),x)
    z=project(v,x.flatten(0,1)).reshape(-1,b,512)
    ref=z[0]
    for j,mode in enumerate(TYPES):
        indexes[mode,"positive"]=1+2*j
        indexes[mode,"negative"]=2+2*j
    result=torch.stack([z[indexes[arm]]-ref for arm in ARMS],1)
    zero=(z[9]-ref).norm(dim=1)
    exact=F.normalize(v(pixels.to(v.conv1.weight.dtype)).float())
    original=(ref-exact).norm(dim=1)
    if zero.max()>.003 or original.max()>.012:
        raise ValueError("A zero intervention or native implementation changed")
    return ref,result,torch.stack((zero,original),1)


@torch.inference_mode()
def run(w,ds,stage):
    items=semantic_inputs(w,ds)
    cfg,qp,qs,pp,gs,q,p,*_=items
    if stage=="trace-photo":
        trace=torch.load(w/"results"/FAMILY/"midpoint-traces.pt",mmap=True,weights_only=True)
        ids=trace["photo"]["image_ids"]
    else:
        ids=make_index(w,ds,items,stage)
    paths=qp if stage.endswith("query") else pp
    reference=q if stage.endswith("query") else p
    desc={"ids":digest([str(paths[int(i)]) for i in ids]),"model":CHECKPOINT,"arms":ARMS,"seeds":SEEDS,
          "directions":"2 * hidden, matched shuffle, two independent isotropic full-grid directions at relative norm .005"}
    folder=w/"cache"/FAMILY/ds/stage
    folder.mkdir(parents=True,exist_ok=True)
    signature=digest(desc)
    if (folder/"protocol.json").is_file():
        if read(folder/"protocol.json")["signature"]!=signature:raise ValueError("Different extraction")
        if (folder/"done").is_file():return
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    reader=PatchInformation(v)
    atomic_json(folder/"protocol.json",desc|{"signature":signature,"images":len(ids)})
    save_tensor({"ids":ids},folder/"ids.pt")
    for start in range(0,len(ids),128):
        file=folder/f"batch-{start:06}.pt"
        if file.exists():continue
        index=ids[start:start+128]
        ds0=Images([paths[int(i)] for i in index],normal_transform(),"rgb")
        dl=DataLoader(ds0,batch_size=12,num_workers=cfg.workers,pin_memory=True,worker_init_fn=configure_image_worker)
        vecs,shift,tests=[],[],[]
        for px in dl:
            ref,delta,diag=extract(v,reader,px.cuda(non_blocking=True))
            vecs.append(ref.cpu());shift.append(delta.half().cpu());tests.append(diag.cpu())
        native=torch.cat(vecs)
        err=(1-(native*reference[index]).sum(1)).abs().max()
        if err>.002:raise ValueError("Image ordering or source changed")
        save_tensor({"ids":index,"native":native,"shift":torch.cat(shift),"numerical_controls":torch.cat(tests)},file)
        print(stage,start+len(index),len(ids),float(err),flush=True)
    (folder/"done").touch()


def load(w,ds,stage):
    folder=w/"cache"/FAMILY/ds/stage
    if not (folder/"done").is_file():raise ValueError("Incomplete intervention bank")
    ids=torch.load(folder/"ids.pt",weights_only=True)["ids"]
    blocks=[torch.load(f,mmap=True,weights_only=True) for f in sorted(folder.glob("batch-*.pt"))]
    data={k:torch.cat([b[k] for b in blocks]) for k in blocks[0]}
    if not torch.equal(ids,data["ids"]):raise ValueError("Image order changed")
    return data,read(folder/"protocol.json")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--stage",choices=("pilot-query","trace-photo","all-query"),required=True)
    opts=p.parse_args()
    run(opts.workspace,opts.dataset,opts.stage)
