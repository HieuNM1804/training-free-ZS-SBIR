"""Prepare independently reproducible responses to held pixel evidence from the original B/32 patch projection."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .data import Images, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .recoverable_patch_information import AMOUNTS,FAMILY,MODES,OFFSETS,PatchInformation
from .response_geometry_views import FAMILY as GEOMETRY
from .retrieval_diagnosis import stratified_sample


def names(rows):
    return [f"{m}/{a:g}" for m,a in rows]


def make_index(w,dataset,inputs,stage):
    cfg,qp,qs,pp,gs,*_=inputs
    if stage=="pilot-query":
        ids=[qi[stratified_sample(y,36,63241)] for qi,y in qs.values()]
        return torch.cat(ids).unique(sorted=True)
    if stage=="all-query":
        return torch.arange(len(qp))
    if stage=="all-photo":
        return torch.arange(len(pp))
    if dataset!="sketchy":
        raise ValueError("Matched diagnostic candidates are defined on the two development splits")
    spec=w/"cache"/GEOMETRY/dataset/"plan.pt"
    pair=torch.load(spec,weights_only=True,mmap=True)
    items=[]
    for s,row in pair["splits"].items():
        si=row["ids"]
        qids=qs[s][0][si]
        pids=gs[s][0]
        by_name={ (Path(pp[int(i)]).parent.name,Path(pp[int(i)]).stem): int(i) for i in pids}
        trueid=[]
        for i in qids.tolist():
            p=Path(qp[i]); stem,sep,version=p.stem.rpartition("-")
            source=by_name.get((p.parent.name,stem))
            if sep and version.isdigit() and source is not None:trueid.append(source)
        items.extend((torch.tensor(trueid),row["pairs"][row["pairs"]>=0]))
    return torch.cat(items).unique(sorted=True)


@torch.inference_mode()
def run(w,dataset,stage):
    inputs=semantic_inputs(w,dataset)
    cfg,qp,qs,pp,gs,q,p,*_=inputs
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ids=make_index(w,dataset,inputs,stage)
    is_query=stage.endswith("query")
    paths=qp if is_query else pp
    previous=q if is_query else p
    selected=[paths[i] for i in ids.tolist()]
    rows=[(m,a) for m in MODES for a in AMOUNTS]
    folder=w/"cache"/FAMILY/dataset/"native-anchor-v2"/stage
    folder.mkdir(parents=True,exist_ok=True)
    stamp={"checkpoint":CHECKPOINT,"paths":digest(list(map(str,selected))),"readouts":names(rows),"offsets":list(map(list,OFFSETS)),
           "operation":"achromatic native-null intervention and exact native CLIP convolution anchor v2"}
    sig=digest(stamp)
    if (folder/"manifest.json").exists():
        if read(folder/"manifest.json")["signature"]!=sig:raise ValueError("Feature manifest changed")
        if (folder/"done").exists():return
    cut=256
    save_tensor({"ids":ids},folder/"ids.pt")
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    reader=PatchInformation(vision)
    atomic_json(folder/"manifest.json",stamp|{"signature":sig,"tokenizer":reader.info,"count":len(ids),"shard":cut})
    print(stage,dataset,len(ids),reader.info,flush=True)
    for first in range(0,len(ids),cut):
        dest=folder/f"{first:06}.pt"
        if dest.exists():continue
        pick=ids[first:first+cut]
        loader=DataLoader(Images(selected[first:first+cut],normal_transform(),"rgb"),batch_size=36,num_workers=cfg.workers,
                          worker_init_fn=configure_image_worker,pin_memory=True)
        embeds,changes,diagnosis=[],[],[]
        tests=[]
        offset=0
        for pixels in loader:
            z,variants,fields,info=reader.measure(pixels.cuda(non_blocking=True),rows,check=(offset==0))
            ref=previous[pick[offset:offset+len(z)]].cuda()
            err=float((1-(ref*z).sum(1)).abs().max())
            if err>.002:raise ValueError(f"Source or native feature changed at row {first+offset}, cosine error {err:g}")
            if info:tests.append(info+[err])
            embeds.append(z.cpu())
            changes.append(torch.stack([variants[x]-z for x in rows],1).half().cpu())
            diagnosis.append(fields.cpu())
            offset+=len(z)
        if offset!=len(pick):raise ValueError("Lost a source image")
        arr=torch.cat(diagnosis)
        save_tensor({"ids":pick,"native":torch.cat(embeds),"change":torch.cat(changes),"diagnosis":arr,"checks":torch.tensor(tests)},dest)
        print(stage,first+len(pick),len(ids),"invisible energy fraction",float(arr[:,0].mean()),flush=True)
    (folder/"done").touch()


def restore(w,dataset,stage):
    base=w/"cache"/FAMILY/dataset/"native-anchor-v2"/stage
    desc=read(base/"manifest.json")
    ids=torch.load(base/"ids.pt",mmap=True,weights_only=True)["ids"]
    if not (base/"done").exists() or len(ids)!=desc["count"]:raise ValueError("Incomplete feature bank")
    files=[torch.load(base/f"{offset:06}.pt",mmap=True,weights_only=True) for offset in range(0,len(ids),desc["shard"])]
    data={k:torch.cat([p[k] for p in files]) for k in ("ids","native","change","diagnosis","checks")}
    if not torch.equal(ids,data["ids"]):raise ValueError("Unexpected order")
    return data,desc


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    cli.add_argument("--stage",choices=("pilot-query","diagnostic-photo","all-photo","all-query"),required=True)
    cli_args=cli.parse_args()
    run(cli_args.workspace,cli_args.dataset,cli_args.stage)
