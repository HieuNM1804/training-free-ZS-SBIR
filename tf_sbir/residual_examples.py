"""Produce a deterministic cross-split contact sheet of the old and newly locked first ranks."""

import argparse
from pathlib import Path
import random

from PIL import Image,ImageDraw,ImageOps
import torch

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json,read
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS,UNSEEN_CLASSES
from .residual_benchmark import INCUMBENT


def make(workspace):
    root=workspace/"results"; out=root/"residual-study"
    decision=read(out/"final-selection.json")
    primary=decision["primary"]
    selections=[]
    rng=random.Random(16883)
    for dataset in ("sketchy","tuberlin"):
        a=runtime(workspace,dataset)
        splits=SKETCHY_SPLITS if dataset=="sketchy" else ("tuberlin",)
        qp,qs=merge_inputs(a.root,splits,"sketch"); pp,gs=merge_inputs(a.root,splits,"photo")
        for s in splits:
            held=s=="tuberlin"
            qi,ql=qs[s]; pi,pl=gs[s]
            old=decision["families"][INCUMBENT]["scores"][0]["method"]
            new=decision["families"][primary]["scores"][0]["method"]
            before=root/INCUMBENT/("heldout" if held else "sketchy")/"top10"/f"{s}-{old}.pt"
            if primary==INCUMBENT:
                after=before
            else:
                candidates=("confirm/heldout",) if held else ("sketchy","confirm/sketchy")
                after=next(p for t in candidates if (p:=root/primary/t/"top10"/f"{s}-{new}.pt").exists())
            a1=torch.load(before,weights_only=True).long()[:,0]; a2=torch.load(after,weights_only=True).long()[:,0]
            olds,news=pl[a1]==ql,pl[a2]==ql
            for name,mask in (("rescued",~olds&news),("regressed",olds&~news),("still wrong",~olds&~news)):
                ids=mask.nonzero().flatten().tolist()
                if not ids:
                    continue
                idx=rng.choice(ids)
                paths=[qp[int(qi[idx])], pp[int(pi[a1[idx]])], pp[int(pi[a2[idx]])]]
                labels=[UNSEEN_CLASSES[s][i] for i in (int(ql[idx]),int(pl[a1[idx]]),int(pl[a2[idx]]))]
                selections.append((s,name,paths,labels))
    image=Image.new("RGB",(890,35+len(selections)*132),"white")
    draw=ImageDraw.Draw(image)
    for x,name in zip((8,153,396,640),("Split / case","Sketch","Previous first photo","Selected first photo")):
        draw.text((x,10),name,fill="#233443")
    examples=[]
    for row,(split,case,paths,labels) in enumerate(selections):
        y=35+132*row
        draw.line((8,y,872,y),fill="#c5cbd1")
        draw.text((8,y+40),split,fill="#263653")
        draw.text((8,y+57),case,fill="#263653")
        for x,path,label in zip((151,394,637),paths,labels):
            with Image.open(path) as im:
                thumb=ImageOps.contain(im.convert("RGB"),(219,92))
            image.paste(thumb,(x+(219-thumb.width)//2,y+4))
            draw.text((x+2,y+96),label,fill="#111820")
            draw.text((x+2,y+111),Path(path).name,fill="#37424b")
        examples.append({"split":split,"case":case,"class_labels":labels,
                         "paths":["/".join(Path(p).parts[-3:]) for p in paths]})
    image.save(out/"qualitative-cases.jpg",quality=90)
    atomic_json(out/"qualitative-cases.json",{"seed":16883,"examples":examples})


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    make(ap.parse_args().workspace)
