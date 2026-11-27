"""Prepare blinded review boards of actual pilot error pairs; do not publish the raw photos."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import torch

from .concept_benchmark import atomic_json,checksum
from .data import UNSEEN_CLASSES
from .kaggle_unseen_features import DEFAULT_CACHE,index_file
from .native_block_probe import SEED
from .wordnet_renaming_pilot import OUT,SPLITS


def key(*parts):
    return hashlib.sha256("/".join(map(str,(SEED,*parts))).encode()).digest()


def candidates(split,index,revision):
    pieces=[torch.load(OUT/f"{split}-{i}.pt",weights_only=True,mmap=True) for i in (0,1)]
    mfile=OUT/"matched"/f"{split}.pt"
    meta=json.loads(mfile.with_suffix(".json").read_bytes())
    if meta["revision"]!=revision or checksum(mfile)!=meta["tensor_sha256"]:
        raise ValueError("Unverified sketch and source metadata")
    matched=torch.load(mfile,mmap=True,weights_only=True)
    order=torch.cat([p["positions"] for p in pieces]).argsort()
    def combine(k):
        return torch.cat([p[k] for p in pieces])[order]
    qids=combine("query_ids")
    if not torch.equal(qids,matched["query_ids"]):
        raise ValueError("Sketch ordering failed to match")
    rgb=pieces[0]["gallery_ids"][combine("opponent")]
    dy=combine("candidate_deltas")
    classes=combine("classes")
    paths={k:index[k]["paths"] for k in ("sketch","photo")}
    rows={}
    for i in range(len(qids)):
        source=matched["source_gallery_ids"][i].item()
        pos,neg=rgb[i].tolist()
        if pos==source or dy[i,0]*dy[i,1]<=0:
            continue
        group="both_prefer_true" if dy[i,0]>0 else "both_prefer_wrong"
        cls=UNSEEN_CLASSES[split][classes[i].item()]
        address=qids[i].item()
        rows.setdefault(cls,{}).setdefault(group,[]).append({"split":split,"class":cls,
                    "sketch_id":address,"source_id":source,"true_id":pos,"false_id":neg,"group":group,
                    "sketch":paths["sketch"][address],"true":paths["photo"][pos],"false":paths["photo"][neg]})
    return rows


def select(pools):
    used_photos,used_sketches,used_classes=set(),set(),set()
    selected=[]
    for split,rows in pools.items():
        start=len(selected)
        for cls in sorted(rows,key=lambda c:key(split,c)):
            if cls in used_classes or set(rows[cls])!={"both_prefer_wrong","both_prefer_true"}:
                continue
            trial=[]
            for group in ("both_prefer_wrong","both_prefer_true"):
                blocked=used_photos|{r[k] for r in trial for k in ("source_id","true_id","false_id")}
                for row in sorted(rows[cls][group],key=lambda r:key(split,group,r["sketch_id"])):
                    images={row[k] for k in ("source_id","true_id","false_id")}
                    if row["sketch_id"] not in used_sketches and not images & blocked:
                        trial.append(row)
                        break
            if len(trial)!=2:
                continue
            for row in trial:
                used_photos.update(row[k] for k in ("source_id","true_id","false_id"))
                used_sketches.add(row["sketch_id"])
            selected.extend(trial)
            used_classes.add(cls)
            if len(selected)-start==12:
                break
        if len(selected)-start!=12:
            raise ValueError("Insufficient disjoint panels")
    return sorted(selected,key=lambda r:key("board",r["split"],r["sketch_id"]))


def render(revision):
    base=json.loads((DEFAULT_CACHE/"manifest.json").read_bytes())["datasets"]["sketchy"]
    index=torch.load(index_file(DEFAULT_CACHE,"sketchy",base),weights_only=True)
    root=Path(base["root"])
    selected=select({s:candidates(s,index,revision) for s in SPLITS})
    output=OUT/"inspection"
    output.mkdir(exist_ok=True,parents=True)
    for start in (0,6,12,18):
        canvas=Image.new("RGB",(760,6*196+25),"white")
        draw=ImageDraw.Draw(canvas)
        for x,name in ((40,"Sketch"),(285,"Photo A"),(527,"Photo B")):
            draw.text((x,4),name,fill="black")
        for i,r in enumerate(selected[start:start+6]):
            r["case"]=start+i+1
            r["correct_side"]="A" if key("side",r["split"],r["sketch_id"])[0]%2==0 else "B"
            sequence=((r["true"],r["false"]) if r["correct_side"]=="A" else (r["false"],r["true"]))
            top=25+i*196
            draw.text((4,top+4),str(r["case"]),fill="black")
            for col,rel in enumerate((r["sketch"],*sequence)):
                with Image.open(root/rel) as im:
                    frame=ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"),(230,178))
                    canvas.paste(frame,(30+240*col+(230-frame.width)//2,top+(178-frame.height)//2))
            draw.line((0,top+188,759,top+188),fill="#bbb")
        canvas.save(output/f"board-{start//6+1}.jpg",quality=90)
    atomic_json(output/"key.json",selected)
    atomic_json(output/"manifest.json",{"seed":SEED,"matched_revision":revision,"cases":len(selected),
            "key_sha256":checksum(output/"key.json"),"boards":{p.name:checksum(p) for p in sorted(output.glob("board-*.jpg"))}})


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-revision",required=True)
    render(parser.parse_args().matched_revision)
