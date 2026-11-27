"""Build source-disjoint discovery boards and independently shuffled annotation boards."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .kaggle_unseen_features import DEFAULT_CACHE, index_file
from .wordnet_visual_inspection import key


OUT = DEFAULT_CACHE.parent/"results"/"visible-attribute-context"
HERE = Path(__file__).resolve().parents[1]


def sample():
    record = json.loads((DEFAULT_CACHE/"manifest.json").read_bytes())["datasets"]["sketchy"]
    index = torch.load(index_file(DEFAULT_CACHE,"sketchy",record),weights_only=True)
    previous = json.loads((HERE/"docs/results/wordnet-description-renaming/blind-selection-key.json").read_bytes())
    excluded = {r[k] for r in previous for k in ("source_id","true_id","false_id")}
    previous_q = {r["sketch_id"] for r in previous}
    by_path = {str(Path(s).with_suffix("")):i for i,s in enumerate(index["photo"]["paths"])}
    drawings = defaultdict(list)
    for i,s in enumerate(index["sketch"]["paths"]):
        path = Path(s)
        prefix,sep,num = path.stem.rpartition("-")
        name = str(Path("photo")/path.parent.name/prefix)
        p = by_path.get(name)
        if p is not None and p not in excluded and i not in previous_q and sep and num.isdigit():
            drawings[p].append(i)
    selected = []
    classes_used = set()
    for split in SKETCHY_SPLITS:
        options = sorted(set(UNSEEN_CLASSES[split])-classes_used, key=lambda c:key("visible/classes",split,c))
        valid_photos = set(index["photo"]["views"][split][0].tolist())
        for cls in options[:8]:
            classes_used.add(cls)
            objs = [i for i in drawings if i in valid_photos and Path(index["photo"]["paths"][i]).parent.name==cls]
            objs.sort(key=lambda i:key("visible/subjects",split,i))
            if len(objs)<16:
                raise ValueError("Not enough subjects for fixed panel")
            rows = []
            for p in objs[:16]:
                q = min(drawings[p],key=lambda i:key("visible/sketch",i))
                rows.append({"photo":p,"sketch":q})
            selected.append({"split":split,"class":cls,"discovery":rows[:6],"confirm":rows[6:]})
    return selected, index, Path(record["root"])


def render(stage,revision):
    selected,index,base=sample()
    ids=[r["photo"] for c in selected for arm in ("discovery","confirm") for r in c[arm]]
    if len(set(ids))!=256: raise ValueError("Repeated photo or incomplete panel")
    folder=OUT/"boards"/stage
    folder.mkdir(exist_ok=True,parents=True)
    data=[]
    for c in selected:
        if stage=="discovery":
            photos=sketches=c["discovery"]
        else:
            photos=sorted(c["confirm"],key=lambda r:key("visible/photo-order",r["photo"]))
            sketches=sorted(c["confirm"],key=lambda r:key("visible/sketch-order",r["sketch"]))
        data.append({"class":c["class"],"split":c["split"],"photo":[r["photo"] for r in photos],
                     "sketch":[r["sketch"] for r in sketches]})
    cols=6 if stage=="discovery" else 10
    tile=132 if stage=="discovery" else 128
    # Each page displays two classes. The confirm modalities and column orders are independent.
    modes=(("sketch","photo"),) if stage=="discovery" else (("sketch",),("photo",))
    per_board=2 if stage=="discovery" else 4
    for start in range(0,len(data),per_board):
        for channels in modes:
            name="-".join(channels)
            h=len(channels)*tile+36
            board=Image.new("RGB",(12+cols*tile,h*per_board),"white")
            draw=ImageDraw.Draw(board)
            for slot,c in enumerate(data[start:start+per_board]):
                y=slot*h
                draw.text((6,y+2),f"{c['class']}  {name}",fill="black")
                for j in range(cols):
                    draw.text((8+j*tile,y+16),str(j+1),fill="black")
                for mi,mode in enumerate(channels):
                    for j,i in enumerate(c[mode]):
                        with Image.open(base/index[mode]["paths"][i]) as im:
                            frame=im.convert("RGB").resize((tile-4,tile-4),Image.Resampling.BICUBIC)
                        board.paste(frame,(6+j*tile,y+31+mi*tile))
            board.save(folder/f"{name}-{1+start//per_board:02d}.jpg",quality=92)
    atomic_json(folder/"mapping.json",data)
    atomic_json(folder/"manifest.json",{"stage":stage,"revision":revision,"classes":[r["class"] for r in data],
            "seed":42,"mapping_sha256":checksum(folder/"mapping.json"),
            "boards":{p.name:checksum(p) for p in sorted(folder.glob("*.jpg"))}})
    atomic_json(OUT/"subject-ids.json",selected)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("discovery","confirm"),required=True)
    parser.add_argument("--revision",required=True)
    args=parser.parse_args()
    render(args.stage,args.revision)
