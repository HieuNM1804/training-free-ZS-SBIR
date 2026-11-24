"""Read-only audit of native Sketchy gallery pixels, source siblings and their effect on diagnostic cohorts."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile, is_zipfile

import numpy as np
from PIL import Image, ImageChops, ImageStat
import requests
import torch

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .encoding import save_tensor


FAMILY="semantic-gallery-input-audit"


def native_pixels(name):
    with Image.open(name) as im:
        rgb=im.convert("RGB")
        r,g,b=rgb.split()
        gray=ImageChops.difference(r,g).getbbox() is None and ImageChops.difference(b,g).getbbox() is None
        hist=rgb.convert("L").histogram()
        pixels=sum(hist)
        levels=[sum(hist[:k+1])/pixels for k in (8,16,32)]
        return [int(gray),int(sum(hist[33:])>0),*levels,sum(i*n for i,n in enumerate(hist))/pixels]


def distance(names):
    with Image.open(names[0]) as a, Image.open(names[1]) as b:
        x,y=a.convert("RGB"),b.convert("RGB")
        if x.size!=y.size:
            return None
        return float(np.mean(ImageStat.Stat(ImageChops.difference(x,y)).mean))


def verify_public(root,name):
    rel=Path(name).relative_to(root).as_posix()
    url="https://www.kaggle.com/api/v1/datasets/download/b20dccn616nguynhutun/sketchy/"+quote("Sketchy/"+rel,safe="")
    local=Path(name).read_bytes()
    result=dict(file=rel,sha256=hashlib.sha256(local).hexdigest())
    try:
        with requests.get(url,params={"datasetVersionNumber":1},timeout=15,stream=True) as resp:
            result["http"]=resp.status_code
            resp.raise_for_status()
            data=bytearray()
            for chunk in resp.iter_content(65536):
                data.extend(chunk)
                if len(data)>2_000_000:
                    raise ValueError("Expected a single small benchmark JPEG")
        raw=bytes(data)
        if is_zipfile(io.BytesIO(raw)):
            with ZipFile(io.BytesIO(raw)) as z:
                if len(z.namelist())!=1:
                    raise ValueError("Expected one member")
                raw=z.read(z.namelist()[0])
        result["public_sha256"]=hashlib.sha256(raw).hexdigest()
        result["identical"]=raw==local
    except (requests.RequestException,ValueError) as error:
        result["error"]=type(error).__name__
    return result


def distribution(values):
    data=np.asarray([v for v in values if v is not None],dtype=float)
    if not len(data):
        return {"n":0}
    return dict(n=len(data),quantiles=np.quantile(data,[0,.25,.5,.75,.95,1]).tolist(),
                below={str(t):int((data<=t).sum()) for t in (3,5,8)})


def pixel_summary(x):
    gray=x[:,0].bool()
    nonblank=x[:,1].bool()
    sensitivities={}
    for i,k in enumerate((8,16,32)):
        sensitivities[str(k)]={str(f):int((gray & nonblank & (x[:,2+i]>=f)).sum()) for f in (.8,.9,.95)}
    return dict(n=len(x),achromatic=int(gray.sum()),dark_achromatic=sensitivities)


def analyze_pairs(rows,positions,source_id,siblings,dark):
    def tab(group):
        out=dict(decisions=len(group))
        if group and "invariant" in group[0]:
            for arm in ("invariant","pooled","direct"):
                out[arm]={k:float(np.mean([r[arm][k][0] for r in group]))
                          for k in ("source_source","sketch_source")}
        return out
    flags=[]
    for r in rows:
        src=source_id[r["q"]]
        named=siblings.get(src)
        flag=named is not None and named in (r["pos"],r["neg"])
        flags.append((flag,bool(dark[positions[r["pos"]]]),bool(dark[positions[r["neg"]]])))
    subset=lambda predicate:[r for r,f in zip(rows,flags) if predicate(*f)]
    without=subset(lambda sibling,positive,negative:not sibling)
    return dict(all=tab(rows),sibling_of_query_source=tab(subset(lambda s,p,n:s)),without_sibling=tab(without),
                positive_dark=sum(p for s,p,n in flags),negative_dark=sum(n for s,p,n in flags),
                either_dark=tab(subset(lambda s,p,n:p or n)),neither_dark=tab(subset(lambda s,p,n:not p and not n)),
                neither_dark_nor_sibling=tab(subset(lambda s,p,n:not (s or p or n))))


def run():
    work=Path("/content/sbir")
    root=work/"results"/FAMILY
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(work,"sketchy")
    qpaths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,"sketch")
    photos,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,"photo")
    with ThreadPoolExecutor(max_workers=2) as pool:
        values=torch.tensor(list(pool.map(native_pixels,photos)),dtype=torch.double)
    dark=values[:,0].bool() & values[:,1].bool() & (values[:,3]>=.9)
    gallery=list(map(Path,photos))
    by_name={p:i for i,p in enumerate(gallery)}
    by_stem={(p.parent.name,p.stem):i for i,p in enumerate(gallery)}
    source_lookup={}
    for i,f in enumerate(map(Path,qpaths)):
        source_lookup[i]=by_stem[f.parent.name,f.stem.rsplit("-",1)[0]]
    sources=sorted(set(source_lookup.values()))
    pairs=[(i,by_name[p.with_name(p.stem+"1"+p.suffix)]) for i in sources if
           (p:=gallery[i]).with_name(p.stem+"1"+p.suffix) in by_name]
    category=defaultdict(list)
    for i in sources:
        category[gallery[i].parent.name].append(i)
    rng=np.random.default_rng(42)
    controls=[]
    for a,b in pairs:
        alternative=[i for i in category[gallery[a].parent.name] if i not in (a,b)]
        controls.append(int(rng.choice(alternative)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        observed=list(pool.map(distance,[(gallery[a],gallery[b]) for a,b in pairs]))
        unrelated=list(pool.map(distance,[(gallery[a],gallery[c]) for (a,_),c in zip(pairs,controls)]))
    confirmed={a:b for (a,b),v in zip(pairs,observed) if v is not None and v<=5}
    public={Path(cfg.root)/"photo/scissors/ext_1.jpg"}
    check=work/"results/semantic-invariant-question"
    examples=json.loads((check/"examples.json").read_text())["examples"]
    for items in examples.values():
        for r in items[:1]:
            public.update((gallery[r["positive"]],gallery[r["negative"]]))
    if pairs:
        public.update(gallery[i] for i in pairs[0])
    proof=[verify_public(Path(cfg.root),p) for p in sorted(public)]
    atomic_json(root/"public-byte-check.json",dict(dataset="b20dccn616nguynhutun/sketchy",version=1,files=proof))
    for split,(ids,labs) in gs.items():
        qi=qs[split][0]
        idset=set(ids.tolist())
        named=set(source_lookup[q] for q in qi.tolist())
        keep=[i for i,(a,b) in enumerate(pairs) if a in named and b in idset]
        annotated={a:b for a,b in confirmed.items() if a in named and b in idset}
        per_class={gallery[int(ids[labs==k][0])].parent.name:pixel_summary(values[ids[labs==k]]) for k in labs.unique()}
        ext=torch.tensor([gallery[i].stem.startswith("ext_") for i in ids.tolist()])
        meta=torch.load(work/"results/gcwe-action-category-transfer"/(split+".pt"),weights_only=True)["meta"]
        all_boundary=[dict(q=q,pos=a if dr else b,neg=b if dr else a) for q,a,b,dr,*_ in meta.tolist()]
        transfer_file=check/(split+"-transfer-pairs.json")
        records=json.loads(transfer_file.read_text()) if transfer_file.exists() else [
            {k:r[k] for k in ("q","pos","neg")} for r in json.loads((check/(split+"-pairs.json")).read_text())["boundary"]]
        identity={i:i for i in idset}
        stats=dict(commit=revision(),seed=42,split=split,all_gallery=pixel_summary(values[ids]),
                   ext_named=pixel_summary(values[ids[ext]]),other_named=pixel_summary(values[ids[~ext]]),
                   source_count=len(named),sibling_candidates=len(keep),
                   sibling_mae=distribution([observed[i] for i in keep]),
                   other_source_mae=distribution([unrelated[i] for i in keep]),classes=per_class,
                   boundary=analyze_pairs(all_boundary,identity,source_lookup,annotated,dark),
                   pilot=analyze_pairs(records,identity,source_lookup,annotated,dark))
        atomic_json(root/(split+".json"),stats)
        save_tensor(dict(ids=ids,pixel_statistics=values[ids],queries=qi,source=torch.tensor([source_lookup[q] for q in qi.tolist()]),
                    siblings=torch.tensor([[*pairs[i],controls[i]] for i in keep]),mae=torch.tensor([
                        [np.nan if observed[i] is None else observed[i],np.nan if unrelated[i] is None else unrelated[i]] for i in keep])),
                    root/(split+".pt"))
        print(split,len(ids),int(dark[ids].sum()),len(annotated),stats["boundary"]["sibling_of_query_source"],flush=True)


if __name__=="__main__":
    run()
