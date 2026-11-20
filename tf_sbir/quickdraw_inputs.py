"""Prepare all unseen QuickDraw-Ext images and encode them on the attached Colab GPU."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import digest
from .crossdomain_inputs import runtime
from .data import UNSEEN_CLASSES, collect_paths
from .encoding import load_global_inputs


def prepare(workspace):
    manifest = workspace / "quickdraw-data.json"
    root = workspace / "datasets/quickdraw-unseen"
    archive = workspace / "datasets/quickdraw-kaggle.zip"
    classes = UNSEEN_CLASSES["quickdraw"]
    if manifest.is_file():
        saved = read(manifest)
        if saved["root"] != str(root) or saved["classes_sha256"] != digest(classes):
            raise ValueError("QuickDraw manifest refers to another dataset")
        return saved
    selected = {}
    prefixes = set()
    counts = Counter()
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            parts = PurePosixPath(item.filename).parts
            if item.is_dir() or len(parts) < 3 or parts[-3] not in ("sketch","photo") or parts[-2] not in classes:
                continue
            if ".." in parts or "\\" in item.filename:
                raise ValueError("Invalid archive path")
            prefixes.add(parts[:-3])
            rel = "/".join(parts[-3:])
            if rel in selected:
                raise ValueError("Duplicate image: "+rel)
            selected[rel] = item
            counts[(parts[-3],parts[-2])] += 1
        if len(prefixes)!=1 or any(counts[(mode,category)]==0 for mode in ("sketch","photo") for category in classes):
            raise ValueError("Expected one complete 30-class QuickDraw split")
        print("Unseen gallery and queries", {mode:sum(counts[(mode,c)] for c in classes) for mode in ("sketch","photo")},flush=True)
        for rel,item in selected.items():
            path = root / rel
            path.parent.mkdir(exist_ok=True,parents=True)
            pending = path.with_name(path.name+".extracting")
            with source.open(item) as inp, pending.open("wb") as out:
                shutil.copyfileobj(inp,out)
            pending.replace(path)
    fingerprint = hashlib.sha256()
    for name,item in sorted(selected.items()):
        fingerprint.update(f"{name}\0{item.file_size}\0{item.CRC}\n".encode())
    metadata = read(workspace/"datasets/quickdraw-metadata.json")
    result = {"root":str(root),"source":metadata,"archive_sha256":checksum(archive),"classes_sha256":digest(classes),
              "image_manifest_sha256":fingerprint.hexdigest(),"counts":{mode:{c:counts[(mode,c)] for c in classes}
                    for mode in ("sketch","photo")}}
    atomic_json(manifest,result)
    return result


def paths(workspace,batch,workers):
    source = prepare(workspace)
    args = runtime(workspace,"quickdraw",batch=batch,workers=workers)
    classes = UNSEEN_CLASSES["quickdraw"]
    q,ql = collect_paths(args.root,classes,"sketch")
    p,pl = collect_paths(args.root,classes,"photo")
    for mode,labels in (("sketch",ql),("photo",pl)):
        if labels.bincount().tolist() != [source["counts"][mode][c] for c in classes]:
            raise ValueError("Disk contents differ from the original unseen split")
    return args,q,ql,p,pl


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--batch",type=int,default=448)
    parser.add_argument("--workers",type=int,default=10)
    options=parser.parse_args()
    args,q,_,p,_=paths(options.workspace,options.batch,options.workers)
    queries,photos,_=load_global_inputs(args,q,p,False)
    atomic_json(options.workspace/"quickdraw-encoded.json",{"queries":len(queries),"gallery":len(photos),
                "dimension":queries.size(1),"class_count":len(UNSEEN_CLASSES["quickdraw"])})
