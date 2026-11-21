"""Independently re-encode all raw TU images with the original OpenAI CLIP implementation."""

import argparse
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import revision
from .crossdomain_inputs import CHECKPOINT
from .data import Images, configure_image_worker, normal_transform
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd


OFFICIAL = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"


def load_reference(w, checkpoint):
    dest = w/"official-clip-audit-source"
    if not dest.exists():
        subprocess.run(["git","clone","--quiet","--no-checkout","https://github.com/openai/CLIP.git",str(dest)],check=True)
    subprocess.run(["git","-C",str(dest),"checkout","--quiet","--detach",OFFICIAL],check=True)
    if subprocess.check_output(["git","-C",str(dest),"rev-parse","HEAD"],text=True).strip()!=OFFICIAL:
        raise ValueError("OpenAI CLIP reference checkout differs")
    sys.path.insert(0,str(dest))
    import clip
    if dest.resolve() not in Path(clip.__file__).resolve().parents:
        raise ValueError("Imported CLIP from another source")
    model, transform = clip.load(checkpoint,"cuda",jit=False)
    return model.eval().requires_grad_(False),transform


def encode(model, paths, transform):
    loader=DataLoader(Images(paths,transform),batch_size=384,num_workers=6,pin_memory=True,
                      worker_init_fn=configure_image_worker)
    features=[]
    for images in loader:
        encoded=model.encode_image(images.cuda(non_blocking=True))
        features.append(F.normalize(encoded.float(),dim=1))
    return torch.cat(features)


@torch.inference_mode()
def run(w):
    a,qp,qs,pp,gs,oldq,oldp=global_inputs(w,"tuberlin")
    qi,ql=qs["tuberlin"]; pi,pl=gs["tuberlin"]
    if (len(qi),len(pi),len(ql.unique()),len(pl.unique()))!=(2130,27262,30,30):
        raise ValueError("TU protocol changed")
    model,preprocess=load_reference(w,a.clip_checkpoint)
    kd,kd_sha=load_kd(w)
    old=read(w/"results/conditional-semantic-index/heldout/tuberlin-native.json")
    report={"revision":revision(),"openai_revision":OFFICIAL,"checkpoint_sha256":CHECKPOINT,
            "official_clip_loaded":True,"kd_file_sha256":kd_sha,"queries":len(qi),"gallery":len(pi),
            "classes":len(ql.unique()),"gpu":torch.cuda.get_device_name(),"stored_native":{x:old[x] for x in ("mAP","precision")},
            "conditions":{}}
    for tag,transform in (("same_square_resize",normal_transform()),("openai_default",preprocess)):
        q=encode(model,[qp[i] for i in qi.tolist()],transform)
        p=encode(model,[pp[i] for i in pi.tolist()],transform)
        m,prec,mk,pk=kd(q,p,ql.cuda(),pl.cuda(),"tuberlin")
        qc=(q*F.normalize(oldq[qi].cuda(),dim=1)).sum(1)
        pc=(p*F.normalize(oldp[pi].cuda(),dim=1)).sum(1)
        item={"mAP":float(m),"precision":float(prec),"map_k":mk,"p_k":pk,
              "cosine_to_cached_embeddings":{"query_mean":float(qc.mean()),"query_min":float(qc.min()),
                                            "photo_mean":float(pc.mean()),"photo_min":float(pc.min())},
              "transform":str(transform)}
        if not torch.isfinite(q).all() or not torch.isfinite(p).all():
            raise ValueError("Official encoder produced nonfinite embeddings")
        report["conditions"][tag]=item
        print(tag,item,flush=True)
        del q,p
    atomic_json(w/"results/strict-public-audit/official-clip-tuberlin.json",report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
