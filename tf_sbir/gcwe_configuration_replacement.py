"""Replace displacement GCWE in the original ranker, holding both other components fixed."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import io
from pathlib import Path
import urllib.request
from types import SimpleNamespace
from zipfile import ZipFile

import torch

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .gallery_panel_resume import region_views, semantic_inputs
from .gcwe_configuration_report import cached
from .gcwe_semantic_configuration import FAMILY as CONFIGURATION, distance_bins, laws, predictions
from .geometry_word_flat import (GLOBAL_WEIGHT,REGIONAL_WEIGHT,CONFIG,Index,KD_FUNCTION_SHA,
    PrefixReplay,index,load_kd,log_score,make_word_bank,regional_readout,score_rows)


FAMILY="gcwe-configuration-replacement"
REFERENCE="405e78cf60dc889a7b03e225ee846a14c44714a5"
REFERENCE_HASH="b5c440e4dc1324c014976949290dcee7995b2487d3f5941c5f64a7420f8123bd"


def original_results(work,split):
    path=work/"assets/locked-gcwe-ablation-405e78c.zip"
    if not path.is_file():
        url=f"https://raw.githubusercontent.com/HieuNM1804/training-free-ZS-SBIR/{REFERENCE}/docs/results/gcwe-displacement-leave-one-out/paired-T4.zip"
        with urllib.request.urlopen(url,timeout=45) as stream:
            reference=stream.read()
        assert hashlib.sha256(reference).hexdigest()==REFERENCE_HASH
        path.write_bytes(reference)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==REFERENCE_HASH
    with ZipFile(path) as archive:
        return torch.load(io.BytesIO(archive.read(split+".pt")),weights_only=True)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    work=Path("/content/sbir")
    out=work/"results"/FAMILY
    cfg,qp,qs,pp,gs,qf,pf,src,tgt,scale=semantic_inputs(work,"sketchy")
    pilot={k:torch.randperm(len(ids),generator=torch.Generator().manual_seed(42))[:1536]
           for k,(ids,_) in qs.items()}
    unique=torch.cat([qs[k][0][ix] for k,ix in pilot.items()]).unique()
    _,background=distance_bins()
    cache=work/"cache"/CONFIGURATION
    qhist=cached([qp[i] for i in unique.tolist()],cache)
    photo_hist=cached(pp,cache)
    views,_=region_views(work,"sampled-sketchy",cfg,qp,unique,qf)
    kd,_=load_kd(work)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(qi,labels) in qs.items():
        ix=pilot[split]
        ids=qi[ix]
        pi,gl=gs[split]
        reference=original_results(work,split)
        assert torch.equal(reference["qi"],qi) and torch.equal(reference["pi"],pi) and torch.equal(reference["labels"],labels)
        photos=pf[pi].cuda()
        stamp=digest(dict(gallery=[str(pp[i]) for i in pi.tolist()],model=CHECKPOINT,source="regional-64/RGB16"))
        words,compact=index(work,split,photos,tgt,scale,stamp[:16])
        bank=make_word_bank(photos,tgt,scale,192)
        graph=Index(SimpleNamespace(p=photos),rgb_only=True)
        graph.prepare(CONFIG)
        groups=[v.cuda() for v in torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).tensor_split(2)]
        models=[]
        for held,ref in zip(groups,reversed(groups)):
            joint,ind=laws(bank,photo_hist[pi],ref,background)
            fake=laws(bank,photo_hist[pi],ref,background,shuffle=True)[0]
            models.append((held,{"joint":joint,"independent":ind,"shuffled":fake}))
        evaluator=PrefixReplay(kd,gl,split)
        stats={}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for b in torch.arange(len(ids)).split(16):
                qid=ids[b]
                local=torch.searchsorted(unique,qid)
                poses=torch.stack((qf[qid].cuda(),views["flipped"][local].cuda().float()),1)
                prior=(scale*(poses@src.T)).softmax(-1).mean(1)
                direct=prior@bank.T
                regional=regional_readout(poses,views["regions"][local].cuda(),src,scale,words,compact,direct)
                observed=qhist[local].cuda().float()/(49*24)
                logits={"without_gcwe":direct}
                for held,distributions in models:
                    destination=bank[held]
                    for kind,law in distributions.items():
                        for rule,posterior in predictions(prior,observed,law,(16,)).items():
                            name=kind+"_"+rule
                            if name not in logits:
                                logits[name]=torch.empty_like(direct)
                            logits[name][:,held]=posterior@destination.T
                names=list(logits)
                propagated=graph.score(CONFIG,torch.cat(list(logits.values()))).split(len(b))
                for name,signal in zip(names,propagated):
                    scores=GLOBAL_WEIGHT*log_score(signal)+REGIONAL_WEIGHT*regional
                    row=score_rows(scores,labels[ix[b]],gl.cuda(),split,evaluator,pool)[:,:2]
                    stats.setdefault(name,[]).append(row)
        raw={k:torch.cat(v).double() for k,v in stats.items()}
        raw["locked_displacement"]=reference["rows"]["same_partition"][ix,:2].double()
        replay=100*(raw["without_gcwe"]-reference["rows"]["semantic_regional"][ix,:2].double())
        if replay.abs().mean(0).max()>.05:
            raise ValueError(f"Baseline replay drift too large: {replay.abs().mean(0)}")
        metrics={k:(100*v.mean(0)).tolist() for k,v in raw.items()}
        effects={}
        for arm in ("joint_one","joint_count16"):
            for ctrl in ("without_gcwe","locked_displacement",arm.replace("joint","independent"),arm.replace("joint","shuffled")):
                delta=100*(raw[arm]-raw[ctrl])
                effects[arm+" minus "+ctrl]=dict(points=delta.mean(0).tolist(),class_ap=class_interval(delta[:,0],labels[ix],seed=42))
        save_tensor(dict(qi=ids,pi=pi,labels=labels[ix],rows=raw,replay=replay),out/(split+"-pilot.pt"))
        atomic_json(out/(split+"-pilot.json"),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
            reference=REFERENCE,evaluator=KD_FUNCTION_SHA,queries=len(ids),gallery=len(pi),metrics=metrics,contrasts=effects,
            replay_mean_absolute_points=replay.abs().mean(0).tolist(),replay_max_absolute_points=replay.abs().amax(0).tolist()))
        print(split,metrics,flush=True)
        del graph,bank,models,joint,ind,fake,destination,distributions,law


if __name__=="__main__":
    run()
