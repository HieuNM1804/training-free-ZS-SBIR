"""Check whether the frozen regional intervention actually removes gallery-dependent errors."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest, revision
from .encoding import save_tensor
from .foundation import load_words
from .gallery_depiction_study import PRIOR_RUN, views
from .gallery_region_panel_study import DIFFERENCES, parse
from .gallery_region_panels import FOLDS, read_regions
from .retrieval_diagnosis import global_inputs
from .semantic import read_words
from .shared_gallery_diagnosis import FAMILY, align, diagnosis


def panel_results(crop, src, scale, packs, base, hist, ids, conditions):
    out = {"regional": base[:, ids]}
    region = (base-.65*hist)/.35
    groups = defaultdict(list)
    for row in conditions:
        if row != "regional":
            f, size, mode, temp, w = parse(row)
            groups[f,size].append((row,mode,temp,w))
    for group, tasks in groups.items():
        temps = sorted({t for _,m,t,_ in tasks if m != "uniform"})
        data, pooled = read_regions(crop, src, packs[group], scale, temps)
        for key, mode, temp, w in tasks:
            if mode in DIFFERENCES:
                ref = data[temp]
                diff = ref["wrong" if mode=="wrong_diff" else "cross"]-ref["same"]
                v = base + (-w if mode=="reverse_diff" else w)*diff
            else:
                local = pooled if mode=="uniform" else data[temp][mode]
                v = base + .35*w*(local-region)
            out[key] = v[:,ids]
    return out


def inverse(scores):
    rank = scores.argsort(dim=1, descending=True, stable=True)
    pos = torch.arange(scores.shape[1], device=scores.device).expand_as(rank)
    return torch.empty_like(rank).scatter_(1, rank, pos)


def fixed_pool(r1, r2, pool, labels, photo_labels):
    # The union is fixed from the baseline rankings before changing methods.
    result = torch.zeros((len(labels), 5), dtype=torch.int64, device=labels.device)
    for i, y in enumerate(labels):
        pos = pool[i] & (photo_labels==y)
        neg = pool[i] & (photo_labels!=y)
        x = r1[i,pos,None] < r1[i,neg][None,:]
        z = r2[i,pos,None] < r2[i,neg][None,:]
        result[i] = torch.stack(((x&z).sum(),(x&~z).sum(),(~x&z).sum(),(~x&~z).sum(),
                                 torch.tensor(x.numel(),device=x.device)))
    return result.cpu()


def report_hard(table):
    mask = table[:,-1] > 0
    values = table[mask,:4].double()/table[mask,-1,None]
    if not len(values):
        raise ValueError("Fixed evaluation pool contains no mixed pairs")
    avg = values.mean(0)
    return {"eligible_queries":int(mask.sum()),"comparison_count":int(table[:,-1].sum()),
            "rank_switch":float(avg[1:3].sum()), "first_correct":float(avg[0]+avg[1]),
            "second_correct":float(avg[0]+avg[2]),"first_correct_only":float(avg[1]),"second_correct_only":float(avg[2])}


@torch.inference_mode()
def run(w):
    root = w/"results"/FAMILY
    lock = read(root/"freeze.json")
    pilot = read(root/"shared-scene-diagnosis.json")
    saved = torch.load(root/"shared-scene-diagnosis.pt", mmap=True, weights_only=True)
    rows = sorted({"regional",lock["nominee"],lock["best_same"],lock["best_average"],
                   lock["best_cross_replacement"],*lock["matched"]})
    cfg, qp, qs, pp, gs, _, _ = global_inputs(w,"sketchy")
    names = read_words(cfg.vocabulary)
    src = load_words(cfg,names)[0].cuda()
    _,scale = bare_words(cfg,names)
    v = views(w,"sketchy",qp)
    s1,s2 = ("sketchy_1","sketchy_2")
    qid,q1,q2=align(qs[s1][0],qs[s2][0])
    pid,p1,p2=align(gs[s1][0],gs[s2][0])
    if (pilot["common_query_path_hash"]!=digest([str(qp[i]) for i in qid]) or
        pilot["common_gallery_path_hash"]!=digest([str(pp[i]) for i in pid])):
        raise ValueError("The common comparisons changed")
    y = saved["query_labels"].cuda()
    from .data import UNSEEN_CLASSES
    char = pilot["classes"]
    photos = torch.tensor([char.index(UNSEEN_CLASSES[s1][int(k)]) for k in gs[s1][1][p1]],device="cuda")
    groups={parse(k)[:2] for k in rows if k!="regional"}
    refs=[]
    for s,p,q in zip((s1,s2),(p1,p2),(q1,q2),strict=True):
        raw = torch.load(w/"cache"/FAMILY/(s+"-donor-indices.pt"),mmap=True,weights_only=True)["banks"]
        banks={g:[[a.cuda() for a in raw[(*g,i)]] for i in range(FOLDS)] for g in groups}
        h=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),mmap=True,weights_only=True)
        b=torch.load(w/"cache"/PRIOR_RUN/"full"/(s+".pt"),mmap=True,weights_only=True)["regional_score"]
        refs.append((banks,p.cuda(),q,h,b))
    all_pairs,hard,change=defaultdict(list),defaultdict(list),defaultdict(list)
    torch.backends.cuda.matmul.allow_tf32=True
    for off in range(0,len(qid),24):
        sl=slice(off,off+24)
        crop=v(qid[sl]).float().contiguous()
        bank_scores=[]
        for banks,p,q,h,b in refs:
            ix=q[sl]
            bank_scores.append(panel_results(crop,src,scale,banks,b[ix].cuda(),h[ix].cuda().clamp_min(1e-30).log(),p,rows))
        a,b=bank_scores
        native=[inverse(z["regional"]) for z in (a,b)]
        use=(native[0]<100)|(native[1]<100)
        for k in rows:
            all_pairs[k].append(diagnosis(a[k],b[k],y[sl],photos)[0])
            r=[inverse(z[k]) for z in (a,b)]
            hard[k].append(fixed_pool(*r,use,y[sl],photos))
            for j,s in enumerate((s1,s2)):
                change[k+"/"+s].append(fixed_pool(native[j],r[j],use,y[sl],photos))
        if off%384==0:
            print("matched gallery images",off,len(qid),flush=True)
    all_pairs={k:torch.cat(v) for k,v in all_pairs.items()}
    hard={k:torch.cat(v) for k,v in hard.items()}
    change={k:torch.cat(v) for k,v in change.items()}
    err=float((all_pairs["regional"]-saved["measurements"]["locked"]).abs().max())
    if err>1e-6:
        raise ValueError("Previously saved comparison is not reproduced")
    def report(t):
        a=t.double().mean(0)
        return {"rank_switch":float(a[1]+a[2]),"correct_Ext1":float(a[0]+a[1]),"correct_Ext2":float(a[0]+a[2]),
                "top100_overlap":float(a[4]),"top1_class_agree":float(a[6])}
    out=root/"shared-scene-intervention.pt"
    save_tensor({"labels":y.cpu(),"query_ids":qid,"photo_ids":pid,"full_pairs":all_pairs,"hard_pairs":hard,
                 "pre_post_in_hard_pool":change},out)
    data={"revision":revision(),"freeze":lock["sha256"],"nominee":lock["nominee"],"queries":len(qid),
          "gallery":len(pid),"classes":char,"full":{k:report(v) for k,v in all_pairs.items()},
          "fixed_top100_union":{k:report_hard(v) for k,v in hard.items()},
          "original_and_edited_in_fixed_pool":{k:report_hard(v) for k,v in change.items()},
          "reference_error":err,"per_query_sha256":checksum(out)}
    atomic_json(out.with_suffix(".json"),data)
    print("matched gallery diagnosis",lock["nominee"],data["full"][lock["nominee"]],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
