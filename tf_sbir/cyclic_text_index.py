"""Build independently checkable, context-sensitive triangular description probes for a gallery."""

import argparse
from itertools import combinations
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments
from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest
from .crossdomain_inputs import CHECKPOINT
from .encoding import load_frozen_clip, save_tensor
from .foundation import load_words
from .retrieval_diagnosis import global_inputs
from .semantic import read_words

FAMILY = "cyclic-description-verification"
FORMS = {"contrast": "{} of {}, rather than {}.", "background": "{} of {}, with {} in the background.",
         "list": "{} of {} and {}."}


def triangles(photos, target, scale):
    geometry = F.normalize(photos - photos.mean(0))
    labels = cluster_assignments(geometry,64)
    _, labels = labels.unique(sorted=True,return_inverse=True)
    n = int(labels.max()) + 1
    votes = photos.new_zeros((n,len(target)))
    centers = photos.new_zeros((n,photos.size(1))).index_add_(0,labels,geometry)
    for start in range(0,len(photos),192):
        loc=slice(start,start+192)
        votes.index_add_(0,labels[loc],((photos[loc]@target.T)*scale).softmax(1))
    names = votes.topk(5).indices.cpu()
    pair = F.normalize(centers)@F.normalize(centers).T
    pair.fill_diagonal_(-torch.inf)
    rivals = pair.topk(2).indices.cpu()
    triplets = set()
    for i, (a,b) in enumerate(rivals.tolist()):
        found = sorted(set(names[i].tolist()+names[a,:3].tolist()+names[b,:3].tolist()))
        triplets.update(combinations(found,3))
    without_duplicates = sorted(triplets)
    terms = sorted({v for three in triplets for v in three})
    lookup = {v:i for i,v in enumerate(terms)}
    compact = [tuple(lookup[w] for w in row) for row in without_duplicates]
    pairs = sorted({pair for row in compact for pair in combinations(row,2)})
    edge_id = {v:i for i,v in enumerate(pairs)}
    tri_edge = [(edge_id[a,b],edge_id[b,c],edge_id[a,c]) for a,b,c in compact]
    return torch.tensor(terms),torch.tensor(compact),torch.tensor(pairs),torch.tensor(tri_edge)


def pair_directions(model, tokenizer, names, pairs, kind, domain):
    template = FORMS[kind]
    phrases = []
    for a,b in pairs.tolist():
        phrases.extend((template.format(domain,names[a],names[b]),template.format(domain,names[b],names[a])))
    blocks = []
    for start in range(0,len(phrases),384):
        tokens = tokenizer(phrases[start:start+384]).cuda()
        blocks.append(F.normalize(model.encode_text(tokens).float()).cpu())
    encoded = torch.cat(blocks)
    return encoded[::2] - encoded[1::2]


def cycle_vectors(edge, edge_ids):
    return (edge[edge_ids[:,0]] + edge[edge_ids[:,1]] - edge[edge_ids[:,2]])/3


def residual_directions(cycle, single, tri, covariance):
    gram = single @ covariance @ single.T
    local = gram[tri[:,:,None],tri[:,None,:]].double()
    cov_with_words = (cycle @ covariance @ single.T).gather(1,tri)
    beta = (torch.linalg.pinv(local,hermitian=True,rtol=1e-10) @ cov_with_words.double()[:,:,None]).squeeze(-1).float()
    residual = cycle - (beta[:,:,None] * single[tri]).sum(1)
    remain = (residual @ covariance @ single.T).gather(1,tri)
    denom = ((residual @ covariance * residual).sum(-1)[:,None].clamp_min(1e-16) * gram.diagonal()[tri].clamp_min(1e-16)).sqrt()
    err = float((remain/denom).abs().amax())
    if err > .005:
        raise ValueError(f"Failed to remove individual noun response: {err}")
    return residual, err


@torch.inference_mode()
def run(w, ds):
    cfg,_,_,paths,galleries,_,praw = global_inputs(w,ds)
    words = read_words(cfg.vocabulary)
    if len(words)!=20931:raise ValueError("Different vocabulary")
    source,target = (x.cuda() for x in load_words(cfg,words))
    _,scale=bare_words(cfg,words)
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
    tokenizer=open_clip.get_tokenizer("ViT-B-32")
    for s,(pi,_) in galleries.items():
        dest=w/"cache"/FAMILY/(s+"-directions.pt")
        desc=dest.with_suffix(".json")
        sig=digest({"checkpoint":CHECKPOINT,"gallery":digest([str(paths[j]) for j in pi.tolist()]),
                    "public_vocabulary":checksum(cfg.vocabulary),"forms":FORMS,"panel":"5,3,3 in 64 RGB groups v1"})
        if dest.is_file() and desc.is_file() and read(desc)["signature"]==sig:continue
        x=praw[pi].cuda()
        terms,tri,pairs,edge_ids=triangles(x,target,scale)
        rows=tri.cuda()
        cyclic_ids=edge_ids.cuda()
        xc=x-x.mean(0)
        cov=xc.T@xc/len(xc)
        core={"terms":terms,"triangles":tri,"pairs":pairs,"edges_of_triangle":edge_ids}
        check={}
        plain_nouns=source[terms.cuda()]
        additive=plain_nouns[pairs[:,0]]-plain_nouns[pairs[:,1]]
        null=float(cycle_vectors(additive,cyclic_ids).abs().amax())
        if null>2e-7:raise ValueError("Nonzero cycle for a difference of unary scores")
        print("probes",s,len(terms),len(pairs),len(tri),flush=True)
        for form in FORMS:
            data={}
            for name,domain,single in (("source","a sketch",source[terms.cuda()]),("target","a photo",target[terms.cuda()])):
                edges=pair_directions(model,tokenizer,[words[i] for i in terms.tolist()],pairs,form,domain).cuda()
                cycle=cycle_vectors(edges,cyclic_ids)
                residual,err=residual_directions(cycle,single,rows,cov)
                fraction=float(cycle.square().sum(-1).mean()/edges.square().sum(-1)[cyclic_ids].mean())
                check[form+"/"+name]={"edge_energy_fraction_in_centered_cycle":fraction,"max_unary_gallery_correlation":err,
                                      "residual_energy_share":float(residual.square().sum()/cycle.square().sum().clamp_min(1e-12))}
                data[name]={"edges":edges.cpu(),"cycle":cycle.cpu(),"residual":residual.cpu()}
            core[form]=data
        save_tensor(core,dest)
        atomic_json(desc,{"signature":sig,"names":len(terms),"triangles":len(tri),"pairs":len(pairs),"unary_null_error":null,"diagnosis":check})
        print("indexed",s,check,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    a=p.parse_args()
    run(a.workspace,a.dataset)
