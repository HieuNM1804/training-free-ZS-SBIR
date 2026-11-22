"""Test whether the nontransitive component of a public-name comparison carries retrieval signal."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .cyclic_text_index import FAMILY,FORMS
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import paired_delta
from .semantic import read_words
from .verification_inputs import get_inputs
from .word_conditional_verifier import zscore
from .crossdomain_inputs import runtime

WEIGHTS=(.025,.1,.35,.75)
TEMPER=(0.,.35,1.)
BASE="regional"


def inferred_reliability(a, b, source, target, left, right):
    def standard(x):
        x=x-x.mean(0)
        return x / x.square().mean(0).clamp_min(1e-14).sqrt()
    out=[]
    for ix in (left,right):
        # Compare opposite-domain prompts on two renderings of the same photograph.
        correlation = .5*((standard(a[ix]@source.T)*standard(b[ix]@target.T)).mean(0)
                         +(standard(b[ix]@source.T)*standard(a[ix]@target.T)).mean(0))
        out.append(correlation)
    return torch.minimum(out[0],out[1]).clamp(0,1), out


class ProbeBank:
    def __init__(self, bank, form, names, a, b):
        self.tri=bank["triangles"].cuda()
        self.pairs=bank["pairs"].cuda()
        self.ix=bank["edges_of_triangle"].cuda()
        self.degree=self.tri.flatten().bincount(minlength=len(names[0]))
        self.u=names
        self.ve=tuple(bank[form][domain]["edges"].cuda() for domain in ("source","target"))
        self.vc=tuple(bank[form][domain]["cycle"].cuda() for domain in ("source","target"))
        residual=tuple(bank[form][domain]["residual"].cuda() for domain in ("source","target"))
        mean = a.mean(0)
        self.center = mean
        cov=(a-mean).T@(a-mean)/len(a)
        self.res=tuple(v/(v@cov*v).sum(1,keepdim=True).clamp_min(1e-12).sqrt() for v in residual)
        perm=torch.randperm(len(a),device=a.device,generator=torch.Generator(device=a.device).manual_seed(50271))
        self.verified, halves=inferred_reliability(a,b,*self.res,perm[::2],perm[1::2])
        self.shuffle=torch.randperm(len(self.tri),generator=torch.Generator().manual_seed(7821)).cuda()
        self.diagnostics={"median_positive_repeatability":float(self.verified.median()),
                          "share_positive_in_both_panels":float((self.verified>0).float().mean()),
                          "agreement_across_two_panels":float(torch.corrcoef(torch.stack(halves))[0,1])}

    def weights(self,q,temp,scale):
        prior=((q@self.u[0].T)*(temp*scale)).softmax(1)
        tri=(prior/self.degree[None])[:,self.tri].sum(2)
        edge=q.new_zeros((len(q),len(self.pairs)))
        for k in range(3):
            edge.index_add_(1,self.ix[:,k],tri/3)
        if float((tri.sum(1)-1).abs().max())>1e-5 or float((edge.sum(1)-1).abs().max())>1e-5:
            raise ValueError("A public name gained mass from appearing in multiple triangles")
        return prior,tri,edge

    def queries(self,q,weights):
        word,tri,edge=weights
        x=q-self.center
        full=((x@self.ve[0].T)*edge)@self.ve[1]
        cycle=((x@self.vc[0].T)*tri)@self.vc[1]
        rr=(x@self.res[0].T)*tri
        return {"unary":((x@self.u[0].T)*word)@self.u[1],
                "edges":full, "transitive":full-cycle, "cycle":cycle,
                "residual":rr@self.res[1], "verified":(rr*self.verified)@self.res[1],
                "shuffled":(rr*self.verified)@self.res[1][self.shuffle]}

    def identity_error(self,q,p,scale):
        part=q[:2]
        target=p[:5]
        _,tri,_=weights=self.weights(part,.35,scale)
        x=(part-self.center)@self.vc[0].T
        y=(target-self.center)@self.vc[1].T
        explicit=(x.double()[:,None]*y.double()[None]*tri.double()[:,None]).sum(2)
        collapsed=self.queries(part,weights)["cycle"]@(target-self.center).T
        return float((collapsed.double()-explicit).abs().max())


def key(form,method,temp,view,a):
    return f"{form}/{method}/{temp:g}/{view}/{a:g}"


@torch.inference_mode()
def run(w):
    root=w/"results"/FAMILY
    cfg=runtime(w,"sketchy",batch=384,workers=6)
    words=read_words(cfg.vocabulary)
    if len(words)!=20931:raise ValueError("Changed external vocabulary")
    source,target=(x.cuda() for x in load_words(cfg,words))
    _,scale=bare_words(cfg,words)
    rev=revision()
    spec={"revision":rev,"checkpoint":CHECKPOINT,"temps":TEMPER,"weights":WEIGHTS,"gallery_forms":FORMS,
          "seed":63241,"pilot_per_class":36,"primary":"verified with contrast or background syntax",
          "controls":"symmetric list, unaries, transitive votes, ordinary cycles, residual, shuffled cycles"}
    sig=digest(spec)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Changed protocol")
    atomic_json(root/"protocol.json",spec|{"sha256":sig})
    table={}
    for s,(data,y,g) in get_inputs(w,"sketchy").items():
        file=root/("pilot-"+s+".json")
        if file.exists():
            prev=read(file)
            if prev["signature"]!=sig or prev["per_query_sha256"]!=checksum(file.with_suffix(".pt")):raise ValueError("Stale pilot")
            table[s]=prev["values"];continue
        name=w/"cache"/FAMILY/(s+"-directions.pt")
        stored=torch.load(name,weights_only=True,mmap=True)
        terms=stored["terms"].cuda()
        a,b=(data[k].cuda().float() for k in ("gallery_native","gallery_detail"))
        ids=stratified_sample(y,36,63241)
        labels=g.cuda()
        rankings=defaultdict(list)
        all_diag={}
        for form in FORMS:
            bank=ProbeBank(stored,form,(source[terms],target[terms]),a,b)
            centered_a,centered_b=a-bank.center,b-bank.center
            err=bank.identity_error(data["query_native"][ids[:2]].cuda(),a[:5],scale)
            if err>2e-5:raise ValueError("The collapsed cyclic score differs from its exact triangle sum")
            all_diag[form]=bank.diagnostics|{"collapsed_absolute_error":err}
            for ix in ids.split(64):
                qa,qb=(data[k][ix].cuda().float() for k in ("query_native","query_detail"))
                truth=y[ix].cuda()
                base=zscore(data["regional_score"][ix].cuda())
                if form=="contrast":rankings[BASE].append(map_values(base,truth,labels,s)[0].cpu())
                for temp in TEMPER:
                    weights=bank.weights(qa,temp,scale)
                    whole,detail=bank.queries(qa,weights),bank.queries(qb,weights)
                    for method in whole:
                        if method=="unary" and form!="contrast":continue
                        native=whole[method]@centered_a.T
                        crossed=.5*(whole[method]@centered_b.T+detail[method]@centered_a.T)
                        for view,x in (("whole",native),("crossed",crossed)):
                            correction=zscore(x)
                            for alpha in WEIGHTS:
                                score=base+alpha*correction
                                rankings[key(form,method,temp,view,alpha)].append(map_values(score,truth,labels,s)[0].cpu())
            print(s,form,all_diag[form],flush=True)
        metrics={k:torch.cat(v) for k,v in rankings.items()}
        vals={k:v.mean(0).tolist() for k,v in metrics.items()}
        table[s]=vals
        file_pt=file.with_suffix(".pt")
        save_tensor({"metrics":metrics,"query_labels":y[ids],"split_ids":ids},file_pt)
        atomic_json(file,{"signature":sig,"values":vals,"queries":len(ids),"gallery":len(g),"checks":all_diag,
                          "index":read(name.with_suffix(".json")),"per_query_sha256":checksum(file_pt)})
    means={k:[sum(table[s][k][i] for s in table)/2 for i in (0,1)] for k in table["sketchy_1"]}
    top=max((k for k in means if "/verified/" in k and not k.startswith("list/")),key=means.get)
    match=top.split("/")
    comparable={"proposed":top,"matched_list":"/".join(["list",*match[1:]]),
                "matched_transitive":"/".join([match[0],"transitive",*match[2:]]),
                "matched_output_unary":"/".join(["contrast","unary",*match[2:]]),"base":BASE}
    paired={}
    for s in table:
        p=torch.load(root/("pilot-"+s+".pt"),weights_only=True)
        paired[s]={m:paired_delta(p["metrics"][top],p["metrics"][k],p["query_labels"]) for m,k in comparable.items()}
    atomic_json(root/"pilot.json",{"means":means,"splits":table,"nominee":top,"controls":comparable,"paired":paired})
    print("joint candidate",top,means[top],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
