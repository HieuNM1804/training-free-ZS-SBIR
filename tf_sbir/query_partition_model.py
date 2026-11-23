"""Out-of-fold retrieval on a gallery tree with query-conditioned resolution and nested disagreement correction."""
from collections import defaultdict
from dataclasses import dataclass
import itertools

import torch

from .query_partition_tree import cut_read,fold_sums,means,rank_gaussian,stop_weights,zscore


@dataclass(frozen=True,order=True)
class Row:
    base:str
    kind:str
    other:str
    gate:str
    variance:float
    scale:float
    gain:float

    def label(self):
        return "/".join((self.base,self.kind,self.other,self.gate,f"{self.variance:g}",f"{self.scale:g}",f"{self.gain:g}"))

    @classmethod
    def parse(cls,name):
        b,k,o,g,v,t,a=name.split("/")
        return cls(b,k,o,g,float(v),float(t),float(a))


BASES=("lex","fragments","regional","wordwalk")
PAIRINGS=(("lex","native"),("lex","fragments"),("regional","native"),("regional","fragments"),("wordwalk","native"),("fragments","native"))
NOISE=(.5,2.,8.)
POOL_GAIN=(.35,.7,.92,1.)
SHIFT_GAIN=(.15,.45,.9)
FIXED=tuple("level"+str(i) for i in range(5))


def grid():
    rows=set()
    for base,other in PAIRINGS:
        full=(base,other) in (("lex","native"),("regional","fragments"),("fragments","native"))
        for kind in ("pool","shift"):
            temps=(0.,1.5,3.) if kind=="pool" else (0.,)
            gains=POOL_GAIN if kind=="pool" else SHIFT_GAIN
            gated=("other","read","permgate","permtree") if full else ("other","read")
            for gate in (*gated,*FIXED,"prior"):
                variances=NOISE if gate in gated else (2.,)
                for v,t,gain in itertools.product(variances,temps,gains):
                    # Direct and prior controls of pooling only need one copy for each base.
                    if kind=="pool" and gate not in ("other","permgate","permtree") and other=="fragments" and base=="lex":continue
                    rows.add(Row(base,kind,other,gate,v,t,gain))
            if kind=="shift":
                rows.update(Row(base,kind,other,"ungrouped",2.,0.,a) for a in gains)
    return sorted(r.label() for r in rows)


def projected(a,b):
    # Residual within this donor fold; none of the response or selection statistics use the candidate.
    rho=(a*b).sum(1,keepdim=True)/a.square().sum(1,keepdim=True).clamp_min(1e-20)
    return b-rho*a


def aggregate_cut(tree,mean,stop,target,level):
    if level.startswith("level"):
        depth=min(int(level[5:]),len(mean)-1)
        return mean[depth][:,tree.ids[depth][target]]
    if level=="prior":
        if len(mean)==1:return mean[0][:,tree.ids[0][target]]
        fixed=[torch.ones_like(mean[0]),*(torch.full_like(z,.5) for z in mean[1:])]
        return cut_read(tree,mean,fixed,target)
    return cut_read(tree,mean,stop,target)


def make_fold_cache(c,raw):
    parts=c["folds"]
    ranks=[{k:rank_gaussian(v[:,idx]) for k,v in raw.items()} for idx in parts]
    # Gate positions are permuted within their own donor set. Ranks and marginal distribution are preserved.
    permutations=[torch.randperm(len(idx),generator=torch.Generator().manual_seed(64331+i)).to(idx.device) for i,idx in enumerate(parts)]
    return ranks,permutations


def predictions(c,raw,selected):
    by_pair=defaultdict(list)
    for name in selected:
        r=Row.parse(name);by_pair[(r.base,r.kind,r.other)].append(r)
    ranks,perms=make_fold_cache(c,raw)
    folds=c["folds"];real=c["tree"];fake=c["shuffled"]
    global_ranks={k:rank_gaussian(x) for k,x in raw.items()}
    for (base,kind,other),rows in by_pair.items():
        groups=defaultdict(list)
        for r in rows:groups[(r.gate,r.variance,r.scale)].append(r)
        orig=zscore(raw[base]) if kind=="pool" else global_ranks[base]
        predictions={}
        if ("ungrouped",2.,0.) in groups:
            predictions[("ungrouped",2.,0.)]=projected(orig,global_ranks[other])
        structured=[k for k in groups if k[0]!="ungrouped"]
        if structured:
            for k in structured:predictions[k]=torch.zeros_like(orig)
            gate_cache={};read_cache={}
            for h,j,k in ((h,(h+a)%3,(h+3-a)%3) for h in range(3) for a in (1,2)):
                target=folds[h];gate_ids=folds[j];reader_ids=folds[k]
                rr=ranks[k];gr=ranks[j]
                rv=zscore(raw[base][:,reader_ids]) if kind=="pool" else projected(rr[base],rr[other])
                for gate,variance,t in structured:
                    tree=fake if gate=="permtree" else real
                    rkey=(gate=="permtree",k,t)
                    if rkey not in read_cache:
                        data=rv if not t else (t*rv).clamp(-50,50).exp()
                        total,num=fold_sums(data,tree,reader_ids)
                        read_cache[rkey]=means(tree,total,num)
                    stop=None
                    if gate not in FIXED and gate!="prior":
                        gkey=(gate,j,variance)
                        if gkey not in gate_cache:
                            if gate=="read":
                                x=gr[base] if kind=="pool" else projected(gr[base],gr[other])
                            else:
                                x=gr[other]
                            if gate=="permgate":x=x[:,perms[j]]
                            total,num=fold_sums(x,tree,gate_ids)
                            gate_cache[gkey]=stop_weights(tree,total,num,variance)
                        stop=gate_cache[gkey]
                    value=aggregate_cut(tree,read_cache[rkey],stop,target,gate)
                    # Pooled evidence is averaged over independent fold assignments before taking its log.
                    predictions[(gate,variance,t)][:,target]+=.5*value
        for key,group in groups.items():
            est=predictions[key]
            if key[-1]:est=est.clamp_min(1e-30).log()/key[-1]
            for row in group:
                yield row.label(),(row.gain*est+(1-row.gain)*orig if kind=="pool" else orig+row.gain*est),raw[base]
