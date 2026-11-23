"""Marginalize persistent or newly sampled public words through all decisions in a gallery hierarchy."""
from collections import defaultdict
from dataclasses import dataclass

import torch


MODES=("stay","new","half","retain80","cv","cv-flat","cv-shuffle","cv-reverse","cv-bg","cv-info","info-new","info-flat","root-info")
DEPTHS=(0,1,2)
WORDS=(32,128)


def collect(table,words):
    m=table.size(0);b,k=words.shape
    return table.index_select(1,words.flatten()).reshape(m,b,k).transpose(0,1).float()


def pool(x,parent,n,prior):
    return x.new_zeros((x.size(0),n,x.size(2))).index_add_(1,parent,x*prior[None,:,None])


def reliability(a,b,fa,fb,parent,n,prior):
    # Symmetrized validation risk, predicting the sibling response of this word from A to B and B to A.
    da=a-fa;db=b-fb
    numerator=pool(da*(b-fa)+db*(a-fb),parent,n,prior)
    denominator=pool(da.square()+db.square(),parent,n,prior)
    return torch.where(denominator>1e-10,(numerator/denominator.clamp_min(1e-10)).clamp(0.,1.),torch.zeros_like(denominator))


@dataclass(frozen=True,order=True)
class Recipe:
    size:int
    depth:int
    mode:str
    readout:str
    within:float
    gain:float

    def label(self):
        return f"{self.size}/{self.depth}/{self.mode}/{self.readout}/{self.within:g}/{self.gain:g}"

    @classmethod
    def parse(cls,name):
        k,l,m,r,t,s=name.split("/")
        return cls(int(k),int(l),m,r,float(t),float(s))


def grid():
    out={"base/direct","base/regional"}
    for k in WORDS:
        for l in DEPTHS:
            for mode in MODES:
                for within in (0.,.35,.75,1.):
                    for gain in ((1.,) if mode=="stay" else (.5,1.,2.)):
                        out.add(Recipe(k,l,mode,"cell",within,gain).label())
                for scheme in ("regional","regional-cell"):
                    for gain in ((1.,) if mode=="stay" else (-.5,.3,1.,2.)):
                        out.add(Recipe(k,l,mode,scheme,0.,gain).label())
    return sorted(out)


def forward(tree,levels,closed,words,prior_word,needed,depths):
    # States are joint masses divided by the size prior of the node. They need not sum to one within a node.
    modes=set(needed)|{"stay"}
    if "cv-flat" in modes:modes.add("cv")
    if "info-flat" in modes:modes.add("cv-info")
    states={m:prior_word[:,None] for m in modes}
    original=prior_word[:,None]
    perm=torch.randperm(words.size(1),generator=torch.Generator().manual_seed(53987)).to(words.device)
    identity=0.;mass_error=0.;out={}
    for j in reversed(range(min(depths),len(levels))):
        up=tree.parents[j];pri=levels[j]["prior"];n=tree.sizes[j+1]
        ratio=collect(levels[j]["full"],words)
        a=collect(levels[j]["A"],words);b=collect(levels[j]["B"],words)
        # The half profiles are stored in BF16. Restore the probability constraints before cross-validation.
        a=a/pool(a,up,n,pri)[:,up].clamp_min(1e-30);b=b/pool(b,up,n,pri)[:,up].clamp_min(1e-30)
        info=.5*pool(a*b.clamp_min(1e-20).log()+b*a.clamp_min(1e-20).log(),up,n,pri)
        fresh_info=(original.clamp_min(1e-30).log()+info).softmax(2)
        def coef(fresh):
            fc=fresh.expand(-1,n,-1)[:,up]
            return reliability(a,b,(fc*a).sum(2,keepdim=True),(fc*b).sum(2,keepdim=True),up,n,pri)
        cv=coef(original);ci=coef(fresh_info)
        cb=reliability(a,b,1.,1.,up,n,pri) if "cv-bg" in modes else None
        def weighted(m,gate):
            x=states[m]
            return (x*gate).sum(2,keepdim=True)/x.sum(2,keepdim=True).clamp_min(1e-30)
        gcv=weighted("cv",cv) if "cv-flat" in modes else None
        gci=weighted("cv-info",ci) if "info-flat" in modes else None
        rates={"new":0.,"half":.5,"retain80":.8,"cv":cv,"cv-flat":gcv,"cv-shuffle":cv[:,:,perm],"cv-reverse":1-cv,
               "cv-bg":cb,"cv-info":ci,"info-new":0.,"info-flat":gci}
        for m,state in states.items():
            if n==1:
                renewal=fresh_info if m=="root-info" else state
            elif m in ("stay","root-info"):
                renewal=state
            else:
                rate=rates[m]
                fresh=fresh_info if m in ("cv-info","info-new","info-flat") else original
                keep=state*rate
                renewal=keep+(state-keep).sum(2,keepdim=True)*fresh
            child=renewal[:,up]*ratio
            # Summation of photo mass under each parent checks the Markov transition independently of labels.
            if m in ("cv","cv-info","stay"):
                recovered=pool(child.sum(2,keepdim=True),up,n,pri).squeeze(2)
                rel=(recovered-state.sum(2)).abs()/(1+state.sum(2))
                mass_error=max(mass_error,float(rel.amax()))
            states[m]=child
            if j in depths:out[(j,m)]=child.sum(2).clamp_min(1e-30).log()
        if j in depths:
            direct=(collect(closed[j],words)*original).sum(2).clamp_min(1e-30).log()
            identity=max(identity,float((direct-out[(j,"stay")]).abs().amax()))
    if identity>7e-4 or mass_error>1e-4:raise ValueError(f"Persistence identities failed: {identity:g} {mass_error:g}")
    return out,identity,mass_error


def group_mean(score,idx,n):
    # Recenter for stable log means over competing photos.
    maximum=score.amax(1,keepdim=True)
    val=(score-maximum).exp()
    sums=score.new_zeros((len(score),n)).index_add_(1,idx,val)
    count=torch.bincount(idx,minlength=n).float()
    return (sums/count[None]).clamp_min(1e-30).log()+maximum


def outputs(tree,bank,closed,logu,lex,regional,rows):
    recipes=defaultdict(list)
    for k in rows:
        if not k.startswith("base/"):
            r=Recipe.parse(k);recipes[r.size].append(r)
    if "base/direct" in rows:yield "base/direct",lex,"direct",None
    if "base/regional" in rows:yield "base/regional",regional,"regional",None
    top_values,top_words=logu.topk(max(recipes),dim=1)
    norm_lex=lex-lex.logsumexp(1,keepdim=True)+torch.tensor(lex.size(1),device=lex.device).float().log()
    for k,group in recipes.items():
        words=top_words[:,:k];u=top_values[:,:k].softmax(1)
        depths={r.depth for r in group}
        out,err,energy=forward(tree,bank,closed,words,u,{r.mode for r in group},depths)
        coverage=(top_values[:,:k].logsumexp(1)-logu.logsumexp(1)).exp()
        for j in sorted(depths):
            ids=tree.ids[j];m=tree.sizes[j]
            group_lex=group_mean(norm_lex,ids,m)
            finite=out[(j,"stay")]
            residual=norm_lex-group_lex[:,ids]
            ctr=regional.new_zeros((len(regional),m)).index_add_(1,ids,regional)/torch.bincount(ids,minlength=m)[None]
            pooled_reg=.65*regional+.35*ctr[:,ids]
            for r in (v for v in group if v.depth==j):
                shift=(out[(j,r.mode)]-finite)[:,ids]
                if r.readout=="cell":val=finite[:,ids]+r.within*residual+r.gain*shift
                elif r.readout=="regional":val=regional+r.gain*shift
                else:val=pooled_reg+r.gain*shift
                yield r.label(),val,("direct" if r.readout=="cell" else "regional"),(err,energy,float(coverage.mean()))
