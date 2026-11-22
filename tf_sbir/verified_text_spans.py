"""Read a probability mixture of equal-length object spans before frozen CLIP text attention."""

from dataclasses import dataclass

import open_clip
from open_clip.model import text_global_pool
import torch
import torch.nn.functional as F

from .cluster_mine import cluster_assignments
from .foundation import DEFAULT_DOMAIN_TEMPLATES


@dataclass
class Bucket:
    group: int
    words: list
    mass: float
    weights: list


def inside(ids):
    eos=int(ids.argmax())
    return tuple(int(x) for x in ids[1:eos])


class ObjectSpans:
    def __init__(self,model,words):
        self.model=model
        tokenizer=open_clip.get_tokenizer("ViT-B-32")
        self.text_ids=[]
        self.width={}
        raw=tokenizer(words)
        raw=[inside(row) for row in raw]
        valid=[True]*len(words)
        for templates in DEFAULT_DOMAIN_TEMPLATES:
            for form in templates:
                prefix,suffix=form.split("{}")
                pre=inside(tokenizer([prefix])[0]);post=inside(tokenizer([suffix])[0])
                tokens=tokenizer([form.format(w) for w in words])
                for j,row in enumerate(tokens):
                    if inside(row)!=pre+raw[j]+post: valid[j]=False
                self.text_ids.append(tokens)
        for i,sub in enumerate(raw):
            self.width[i]=len(sub) if valid[i] and sub else -1-i

    def encode(self,buckets):
        model=self.model
        both=[]
        for pairs in self.text_ids:
            runs=[]
            for start in range(0,len(buckets),24):
                batch=buckets[start:start+24]
                max_n=max(len(b.words) for b in batch)
                words=[];masses=[]
                for b in batch:
                    n=len(b.words)
                    words.append(b.words+[b.words[0]]*(max_n-n))
                    masses.append(b.weights+[0.]*(max_n-n))
                ids=pairs[torch.tensor(words)].cuda()
                mass=torch.tensor(masses,device=ids.device)
                reference=ids[:,0]
                if (ids.argmax(-1)!=reference.argmax(-1)[:,None]).any():raise ValueError("Incompatible object spans")
                tok=model.token_embedding(ids)
                content=(tok.float()*mass[:,:,None,None]).sum(1).to(model.transformer.get_cast_dtype())
                content=content+model.positional_embedding.to(content)
                hidden=model.transformer(content,attn_mask=model.attn_mask)
                hidden=model.ln_final(hidden)
                out=text_global_pool(hidden,reference,model.text_pool_type,eos_token_id=getattr(model,"text_eos_id",None))
                if isinstance(model.text_projection,torch.nn.Linear):out=model.text_projection(out)
                else:out=out@model.text_projection
                runs.append(F.normalize(out.float()))
            both.append(torch.cat(runs))
        return F.normalize(both[0]+both[1]),F.normalize(both[2]+both[3])


def names(coarse,k,width):
    buckets=[]
    for g,row in enumerate(coarse):
        vals,indices=row.topk(k)
        vals=vals/vals.sum().clamp_min(1e-30)
        parts={}
        for w,m in zip(indices.tolist(),vals.tolist(),strict=True):
            parts.setdefault(width[w],[]).append((w,m))
        for group in parts.values():
            total=sum(m for _,m in group)
            if total>0:
                buckets.append(Bucket(g,[w for w,_ in group],total,[m/total for _,m in group]))
    return buckets


class Readout:
    def __init__(self,spans,coarse,heldout,labels,source,target,scale,k):
        self.buckets=names(coarse,k,spans.width)
        self.count=len(coarse)
        self.src,self.tgt=spans.encode(self.buckets)
        self.mass=source.new_tensor([b.mass for b in self.buckets])
        self.g=torch.tensor([b.group for b in self.buckets],device=source.device)
        all_ids=sorted({w for b in self.buckets for w in b.words})
        self.terms=torch.tensor(all_ids,device=source.device)
        weights=source.new_zeros((len(self.buckets),len(all_ids)))
        mapping={w:i for i,w in enumerate(all_ids)}
        for j,b in enumerate(self.buckets):
            ix=[mapping[w] for w in b.words]
            weights[j,ix]=source.new_tensor(b.weights)
        self.weights=weights
        self.output=F.normalize(weights@source[self.terms])
        self.output_target=F.normalize(weights@target[self.terms])
        hard_logit=(heldout@target[self.terms].T)*scale
        fused=(heldout@self.tgt.T)*scale
        z=hard_logit.amax(1,keepdim=True)
        hard=(hard_logit-z).exp()@weights.T
        hard=hard.clamp_min(1e-30).log()+z
        local=(fused-hard)
        cls=labels[:,None]==self.g
        count=cls.sum(0)
        self.positive=(local*cls).sum(0)/count.clamp_min(1)
        self.verified=self.positive-(fused.logsumexp(0)-hard.logsumexp(0))
        self.verified=torch.where(count>0,self.verified,0.)
        single=torch.tensor([len(b.words)==1 for b in self.buckets],device=source.device)
        if single.any():
            near=(self.src[single]*self.output[single]).sum(-1)
            terr=float((1-near).abs().max())
            if terr>.002:raise ValueError("A pure object did not reproduce the native text encoder")
        else:terr=0.
        mixed=~single
        self.diagnostics={"groups":self.count,"buckets":len(self.g),"mixed_buckets":int(mixed.sum()),
                          "mixed_probability_share":float(self.mass[mixed].sum()/self.count),
                          "heldout_positive_gain_share":float((self.verified[mixed]>0).float().mean()) if mixed.any() else 0.,
                          "heldout_gain_mean_nats":float(self.verified[mixed].mean()) if mixed.any() else 0.,
                          "onehot_native_max_one_minus_cosine":terr}

    def responses(self,x,src,scale):
        # Use one offset per query and fold across all its views and representations.
        # Restore it before combining the independently constructed panels.
        z=x@src[self.terms].T*scale
        zs=x@self.src.T*scale
        zo=x@self.output.T*scale
        shift=torch.stack([v.amax((1,2)) for v in (z,zs,zo)]).amax(0)[:,None,None]
        if x.size(1)==13:
            pri=z.new_tensor([1/18]*9+[1/8]*4)[None,:,None]
        else:pri=1.
        h=((z-shift).exp()*pri).sum(1)@self.weights.T
        soft=((zs-shift).exp()*pri).sum(1)
        avg=((zo-shift).exp()*pri).sum(1)
        return h,soft,avg,shift[:,0]

    def groups(self,evidence,kind,temp):
        h,soft,avg,shift=evidence
        if kind=="hard":weight=0.
        elif kind in ("always","output"):weight=1.
        elif kind=="shared":weight=.5
        elif kind=="local":weight=(temp*self.positive).sigmoid()
        elif kind=="reverse":weight=(-temp*self.verified).sigmoid()
        elif kind in ("verified","verified_output"):weight=(temp*self.verified).sigmoid()
        else:raise ValueError(kind)
        target=avg if kind in ("output","verified_output") else soft
        parts=((1-weight)*h+weight*target)*self.mass
        group=h.new_zeros((len(h),self.count)).index_add_(1,self.g,parts)
        return group.clamp_min(1e-30).log()+shift


def prepare(photos,target,source,spans,scale,k):
    centered=F.normalize(photos-photos.mean(0))
    labels=cluster_assignments(centered,64)
    _,labels=labels.unique(sorted=True,return_inverse=True)
    count=int(labels.max())+1
    # Stratify by the inferred groups so every held-out panel contains a
    # representative whenever there are at least two photos in that group.
    rng=torch.Generator().manual_seed(81321)
    parts=[[],[]]
    for g in range(count):
        ix=(labels==g).nonzero().flatten().cpu()
        perm=ix[torch.randperm(len(ix),generator=rng)]
        if len(perm)<2:raise ValueError("A gallery group cannot be cross-fitted")
        for j in (0,1):parts[j].append(perm[j::2])
    parts=[torch.cat(p).cuda() for p in parts]
    banks=[]
    for train,test in (parts,parts[::-1]):
        coarse=photos.new_zeros((count,len(target)))
        for ids in train.split(256):
            prob=((photos[ids]@target.T)*scale).softmax(1)
            coarse.index_add_(0,labels[ids],prob)
        coarse/=labels[train].bincount()[:,None]
        banks.append(Readout(spans,coarse,photos[test],labels[test],source,target,scale,k))
    return labels,banks
