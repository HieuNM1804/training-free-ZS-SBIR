"""Closed-form two-by-two minimum-collision matching of lexical explanations."""

import torch

from .domain_neutral_witness import DomainNeutralIndex


def matching_score(collision, direct, direct2, witness1, witness2, same2):
    # Exact expectation over the two possible matchings of two independent
    # i.i.d. word pairs. Ties between the matchings are broken uniformly.
    return ((1-collision)*(direct*witness1-same2) + direct2*witness1
            + direct*witness2 - direct2*witness2)


def joint_distribution(b):
    """Small reference only. Rows and columns both have marginal b."""
    pi,pj=b[:,None],b[None,:]
    c=b.square().sum()
    return pi*pj*(1-c+pi+pj-pi*pj)-torch.diag(b.square()*(1-c))


class ComplementaryIndex(DomainNeutralIndex):
    def __init__(self,gallery_images,public_words,logit_scale):
        super().__init__(gallery_images,public_words,logit_scale)
        b=self.model.profile
        self.squared=b.square()
        self.correction2=self.model.correction*b
        self.collision=self.squared.sum(1)
        self.correction2_sum=self.correction2.sum(1)
        self.complement=(1-self.collision).clamp_min(0)
        self.removed=(self.squared*(1-b).square()).sum(1)

    @torch.inference_mode()
    def score(self,query_images):
        return self.score_likelihood(self.likelihood(query_images))

    def score_likelihood(self,l):
        bank=self.model
        b,d=bank.profile,bank.correction
        e=l@b.T
        if len(b)==1:
            return {k:e for k in ("linear","same_word","independent","paired","reverse_pairing",
                                   "reject_repeats","rejection_marginals","collision_only")}
        anchor=(e@b)/bank.count.clamp_min(torch.finfo(b.dtype).tiny)
        joint=l*anchor
        e2=l@self.squared.T
        h1=anchor@b.T+anchor@d.T-e*bank.correction_sum
        h2=anchor@self.squared.T+anchor@self.correction2.T-e*self.correction2_sum
        same1=joint@b.T+joint@d.T-e*(l@d.T)
        same2=joint@self.squared.T+joint@self.correction2.T-e*(l@self.correction2.T)
        if bank.special.numel():
            ids=bank.owners.expand(len(l),-1)
            mass=b[bank.owners,bank.special]
            diff=(e@bank.others-anchor[:,bank.special])*mass
            h1.scatter_add_(1,ids,diff)
            h2.scatter_add_(1,ids,diff*mass)
            same1.scatter_add_(1,ids,diff*l[:,bank.special])
            same2.scatter_add_(1,ids,diff*(l[:,bank.special]*mass))
        independent=e*h1
        paired=matching_score(self.collision,e,e2,h1,h2,same2)
        # Conditional-without-replacement and its independent, same-marginal
        # control are defined only when two different names have positive mass.
        # At a point mass all possible couplings with preserved marginals coincide.
        valid=self.complement>torch.finfo(b.dtype).eps
        support=torch.where(valid,self.complement,1.)
        distinct=torch.where(valid,(independent-same2)/support,independent)
        new_margins=torch.where(valid,((e-e2)/support)*((h1-h2)/support),independent)
        return {"linear":e,"same_word":same1,"independent":independent,"paired":paired,
                "reverse_pairing":2*independent-paired,"reject_repeats":distinct,
                "rejection_marginals":new_margins,"collision_only":same2/self.collision.clamp_min(1e-30)}

    @torch.inference_mode()
    def explain_pair(self,query_image,first,second):
        # Apply the same joint matrix to a vector without building a W-by-W matrix.
        l=self.likelihood(query_image)
        if len(l)!=1:
            raise ValueError("Pair explanations take one query")
        def transition(b,x):
            b2=b.square()
            c=b2.sum()
            bx=b@x
            b2x=b2@x
            return b*((1-c)*bx+b2x)+b2*(bx-b2x)-(1-c)*b2*x
        def score_and_drop(candidate):
            b=self.model.profile
            own=b[candidate].double()
            query=l[0].double()
            if len(b)==1:
                return own@query,own*query
            def blocks():
                for part in (b[:candidate],b[candidate+1:]):
                    for block in part.split(128):
                        if len(block):
                            yield block.double()
            count=torch.zeros_like(own)
            weighted=torch.zeros_like(own)
            for block in blocks():
                count+=block.sum(0)
                weighted+=(block@query)@block
            divisor=count.clamp_min(torch.finfo(count.dtype).tiny)
            inv=torch.where(count>0,divisor.reciprocal(),0.)
            h=weighted*inv
            forward=transition(own,h)
            route=transition(own,query)*inv
            backwards=torch.zeros_like(own)
            diag=torch.zeros_like(own)
            b2=own.square()
            c=b2.sum()
            # diag(J M) uses two low-rank moments and one diagonal.
            # M = diag(count)^(-1) B_other^T B_other.
            for block in blocks():
                backwards+=(block@route)@block
                first_m=(block@(own*inv))@block
                second_m=(block@(b2*inv))@block
                diag+=(own*(1-c)+b2)*first_m + (own-b2)*second_m
                diag-=(1-c)*b2*inv*block.square().sum(0)
            return query@forward,query*(forward+backwards-query*diag)
        s1,d1=score_and_drop(first)
        s2,d2=score_and_drop(second)
        gap=s1-s2
        margins=gap-(d1-d2)
        return {"original_margin":gap,"margin_after_each_deletion":margins,
                "most_damaging_word":int(margins.argmin())}
