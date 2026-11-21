"""A word-conditioned, training-free retrieval score for arbitrary image domains."""

import math

import torch
import torch.nn.functional as F


def native_diffusion(features, profiles, scale):
    """All non-self images vote with the frozen model's native cosine scale."""
    n = len(features)
    if n == 1:
        return profiles, {"effective_neighbors": [1.,1.,1.]}
    output = torch.empty_like(profiles)
    counts = []
    for start in range(0,n,256):
        x = features[start:start+256]
        logits = (x @ features.T) * scale
        row = torch.arange(len(x),device=x.device)
        logits[row,start+row] = -torch.inf
        transition = logits.softmax(1)
        output[start:start+len(x)] = transition @ profiles
        counts.append(transition.square().sum(1).reciprocal())
    count = torch.cat(counts)
    quantiles = count.quantile(count.new_tensor([.1,.5,.9])).tolist()
    return output, {"effective_neighbors":quantiles,"fraction_below_two":float((count < 2).float().mean())}


class WordWitness:
    """Gallery observations as row-normalized distributions over any shared lexicon."""
    def __init__(self, profiles):
        if profiles.ndim!=2 or min(profiles.shape)<1 or not torch.isfinite(profiles).all() or (profiles<0).any():
            raise ValueError("Expected finite nonnegative gallery profiles")
        sums = profiles.sum(1,keepdim=True)
        if (sums<=0).any():
            raise ValueError("Empty gallery profile")
        self.profile = profiles / sums
        self.count = self.profile.sum(0,dtype=torch.double).float()
        self.correction = torch.empty_like(self.profile)
        # A word can have at most one photo carrying over half of its mass.
        # Evaluate that entry's leave-one-out expectation from the other photos
        # directly, so near-identical large positive sums never have to cancel.
        largest, owners = self.profile.max(0)
        self.special = (largest > .5*self.count).nonzero().flatten()
        self.owners = owners[self.special]
        for start in range(0,len(self.profile),256):
            b = self.profile[start:start+256]
            positive = (2*b<=self.count) & (b>0)
            denom = torch.where(positive,self.count-b,1.)
            self.correction[start:start+len(b)] = torch.where(positive,b.square()/denom,0.)
        self.correction_sum = self.correction.sum(1)
        self.others = self.profile[:,self.special].clone()
        col = torch.arange(len(self.special),device=profiles.device)
        self.others[self.owners,col] = 0.
        # Zero mass in the other photos means that this word has no witness.
        self.others /= self.others.sum(0,keepdim=True).clamp_min(torch.finfo(profiles.dtype).tiny)

    def scores(self, likelihood):
        b, d = self.profile, self.correction
        e = likelihood @ b.T
        if len(b)==1:
            return {"linear":e,"independent":e,"witness":e}
        response = (e @ b) / self.count.clamp_min(torch.finfo(b.dtype).tiny)
        joint = likelihood * response
        witnessed = joint @ b.T + joint @ d.T - e * (likelihood @ d.T)
        separate = response @ b.T + response @ d.T - e * self.correction_sum
        if self.special.numel():
            response_change = (e @ self.others - response[:,self.special]) * b[self.owners,self.special]
            ids = self.owners.expand(len(e),-1)
            witnessed.scatter_add_(1,ids,response_change*likelihood[:,self.special])
            separate.scatter_add_(1,ids,response_change)
        return {"linear":e,"independent":e*separate,"witness":witnessed}


class DomainNeutralIndex:
    """Build from gallery image vectors, isolated public-word vectors and CLIP scale."""
    @torch.inference_mode()
    def __init__(self, gallery_images, public_words, logit_scale):
        if gallery_images.ndim!=2 or public_words.ndim!=2 or not len(gallery_images) or not len(public_words):
            raise ValueError("Provide a nonempty matrix of gallery images and a nonempty matrix of words")
        if gallery_images.size(1)!=public_words.size(1) or gallery_images.device!=public_words.device:
            raise ValueError("Gallery and word embeddings must share a dimension and device")
        self.words = F.normalize(public_words.float())
        self.scale = float(logit_scale)
        if not math.isfinite(self.scale) or self.scale<=0:
            raise ValueError("The frozen logit scale must be positive and finite")
        gallery = F.normalize(gallery_images.float())
        profiles = ((gallery @ self.words.T)*self.scale).softmax(1)
        profiles,self.graph_statistics = native_diffusion(gallery,profiles,self.scale)
        self.model = WordWitness(profiles)

    @torch.inference_mode()
    def score(self, query_images):
        if query_images.ndim==1:
            query_images=query_images.unsqueeze(0)
        if query_images.ndim!=2 or query_images.size(1)!=self.words.size(1):
            raise ValueError("Expected one query vector or a matrix of query vectors")
        logits = (F.normalize(query_images.float()) @ self.words.T)*self.scale
        likelihood = (logits-logits.amax(1,keepdim=True)).exp()
        return self.model.scores(likelihood)
