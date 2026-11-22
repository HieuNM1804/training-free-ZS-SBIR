"""Which part of a word's depiction preference is observable in the target photo likelihood?"""

from dataclasses import dataclass

import torch


def shuffle_photos(bank, seed):
    n, words = bank.shape
    rng = torch.Generator(device=bank.device).manual_seed(seed)
    reorder = torch.randperm(n, device=bank.device, generator=rng)
    shifts = torch.randint(n, (words,), device=bank.device, generator=rng)
    rows = torch.arange(n, device=bank.device)[:, None]
    shuffled = torch.empty_like(bank)
    for begin in range(0, words, 96):
        stop = min(begin+96, words)
        offsets = (rows + shifts[None, begin:stop]) % n
        cols = torch.arange(begin, stop, device=bank.device)[None]
        shuffled[:, begin:stop] = bank[reorder[offsets], cols]
    return shuffled


@dataclass
class Depiction:
    centered: torch.Tensor
    observable: torch.Tensor
    linear: torch.Tensor
    checks: torch.Tensor
    statistics: torch.Tensor

    def columns(self, kind):
        if kind == "visible":
            return self.observable
        if kind == "scalar":
            return self.linear
        if kind == "raw":
            return self.centered
        raise ValueError(kind)


class GalleryChannel:
    def __init__(self, bank):
        self.bank = bank.contiguous()
        self.total = self.bank.sum(0, dtype=torch.float64).float()
        if torch.any(self.total <= 0) or not torch.isfinite(self.total).all():
            raise ValueError("A public name has no finite gallery mass")
        self.log_total = self.total.log()

    def analyze(self, logits, difference):
        weights = (logits-logits.amax(1,keepdim=True)).exp()
        joint_w = weights*self.total
        prior = joint_w / joint_w.sum(1,keepdim=True)
        centered = difference-(difference*prior).sum(1,keepdim=True)
        image = weights @ self.bank.T
        posterior_mean = ((weights*centered) @ self.bank.T) / image.clamp_min(1e-30)
        observed = (posterior_mean @ self.bank) / self.total
        photo_prior = image / image.sum(1,keepdim=True).clamp_min(1e-30)
        average = (observed*prior).sum(1,keepdim=True)
        observed = observed-average
        variance = (prior*centered.square()).sum(1)
        attributable = (photo_prior*posterior_mean.square()).sum(1)
        slope = torch.where(variance>1e-10,attributable/variance.clamp_min(1e-10),0.)
        linear = slope[:,None]*centered
        # The change in observed gallery log mass is the posterior mean. Its
        # variance equals the word-space inner product of d and its round trip.
        inner = (prior*centered*observed).sum(1)
        relative = (inner-attributable).abs()/variance.clamp_min(1e-5)
        remainder = (prior*(observed-linear).square()).sum(1)/variance.clamp_min(1e-8)
        checks = torch.stack((relative,(-slope).clamp_min(0),(slope-1).clamp_min(0)),1)
        stats = torch.stack((slope,remainder.sqrt(),variance.sqrt()),1)
        return Depiction(centered,observed,linear,checks,stats)

    def adjusted(self, logits, change):
        # Preserve total target likelihood of this individual view. Reweighting
        # words cannot simply brighten a crop relative to all the other crops.
        after = logits+change
        shift = torch.logsumexp(logits+self.log_total,1)-torch.logsumexp(after+self.log_total,1)
        return after+shift[:,None]


def region_score(logits, bank, batch):
    # 2 corresponding poses × 13 regions; this is the locked same-view reader.
    x = logits.view(batch,2,13,-1)
    weights = (x-x.amax((1,2,3),keepdim=True)).exp()
    scores = (weights.flatten(0,2)@bank.T).view(batch,2,13,-1).clamp_min_(1e-30)
    prior = scores.new_tensor([1/18]*9+[1/8]*4).view(1,1,13,1)
    multiplier = (scores / scores.mean(-1,keepdim=True).clamp_min(1e-30))*prior
    chosen = (scores*multiplier).sum(2) / multiplier.sum(2).clamp_min(1e-30)
    return chosen.mean(1).clamp_min(1e-30).log()


def global_score(logits, bank):
    w = (logits-logits.amax(1,keepdim=True)).exp()
    return (w@bank.T).clamp_min_(1e-30).log()


def algebra_checks(device):
    dtype = torch.float32
    # Indistinguishable words share all target photos. Equal and opposite
    # depiction effects then produce zero observable effect.
    bank = torch.tensor([[.6,.6,0.],[.4,.4,0.],[0.,0.,1.]],device=device,dtype=dtype)
    zero = torch.zeros((1,3),device=device)
    contrast = torch.tensor([[1.,-1.,0.]],device=device)
    alias = GalleryChannel(bank).analyze(zero,contrast)
    identity = GalleryChannel(torch.eye(3,device=device)).analyze(zero,contrast)
    l = torch.tensor([[.5,0.,-.1]],device=device)
    d = torch.tensor([[-.9,.3,1.1]],device=device)
    chan = GalleryChannel(bank)
    obs = chan.analyze(l,d)
    exact_image = (l.exp()@bank.T)
    raw_centered = d-((l.exp()*chan.total*d).sum(1,keepdim=True)/(l.exp()*chan.total).sum(1,keepdim=True))
    predicted = ((l.exp()*raw_centered)@bank.T) / exact_image
    def posterior(x):
        y=x.exp()@bank.T
        return (y/y.sum(1,keepdim=True)).log()
    step = .001
    numerical = (posterior(l+step*d)-posterior(l-step*d))/(2*step)
    adjusted = chan.adjusted(l,.61*obs.observable)
    scale_err = ((adjusted.exp()@bank.T).sum() / exact_image.sum()-1).abs().item()
    info={"alias_error":alias.observable.abs().max().item(),
          "identity_error":(identity.observable-contrast).abs().max().item(),
          "finite_difference_error":(predicted-numerical).abs().max().item(),
          "mass_error":scale_err,"inner_product_error":obs.checks.abs().max().item()}
    if max(info.values())>.001:
        raise ValueError(f"Gallery observability identities failed {info}")
    return info
