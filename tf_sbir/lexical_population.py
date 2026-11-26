"""Identify latent visual populations from incomplete, domain-dependent public names."""

from dataclasses import dataclass

import torch

from .replicated_concept_simplex import Observations, select_anchors


LANDMARKS = 1024


def split_visual_experiment(photos, max_neighbors, seed, landmarks=LANDMARKS):
    """Disjoint probe images and two donor populations. Condition on the fixed probes."""
    perm = torch.randperm(len(photos), generator=torch.Generator().manual_seed(seed)).to(photos.device)
    locations = perm[:landmarks]
    halves = perm[landmarks:].chunk(2)
    observations = []
    for candidates in halves:
        projected = photos[locations] @ photos[candidates].T
        observations.append(candidates[projected.topk(max_neighbors, dim=1).indices])
    return locations, observations


def aggregate_native(native, neighbors, k):
    result = native.new_empty(len(neighbors), native.size(1))
    for start in range(0,len(neighbors),128):
        sl = slice(start, min(len(neighbors),start+128))
        result[sl] = native[neighbors[sl,:k]].mean(1)
    return result


@dataclass
class Population:
    obs: Observations
    rates: tuple
    densities: tuple


def signature(readers, donor_lists, k):
    # The (uniform) sampling unit is a probe. Each column is the likelihood of its noun.
    fields = [torch.stack([aggregate_native(native, ids, k) for ids in donor_lists]) for native in readers]
    rates = tuple(f.mean((0,1)) for f in fields)
    half_shapes = [torch.nn.functional.normalize(f.transpose(1,2),p=1,dim=-1,eps=1e-24) for f in fields]
    pooled = tuple(torch.nn.functional.normalize(f.mean(0).T,p=1,dim=1,eps=1e-24).contiguous() for f in fields)
    obs = Observations(half_shapes[0].double(),half_shapes[1].double(),(rates[0]*rates[1]).sqrt().double(),
                       pooled[0],pooled[1],rates[1])
    return Population(obs,rates,pooled)


def prevalence(pop, ids, *, single=False, iterations=200):
    """Maximum likelihood of uniform reference-probe exposure in both naming experiments."""
    # Columns sum to one over probe positions for each domain; combining both experiments
    # assigns equal probability to choosing either reader. EM estimates only mixture masses.
    components = [d[ids] for d in (pop.densities[1:] if single else pop.densities)]
    emissions = torch.cat(components,dim=1).T.double().contiguous()/len(components)
    prior = emissions.new_full((len(ids),),1/len(ids))
    for step in range(iterations):
        likelihood = (emissions @ prior).clamp_min(1e-24)
        update = prior * ((1/likelihood) @ emissions) / len(emissions)
        update = update / update.sum()
        if step % 16 == 15 and (update-prior).abs().max() < 1e-7:
            prior = update
            break
        prior = update
    error = -(len(emissions) * (emissions @ prior)).clamp_min(1e-24).log().mean().item()
    return prior.float(), error


def condition(evidence, ratio):
    mass = evidence * ratio
    return mass / mass.sum(1,keepdim=True).clamp_min(1e-30)


def codes(pop, source, target, ids, *, correction="both"):
    prior, deviance = prevalence(pop,ids,single=correction=="target_fit")
    source_ratio = prior / pop.rates[0][ids].clamp_min(1e-20)
    target_ratio = prior / pop.rates[1][ids].clamp_min(1e-20)
    ones = torch.ones_like(prior)
    if correction == "raw":
        source_ratio = target_ratio = ones
    elif correction == "target_only":
        source_ratio = ones
    elif correction == "source_only":
        target_ratio = ones
    elif correction == "flat_prior":
        source_ratio = 1/pop.rates[0][ids].clamp_min(1e-20)
        target_ratio = 1/pop.rates[1][ids].clamp_min(1e-20)
    elif correction not in ("both","target_fit"):
        raise ValueError(correction)
    return condition(source[:,ids],source_ratio), condition(target[:,ids],target_ratio), {
        "kl_uniform_to_reconstructed":deviance,"prior":prior.cpu().tolist(),
        "effective_components":float(prior.square().sum().reciprocal()),
    }


def toy():
    # Two possible subjects have different chances of eliciting their identifying word.
    centers = torch.tensor([[.9,.1],[.1,.9]])
    pi = torch.tensor([.5,.5])
    report = torch.tensor([.2,.8])
    density = centers.contiguous()
    pop = Population(None,(report*pi,report*pi),(density,density))
    corrected,_,_ = codes(pop,torch.tensor([[2/3,1/3]])*report,torch.ones(1,2),torch.arange(2))
    raw = condition(torch.tensor([[2/3,1/3]])*report,torch.ones(2))
    assert corrected.argmax()==0 and raw.argmax()==1 and (corrected-torch.tensor([[2/3,1/3]])).abs().max()<1e-6
    return {"raw":raw.tolist(),"corrected":corrected.tolist()}
