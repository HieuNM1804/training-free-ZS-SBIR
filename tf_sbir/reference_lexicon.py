"""Assign public nouns jointly so independently read images can identify anonymous visual cells."""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


def rectangular_assignment(utility):
    groups, words = linear_sum_assignment(utility.double().cpu().numpy(), maximize=True)
    if not np.array_equal(groups, np.arange(len(utility))):
        raise ValueError("Expected one public name per group")
    return torch.from_numpy(words).to(utility.device)


def divide(images, groups, seed, fit_steps=15):
    """Form a reference task on a random half-gallery. Return held-out observations for naming."""
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(images), generator=rng).to(images.device)
    fit, name = perm.chunk(2)
    basis = images[fit[:groups]]
    fit_images = images[fit]
    for _ in range(fit_steps):
        ids = (fit_images @ basis.T).argmax(1)
        # Matrix reduction also defines deterministic arithmetic on GPU.
        membership = F.one_hot(ids,num_classes=groups).T.float()
        means = F.normalize(membership @ fit_images,dim=1)
        basis = torch.where(membership.sum(1,keepdim=True)>0,means,basis)
    destination = (images @ basis.T).argmax(1)
    support = destination[name]
    if torch.bincount(support,minlength=groups).min()==0:
        # The empty group has no observations with which to define an identification question.
        # Remove that option, then reassign all photos, including its original members.
        present = torch.bincount(support,minlength=groups)>0
        destination = (images @ basis[present].T).argmax(1)
        support = destination[name]
    return name, support, destination


def representative_listeners(support_ids, groups, bank, rng, draws):
    order = support_ids.argsort(stable=True)
    counts = torch.bincount(support_ids,minlength=groups).cpu()
    starts = counts.cumsum(0)-counts
    # Equal-weight cells implement a balanced reference game, independent of gallery class sizes.
    fractions = torch.rand((groups,draws),generator=rng)
    offsets = starts[:,None] + (fractions * counts[:,None]).long()
    return bank[order[offsets.to(order.device)]]


def likelihood(logits, names):
    chosen = logits[:,:,names]
    row = torch.arange(len(names),device=names.device)
    return (chosen[row,:,row].mean() - chosen.logsumexp(-1).mean()).item()


def choose_code(logits, independent_scores, sweeps=3):
    """Exact one-symbol substitutions in a finite code. All other assigned symbols compete."""
    m,draws,words = logits.shape
    mrows = torch.arange(m,device=logits.device)
    names = rectangular_assignment(independent_scores)
    start = names.clone()
    native_score = likelihood(logits,names)
    # Probabilities scaled by a different constant per observation preserve restricted-softmax odds.
    exp = (logits-logits.amax(-1,keepdim=True)).exp()
    numerator = logits.mean(1)
    taken = torch.zeros(words,dtype=torch.bool,device=logits.device)
    taken[names] = True
    edits = 0
    for _ in range(sweeps):
        before = edits
        for g in range(m):
            # Re-summing the remaining probabilities avoids cancellation of a dominant token.
            other_names = names[mrows!=g]
            other = exp[:,:,other_names].sum(-1)
            denominator = (other[:,:,None] + exp).clamp_min(1e-35).log().mean((0,1))
            utility = numerator[g]/m - denominator
            previous = int(names[g])
            taken[previous] = False
            best = int(utility.masked_fill(taken,-torch.inf).argmax())
            if utility[best] > utility[previous]+1e-7:
                names[g], edits = best, edits+1
            taken[names[g]] = True
        if edits==before:
            break
    value=likelihood(logits,names)
    if names.unique().numel()!=m or value+3e-6<native_score:
        raise ValueError("A code exchange reduced the objective")
    return start,names,{"before":native_score,"after":value,"edits":edits}


def reference_task(photos, source_logits, target_logits, queries, groups, seed, draws, sweeps,
                   audit=False, fit_steps=15):
    name, label, owner = divide(photos,groups,seed,fit_steps=fit_steps)
    groups = int(label.max())+1
    membership = F.one_hot(label,num_classes=groups).T.float()
    membership /= membership.sum(1,keepdim=True)
    full = target_logits[name]
    # An independent word's discrimination is measured against other cells for that same word.
    evidence = membership @ full.softmax(1)
    score = evidence.clamp_min(1e-24).log()
    one_vs_all = score - score.logsumexp(0,keepdim=True)
    rng = torch.Generator().manual_seed(seed+1)
    representatives = representative_listeners(label,groups,torch.arange(len(full),device=full.device),rng,draws)
    photo_listeners = full[representatives]
    listeners = torch.cat((photo_listeners,source_logits[name][representatives]),dim=1)
    independent, joint, quality = choose_code(listeners,one_vs_all,sweeps)
    names = {"joint":joint,"one_to_one":independent,"free_names":one_vs_all.argmax(1),
             "common_names":rectangular_assignment(score)}
    if audit:
        names["target_reader_only"] = choose_code(photo_listeners,one_vs_all,sweeps)[1]
        # These photos fitted the visual partition but none entered the verbal objective.
        held_mask = torch.ones(len(owner),dtype=torch.bool,device=owner.device)
        held_mask[name] = False
        held_ids = torch.where(held_mask)[0]
        held_labels = owner[held_ids]
        held_counts = torch.bincount(held_labels,minlength=groups).clamp_min(1)
        quality["held_accuracy"] = {}
        for arm,code in names.items():
            for reader,bank in (("source",source_logits),("target",target_logits)):
                correct = (bank[held_ids][:,code].argmax(1) == held_labels).float()
                quality["held_accuracy"][arm+"-"+reader] = (correct/held_counts[held_labels]).sum().item()/groups
    by_arm = {}
    for arm,code in names.items():
        choice = queries[:,code]
        by_arm[arm] = choice/choice.sum(1,keepdim=True).clamp_min(1e-24)
    # All of these arms receive precisely the same population and all the same query features.
    baseline = queries @ evidence.T
    by_arm["mean_cell"] = baseline
    by_arm["soft_cell"] = baseline / baseline.sum(1,keepdim=True).clamp_min(1e-24)
    return by_arm, owner, names, quality


def fixture():
    # Two independent discriminators agree about the rows, while one of their symbols
    # makes the other row particularly hard to recognize when both must coexist.
    rng = torch.Generator().manual_seed(42)
    for _ in range(64):
        logits = torch.randn((3,4,7),generator=rng)*2
        score = logits.softmax(2).mean(1).log()
        unary = score-score.logsumexp(0,keepdim=True)
        start,final,q = choose_code(logits,unary)
        if q["after"] > q["before"]+.05:
            import itertools
            optimum = max(likelihood(logits,torch.tensor(p)) for p in itertools.permutations(range(7),3))
            assert q["after"] <= optimum+2e-6
            return {"separate":start.tolist(),"joint":final.tolist(),"log_likelihood":q,"global_optimum":optimum}
    raise ValueError("No competition interaction in fixture")
