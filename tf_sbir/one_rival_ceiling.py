"""Attainable retrieval ceilings when only one false category may be demoted in a frozen ranking."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json

import torch

from .branch_view_study import PrefixReplay
from .conditional_agreement_study import Replay
from .description_block_results import Population
from .description_gallery_transfer import OUT
from .gallery_name_rank_decomposition import OUT as DONOR_OUT, semantic_inputs
from .native_block_probe import deterministic
from .photo_name_evaluation import KD
from .photo_name_interaction import record, verified
from .quickdraw_locked import KD_FUNCTION_SHA
from .replicated_simplex_study import evaluator
from .wordnet_renaming_pilot import SPLITS


ARMS = ("semantic", "neighbors_64", "opposite_fold_rival", "first_wrong_rival",
        "best_one_rival", "same_rival_fully_demoted")
SELECTORS = ARMS[2:5]


def optimum(positions, earliest, cutoff, precision_k):
    """Per-row optimum for every possible foil. Positions of real positives are one-based."""
    count = positions.shape[1]
    j = torch.arange(1, count + 1, device=positions.device)
    if not (earliest >= j[None,:,None]).all():
        raise ValueError("Earliest positive positions must be in retrieval order")
    contributions = j[None,:,None] / earliest.double()
    if cutoff is None:
        ap = contributions.mean(1)
        h = torch.full_like(ap, count, dtype=torch.long)
        return ap, (earliest <= precision_k).double().sum(1) / precision_k, h, ap
    if precision_k != cutoff:
        raise ValueError("This finite-cutoff derivation assumes K for precision equals the AP cutoff")
    h0 = (positions <= cutoff).sum(1)
    hmax = (earliest <= cutoff).sum(1)
    means = contributions.cumsum(1) / j[None,:,None]
    allowed = (j[None,:,None] >= h0[:,None,None]) & (j[None,:,None] <= hmax[:,None])
    best, reverse_index = means.masked_fill(~allowed, -torch.inf).flip(1).max(1)
    best = torch.where(hmax > 0, best, 0.)
    h = torch.where(hmax > 0, count - reverse_index, 0)
    clear = means.gather(1, (hmax-1).clamp_min(0)[:,None]).squeeze(1)
    clear = torch.where(hmax > 0, clear, 0.)
    return best, h.double()/precision_k, h, clear


def all_foils(ranked_classes, truth, nclass, cutoff, pk):
    """Compute earliest positions from gallery class counts immediately before each positive."""
    batch, length = ranked_classes.shape
    device = ranked_classes.device
    values = torch.empty((batch,nclass,2), device=device, dtype=torch.float64)
    full_ap = torch.empty_like(values[:,:,0])
    counts = torch.empty((batch,nclass), device=device, dtype=torch.long)
    cats = torch.arange(nclass,device=device)
    rank = torch.arange(1,length+1,device=device)
    for y in truth.unique().tolist():
        take = torch.where(truth == y)[0]
        # Bound transient memory while retaining the original whole-batch matrix multiplication and rank order.
        for idx in take.split(48):
            table = ranked_classes[idx]
            positions = rank.expand(len(idx),-1)[table == y].reshape(len(idx),-1)
            prefix = (table[:,:,None] == cats).cumsum(1,dtype=torch.int32)
            early = positions[:,:,None] - prefix.gather(1,(positions-1)[:,:,None].expand(-1,-1,nclass))
            early[:,:,y] = positions
            ap, precision, h, fully = optimum(positions, early, cutoff, pk)
            values[idx] = torch.stack((ap,precision),dim=2)
            counts[idx], full_ap[idx] = h, fully
    return values, counts, full_ap


def best_rival(values, truth):
    roles = torch.arange(values.shape[1],device=values.device)
    ap = values[:,:,0].masked_fill(roles == truth[:,None], -torch.inf)
    precision = values[:,:,1].masked_fill(ap != ap.amax(1,keepdim=True), -torch.inf)
    return precision.argmax(1)


def witness(order, ranked, truth, foil, keep_hits):
    hits = ranked == truth[:,None]
    ranks = torch.arange(ranked.shape[1],device=ranked.device)
    end = (hits & (hits.cumsum(1) == keep_hits[:,None])).int().argmax(1)
    prefix = (keep_hits[:,None] > 0) & (ranks <= end[:,None])
    bucket = torch.where(prefix,(ranked == foil[:,None]).long(),2)
    return order.gather(1,bucket.argsort(dim=1,stable=True))


def brute_check():
    """Enumerate all legal positions of indistinguishable foils against a fixed nonfoil order."""
    k = 3
    max_error, cases, candidates = 0., 0, 0
    for src in itertools.product((0,1,2), repeat=6):
        if not all(i in src for i in (0,1,2)):
            continue
        core = [a for a in src if a != 2]
        old_foil = [i for i,a in enumerate(src) if a == 2]
        pos = torch.tensor([[i+1 for i,a in enumerate(src) if a == 1]])
        early = torch.tensor([[[i+1] for i,v in enumerate(core) if v == 1]])
        def ap_p(seq, cutoff, P):
            h = [i+1 for i,v in enumerate(seq[:cutoff]) if v == 1]
            return sum(j/r for j,r in enumerate(h,1))/max(1,len(h)), sum(v == 1 for v in seq[:P])/P
        best = {k:(0.,0.), None:(0.,0.)}
        for slots in itertools.combinations(range(len(src)),len(old_foil)):
            if any(after < before for after,before in zip(slots,old_foil)):
                continue
            other = iter(core)
            seq = [2 if i in slots else next(other) for i in range(len(src))]
            candidates += 1
            for cap in best:
                best[cap] = max(best[cap],ap_p(seq,cap,k))
        for cap,real in best.items():
            value, prec, _, _ = optimum(pos,early,cap,k)
            max_error = max(max_error,abs(float(value)-real[0]),abs(float(prec)-real[1]))
        cases += 1
    if max_error > 1e-12:
        raise ValueError(f"Exhaustive finite-cutoff certificate failed: {max_error}")
    return {"initial_sequences":cases,"legal_edits":candidates,"maximum_error":max_error}


def native_foils(split, qi, labels, rival_revision):
    assigned = torch.full_like(qi,-1)
    for held in (0,1):
        data = verified(OUT/f"wrong-alternatives-{split}-{held}.pt",rival_revision)
        perm = qi.argsort()
        where = perm[torch.searchsorted(qi[perm],data["query_ids"])]
        if not torch.equal(qi[where],data["query_ids"]) or (assigned[where] != -1).any():
            raise ValueError("Fold queries do not form a subset with unique assignments")
        rivals = data["hard_foils"]["all_classes"]
        assigned[where] = torch.tensor([rivals[int(y)] for y in labels[where]])
    if (assigned < 0).any() or (assigned == labels).any():
        raise ValueError("Every query needs a distinct native foil")
    return assigned


@torch.inference_mode()
def run(a):
    split,part = a.split,a.part
    qi,labels,gi,classes,sources,_,q,p,b = semantic_inputs(split,a.name_revision,a.native_revision)
    assigned = native_foils(split,qi,labels,a.rival_revision)
    previous = verified(DONOR_OUT/f"summary-{split}.pt",a.baseline_revision)
    if not torch.equal(qi,previous["query_ids"]) or not torch.equal(gi,previous["gallery_ids"]):
        raise ValueError("Baseline archive and current populations must agree")
    kd = evaluator(KD)
    replay = PrefixReplay(kd,classes,split)
    full_replay = Replay(kd,classes,split)
    cutoff, pk = (200,200) if split == "sketchy_2" else (None,100)
    take = torch.arange(part,len(qi),2)
    gt = classes.cuda()
    p,b = p.T.contiguous(),b.T.contiguous()
    vals, counts, clears, choices = [], [], [], []
    rows = {k:[] for k in ARMS}
    maximum, full_error = 0., 0.
    for ids in take.split(192):
        truth = labels[ids].cuda()
        bs = len(ids)
        rank = (q[ids]@p).topk(len(gi),dim=1).indices
        ranked = gt[rank]
        table,h,fully = all_foils(ranked,truth,len(classes.unique()),cutoff,pk)
        first_wrong = ranked.gather(1,(ranked != truth[:,None]).int().argmax(1,keepdim=True)).flatten()
        foils = torch.stack((assigned[ids].cuda(),first_wrong,best_rival(table,truth)),1)
        rr = torch.arange(bs,device="cuda")
        orders = {"semantic":rank,"neighbors_64":(q[ids]@b).topk(len(gi),dim=1).indices}
        for j,k in enumerate(SELECTORS):
            f = foils[:,j]
            orders[k] = witness(rank,ranked,truth,f,h[rr,f])
        oracle = foils[:,-1]
        tail = (ranked == oracle[:,None]).long().argsort(dim=1,stable=True)
        orders["same_rival_fully_demoted"] = rank.gather(1,tail)
        with ThreadPoolExecutor(max_workers=5) as pool:
            for k,order in orders.items():
                reported = replay.score(order[:,:cutoff],labels[ids],pool).double()
                rows[k].append(reported)
                if cutoff is not None:
                    original = full_replay.chunk((order[:1].int().cpu(), labels[ids[:1]])).double()
                    full_error = max(full_error, float((original - reported[:1]).abs().max()))
            if full_error > 2e-6:
                raise ValueError("Full-gallery original KD disagrees with prefix evaluation")
        recorded = [rows[k][-1] for k in SELECTORS]
        exact = [table[rr,foils[:,j]].cpu() for j in range(len(SELECTORS))]
        checks = [float((a-b).abs().max()) for a,b in zip(recorded,exact)]
        checks += [float((table[rr,truth].cpu()-rows["semantic"][-1]).abs().max()),
                   float((fully[rr,oracle].cpu()-rows["same_rival_fully_demoted"][-1][:,0]).abs().max())]
        maximum = max(maximum,*checks)
        if maximum > 2e-6:
            raise ValueError(f"Bound not attained by original evaluator: {checks}")
        vals.append(table.cpu());counts.append(h.int().cpu());clears.append(fully.cpu());choices.append(foils.cpu())
    rows = {k:torch.cat(z) for k,z in rows.items()}
    endpoint = {k:float((rows[k] - previous["rows"][prev][take]).abs().max())
                for k,prev in (("semantic","raw"),("neighbors_64","donors"))}
    record(OUT/f"one-rival-{split}-{part}.pt",{"positions":take,"query_ids":qi[take],"labels":labels[take],
           "sources":sources[take],"gallery_ids":gi,"optimum":torch.cat(vals),"positive_count":torch.cat(counts),
           "fully_demoted_AP":torch.cat(clears),"foils":torch.cat(choices),"original_KD":rows},
           a.revision,{"split":split,"part":part,"queries":len(take),"gallery":len(gi),"rival_revision":a.rival_revision,
           "baseline_revision":a.baseline_revision,"name_revision":a.name_revision,"native_revision":a.native_revision,
           "exact_witness_max_error":maximum,"full_original_KD_max_error":full_error,"archived_endpoint_max_error":endpoint,
           "KD_sha256":KD_FUNCTION_SHA,"selectors":SELECTORS})


def collect(a):
    check = brute_check()
    parts = [verified(OUT/f"one-rival-{a.split}-{i}.pt",a.revision) for i in (0,1)]
    meta = [json.loads((OUT/f"one-rival-{a.split}-{i}.json").read_bytes()) for i in (0,1)]
    ix = torch.cat([x["positions"] for x in parts]); order = ix.argsort()
    if not torch.equal(ix[order],torch.arange(len(ix))) or not torch.equal(parts[0]["gallery_ids"],parts[1]["gallery_ids"]):
        raise ValueError("Populations differ or queries are incomplete")
    def joined(key):
        return torch.cat([p[key] for p in parts])[order]
    qi,labels,subjects = (joined(k) for k in ("query_ids","labels","sources"))
    measured = {k:torch.cat([p["original_KD"][k] for p in parts])[order] for k in ARMS}
    pop = Population(labels,subjects)
    results = {}
    for k,row in measured.items():
        delta = row - measured["semantic"]
        results[k] = {"mean_percent":(100*row.mean(0)).tolist(),"delta":pop.effect(delta),
                      "fraction_gain_at_least_four":pop.effect(delta[:,0] >= .04)}
    diff = measured["best_one_rival"][:,0] - measured["same_rival_fully_demoted"][:,0]
    larger = diff > 2e-6
    fixed = measured["opposite_fold_rival"][:,0]
    best = measured["best_one_rival"][:,0]
    if (best+2e-6 < fixed).any() or (best+2e-6 < measured["semantic"][:,0]).any():
        raise ValueError("Oracle lost to an admissible alternative")
    foils = joined("foils")
    info = {"split":a.split,"queries":len(qi),"subjects":len(subjects.unique()),"gallery":len(parts[0]["gallery_ids"]),
            "KD_sha256":KD_FUNCTION_SHA,"results":results,"finite_exhaustion":check,
            "best_ap_exceeds_full_demotion_queries":int(larger.sum()),"max_ap_gain_of_partial_over_full":100*float(diff.max()),
            "same_choice_as_best_percent":(100*(foils[:,:2]==foils[:,2:]).double().mean(0)).tolist(),
            "witness_max_error":max(z["exact_witness_max_error"] for z in meta),
            "full_original_KD_max_error":max(z["full_original_KD_max_error"] for z in meta),
            "archived_endpoint_max_error":{k:max(m["archived_endpoint_max_error"][k] for m in meta) for k in ARMS[:2]}}
    record(OUT/f"one-rival-summary-{a.split}.pt",{"query_ids":qi,"labels":labels,"subjects":subjects,"foils":foils,
           "measured":measured,"gallery_ids":parts[0]["gallery_ids"]},a.revision,info)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",required=True,choices=SPLITS)
    parser.add_argument("--part",type=int,choices=(0,1))
    for key in ("revision","rival-revision","baseline-revision","name-revision","native-revision"):
        parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    deterministic()
    if args.part is None:
        collect(args)
    else:
        run(args)
