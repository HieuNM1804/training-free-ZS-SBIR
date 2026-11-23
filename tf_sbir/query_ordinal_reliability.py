"""Can gallery-only pair choices identify a useful reading of one unseen sketch?"""
import itertools
import json
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_policy import outputs
from .gallery_panel_resume import region_views, semantic_inputs
from .gallery_region_panels import CONFIG, Index
from .positive_foil import base_parts, STRONG
from .branch_view_study import PrefixReplay
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd


READERS = ('visual', 'words', 'crops_visual', 'gallery_words')
PAIRS = list(itertools.combinations(range(len(READERS)), 2))


def pair_votes(scores, index):
    left, right = index
    return (scores[:, :, left] - scores[:, :, right]).sign()


def fit(votes, design_inverse):
    # If errors of ordinal readers are independent given the true orientation,
    # E[v_i v_j]=r_i r_j, where r_i is twice accuracy minus one.
    agreement = torch.stack([(votes[:, i] * votes[:, j]).mean(-1) for i,j in PAIRS], 1)
    positive = (agreement > 0).all(1)
    raw = (agreement.clamp_min(1e-6).log() @ design_inverse.T).exp()
    feasible = positive & (raw < 1).all(1)
    predicted = torch.stack([raw[:,i]*raw[:,j] for i,j in PAIRS], 1)
    residual = (predicted - agreement).abs().mean(1)
    # Do not repair infeasible fits by silently projecting their reliabilities.
    reliability = torch.where(feasible[:, None], raw, .5)
    weight = torch.logit((1 + reliability)/2)
    return weight/weight.sum(1,keepdim=True), feasible, residual


def percentiles(scores):
    order = scores.argsort(dim=-1, stable=True)
    # Public posterior scores are continuous. Assign equal ranks to occasional ties.
    values = scores.gather(-1, order)
    steps = torch.arange(values.size(-1),device=values.device).expand_as(order)
    start = steps.masked_fill(values == values.roll(1,-1),0)
    end = steps.masked_fill(values == values.roll(-1,-1),values.size(-1)-1)
    start[...,0]=0
    end[...,-1]=values.size(-1)-1
    middle=(start.cummax(-1).values + end.flip((-1,)).cummin(-1).values.flip((-1,))).float()/2
    return torch.zeros_like(scores).scatter_(-1,order,middle)/(scores.size(-1)-1)


def evaluate(scores, truth, train_ids, test_ids, index, inverse):
    train=pair_votes(scores[:,:,train_ids],index)
    weight,valid,residual=fit(train,inverse)
    held=scores[:,:,test_ids]
    rank=percentiles(held)
    candidates=torch.cat([held,(rank*weight[:,:,None]).sum(1,keepdim=True),
                          rank.mean(1,keepdim=True),(rank*weight.roll(1,1)[:,:,None]).sum(1,keepdim=True)],1)
    votes=pair_votes(candidates,index)
    difference=truth[:,test_ids[index[0]]].float()-truth[:,test_ids[index[1]]].float()
    relevant_pairs=difference.abs().sum(1)
    accuracy=((votes*difference[:,None,:]+1)*.5*difference.abs()[:,None,:]).sum(2)/relevant_pairs[:,None].clamp_min(1)
    # All labels enter through this final evaluation on held-out photos.
    chosen=accuracy.gather(1,weight.argmax(1,keepdim=True))
    preference=torch.cat([(weight[:,i:i+1]-weight[:,j:j+1]).sign()*
                          (accuracy[:,i:i+1]-accuracy[:,j:j+1]).sign() for i,j in PAIRS],1)
    # The incumbent sorted the head in descending score before we split its ranks.
    incumbent=(test_ids[index[1]]-test_ids[index[0]]).sign()
    incumbent_hit=((incumbent*difference+1)*.5*difference.abs()).sum(1)/relevant_pairs.clamp_min(1)
    statistics=torch.cat([valid[:,None],residual[:,None],relevant_pairs[:,None],accuracy,
                          chosen,accuracy[:,:4].amax(1,keepdim=True),preference.mean(1,keepdim=True),
                          incumbent_hit[:,None]],1)
    return statistics.cpu(),weight


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    cfg,paths,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(w,'sketchy')
    result={}
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    design=torch.zeros(len(PAIRS),len(READERS),device='cuda')
    for i,(left,right) in enumerate(PAIRS):
        design[i,[left,right]]=1
    inverse=torch.linalg.pinv(design)
    panels=torch.randperm(256,generator=torch.Generator().manual_seed(599639)).cuda().tensor_split(2)
    index=torch.triu_indices(128,128,1,device='cuda')
    for split,(qi,labels) in qs.items():
        pi,gold=gs[split]
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        views,_=region_views(w,split,cfg,paths,qi,q)
        folder,tree,regional=base_parts(w,split)
        batches=[]
        replay=PrefixReplay(kd,gold,split)
        benchmark={}
        flips={}
        for start in range(0,len(qi),48):
            sl=slice(start,start+48)
            offset=start%192
            if not offset:
                pack=torch.load(folder/'full'/f'{start:06d}.pt',mmap=True,weights_only=True)
            sub=slice(offset,offset+len(qi[sl]))
            assert torch.equal(pack['ids'][sub],torch.arange(start,start+len(qi[sl])))
            reg=regional[sl].cuda()
            _,old,_=next(outputs(tree,pack['densities'][sub].cuda(),reg,reg,[STRONG]))
            sketch=q[qi[sl]].cuda()
            mirror=views['flipped'][sl].float().cuda()
            a=(scale*(sketch@src.T)).softmax(1)
            b=(scale*(mirror@src.T)).softmax(1)
            lexical=((a+b)/2)@bank.T
            diff=graph.score(CONFIG,lexical).clamp_min(1e-30).log()
            original=graph.score(CONFIG,a@bank.T).clamp_min(1e-30).log()
            region=reg+.65*(diff-original)
            reference=region+(old-reg)*region.std(1,keepdim=True)/reg.std(1,keepdim=True)
            order=reference.topk(reference.size(1)).indices
            ids=order[:,:256]
            global_visual=((sketch+mirror)/2)@g.T
            crop_visual=views['regions'][sl].float().cuda().mean((1,2))@g.T
            scores=torch.stack([v.gather(1,ids) for v in (global_visual,lexical,crop_visual,diff)],1)
            truth=gold.cuda()[ids]==labels[sl].cuda()[:,None]
            folds=[evaluate(scores,truth,panels[i],panels[1-i],index,inverse) for i in (0,1)]
            batches.append(torch.stack([z[0] for z in folds],1))
            # Put both folds on a common 256-photo percentile scale. Each item's
            # reliability weights use only pair judgments on the other fold.
            scale_rank=percentiles(scores)
            weighted=torch.empty_like(scores[:,0])
            for i,(_,weight) in enumerate(folds):
                held=panels[1-i]
                weighted[:,held]=(scale_rank[:,:,held]*weight[:,:,None]).sum(1)
            natural=torch.arange(256,device=ids.device).expand_as(ids)
            for name,arr in [('baseline',None),('words',scores[:,1]),('diffused_words',scores[:,3]),
                             ('uniform',scale_rank.mean(1)),('moment_weights',weighted)]:
                positions=natural if arr is None else arr.argsort(dim=1,descending=True,stable=True)
                head=ids.gather(1,positions)
                cutoff=200 if split=='sketchy_2' else order.size(1)
                ranking=torch.cat([head,order[:,256:cutoff]],1)[:,:cutoff]
                benchmark.setdefault(name,[]).append(replay.chunk((ranking.cpu(),labels[sl])))
                ok=truth[:,0]
                proposed=truth.gather(1,positions[:,:1])[:,0]
                flips.setdefault(name,[]).append(torch.stack([~ok&proposed,ok&~proposed],1).cpu())
        data=torch.cat(batches).double()
        report={'queries':len(qi),'gallery':len(g),'readers':list(READERS)}
        for k,mask in [('all_mixed_heads',data[:,:,2]>0),
                       ('identifiable_and_mixed',(data[:,:,0]>0)&(data[:,:,2]>0))]:
            counts=mask.sum(1)
            ok=counts>0
            average=(data*mask[:,:,None]).sum(1)[ok]/counts[ok,None]
            row={'mixed_queries':int(ok.sum()),'panel_reads':int(mask.sum()),
                 'mean_offdiagonal_residual':float(average[:,1].mean()),
                 'held_pair_accuracy_pct':dict(zip(READERS+('moment_weights','uniform','rotated_weights'),(100*average[:,3:10].mean(0)).tolist())),
                 'selected_reader_accuracy_pct':float(100*average[:,10].mean()),
                 'incumbent_pair_accuracy_pct':float(100*average[:,13].mean()),
                 'best_reader_after_labels_pct':float(100*average[:,11].mean()),
                 'reader_order_concordance':float(average[:,12].mean())}
            gap=100*(average[:,7]-average[:,8])
            groups=labels[ok]
            macro=torch.stack([gap[groups==c].mean() for c in groups.unique()])
            boot=torch.randint(len(macro),(10000,len(macro)),generator=torch.Generator().manual_seed(693))
            row['paired_moment_minus_uniform_class_mean']=float(macro.mean())
            row['class_bootstrap95']=macro[boot].mean(1).quantile(torch.tensor([.025,.975],dtype=torch.double)).tolist()
            report[k]=row
        report['feasible_panel_reads']=int(data[:,:,0].sum())
        report['official_metrics']={k:(100*torch.cat(v).double().mean(0)).tolist() for k,v in benchmark.items()}
        report['top1_fixed_damaged']={k:torch.cat(v).sum(0).tolist() for k,v in flips.items()}
        result[split]=report
        print(split,json.dumps(report),flush=True)
    out=w/'results/query-ordinal-reliability'
    out.mkdir(parents=True,exist_ok=True)
    (out/'sketchy.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    run()
