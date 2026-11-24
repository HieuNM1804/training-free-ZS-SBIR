"""Test held-word and held-redraw predictions from one shared photo-to-sketch observation channel."""
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations, product
import multiprocessing
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import torch

from .gallery_corner_observations import read_corners
from .gcwe_relational_completion import CORNERS
from .geometry_word_flat import CHECKPOINT, atomic_json, digest, index, revision, save_tensor, semantic_inputs
from .regional_predictive_observation import cluster_effect
from .regional_scale_position_contrast import complete_windows


SETS = np.asarray(list(product((0.,1.), repeat=4))[1:-1])
PERMS = np.asarray(list(permutations(range(4))))
ARMS = ('channel','registered','best_permutation','constant','swapped_words','reverse')
FOLDS = np.sort(np.random.default_rng(42).permutation(64).reshape(2,32),axis=1)


def deficiency(photo, sketch):
    """min_K max_w TV(Q_w, P_w K), K a Markov kernel (complete finite LP)."""
    nin = photo.shape[1]
    coefficients = -np.einsum('wi,sj->wsij', photo, SETS).reshape(-1,nin*4)
    constraints = np.column_stack((coefficients, -np.ones(len(coefficients))))
    rhs = -(sketch @ SETS.T).flatten()
    equality = np.column_stack((np.kron(np.eye(nin),np.ones((1,4))),np.zeros(nin)))
    res = linprog(np.r_[np.zeros(4*nin),1.],A_ub=constraints,b_ub=rhs,A_eq=equality,
                  b_eq=np.ones(nin),bounds=(0,1),method='highs-ds')
    if not res.success:
        raise ValueError(res.message)
    kernel = res.x[:-1].reshape(nin,4)
    actual = .5*np.abs(photo@kernel-sketch).sum(1).max()
    if abs(actual-res.fun)>3e-6 or np.max(np.abs(kernel.sum(1)-1))>1e-7:
        raise ValueError('The Markov kernel failed its primal feasibility or objective check')
    return kernel, res.fun


def validate():
    rng = np.random.default_rng(42)
    table = np.vstack((np.eye(4),rng.dirichlet(np.ones(4),size=12)))
    k = rng.dirichlet(np.ones(4),size=4)
    observed = table @ k
    fit,cost = deficiency(table,observed)
    np.testing.assert_allclose(fit,k,atol=2e-7)
    assert cost<2e-7
    pick = [2,0,3,1]
    fit2,cost2 = deficiency(table[:,pick],observed[:,pick])
    np.testing.assert_allclose(table[:,pick]@fit2, observed[:,pick],atol=2e-7)
    _,degraded = deficiency(np.full((4,4),.25),np.eye(4))
    assert abs(degraded-.75)<2e-7 and cost2<2e-7
    inverse = deficiency(observed,table)[1]
    assert inverse>0


def utility(prediction, target):
    return -.5*np.abs(prediction-target).sum(axis=-1).mean(axis=-1)


def score_photo(own,redraw,photo):
    output = np.zeros((2,len(ARMS)))
    fitted = []
    for seen,held in (FOLDS,FOLDS[::-1]):
        target = np.stack((own[held],redraw[held]))
        kernel,score = deficiency(photo[seen],own[seen])
        constant,c0 = deficiency(np.ones((len(seen),1)),own[seen])
        reverse,rev_score = deficiency(own[seen],photo[seen])
        maps = photo[seen][:,PERMS].transpose(1,0,2)
        costs = .5*np.abs(maps-own[seen]).sum(2).max(1)
        perm = PERMS[costs.argmin()]
        predicted = photo[held] @ kernel
        shuffled = photo[held.reshape(-1,2)[:,::-1].flatten()] @ kernel
        for j,pred in enumerate((predicted,photo[held],photo[held][:,perm],constant,shuffled)):
            output[:,j] += .5*utility(pred,target)
        output[:,-1] += .5*utility(target@reverse,photo[held])
        fitted.append([score,c0,costs.min(),rev_score])
    return output,np.mean(fitted,axis=0)


def score_source(task):
    own,redraw,true,foil=task
    results=[score_photo(own,redraw,photo) for photo in (true,foil)]
    return (np.stack([r[0] for r in results],axis=-1),
            np.stack([r[1] for r in results],axis=-1))


def summarize(scores, meta, group):
    selected = meta[:,3] == group
    ids = meta[selected,4]
    report = {}
    for j,view in enumerate(('own','redraw')):
        values = scores[selected,j]
        margin = values[:,:,0]-values[:,:,1]
        report[view] = dict(sources=len(ids),pairs=len(np.unique(ids)),
            mean_utility=values.mean(0).tolist(),
            accuracy=((margin>1e-10)+.5*(np.abs(margin)<=1e-10)).mean(0).tolist(),
            contrasts={name:dict(true=cluster_effect(values[:,0,0]-values[:,i,0],ids),
                                margin=cluster_effect(margin[:,0]-margin[:,i],ids))
                       for i,name in enumerate(ARMS) if i})
    return report


def select_sources(trials):
    rng = np.random.default_rng(42)
    rows=[]
    for key in np.unique(trials[:,1:],axis=0):
        sketch_ids = trials[(trials[:,1:]==key).all(1),0]
        own,sibling = rng.choice(np.sort(sketch_ids),2,replace=False)
        src,foil,mode,pair=key
        rows.append((own,src,foil,mode,pair,sibling))
    return np.asarray(rows)


def readout(crops,text,scale):
    # Each word independently names a distribution over four spatial observations, for each pose.
    logits = scale*(crops.cuda().float()@text.T)
    result = logits.softmax(dim=1).mean(dim=2).transpose(1,2).cpu().double().numpy()
    return result / result.sum(axis=-1,keepdims=True)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    w=Path('/content/sbir')
    root=w/'results/semantic-channel-comparison'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    _,_,qs,paths,gs,_,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    photos=read_corners(w,paths)
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0))),mp_context=context) as pool:
        for split,(qi,_) in qs.items():
            pi,_=gs[split]
            ref=torch.load(w/'results/semantic-observation-channel'/(split+'.pt'),weights_only=True)
            assert torch.equal(ref['qi'],qi) and torch.equal(ref['pi'],pi)
            meta=select_sources(ref['trials'].numpy())
            sketch_ids=np.unique(meta[:,[0,5]])
            photo_ids=np.unique(meta[:,[1,2]])
            stamp=digest(dict(gallery=[str(paths[i]) for i in pi.tolist()],model=CHECKPOINT,source='regional-64/RGB16'))
            heads,_=index(w,split,p[pi].cuda(),tgt,scale,stamp[:16])
            assert len(heads)==64
            sketches=np.concatenate([readout(views['regions'][qi[batch]][:,CORNERS],src[heads],scale)
                                     for batch in torch.tensor(sketch_ids).split(128)])
            candidates=np.concatenate([readout(photos[pi[batch]],tgt[heads],scale)
                                       for batch in torch.tensor(photo_ids).split(128)])
            tasks=((sketches[np.searchsorted(sketch_ids,row[0])],sketches[np.searchsorted(sketch_ids,row[5])],
                    candidates[np.searchsorted(photo_ids,row[1])],candidates[np.searchsorted(photo_ids,row[2])]) for row in meta)
            results=list(pool.map(score_source,tasks,chunksize=8))
            scores=np.stack([r[0] for r in results])
            costs=np.stack([r[1] for r in results])
            report={name:summarize(scores,meta,g) for g,name in enumerate(('same_category','different_category'))}
            save_tensor(dict(meta=torch.from_numpy(meta),qi=qi,pi=pi,heads=heads.cpu(),folds=torch.from_numpy(FOLDS),
                        scores=torch.from_numpy(scores),training_cost=torch.from_numpy(costs)),root/(split+'.pt'))
            atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),arms=list(ARMS),
                unit='negative held-word mean total variation',train_columns=['channel','constant','best_permutation','reverse'],
                train_cost=costs.mean(0).tolist(),assessment=report))
            print(split,len(meta),report['same_category']['own']['accuracy'],flush=True)


if __name__=='__main__':
    run()
