"""Trace full-gallery semantic retrieval differences with identical visible patches and native CLIP."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import Images,SKETCHY_SPLITS,configure_image_worker,normal_transform
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip,load_global_inputs,save_tensor
from .foundation import load_words
from .geometry_word_flat import KD_FUNCTION_SHA,PrefixReplay,load_kd,make_word_bank,score_rows
from .semantic import read_words


NAMES=('separate','raw_additive','joint_union','full')


def subsets():
    y,x=torch.meshgrid(torch.arange(7),torch.arange(7),indexing='ij')
    intervals=(x<3,x>3,y<3,y>3)
    views=(*intervals,intervals[0]|intervals[1],intervals[2]|intervals[3],y>=0,y<0)
    assert not (views[0]&views[1]).any() and not (views[2]&views[3]).any()
    assert [int(v.sum()) for v in views]==[21]*4+[42,42,49,0]
    return [torch.cat((torch.zeros(1,dtype=torch.long),v.flatten().nonzero().flatten()+1)).cuda() for v in views]


def encode(paths,vision):
    batches=DataLoader(Images(paths,normal_transform()),batch_size=48,num_workers=2,
        pin_memory=True,worker_init_fn=configure_image_worker)
    masks=subsets()
    result=[]
    replay=0.
    for px in batches:
        px=px.cuda(non_blocking=True).to(vision.conv1.weight.dtype)
        tokens=vision._embeds(px)
        z=torch.empty((len(px),8,512),device=px.device,dtype=torch.float16)
        for count in (1,22,43,50):
            ids=[i for i,m in enumerate(masks) if len(m)==count]
            x=tokens[:,torch.stack([masks[i] for i in ids])].flatten(0,1)
            h=vision.transformer(x)
            z[:,ids]=(vision.ln_post(h[:,0])@vision.proj).reshape(len(px),len(ids),-1)
        if not result:
            diff=F.normalize(z[:,6].float())-F.normalize(vision(px).float())
            replay=float(diff.abs().max())
            assert replay<5e-4
        result.append(z.cpu())
    return torch.cat(result),replay


def read(values,bank,src,scale,half):
    words=(scale*(F.normalize(values.float(),dim=-1)@src.T)).softmax(-1)
    # Masking the query side retains the full candidate marginal for every class.
    left=words*half
    right=words*(~half)
    mass_left=left.sum(-1,keepdim=True)
    mass_right=right.sum(-1,keepdim=True)
    assert mass_left.min()>0 and mass_right.min()>0
    a=left@bank.T
    b=right@bank.T
    return (a+b).double(),(a/mass_left).double(),(b/mass_right).double()


def evaluate(raw,bank,src,scale,half):
    raw=raw.cuda().float()
    add=torch.stack((raw[:,0]+raw[:,1]-raw[:,7],raw[:,2]+raw[:,3]-raw[:,7]),1)
    inputs=torch.cat((raw[:,:7],add),1)
    full,a,b=read(inputs,bank,src,scale,half)
    results={}
    for name,idx in zip(NAMES,[(0,1,2,3),(7,7,8,8),(4,4,5,5),(6,6,6,6)]):
        l,r,t,d=idx
        results[name+'/words']=.5*(full[:,l]*full[:,r]+full[:,t]*full[:,d])
        results[name+'/split']=.25*(a[:,l]*b[:,r]+b[:,l]*a[:,r]+a[:,t]*b[:,d]+b[:,t]*a[:,d])
    torch.testing.assert_close(results['full/words'],full[:,6].square(),atol=0,rtol=0)
    return results,full[:,6]


def boundaries(scores,meta,certified):
    correct=2*meta[:,3:4]-1
    delta=(scores[:,:,0]-scores[:,:,1])*correct
    hits=(delta>0).double()+.5*(delta==0)
    uq,c=meta[:,0].unique(return_counts=True)
    both=(meta[:,0,None]==uq[c==2][None]).T
    bal=both.double()@hits/2
    return dict(pairs=len(meta),both=len(bal),certified=int(certified.sum()),balanced=bal.mean(0).tolist(),
        corrected_certified=hits[certified].sum(0).tolist(),damaged=((1-hits)*(hits[:,:1])).sum(0).tolist(),
        rescued=(hits*(1-hits[:,:1])).sum(0).tolist())


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    root=w/'results/semantic-context-union'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    qpaths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    paths,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    scale=model.logit_scale.exp().float()
    src,tgt=[v.cuda() for v in load_words(cfg,read_words(cfg.vocabulary))]
    anchors=torch.randperm(len(src),generator=torch.Generator().manual_seed(42))[:2]
    half=torch.zeros(len(src),device='cuda',dtype=torch.bool)
    half[(tgt@(tgt[anchors[0]]-tgt[anchors[1]])).argsort(stable=True).tensor_split(2)[0]]=True
    _,photos,_=load_global_inputs(cfg,qpaths,paths,False)
    refs={s:torch.load(w/'results/semantic-marginal-error-audit'/(s+'.pt'),weights_only=True) for s in qs}
    all_ids=torch.cat([r['qi'] for r in refs.values()]).unique()
    pixels,replay=encode([qpaths[i] for i in all_ids.tolist()],model.visual)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for s,(original,all_labels) in qs.items():
        ref=refs[s]
        qi=ref['qi']
        pi,gl=gs[s]
        assert torch.equal(pi,ref['pi'])
        qpos={v:i for i,v in enumerate(original.tolist())}
        labels=all_labels[[qpos[v] for v in qi.tolist()]]
        meta=ref['meta']
        positions={v:i for i,v in enumerate(pi.tolist())}
        cand=torch.tensor([[positions[a],positions[b]] for a,b in meta[:,1:3].tolist()])
        bank=make_word_bank(F.normalize(photos[pi].cuda().float(),dim=-1),tgt,scale,192)
        evaluator=PrefixReplay(kd,gl,s)
        subset=pixels[torch.searchsorted(all_ids,qi)]
        rows,paired={},{}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for ix in torch.arange(len(qi)).split(32):
                out,direct=evaluate(subset[ix],bank,src,scale,half)
                direct_rows=score_rows(direct,labels[ix],gl.cuda(),s,evaluator,pool)[:,:2]
                qi_batch=qi[ix].tolist()
                loc={v:j for j,v in enumerate(qi_batch)}
                mask=torch.isin(meta[:,0],qi[ix])
                bix=torch.tensor([loc[v] for v in meta[mask,0].tolist()],device='cuda',dtype=torch.long)
                candidates=cand[mask].cuda()
                for name,val in out.items():
                    ap=score_rows(val,labels[ix],gl.cuda(),s,evaluator,pool)[:,:2]
                    if name=='full/words':torch.testing.assert_close(ap,direct_rows,atol=2e-6,rtol=1e-6)
                    rows.setdefault(name,[]).append(ap)
                    if name not in paired:paired[name]=torch.zeros(len(meta),2,dtype=torch.double)
                    paired[name][mask]=val[bix[:,None],candidates].clamp_min(1e-30).log().cpu()
        raw={k:torch.cat(v).double() for k,v in rows.items()}
        arms=['separate/split','separate/words','raw_additive/split','raw_additive/words','joint_union/split',
              'joint_union/words','full/split','full/words']
        board=torch.stack([paired[k] for k in arms],1)
        b=boundaries(board,meta,ref['certified_error'])
        prior_drift=(board[:,0]-ref['scores'][:,0]).abs()
        changed=int(((board[:,0,0]>board[:,0,1])!=(ref['scores'][:,0,0]>ref['scores'][:,0,1])).sum())
        contrasts={}
        for v in ('words','split'):
            diffs=[]
            for earlier,later in zip(NAMES,NAMES[1:]):
                change=100*(raw[later+'/'+v]-raw[earlier+'/'+v])
                contrasts[later+' minus '+earlier+'/'+v]=dict(points=change.mean(0).tolist(),
                    class_ap=class_interval(change[:,0],labels,seed=42))
                diffs.append(change)
            torch.testing.assert_close(sum(diffs),100*(raw['full/'+v]-raw['separate/'+v]))
        save_tensor(dict(qi=qi,pi=pi,labels=labels,meta=meta,arms=arms,rows=raw,paired=board),root/(s+'.pt'))
        atomic_json(root/(s+'.json'),dict(commit=revision(),seed=42,evaluator=KD_FUNCTION_SHA,gpu=torch.cuda.get_device_name(),
            queries=len(qi),gallery=len(pi),arms=arms,boundary=b,baseline_pair_replay=dict(changed=changed,
                mean_log_drift=float(prior_drift.mean()),maximum_log_drift=float(prior_drift.max())),
            full_token_replay_max=replay,metrics={k:(100*x.mean(0)).tolist() for k,x in raw.items()},effects=contrasts))
        print(s,contrasts['joint_union minus raw_additive/words'],flush=True)
        del bank


if __name__=='__main__':
    run()
