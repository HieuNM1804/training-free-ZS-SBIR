"""Does lack of joint lexical support on an unseen photo panel identify a semantic rival?"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_policy import outputs
from .gallery_panel_resume import region_views, semantic_inputs
from .gallery_region_panels import CONFIG, Index
from .positive_foil import base_parts, STRONG


def event_bank(terms, reference, dictionary_size):
    """Row i marks reference photos mentioning at least one of photo i's eight words."""
    membership = torch.zeros(dictionary_size,len(reference),device=terms.device,dtype=torch.bool)
    columns = torch.arange(len(reference),device=terms.device)
    membership[reference,columns[:,None]] = True
    return torch.cat([membership[part].any(1).half() for part in terms.split(192)])


def association(events, weights, left, right):
    """Standardized deficit of co-mentions relative to a permutation at fixed margins."""
    a = events[left].float()
    b = events[right].float()
    pa = a.mean(1,keepdim=True)
    pb = b.mean(2)
    cov = (a[:,None,:]*b).mean(2) - pa*pb
    scale = (pa*(1-pa)*pb*(1-pb)).clamp_min(1e-12).sqrt()
    co = -cov/scale
    # Matched reference photos can mention complementary words in different views.
    # The neighbor assignment comes entirely from reference RGB embeddings.
    nb = weights[left].float()
    pn = nb.mean(1,keepdim=True)
    cov = (nb[:,None,:]*b).mean(2) - pn*pb
    return co, -cov/(pn*(1-pn)*pb*(1-pb)).clamp_min(1e-12).sqrt()


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    cfg,paths,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(w,'sketchy')
    out={}
    for s,(qi,y) in qs.items():
        pi,gl=gs[s]
        cls=gl.cuda()
        g=F.normalize(p[pi].cuda())
        readings=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
        photo_words=readings.topk(32).indices
        terms=photo_words[:,:8]
        refs=torch.randperm(len(g),generator=torch.Generator().manual_seed(7736))[:4096].cuda()
        tables=[]
        for r in refs.tensor_split(2):
            e=event_bank(terms,photo_words[r],len(src))
            similarity=g[r]@g[r].T
            similarity.fill_diagonal_(-torch.inf)
            ids=similarity.topk(8).indices
            weights=torch.cat([part[:,ids].mean(2) for part in e.split(128)])
            tables.append((e,weights))
        views,_=region_views(w,s,cfg,paths,qi,q)
        folder,tree,regional=base_parts(w,s)
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        results=[]
        counterfactual=[]
        strata=[]
        for start in range(0,len(qi),48):
            sl=slice(start,start+48)
            offset=start%192
            if offset==0:
                pack=torch.load(folder/'full'/f'{start:06d}.pt',weights_only=True,mmap=True)
            sub=slice(offset,offset+len(y[sl]))
            assert torch.equal(pack['ids'][sub],torch.arange(start,start+len(y[sl])))
            reg=regional[sl].cuda()
            _,old,_=next(outputs(tree,pack['densities'][sub].cuda(),reg,reg,[STRONG]))
            a=(scale*(q[qi[sl]].cuda()@src.T)).softmax(1)
            b=(scale*(views['flipped'][sl].cuda().float()@src.T)).softmax(1)
            post=(a+b)/2
            lex=post@readings.T
            raw=graph.score(CONFIG,a@readings.T).clamp_min(1e-30).log()
            combined=graph.score(CONFIG,lex).clamp_min(1e-30).log()
            new_reg=reg+.65*(combined-raw)
            ranker=new_reg+(old-reg)*new_reg.std(1,keepdim=True)/reg.std(1,keepdim=True)
            ranker[:,refs]=-torch.inf
            lex[:,refs]=-torch.inf
            winner=ranker.argmax(1)
            top=lex.topk(200).indices
            valid=torch.ones_like(top,dtype=torch.bool)
            for word in terms[winner].T:
                valid &= (terms[top]!=word[:,None,None]).all(2)
            # Fixed candidate lists and matched comparison opportunities for all scorers.
            order=valid.int().argsort(dim=1,descending=True,stable=True)[:,:8]
            pool=top.gather(1,order)
            eligible=valid.sum(1)>=8
            c1,n1=association(*tables[0],winner,pool)
            c2,n2=association(*tables[1],winner,pool)
            pos=torch.arange(8,device=g.device).expand_as(pool)
            choices=[pos[:,0],c1.argmax(1),torch.minimum(c1,c2).argmax(1),
                     torch.minimum(n1,n2).argmax(1),torch.minimum(torch.minimum(c1,c2),torch.minimum(n1,n2)).argmax(1)]
            actual=cls[pool]!=cls[winner,None]
            qlabel=y[sl].cuda()
            gold=cls[pool]==qlabel[:,None]
            high=post.topk(32).indices
            omission=(high[:,:,None]!=terms[winner,None,:]).all((1,2))
            records=[]
            by_candidate=[]
            for k in range(pool.size(1)):
                claim=(high[:,:,None]==terms[pool[:,k],None,:]).any(2).any(1)
                switch=omission&claim
                current=cls[winner]==qlabel
                by_candidate.append(torch.stack([actual[:,k],(~current)&gold[:,k],switch&~current&gold[:,k],
                                               switch&current&~gold[:,k],c2[:,k],n2[:,k]],1))
            counterfactual.append(torch.stack(by_candidate,1)[eligible].cpu())
            # Diagnostic stratification after ranking: class and depth of the eighth eligible rival.
            strata.append(torch.stack([qlabel,order[:,-1]//20],1)[eligible].cpu())
            for ix in choices:
                row=torch.arange(len(pool),device=g.device)
                alt=pool[row,ix]
                claim=(high[:,:,None]==terms[alt,None,:]).any(2).any(1)
                switch=omission&claim
                records.append(torch.stack([actual[row,ix],gold[row,ix],switch,cls[winner]==qlabel,
                                            c2[row,ix],n2[row,ix],ix],1))
            results.append(torch.stack(records,1)[eligible].cpu())
        v=torch.cat(results).double()
        options=torch.cat(counterfactual).double()
        stratum=torch.cat(strata)
        _,stratum_ids=stratum.unique(dim=0,return_inverse=True)
        report={}
        for name,z in zip(('first','co_panel_1','co_both','neighbor_both','joint_both'),v.unbind(1)):
            diff,correct,switch,old=z[:,:4].bool().T
            report[name]={'eligible':len(z),'opponent_different_class':int(diff.sum()),
                'incumbent_errors':int((~old).sum()),'correct_alternate_on_errors':int((~old&correct).sum()),
                'switches':int(switch.sum()),'fixed':int((switch&~old&correct).sum()),
                'damaged':int((switch&old&~correct).sum()),'mean_selected_index':float(z[:,6].mean()),
                'reference2_mean_deficit':z[:,4:6].mean(0).tolist()}
            chosen=z[:,6].long()
            expected=torch.zeros(len(v),6,dtype=torch.double)
            for sid in stratum_ids.unique():
                keep=stratum_ids==sid
                weights=chosen[keep].bincount(minlength=8).double()/keep.sum()
                expected[keep]=(options[keep]*weights[None,:,None]).sum(1)
            report[name]['position_matched_null_totals']=expected[:,:4].sum(0).tolist()
            report[name]['position_matched_null_panel2']=expected[:,4:].mean(0).tolist()
            difference=options[torch.arange(len(v)),chosen,0]-expected[:,0]
            macro=torch.stack([difference[stratum[:,0]==c].mean() for c in stratum[:,0].unique()])
            report[name]['class_mean_delta_pp']=100*float(macro.mean())
            ids=torch.randint(len(macro),(10000,len(macro)),generator=torch.Generator().manual_seed(731))
            report[name]['ci95_macro_delta_pp']=(100*macro[ids].mean(1).quantile(torch.tensor([.025,.975],dtype=torch.double))).tolist()
        out[s]={'queries':len(y),'photos':len(pi),'reserved_panels':4096,'outcomes':report,
                'errors_with_correct_rival_available':int(options[:,:,1].amax(1).sum())}
        print(s,json.dumps(out[s]),flush=True)
    dest=w/'results/gallery-concept-exclusion'
    dest.mkdir(exist_ok=True,parents=True)
    (dest/'sketchy.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':
    run()

