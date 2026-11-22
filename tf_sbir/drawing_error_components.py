"""Separate repeatable photographed-object difficulty from differences among independently drawn sketches after official evaluation."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .recoverable_patch_study import word_log
from .semantic_neuron_choice_repeatability import verified_sources
from .semantic_neuron_counterfactuals import restore
from .semantic_neuron_gallery import FAMILY
from .semantic_neuron_retrieval import setup


def grouped(cats,objects):
    groups=defaultdict(lambda:defaultdict(list))
    for i,(y,o) in enumerate(zip(cats.tolist(),objects,strict=True)):
        if o is not None:groups[y][o].append(i)
    return {k:[np.array(rows,dtype=np.int64) for rows in by_obj.values() if len(rows)>=2] for k,by_obj in groups.items()}


def parts(ap,groups):
    byclass=[]
    for c,clusters in sorted(groups.items()):
        if len(clusters)<2:continue
        n=np.asarray([len(ix) for ix in clusters],dtype=np.float64)
        avg=np.array([ap[ix].mean() for ix in clusters])
        sw=sum(float(((ap[ix]-a)**2).sum()) for ix,a in zip(clusters,avg,strict=True))
        pooled=sw/(n.sum()-len(n))
        mb=float((n*(avg-np.average(avg,weights=n))**2).sum()/(len(n)-1))
        n0=(n.sum()-np.sum(n*n)/n.sum())/(len(n)-1)
        byclass.append([c,n.sum(),len(n),pooled,(mb-pooled)/n0])
    rows=torch.tensor(byclass,dtype=torch.double)
    def share(r):
        # Equal category weights. Retain signed moment estimates to avoid an upward object-variance bias.
        within=r[...,3].mean(-1)
        between=r[...,4].mean(-1)
        return within/(within+between).clamp_min(1e-16)
    idx=torch.randint(len(rows),(4000,len(rows)),generator=torch.Generator().manual_seed(94173))
    ci=share(rows[idx]).quantile(torch.tensor([.025,.975],dtype=torch.double))
    return rows,{"within_drawing_variance":float(rows[:,3].mean()),"between_object_variance_within_category":float(rows[:,4].mean()),
                 "share_within_drawings":float(share(rows)),"share_class_bootstrap95":ci.tolist()}


def switches(pred,expected,groups):
    qerr=rescued=samedistr=disagree=equal=tot=0
    correct=pred==expected
    for c,clusters in groups.items():
        for ix in clusters:
            vals=pred[ix];hit=correct[ix]
            n=len(vals);miss=n-int(hit.sum());qerr+=miss
            if hit.any():rescued+=miss
            agree=(vals[:,None]==vals[None,:])
            incorrect=(~hit[:,None])&(~hit[None,:])
            samedistr+=int((agree&incorrect).sum())-miss
            disagree+=int(incorrect.sum())-miss
            equal+=int(agree.sum())-n;tot+=n*(n-1)
    return np.array([qerr,rescued,samedistr,disagree,equal,tot],dtype=np.float64)


def ratios(vec):
    return (vec[1]/max(vec[0],1),vec[2]/max(vec[3],1),vec[4]/max(vec[5],1))


def example_pairs(groups,source_ids,ap,pred,labels,qp,pp,qi,pi,gallery_top):
    classes=[]
    for category,clusters in groups.items():
        eligible=[]
        for ix in clusters:
            good=ix[pred[ix]==category]
            bad=ix[pred[ix]!=category]
            if not len(good) or not len(bad):continue
            correct=int(good[np.abs(ap[good]-np.median(ap[good])).argmin()])
            wrong=int(bad[np.abs(ap[bad]-np.median(ap[bad])).argmin()])
            eligible.append((correct,wrong,float(ap[correct]-ap[wrong])))
        if eligible:
            gaps=np.asarray([t[2] for t in eligible])
            j=int(np.abs(gaps-np.median(gaps)).argmin())
            classes.append((len(eligible),category,eligible[j]))
    result=[]
    def short(path):
        p=Path(path)
        return str(Path(p.parent.name)/p.name)
    for count,category,(first,second,gap) in sorted(classes,key=lambda t:(-t[0],t[1]))[:3]:
        def describe(ix):
            return {"drawing":short(qp[int(qi[ix])]),"top_gallery_photo":short(pp[int(pi[gallery_top[ix]])]),
                    "AP":float(ap[ix]),"top1_is_relevant":bool(pred[ix]==labels[ix])}
        result.append({"number_of_mixed_source_objects_in_category":count,"source":source_ids[first],
                       "correct_drawing":describe(first),"wrong_drawing":describe(second),"AP_gap":gap})
    return result


def report(pred,labels,ap,groups,perms):
    items=[ix for clusters in groups.values() for ix in clusters]
    indices=np.concatenate(items)
    variance,safe=parts(ap.astype(np.float64),groups)
    actual=switches(pred,labels,groups)
    real=ratios(actual)
    rand=[]
    rng=np.random.default_rng(28641)
    byclass=[np.concatenate(clusters) for clusters in groups.values() if clusters]
    for _ in range(perms):
        rotated=pred.copy()
        for target in byclass:rotated[target]=pred[rng.permutation(target)]
        rand.append(ratios(switches(rotated,labels,groups)))
    sh=np.asarray(rand)
    cols=("wrong_query_has_correct_sibling","two_wrong_drawings_same_distractor_class","two_drawings_same_top1_class")
    return {"verified_queries":len(indices),"eligible_source_objects":len(items),"errors_among_verified":int(actual[0]),
            "errors_with_another_correct_drawing":int(actual[1]),"two_wrong_ordered_pairs":int(actual[3]),
            "actual":dict(zip(cols,real,strict=True)),"class_permuted_mean":dict(zip(cols,sh.mean(0).tolist(),strict=True)),
            "class_permuted95":dict(zip(cols,np.quantile(sh,[.025,.975],axis=0).T.tolist(),strict=True)),
            "ap_variance":safe},variance


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    torch.backends.cuda.matmul.allow_tf32=False
    _,qp,qs,pp,gs,*_=inputs
    root=w/"results"/FAMILY
    lock=read(root/"freeze.json")
    main=lock["configuration"]["nominee"]
    arm,channel,gain=main.rsplit("/",2);gain=float(gain)
    data,summary={},{}
    for s,(qi,labels) in qs.items():
        saved=torch.load(root/(s+".pt"),mmap=True,weights_only=True)
        features,desc=restore(w,s,"all-query")
        context=setup(w,s,inputs,features,"all-query")
        pos=desc["arms"].index(arm)
        pred=defaultdict(list)
        gallery_top=[]
        gl=gs[s][1]
        for off in range(0,len(labels),48):
            sl=slice(off,off+48)
            init=context["base"][sl].cuda()
            z=features["native"][sl].cuda()
            d=features["change"][sl,pos].cuda().float()
            rgb=z@context["rgb"].T
            wl=lambda vec:word_log(vec,context["src"],context["bank"],context["scale"])
            wd=wl(z+d)-wl(z)
            rd=context["scale"]*(d@context["rgb"].T)
            change={"words":wd,"rgb":rd,"both":.5*(rd+wd)}[channel]
            for name,table in (("regional",init),("native_rgb",rgb),(main,init+gain*change)):
                chosen=table.argmax(1).cpu()
                pred[name].append(gl[chosen])
                if name=="regional":gallery_top.append(chosen)
        src=verified_sources(qp,pp,qi,gs[s][0])
        groups=grouped(labels,src)
        stats,bins={},{}
        for k,ys in pred.items():
            out=torch.cat(ys)
            mismatch=int(((out==labels)!=saved["top1"][k]).sum())
            if mismatch:raise ValueError(f"Top predictions differ from full official data for {s} {k}: {mismatch}")
            stats[k],var=report(out.numpy(),labels.numpy(),saved["metrics"][k][:,0].numpy(),groups,64)
            bins[k]={"predicted_class":out,"observed_variance_by_class":var}
        data[s]={"labels":labels,"source_groups":{k:[torch.tensor(x) for x in xs] for k,xs in groups.items()},"conditions":bins}
        stats["regional"]["example_subjects"]=example_pairs(groups,src,saved["metrics"]["regional"][:,0].numpy(),
                    bins["regional"]["predicted_class"].numpy(),labels.numpy(),qp,pp,qi,gs[s][0],torch.cat(gallery_top))
        summary[s]=stats
        print(s,stats["regional"]["actual"],stats["regional"]["ap_variance"],flush=True)
        del context,features
    file=root/"drawing-error-components.pt"
    save_tensor(data,file)
    atomic_json(file.with_suffix(".json"),{"revision":revision(),"freeze":lock["sha256"],"per_query_sha256":checksum(file),"results":summary,
                "scope":"post-hoc variance and error agreement. Predictions and AP come from independent single-query scores. Only drawings with verified camera-source IDs and at least two observed depictions are included. Variance is equal-category after conditioning on category."})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(cli.parse_args().workspace)
