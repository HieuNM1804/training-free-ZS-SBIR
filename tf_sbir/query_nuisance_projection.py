"""Estimate one nuisance coefficient for a query from unlabeled differences between neighboring gallery photos."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .map_diagnosis import map_values
from .query_fragment_likelihood import GalleryWalk
from .query_fragment_reference_inputs import dataset,load
from .query_fragment_reference_study import split_context
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd,paired_delta


FAMILY="query-conditioned-nuisance-projection"
REFS=("appearance","area_shuffle","grid","scrambled","empty_token","mean_colour","fixed_white","fixed_neutral")
METRICS=("fixed","euclidean","global","within","permuted")
PRIMARY="appearance/within/walk"
AUDIT={"native","native/walk",PRIMARY,"appearance/within/direct","appearance/fixed/walk","appearance/global/walk",
       "appearance/permuted/walk","appearance/euclidean/walk","mean_colour/within/walk","fixed_white/within/walk",
       "area_shuffle/within/walk","scrambled/within/walk"}
TEMPERATURE=64.
FIXED=.8


def edge_covariance(photos,i,j,weight):
    n=len(photos)
    x=photos.double()
    edges=torch.sparse_coo_tensor(torch.stack((i,j)),weight.double(),(n,n)).coalesce()
    degrees=torch.zeros(n,device=x.device,dtype=x.dtype)
    degrees.index_add_(0,i,weight.double())
    degrees.index_add_(0,j,weight.double())
    lp=degrees[:,None]*x-torch.sparse.mm(edges,x)-torch.sparse.mm(edges.transpose(0,1),x)
    cov=x.T@lp/weight.sum(dtype=torch.float64)
    return (cov+cov.T)*.5


def metrics(photos,graph):
    x=photos.double()
    centered=x-x.mean(0)
    local=graph.index.kernels[0].local.to_sparse_coo().coalesce()
    i,j=local.indices()
    weight=local.values()
    perm=torch.randperm(len(x),generator=torch.Generator().manual_seed(57239)).to(x.device)
    result={"euclidean":torch.eye(x.size(1),device=x.device,dtype=x.dtype),
            "global":centered.T@centered/len(x),"within":edge_covariance(x,i,j,weight),
            "permuted":edge_covariance(x[perm],i,j,weight)}
    # Induced edge graphs have no photos in common between A and B.
    half=perm<len(x)//2
    for name,mask in (("A",half[i]&half[j]),("B",~half[i]&~half[j])):
        result[name]=edge_covariance(x,i[mask],j[mask],weight[mask])
    test=x[:2].T.contiguous()
    a=x@test
    explicit=((a[i]-a[j]).T*weight.double())@(a[i]-a[j])/weight.sum(dtype=torch.float64)
    maxerr=float((test.T@result["within"]@test-explicit).abs().max())
    if maxerr>1e-9:raise ValueError("Laplacian covariance differs from edge differences")
    return result,(i,j,weight),maxerr


def corrections(q,f,matrices):
    q=q.double();f=f.double()
    beta={};ortho={}
    for name,c in matrices.items():
        cf=f@c
        denom=(cf*f).sum(1,keepdim=True)
        numer=(cf*q).sum(1,keepdim=True)
        b=torch.where(denom>1e-14,numer/denom.clamp_min(1e-14),torch.zeros_like(numer))
        coef=((q-b*f)*cf).sum(1,keepdim=True)/denom.clamp_min(1e-14)
        if float(coef.abs().amax())>1e-7:raise ValueError("Residual still contains the measured linear component")
        beta[name]=b
        ortho[name]=coef.squeeze(1)
    beta["fixed"]=q.new_full((len(q),1),FIXED)
    return beta,ortho


def view(bank,index,name):
    if name=="scrambled":
        a=bank["modes"]["permuted_fragments_A"][index].float().mean(1)
        b=bank["modes"]["permuted_fragments_B"][index].float().mean(1)
        return (.5*(a+b)).cuda()
    return bank["modes"][name][index].float().mean(1).cuda()


def match_pair(matrix,labels,gt):
    order=matrix.argsort(dim=1,descending=True,stable=True)
    good=labels[:,None]==gt[order]
    first=good.int().argmax(1,keepdim=True)
    wrong=(~good).int().argmax(1,keepdim=True)
    return ~good[:,0],order.gather(1,first),order.gather(1,wrong)


def digest_beta(data):
    q=torch.tensor([.1,.5,.9],dtype=data.dtype)
    return {"mean":float(data.mean()),"q10_q50_q90":data.quantile(q).tolist()}


@torch.inference_mode()
def run(w,stage,ds):
    if stage=="pilot" and ds!="sketchy":raise ValueError("No tuning on TU")
    root=w/"results"/FAMILY
    root.mkdir(parents=True,exist_ok=True)
    spec={"code":revision(),"checkpoint":CHECKPOINT,"evaluator":KD_FUNCTION_SHA,"primary":PRIMARY,"controls":sorted(AUDIT),
          "temperature":TEMPERATURE,"previous_sketchy_coefficient":FIXED,
          "design":"projection coefficient is estimated from edge-weighted score differences within the unlabeled gallery"}
    sig=digest(spec)
    protocol=root/"protocol.json"
    if stage=="pilot":
        if protocol.is_file() and read(protocol)["signature"]!=sig:raise ValueError("Altered pilot")
        atomic_json(protocol,spec|{"signature":sig,"device":torch.cuda.get_device_name()})
    else:
        lock=read(root/"freeze.json")
        if lock["signature"]!=digest(lock["config"]) or lock["config"]["pilot_signature"]!=sig:raise ValueError("Incompatible transfer lock")
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong KD evaluator")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    inputs=dataset(w,ds)
    bank=load(w,ds,"pilot" if stage=="pilot" else "full")
    rows={"native","native/walk",*(f"{r}/{m}/{op}" for r in REFS for m in METRICS for op in ("direct","walk"))} if stage=="pilot" else set(lock["config"]["rows"])
    verified=rows&AUDIT if stage=="pilot" else rows
    for s in inputs[2]:
        dest=root/(("pilot-" if stage=="pilot" else "")+s+".json")
        if dest.is_file():
            prior=read(dest)
            if prior["signature"]!=sig or checksum(dest.with_suffix(".pt"))!=prior["per_query_sha256"]:raise ValueError("Corrupt checkpoint")
            continue
        p=split_context(w,s,ds,inputs,bank,"pilot" if stage=="pilot" else "official",False)
        gallery=p["gallery"];gt=p["gl"].cuda()
        graph=GalleryWalk(gallery)
        matrices,edges,graph_error=metrics(gallery,graph)
        replay=Replay(kd,p["gl"],s)
        scores,tops,hold,coeff=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
        fixed_margins=defaultdict(list);leads=defaultdict(list)
        maxerr=0.;maxorth=0.;errids=[];pairids=[]
        workers=min(6,max(1,len(os.sched_getaffinity(0))-1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for off in range(0,len(p["ids"]),64):
                sl=slice(off,off+64)
                query=p["q"][sl];y=p["labels"][sl]
                orig=query@gallery.T
                error,pos,neg=match_pair(orig,y.cuda(),gt)
                errids.append(error.cpu());pairids.append(torch.cat((pos,neg),1).int().cpu())
                def record(name,matrix):
                    nonlocal maxerr
                    if name not in rows:return
                    val,order,good,_=map_values(matrix,y.cuda(),gt,s)
                    if name in verified:
                        exact=replay.score(order,y,pool)
                        err=float((exact-val.cpu()).abs().max());maxerr=max(maxerr,err)
                        if err>2e-6:raise ValueError("Different KD metrics")
                        val=exact
                        leads[name].append(order[:,:5].int().cpu())
                    scores[name].append(val.cpu());tops[name].append(good[:,0].cpu())
                    fixed_margins[name].append((matrix.gather(1,pos)-matrix.gather(1,neg)).squeeze(1).cpu())
                record("native",orig)
                if "native/walk" in rows:record("native/walk",graph.log(TEMPERATURE*orig))
                for ref in REFS:
                    if not any(k.startswith(ref+"/") for k in rows):continue
                    f=view(bank,p["pos"][sl],ref)
                    betas,orth=corrections(query,f,matrices)
                    maxorth=max(maxorth,max(float(o.abs().amax()) for o in orth.values()))
                    hold[ref].append(torch.cat((betas["A"],betas["B"]),1).cpu())
                    for metric in METRICS:
                        name=ref+"/"+metric
                        if name+"/direct" not in rows and name+"/walk" not in rows:continue
                        b=betas[metric];coeff[name].append(b.squeeze(1).cpu())
                        residual=(query.double()-b*f.double()).float()
                        matrix=residual@gallery.T
                        record(name+"/direct",matrix)
                        if name+"/walk" in rows:record(name+"/walk",graph.log(TEMPERATURE*matrix))
                if off%1536==0:print(s,off,len(p["ids"]),flush=True)
        m={k:torch.cat(v) for k,v in scores.items()};top={k:torch.cat(v) for k,v in tops.items()}
        margins={k:torch.cat(v) for k,v in fixed_margins.items()};wrong=torch.cat(errids)
        if set(m)!=rows:raise ValueError("Missing readout")
        local_i,local_j,ww=edges
        purity=float((ww*(gt[local_i]==gt[local_j])).sum()/ww.sum())
        by_class=p["labels"].unique(return_inverse=True)[1]
        beta={k:torch.cat(v) for k,v in coeff.items()}
        halves={k:torch.cat(v) for k,v in hold.items()}
        save_tensor({"ids":p["ids"],"labels":p["labels"],"metrics":m,"top1":top,"first_five":{k:torch.cat(v) for k,v in leads.items()},
                     "native_top1_error":wrong,"native_best_positive_negative":torch.cat(pairids),"margin":margins,"beta":beta,"split_beta":halves},dest.with_suffix(".pt"))
        summary={"signature":sig,"queries":len(wrong),"gallery":len(gt),"classes":len(p["labels"].unique()),
                 "values":{k:v.double().mean(0).tolist() for k,v in m.items()},"verified":sorted(verified),"KD_max_error":maxerr,
                 "pair_covariance_check":graph_error,"orthogonality_check":maxorth,"edge_purity_after_scoring":purity,
                 "errors_in_original":int(wrong.sum()),"best_positive_above_original_top_negative":{k:int((wrong&(v>0)).sum()) for k,v in margins.items()},
                 "beta":{k:digest_beta(v) for k,v in beta.items()},
                 "between_disjoint_galleries":{k:{"median_absolute_beta_difference":float((v[:,0]-v[:,1]).abs().median()),
                    "correlation":float(torch.corrcoef(v.T)[0,1])} for k,v in halves.items()},
                 "paired":{k:paired_delta(m[PRIMARY],m[k],by_class) for k in sorted(verified)},
                 "primary_top1_vs":{k:{"fix":int((top[PRIMARY]&~top[k]).sum()),"loss":int((~top[PRIMARY]&top[k]).sum())} for k in sorted(verified)},
                 "per_query_sha256":checksum(dest.with_suffix(".pt"))}
        atomic_json(dest,summary);print("saved",s,flush=True)
        del p,graph,matrices


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    parser.add_argument("--stage",choices=("pilot","official"),default="pilot")
    args=parser.parse_args();run(args.workspace,args.stage,args.dataset)
