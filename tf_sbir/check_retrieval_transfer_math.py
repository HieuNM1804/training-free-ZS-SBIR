"""Test paper equations against small, independently computed references on the Colab GPU."""

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.nn import functional as F
from .collect_image_retrieval_round import source
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .map_study import Data


@torch.inference_mode()
def run(w,specs):
    modules={name:source(w,name,{"revision":sha}) for name,sha in specs}
    torch.manual_seed(1984)
    if not torch.cuda.is_available() or not Path("/content").is_dir():
        raise RuntimeError("Colab CUDA only")
    torch.set_float32_matmul_precision("highest")
    unit=lambda n:F.normalize(torch.randn(n,20,device="cuda"),dim=1)
    q,p,ts,tp=unit(6),unit(19),unit(143),unit(143)
    d=Data(q,p,ts,tp,8.)
    d.literal=unit(143);d.domain_text=unit(1)
    checks={}
    white=modules["retrieval-semantic-whitening"]
    center,eig,axes=white.basis(d.literal)
    z=(d.literal-center)@(axes*eig.rsqrt())
    checks["public_whitening_identity_covariance"]=float((z.T@z/(len(z)-1)-torch.eye(20,device="cuda")).abs().max())
    tree=modules["retrieval-vocabulary-tree"]
    ix=tree.Index(d)
    def dense_tree_response(response,levels):
        mass,ids=response.topk(16)
        mass/=mass.sum(1,keepdim=True)
        tables=[]
        for lv in levels:
            count=lv.bincount().float()
            table=mass.new_zeros(len(mass),len(count))
            for i in range(len(mass)):
                for j in range(ids.shape[1]):
                    table[i,lv[ids[i,j]]]+=mass[i,j]
            tables.append(table*(len(lv)/count).log())
        return torch.cat(tables,1)
    blocks=[*ix.levels,ix.ids]
    qhist,phist=(dense_tree_response(x,blocks) for x in (d.query_emission,d.a))
    for name,norm in (("intersection",1),("cosine",2)):
        qn,pn=(F.normalize(x,p=norm,dim=1) for x in (qhist,phist))
        reference=torch.minimum(qn[:,None],pn[None]).sum(2) if norm==1 else qn@pn.T
        checks["tree_exact_"+name]=float((ix.score(tree.Config(metric=name))-reference).abs().max())
    reg=modules["retrieval-democratic-regions"]
    regions=torch.tensor([[[1.,0.]]*4+[[0.,1.]]],device="cuda")
    target=F.normalize(torch.tensor([[2.,1.]],device="cuda"),dim=1)
    checks["democratic_four_identical_descriptors"]=float((reg.aggregate(regions,"democratic")-target).abs().max())
    mix=modules["retrieval-composed-modality"]
    mixed=mix.Index(d);c=mix.Config(.1,.8,"domain","three")
    h=F.normalize((1-c.alpha)*q+c.alpha*d.domain_text,dim=1)
    words=tp[d.a.topk(3,dim=1).indices]
    by_caption=torch.einsum("bd,nkd->bnk",h,words).mean(2)
    checks["weimocir_two_equations"]=float((mixed.score(c)-((1-c.beta)*(h@p.T)+c.beta*by_caption)).abs().max())
    soft=modules["retrieval-soft-assignment"]
    sx=soft.Index(d)
    def gaussian(x,bank,factor):
        distance=(x[:,None]-bank[None]).square().sum(2)
        values,ids=distance.topk(5,dim=1,largest=False)
        weights=(-.5*d.scale*factor*values).softmax(1)
        return distance.new_zeros(distance.shape).scatter_(1,ids,weights)
    qa,pa=gaussian(q,ts,.5),gaussian(p,tp,2.)
    for name,norm in (("mass",1),("cosine",2)):
        expect=F.normalize(qa,p=norm,dim=1)@F.normalize(pa,p=norm,dim=1).T
        checks["multiple_assignment_"+name]=float((sx.score(soft.Config(.5,2.,5,name))-expect).abs().max())
    rr=modules["retrieval-reciprocal-encoding"]
    gen=np.random.default_rng(387)
    qr=np.maximum(0.,gen.normal(size=(8,23))-.6).astype("f4")
    gr=np.maximum(0.,gen.normal(size=(12,23))-.6).astype("f4")
    qcsr=csr_matrix(qr);gcol=csr_matrix(gr).tocsc()
    overlap=rr.overlap_block(qcsr.indptr,qcsr.indices,qcsr.data,gcol.indptr,gcol.indices,gcol.data,gr.sum(1),0,len(qr))
    exact=np.minimum(qr[:,None],gr).sum(2)/np.maximum(qr[:,None],gr).sum(2)
    checks["weighted_jaccard_sparse_vs_dense"]=float(np.abs(overlap-exact).max())
    core=csr_matrix((gen.random((9,19))>.56).astype("f4"))
    half=csr_matrix((gen.random((19,19))>.73).astype("f4"))
    ptr,idx=rr.expand_indices(core.indptr,core.indices,half.indptr,half.indices,19)
    for row in range(9):
        cset=set(core.indices[core.indptr[row]:core.indptr[row+1]])
        true=set(cset)
        for item in cset:
            hset=set(half.indices[half.indptr[item]:half.indptr[item+1]])
            if 3*len(hset&cset)>2*len(hset):
                true.update(hset)
        if set(idx[ptr[row]:ptr[row+1]])!=true:
            raise ValueError("k-reciprocal expansion differs from Eq. 4")
    probes,gallery=unit(7),unit(144)
    gi,gd=rr.nearest(gallery,gallery,True);qi,qd=rr.nearest(probes,gallery,False)
    k=20
    for i in range(len(probes)):
        augmented=torch.cat((gallery,probes[i:i+1]))
        distances=(1.-augmented@augmented.T)*.5
        distances.fill_diagonal_(torch.inf)
        neighbors=distances.topk(k,largest=False).indices.cpu().numpy()
        exact={int(j) for j in neighbors[-1] if len(gallery) in neighbors[j]}
        efficient=set(qi[i,:k][qd[i,:k]<=gd[qi[i,:k],k-1]])
        if exact!=efficient:
            raise ValueError("Frozen-gallery reciprocal query membership mismatch")
    if max(checks.values())>1e-4:
        raise ValueError(checks)
    atomic_json(w/"results/image-retrieval-round/equation-checks.json",{"revision":revision(),
         "source_revisions":dict(specs),"max_abs":checks,"explicit_query_insertions":7,"set_expansion_rows":9})
    print(checks,flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--method",action="append",required=True)
    args=parser.parse_args()
    run(args.workspace,[s.split("=",1) for s in args.method])
