"""Counterfactual differences of native CLS attention readouts on two patch banks."""

import numpy as np
import torch
import torch.nn.functional as F

from .attention_witness_cache import bank_path


MODES=("exchange","uncentered_exchange","conditional","native","uniform","common_reader","unrelated_reader")


def gather_bank(folder,field,layer,rows):
    array=np.load(folder/f"{field}-{layer}.npy",mmap_mode="r")
    out=torch.empty((len(rows),*array.shape[1:]),device="cuda",dtype=torch.float16)
    for start in range(0,len(rows),256):
        ix=rows[start:start+256].numpy()
        out[start:start+len(ix)]=torch.from_numpy(array[ix]).cuda()
    return out


def own_read(r,kv):
    k=kv[:,:,:,0,:]
    v=kv[:,:,:,1,:]
    alpha=(r[:,:,None,:]*k).sum(-1,dtype=torch.float32).mul_(r.size(-1)**-.5).softmax(-1).to(v.dtype)
    return (alpha[...,None]*v).sum(2,dtype=torch.float32)


def reference_mean(readers,refs):
    out=[]
    for r in readers.split(96):
        weights=torch.einsum("bhd,khtd->bkht",r,refs[:,:,:,0,:]).float().mul_(r.size(-1)**-.5)
        alpha=weights.softmax(3).to(refs.dtype)
        v=torch.einsum("bkht,khtd->bkhd",alpha,refs[:,:,:,1,:])
        out.append(v.float().mean(1))
    return torch.cat(out)


def similarity(a,b):
    return F.cosine_similarity(a,b,dim=-1,eps=1e-8).mean(-1)


class Tokens:
    def __init__(self,folder,layer,rows):
        self.r=gather_bank(folder,"reader",layer,rows)
        self.kv=gather_bank(folder,"patch",layer,rows)
        self.native=torch.cat([own_read(self.r[start:start+96],self.kv[start:start+96])
                               for start in range(0,len(rows),96)])
        self.mean=torch.cat([block[:,:,:,1,:].float().mean(2) for block in self.kv.split(96)])

    def prepare(self,refs,common):
        self.reference=reference_mean(self.r,refs)
        self.common=torch.cat([own_read(common.expand(len(block),-1,-1),block)
                               for block in self.kv.split(96)])


class PatchExchange:
    def __init__(self,w,dataset,layer,qi,pi):
        self.query=Tokens(bank_path(w,dataset,"query"),layer,qi)
        self.gallery=Tokens(bank_path(w,dataset,"gallery"),layer,pi)
        seed=torch.Generator().manual_seed(67327)
        order=torch.randperm(len(pi),generator=seed).cuda()
        ref=order[:64]
        self.in_ref=torch.zeros(len(pi),device=order.device,dtype=torch.bool)
        self.in_ref[ref]=True
        # A fixed cycle in a random gallery ordering guarantees a distinct control reader.
        self.alternative=torch.empty_like(order)
        self.alternative[order]=order.roll(1)
        refs=self.gallery.kv[ref]
        common=self.gallery.r.float().mean(0,keepdim=True).half()
        self.query.prepare(refs,common)
        self.gallery.prepare(refs,common)

    @staticmethod
    def crossed(src_reader,tgt_readers,src_kv,tgt_kv):
        sq=torch.einsum("bhd,bkhtd->bkht",src_reader,tgt_kv[:,:,:,:,0,:])
        qr=(sq.float()*src_reader.size(-1)**-.5).softmax(3).to(src_reader.dtype)
        q_t=torch.einsum("bkht,bkhtd->bkhd",qr,tgt_kv[:,:,:,:,1,:]).float()
        tq=torch.einsum("bkhd,bhtd->bkht",tgt_readers,src_kv[:,:,:,0,:])
        tr=(tq.float()*src_reader.size(-1)**-.5).softmax(3).to(src_reader.dtype)
        t_q=torch.einsum("bkht,bhtd->bkhd",tr,src_kv[:,:,:,1,:]).float()
        return q_t,t_q

    def scores(self,query_ids,candidates,all_controls=True):
        q,g=self.query,self.gallery
        output={key:[] for key in (MODES if all_controls else ("exchange","conditional"))}
        for i in range(0,len(query_ids),8):
            ix=query_ids[i:i+8]
            ids=candidates[i:i+len(ix)]
            row={key:[] for key in output}
            for col in range(0,ids.size(1),48):
                selected=ids[:,col:col+48]
                r=g.r[selected]
                patches=g.kv[selected]
                q_t,t_q=self.crossed(q.r[ix],r,q.kv[ix],patches)
                ownq=q.native[ix,None]
                ownt=g.native[selected]
                left=ownq-t_q
                right=q_t-ownt
                refdiff=q.reference[ix,None]-g.reference[selected]
                # For a panel member remove the tested donor's contribution.
                panel=self.in_ref[selected][...,None,None]
                refdiff=torch.where(panel,(64*refdiff-right)/63,refdiff)
                row["exchange"].append(similarity(left-refdiff,right-refdiff))
                row["conditional"].append(.5*(similarity(ownq-q.mean[ix,None],q_t-g.mean[selected])
                    +similarity(t_q-q.mean[ix,None],ownt-g.mean[selected])))
                if all_controls:
                    row["uncentered_exchange"].append(similarity(left,right))
                    row["native"].append(similarity(ownq-q.mean[ix,None],ownt-g.mean[selected]))
                    row["uniform"].append(similarity(q.mean[ix,None],g.mean[selected]))
                    row["common_reader"].append(similarity(q.common[ix,None]-q.mean[ix,None],g.common[selected]-g.mean[selected]))
                    other=self.alternative[selected]
                    other_r=g.r[other]
                    att=torch.einsum("bkhd,bkhtd->bkht",other_r,patches[:,:,:,:,0,:])
                    weight=(att.float()*r.size(-1)**-.5).softmax(3).to(r.dtype)
                    other_t=torch.einsum("bkht,bkhtd->bkhd",weight,patches[:,:,:,:,1,:]).float()
                    att=torch.einsum("bkhd,bhtd->bkht",other_r,q.kv[ix][:,:,:,0,:])
                    weight=(att.float()*r.size(-1)**-.5).softmax(3).to(r.dtype)
                    other_q=torch.einsum("bkht,bhtd->bkhd",weight,q.kv[ix][:,:,:,1,:]).float()
                    ref=q.reference[ix,None]-g.reference[other]
                    alt_right=q_t-other_t
                    ref=torch.where(panel,(64*ref-alt_right)/63,ref)
                    row["unrelated_reader"].append(similarity(ownq-other_q-ref,alt_right-ref))
            for key in output:
                output[key].append(torch.cat(row[key],dim=1))
        return {key:torch.cat(blocks) for key,blocks in output.items()}
