"""Replay the unmodified KD evaluator on unique binary relevance sequences using CPU-only processes."""
import argparse
from collections import OrderedDict,deque
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import hashlib
import importlib.util
import inspect
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch


def initialize_worker(workspace,expected,split,n):
    global exact_fn,feature,gallery,one,category,count
    torch.set_num_threads(1)
    module=importlib.util.spec_from_file_location("original_kd_cpu_only",Path(workspace)/"inquiry_kd_reference.py")
    original=importlib.util.module_from_spec(module);module.loader.exec_module(original)
    exact_fn=original._retrieval_metrics
    if hashlib.sha256(inspect.getsource(exact_fn).strip().encode()).hexdigest()!=expected:raise ValueError("Incorrect original evaluator")
    cosines=torch.linspace(.75,-.75,n)
    feature=torch.tensor([[1.,0.]])
    gallery=torch.stack((cosines,(1-cosines.square()).sqrt()),1)
    category=split;count=min(n,200) if split=="sketchy_2" else n
    gallery=gallery[:count]
    if not (torch.nn.functional.cosine_similarity(feature,gallery)[:-1]>torch.nn.functional.cosine_similarity(feature,gallery)[1:]).all():
        raise ValueError("Synthetic features did not preserve rank order")
    one=torch.ones(1,dtype=torch.long)


def exact_bits(batch):
    out=[]
    for block in batch:
        labels=torch.from_numpy(np.unpackbits(np.frombuffer(block,dtype=np.uint8),count=count)).bool()
        ap,precision,k,p=exact_fn(feature,gallery,one,labels,category)
        if (k,p)!=((200,200) if category=="sketchy_2" else (0,100)):raise ValueError("KD cutoffs changed")
        out.append((float(ap),float(precision)))
    return out


class PackedReplay:
    def __init__(self,workspace,digest,split,labels,workers):
        self.n=len(labels);self.cut=min(self.n,200) if split=="sketchy_2" else self.n
        self.labels=labels
        self.cache=OrderedDict();self.running=deque();self.limit=workers*3;self.capacity=32768
        self.calls=0;self.ranks=0;self.maxerr=0.
        context=mp.get_context("forkserver")
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=initialize_worker,
                                      initargs=(str(workspace),digest,split,self.n))

    def submit(self,rank,y,expected,dest):
        # _retrieval_metrics only observes class identities through target.eq(query_label).
        correct=self.labels[rank[:,:self.cut]].eq(y.to(rank.device)[:,None])
        packed=np.packbits(correct.cpu().numpy(),axis=1)
        keys=[r.tobytes() for r in packed]
        self.ranks+=len(keys)
        missing=list(dict.fromkeys(k for k in keys if k not in self.cache))
        for start in range(0,len(missing),16):
            section=missing[start:start+16]
            future=self.pool.submit(exact_bits,section)
            for j,k in enumerate(section):self.cache[k]=(future,j)
        self.calls+=len(missing)
        tasks=[self.cache[k] for k in keys]
        for k in keys:self.cache.move_to_end(k)
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        self.running.append((tasks,expected.cpu(),dest))
        if len(self.running)>=self.limit:self.collect()

    def collect(self):
        tasks,expected,dest=self.running.popleft()
        got=torch.tensor([future.result()[idx] for future,idx in tasks])
        self.maxerr=max(self.maxerr,float((expected-got).abs().amax()))
        if self.maxerr>2e-6:raise ValueError("Packed original evaluator differs from the full ranking")
        dest.append(got)

    def finish(self):
        while self.running:self.collect()
        self.pool.shutdown(wait=True);return self.maxerr


def benchmark(w):
    from .attention_exchange_inputs import FAMILY
    from .conditional_agreement_study import Replay
    from .quickdraw_locked import KD_FUNCTION_SHA
    from .retrieval_evidence import load_kd
    kd,_=load_kd(w)
    for s in ("sketchy_2","sketchy_1"):
        d=torch.load(w/"cache"/FAMILY/"panel"/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        labels=d["gl"];y=d["labels"];ranks=list(d["plans"].values())
        # Repeat the last ranking once to test exact deduplication. The other rankings are distinct.
        order=torch.cat([*ranks,ranks[-1]],0).cuda();yy=y.repeat(len(ranks)+1)
        old=Replay(kd,labels,s);workers=min(4,len(os.sched_getaffinity(0)))
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:prior=old.score(order,yy,pool)
        elapsed_old=time.monotonic()-start
        now=PackedReplay(w,KD_FUNCTION_SHA,s,labels.cuda(),workers)
        output=[];start=time.monotonic()
        for i in range(0,len(yy),64):now.submit(order[i:i+64],yy[i:i+64],prior[i:i+64],output)
        error=now.finish();elapsed_new=time.monotonic()-start
        if not torch.equal(prior,torch.cat(output)):raise ValueError("New replay differs from the unmodified full-rank reference")
        print(s,{"identical":True,"rankings":now.ranks,"distinct_cpu_calls":now.calls,"max_error":error,
                 "old_seconds":elapsed_old,"packed_seconds_with_startup":elapsed_new,"workers":workers},flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    benchmark(ap.parse_args().workspace)
