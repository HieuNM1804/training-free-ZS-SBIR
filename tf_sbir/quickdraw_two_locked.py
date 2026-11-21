"""Score two previously frozen retrieval methods over the full QuickDraw split."""
from concurrent.futures import ThreadPoolExecutor
from importlib import util
import hashlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

w=Path("/content/sbir")
this=Path(__file__).resolve().parents[1]
cross=w/"worktrees/quickdraw-confirm"
sys.path.insert(0,str(cross))
from tf_sbir.quickdraw_locked import OfficialMetrics, paths, load_global_inputs, read_words, bare_words, load_words, load_kd, KD_FUNCTION_SHA, Index, Config, emission, target_inputs, read, digest, save_tensor, atomic_json, CHECKPOINT, checksum

spec=util.spec_from_file_location("tf_sbir.frozen_translation",this/"tf_sbir/public_semantic_translation.py")
tr=util.module_from_spec(spec);sys.modules[spec.name]=tr;spec.loader.exec_module(tr)

@torch.inference_mode()
def run():
    a,qs,ql,gs,gl=paths(w,448,10)
    q,p,_=load_global_inputs(a,qs,gs,False)
    assert (len(q),len(p),len(ql.unique()),len(gl.unique()))==(91491,56407,30,30)
    previous=read(cross/"docs/results/quickdraw-locked-map/quickdraw.json")
    data=read(w/"quickdraw-data.json")
    assert all(data[k]==previous[k] for k in ("image_manifest_sha256","archive_sha256"))
    lock=read(this/"docs/results/public-semantic-translation/freeze.json")
    sig=lock.pop("sha256");assert digest(lock)==sig
    joint=read(this/"docs/results/public-semantic-translation/strict-joint.json")
    assert joint["families"]["public-semantic-translation"]["freeze"]==sig
    words=read_words(a.vocabulary);assert checksum(a.vocabulary)==previous["external_words_sha256"]
    lit,scale=bare_words(a,words);src,tgt=(v.cuda() for v in load_words(a,words))
    posterior=target_inputs(p.cuda(),tgt,scale);posterior/=posterior.sum(1,keepdim=True)
    lex=tr.Lexicon(lit.cuda(),scale);tc=tr.Config(**lock["winner"])
    gc=Config(**previous["configuration"]);graph=Index(SimpleNamespace(p=p.cuda()),rgb_only=True);graph.prepare(gc)
    kd,sha=load_kd(w);assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    evaluator=OfficialMetrics(kd,gl);root=w/"results/quickdraw-two-locked"
    names=["public-semantic-translation","cross-diffusion-map"]
    out={}
    for name in names:
        values=[];errors=[];labels=gl.cuda()
        with ThreadPoolExecutor(max_workers=10) as pool:
            for start in range(0,len(q),1536):
                chunk=root/name/f"{start:06d}.pt"
                if not chunk.exists():
                    x=emission(q[start:start+1536].cuda(),src,scale)
                    y=lex.translate(x,tc) if name==names[0] else None
                    futures=[];fast=[]
                    for pos in range(0,len(x),192):
                        base=x[pos:pos+192]@posterior.T
                        scores=base.lerp(y[pos:pos+192]@posterior.T,tc.mass) if y is not None else graph.score(gc,base)
                        order=scores.argsort(dim=1,descending=True,stable=True)
                        query=ql[start+pos:start+pos+len(base)]
                        truth=labels[order]==query.cuda()[:,None]
                        pr=truth.cumsum(1)/torch.arange(1,len(p)+1,device=p.device).cuda()
                        fast.append(torch.stack(((truth*pr).sum(1)/truth.sum(1),truth[:,:200].float().mean(1)),1).cpu())
                        order=order.int().cpu()
                        futures.extend(pool.submit(evaluator,b,l) for b,l in zip(order.split(24),query.split(24)))
                    exact=torch.cat([f.result() for f in futures]);err=(exact-torch.cat(fast)).abs().amax(0)
                    assert float(err.max())<2e-6
                    save_tensor((exact,err),chunk)
                v,e=torch.load(chunk,weights_only=True);values.append(v);errors.append(e)
                print(name,start+len(v),flush=True)
        v=torch.cat(values);assert len(v)==len(q)
        out[name]={"configuration":lock["winner"] if name==names[0] else previous["configuration"],
                   "freeze":sig if name==names[0] else previous["freeze_sha256"],"mAP":float(v[:,0].mean()),
                   "P200":float(v[:,1].mean()),"KD_queries":len(v),"max_error":torch.stack(errors).amax(0).tolist()}
        save_tensor(v,root/(name+".pt"))
        atomic_json(root/"scores.json",{"methods":out,"queries":len(q),"gallery":len(p),"classes":30,"checkpoint":CHECKPOINT,
            "kd_function":KD_FUNCTION_SHA,"manifest":data["image_manifest_sha256"],"gpu":torch.cuda.get_device_name()})

if __name__=="__main__":
    run()
