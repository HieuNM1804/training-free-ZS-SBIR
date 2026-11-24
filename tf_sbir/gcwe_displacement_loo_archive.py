"""Finalize every already-computed evaluator row, including the original strict replay discrepancy."""
import hashlib
import inspect
from pathlib import Path

import torch
from .gcwe_conditioned_donor_test import paired_replay
from .geometry_word_flat import COUNTS, KD_FUNCTION_SHA, atomic_json, class_interval, digest, load_kd, revision, save_tensor


SCORER = 'f05394c9904ec948e0ea03ec1568cb94677fb63c'
FAMILY = 'gcwe-displacement-leave-one-out'


def run():
    w = Path('/content/sbir')
    root = w/'results'/FAMILY
    kd, _ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split, (n, g, c) in COUNTS.items():
        locked = torch.load(w/'results/gcwe-displacement-full'/(split+'.pt'), weights_only=True)
        qi, pi, labels = (locked[k] for k in ('qi','pi','labels'))
        assert (len(qi),len(pi),len(labels.unique()))==(n,g,c)
        recipe = digest(dict(commit=SCORER, family=FAMILY, device=torch.cuda.get_device_name(),
                             seed=42,evaluator=KD_FUNCTION_SHA, ids=qi.tolist(),photos=pi.tolist()))
        folder = root/split/recipe
        expected = [f'{i:05d}.pt' for i in range(0,n,256)]
        assert sorted(p.name for p in folder.glob('*.pt'))==expected
        rows = {}
        for f in expected:
            off = int(f[:-3])
            shard = torch.load(folder/f, weights_only=True)
            assert torch.equal(shard['ids'],qi[off:off+256])
            assert set(shard['metrics'])=={'original','semantic_regional','same_partition','without_semantic','without_regional'}
            for k,r in shard['metrics'].items():
                assert r.shape==(len(shard['ids']),4) and torch.isfinite(r).all()
                rows.setdefault(k,[]).append(r[:,:2])
        rows = {k:torch.cat(v) for k,v in rows.items()}
        audit = paired_replay(rows,locked,('same_partition','semantic_regional','original'))
        contrasts = {}
        for arm in ('without_semantic','without_regional','semantic_regional'):
            diff = 100*(rows['same_partition'].double()-rows[arm].double())
            contrasts['same_partition minus '+arm] = dict(points=diff.mean(0).tolist(),
                class_delta=class_interval(diff[:,0],labels,seed=42))
        save_tensor(dict(recipe=recipe,qi=qi,pi=pi,labels=labels,rows=rows),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=SCORER,archive_commit=revision(),recipe=recipe,device=torch.cuda.get_device_name(),
            torch=str(torch.__version__),seed=42,sample_rows=n,gallery=g,classes=c,evaluator_sha=KD_FUNCTION_SHA,
            paired_replay=audit,metrics={k:(100*v.double().mean(0)).tolist() for k,v in rows.items()},contrasts=contrasts))
        print(split, n, flush=True)


if __name__=='__main__':
    run()
