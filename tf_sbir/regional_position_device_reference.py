"""Replay the locked pre-GCWE global-plus-regional score on this GPU for the prospective pilot."""
import hashlib
import inspect
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .gallery_panel_resume import region_views
from .geometry_word_flat import (
    CHECKPOINT, CONFIG, GLOBAL_WEIGHT, REGIONAL_WEIGHT, KD_FUNCTION_SHA, Index,
    PrefixReplay, digest, index, load_kd, log_score, make_word_bank, regional_readout,
    revision, save_tensor, score_rows, semantic_inputs,
)


@torch.inference_mode()
def run():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    w = Path('/content/sbir')
    dataset = 'sketchy'
    cfg, qp, qs, pp, gs, q, p, src, tgt, scale = semantic_inputs(w, dataset)
    kd, _ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() == KD_FUNCTION_SHA
    rows = {s: torch.randperm(len(ids), generator=torch.Generator().manual_seed(42))[:1536].sort().values
            for s,(ids,_) in qs.items()}
    needed = torch.cat([ids[rows[s]] for s,(ids,_) in qs.items()]).unique(sorted=True)
    views,_ = region_views(w, 'sampled-'+dataset, cfg, qp, needed, q)
    for s,(qi,labels) in qs.items():
        ids = rows[s]
        pi, gl = gs[s]
        g = p[pi].cuda()
        path_list = [str(pp[i]) for i in pi.tolist()]
        stamp = digest({'gallery':path_list, 'model':CHECKPOINT, 'source':'regional-64/RGB16'})
        words, region_bank = index(w,s,g,tgt,scale,stamp[:16])
        graph = Index(SimpleNamespace(p=g),rgb_only=True)
        bank = make_word_bank(g,tgt,scale,256)
        replay = PrefixReplay(kd,gl,s)
        metrics = []
        # Same arithmetic and batch size as the original seed-42 flat baseline.
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for group in ids.split(16):
                source_ids = qi[group]
                take = torch.searchsorted(needed,source_ids)
                assert torch.equal(needed[take],source_ids)
                poses = torch.stack((q[source_ids].cuda(),views['flipped'][take].float().cuda()),1)
                crops = views['regions'][take].cuda()
                post = (scale*(poses@src.T)).softmax(2)
                first_d = graph.score(CONFIG,post[:,0]@bank.T)
                lex = log_score(graph.score(CONFIG,post.mean(1)@bank.T))
                regional = regional_readout(poses,crops,src,scale,words,region_bank,first_d)
                reg = GLOBAL_WEIGHT*log_score(first_d) + REGIONAL_WEIGHT*regional
                combined = reg + GLOBAL_WEIGHT*(lex-log_score(first_d))
                metrics.append(score_rows(combined,labels[group],gl.cuda(),s,replay,pool)[:,:2])
        save_tensor({'qi':qi[ids],'pi':pi,'labels':labels[ids],'kd':torch.cat(metrics),
                     'device':torch.cuda.get_device_name(),'commit':revision(),'seed':42},w/'results/regional-position-device-reference'/(s+'.pt'))
        del bank,graph,region_bank


if __name__ == '__main__':
    run()
