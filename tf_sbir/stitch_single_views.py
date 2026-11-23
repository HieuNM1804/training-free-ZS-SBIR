"""Cache ten reproducible RGB crops/flips per gallery image for standalone set scoring."""
import argparse
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, RandomResizedCrop

from .benchmark import merge_inputs
from .crossdomain_inputs import runtime
from .data import Images,SKETCHY_SPLITS,configure_image_worker,normal_transform
from .encoding import load_frozen_clip,save_tensor


class PhotoViews(Images):
    def __init__(self,paths,root):
        super().__init__(paths,normal_transform(),'rgb')
        self.root=root
        self.crop=RandomResizedCrop(224,scale=(.5,1.),ratio=(.75,4/3),interpolation=InterpolationMode.BICUBIC,antialias=True)

    def __getitem__(self,i):
        image=super().__getitem__(i)
        key=Path(self.paths[i]).relative_to(self.root).as_posix()
        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            more=[]
            for _ in range(9):
                a=self.crop(image)
                more.append(a.flip((-1,)) if float(torch.rand(()))<.5 else a)
        return torch.stack([image,*more])


def storage(w,ds,paths):
    h=hashlib.sha256(('b32/normalized-RGB-10-crops-0.5-v1\n'+'\n'.join(map(str,paths))).encode()).hexdigest()[:20]
    return w/'cache/stitch-isolated-alignment'/(ds+'-'+h)


@torch.inference_mode()
def encode(w,ds):
    cfg=runtime(w,ds,batch=160,workers=min(2,len(os.sched_getaffinity(0))))
    splits=SKETCHY_SPLITS if ds=='sketchy' else ('tuberlin',)
    photos,_=merge_inputs(cfg.root,splits,'photo')
    root=storage(w,ds,photos)
    root.mkdir(parents=True,exist_ok=True)
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    # Materialize only small pixel batches per worker. Use the shared gallery index across Sketchy splits.
    for start in range(0,len(photos),512):
        dst=root/f'{start:06d}.pt'
        if dst.exists():continue
        ds_views=PhotoViews(photos[start:start+512],cfg.root)
        dl=DataLoader(ds_views,batch_size=12,num_workers=cfg.workers,pin_memory=True,
                      prefetch_factor=1,worker_init_fn=configure_image_worker)
        records=[]
        for px in dl:
            z=F.normalize(v(px.flatten(0,1).cuda(non_blocking=True).to(v.conv1.weight.dtype)).float(),dim=1)
            records.append(z.reshape(len(px),10,512).half().cpu())
        save_tensor(torch.cat(records),dst)
        if start%4096==0:print(ds,'views',min(start+512,len(photos)),len(photos),flush=True)
    return root


def load_views(w,ds,paths):
    root=storage(w,ds,paths)
    blocks=[torch.load(root/f'{i:06d}.pt',weights_only=True,mmap=True) for i in range(0,len(paths),512)]
    out=torch.cat(blocks)
    if out.shape!=(len(paths),10,512):raise ValueError('Gallery view cache incomplete')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['sketchy','tuberlin'],required=True)
    encode(Path('/content/sbir'),p.parse_args().dataset)
