"""Cache independently encoded gallery observations at the four existing sketch-crop locations."""
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .benchmark import merge_inputs
from .crossdomain_inputs import CHECKPOINT,runtime
from .crossdomain_benchmark import digest
from .data import Images,SKETCHY_SPLITS,configure_image_worker,normal_transform
from .encoding import load_frozen_clip,save_tensor


def extract(vision,pixels):
    crops=torch.stack([pixels[:,:,y:y+96,x:x+96] for y,x in ((0,0),(0,128),(128,0),(128,128))],1)
    zoom=F.interpolate(crops.flatten(0,1),size=(224,224),mode='bicubic',align_corners=False,antialias=True)
    poses=torch.stack((zoom,zoom.flip((-1,))),1).flatten(0,1)
    feats=F.normalize(vision(poses.to(vision.conv1.weight.dtype)).float()).half()
    return feats.reshape(len(pixels),4,2,-1)


def directory(w,paths):
    key=digest(dict(paths=list(map(str,paths)),model=CHECKPOINT,version='original-3x3-corner-pair/v1'))
    return w/'cache/independent-gallery-corners'/key


def read_corners(w,paths):
    dest=directory(w,paths)
    records=[]
    for start in range(0,len(paths),768):
        item=torch.load(dest/f'{start:05d}.pt',weights_only=True,mmap=True)
        assert item['paths']==list(map(str,paths[start:start+768]))
        records.append(item['features'])
    return torch.cat(records)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    cfg=runtime(w,'sketchy')
    paths,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    dest=directory(w,paths)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    for start in range(0,len(paths),768):
        output=dest/f'{start:05d}.pt'
        subset=list(map(str,paths[start:start+768]))
        if output.exists():
            item=torch.load(output,mmap=True,weights_only=True)
            if item['paths']==subset and item['features'].shape==(len(subset),4,2,512):
                continue
        loader=DataLoader(Images(subset,normal_transform(),'rgb'),batch_size=32,num_workers=2,
            pin_memory=True,worker_init_fn=configure_image_worker,generator=torch.Generator().manual_seed(42))
        result=torch.cat([extract(vision,px.cuda(non_blocking=True)).cpu() for px in loader])
        if not result.isfinite().all() or result.shape!=(len(subset),4,2,512):
            raise ValueError('Invalid frozen gallery features')
        save_tensor(dict(paths=subset,features=result),output)
        print(start+len(subset),'/',len(paths),flush=True)


if __name__=='__main__':
    run()
