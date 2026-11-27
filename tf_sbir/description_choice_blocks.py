"""Record unmodified CLS trajectories and their normalized visual projections on Kaggle."""

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from .concept_benchmark import atomic_json, checksum
from .crossdomain_inputs import CHECKPOINT
from .data import CLIP_MEAN, CLIP_STD, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, index_file
from .native_block_probe import SEED, deterministic


OUT = DEFAULT_CACHE.parent / "results" / "description-choice-blocks"
PARTS = 4
PILOT = DEFAULT_CACHE.parent / "results" / "wordnet-description-renaming"


class RawImages(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.resize = normal_transform().transforms[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            return pil_to_tensor(self.resize(image.convert("RGB")))


def inputs():
    meta = json.loads((DEFAULT_CACHE/"manifest.json").read_bytes())
    if meta["sha256"] != CHECKPOINT:
        raise ValueError("Unexpected image encoder checkpoint")
    spec = meta["datasets"]["sketchy"]
    index = torch.load(index_file(DEFAULT_CACHE,"sketchy",spec),weights_only=True)
    gallery = {str(Path(p).with_suffix("")):i for i,p in enumerate(index["photo"]["paths"])}
    sources = []
    for path in index["sketch"]["paths"]:
        p = Path(path)
        stem,sep,number = p.stem.rpartition("-")
        if not sep or not number.isdigit():
            raise ValueError("Unrecognized source identity")
        sources.append(gallery[str(Path("photo")/p.parent.name/stem)])
    return meta,index,torch.tensor(sources)


class Trajectory:
    def __init__(self,model):
        self.model = model
        self.visual = model.visual
        self.batch_first = self.visual.transformer.batch_first
        if self.visual.pool_type != "tok" or self.visual.attn_pool is not None:
            raise ValueError("The ledger assumes native CLS pooling")
        self.proj = self.visual.proj.float()
        self.bias = self.visual.ln_post.bias.float() @ self.proj
        self.weight = self.visual.ln_post.weight.float()
        self.eps = self.visual.ln_post.eps
        self.states = []
        blocks = self.visual.transformer.resblocks
        self.hooks = [blocks[0].register_forward_pre_hook(self.enter)]
        self.hooks.extend(b.register_forward_hook(self.leave) for b in blocks)

    def record(self,x):
        self.states.append((x[:,0] if self.batch_first else x[0]).clone())

    def enter(self,block,args):
        self.record(args[0])

    def leave(self,block,args,out):
        self.record(out)

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __call__(self,pixels):
        self.states.clear()
        native = normalize(self.model.encode_image(pixels).float(),dim=1)
        h = torch.stack(self.states,1).float()
        self.states.clear()
        if h.shape[1:]!=(13,768):
            raise ValueError(f"Unexpected hidden state dimensions {h.shape}")
        h = h-h.mean(-1,keepdim=True)
        sig = (h.square().mean(-1,keepdim=True)+self.eps).sqrt()
        g = (h*self.weight) @ self.proj
        denominator = (g[:,-1]+sig[:,-1]*self.bias).norm(dim=1)[:,None,None]
        prefix = (g+sig[:,-1:]*self.bias)/denominator
        local_shift = ((sig-sig[:,-1:])/denominator)
        local_norm = (prefix+local_shift*self.bias).norm(dim=-1)
        diff = (prefix[:,-1]-native).norm(dim=1)
        if diff.max() > .003 or not torch.isfinite(local_norm).all() or local_norm.min()<=0:
            raise ValueError(f"Final visual vector not reconstructed, maximum L2 = {diff.max().item()}")
        return {"prefix":prefix.cpu(),"native":native.cpu(),"shift":local_shift.squeeze(-1).cpu(),
                "norm":local_norm.cpu(),"l2_error":diff.cpu()}


@torch.inference_mode()
def extract(part,revision,batch,workers):
    meta,index,sources = inputs()
    base = Path(meta["datasets"]["sketchy"]["root"])
    model = load_frozen_clip(meta["checkpoint"],torch.device("cuda"))
    ledger = Trajectory(model)
    mean = torch.tensor(CLIP_MEAN,device="cuda")[None,:,None,None]
    std = torch.tensor(CLIP_STD,device="cuda")[None,:,None,None]
    try:
        for mode,all_ids in (("sketch",torch.arange(len(sources))),("photo",sources.unique())):
            ids = all_ids[part::PARTS]
            dest = OUT/f"native-{mode}-{part}.pt"
            if dest.with_suffix(".json").is_file():
                old=json.loads(dest.with_suffix(".json").read_bytes())
                if old["revision"]!=revision or checksum(dest)!=old["sha256"]:raise ValueError("Incompatible cached native reads")
                continue
            paths=[base/index[mode]["paths"][i] for i in ids.tolist()]
            loader=DataLoader(RawImages(paths),batch_size=batch,num_workers=workers,pin_memory=True,prefetch_factor=1,
                              shuffle=False,generator=torch.Generator().manual_seed(SEED),worker_init_fn=configure_image_worker)
            views=("rgb","mirror") if mode=="sketch" else ("rgb",)
            output={v:[] for v in views}
            for i,pixels in enumerate(loader):
                x=pixels.cuda(non_blocking=True).float().div_(255).sub_(mean).div_(std).to(model.visual.conv1.weight.dtype)
                for v in views:
                    output[v].append(ledger(x if v=="rgb" else x.flip((-1,))))
                if i%12==0: print(mode,part,i+1,"/",len(loader),flush=True)
            store={"ids":ids,"bias":ledger.bias.cpu()}
            max_error={}
            for v in views:
                store[v]={k:torch.cat([row[k] for row in output[v]]) for k in output[v][0]}
                max_error[v]=store[v]["l2_error"].max().item()
            save_tensor(store,dest)
            atomic_json(dest.with_suffix(".json"),{"part":part,"mode":mode,"rows":len(ids),"revision":revision,
                        "seed":SEED,"sha256":checksum(dest),"maximum_l2_error":max_error,"device":torch.cuda.get_device_name()})
            print(mode,part,"complete",len(ids),max_error,flush=True)
    finally:
        ledger.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part",type=int,required=True,choices=range(PARTS))
    p.add_argument("--revision",required=True)
    p.add_argument("--batch",type=int,default=256)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    deterministic()
    extract(a.part,a.revision,a.batch,a.workers)
