"""Apply WinCLIP's masked-window CLS embeddings to photo likelihood estimation."""

from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import Images,configure_image_worker,normal_transform
from .encoding import global_cache_name,load_frozen_clip,save_tensor


@dataclass(frozen=True)
class Config:
    side: int = 3
    encoding: str = "masked"
    mix: float = .5


def canonical():
    return Config()


def candidates():
    return [Config(side,"masked",mix) for side in (2,3,5) for mix in (.25,.5,1.)]


def controls(c):
    return {"primary":c,"full_frame":replace(c,mix=0.),"window_only":replace(c,mix=1.),
            "penultimate_patches":replace(c,encoding="patch"),
            "shuffled_window_assignment":replace(c,encoding="shuffle"),
            "small_windows":replace(c,side=2),"medium_windows":replace(c,side=3),
            "large_windows":replace(c,side=5)}


def cache_key(c):
    return c.side,c.encoding,c.mix


def windows(side):
    """Index windows of visible patches and retain positions in the original 7x7 grid."""
    full=torch.arange(1,50,dtype=torch.long).reshape(7,7)
    indices=full.unfold(0,side,1).unfold(1,side,1).reshape(-1,side*side)
    return F.pad(indices,(1,0))


@torch.inference_mode()
def encode(args,paths):
    cached=args.cache/global_cache_name(args.root,paths,"photo","winclip-2-3-5-v1","fp16",args.clip_checkpoint)
    if cached.is_file():
        return torch.load(cached,weights_only=True,mmap=True)
    model=load_frozen_clip(args.clip_checkpoint,torch.device("cuda"))
    visual=model.visual
    masks={side:windows(side).cuda() for side in (2,3,5)}
    ds=Images(paths,normal_transform())
    loader=DataLoader(ds,batch_size=96,num_workers=args.workers,pin_memory=True,worker_init_fn=configure_image_worker)
    output={key:[] for key in ("masked2","masked3","masked5","patch")}
    for batch in tqdm(loader,desc="WinCLIP masked photo windows",mininterval=2.):
        images=batch.to("cuda",dtype=visual.conv1.weight.dtype,non_blocking=True)
        embedded=visual._embeds(images)
        patch=visual.transformer(embedded)[:,1:]
        output["patch"].append(F.normalize((visual.ln_post(patch)@visual.proj).float(),dim=-1).half().cpu())
        for side,indices in masks.items():
            tokens=embedded[:,indices].flatten(0,1)
            encoded=[]
            for view in tokens.split(768):
                pooled,_=visual._pool(visual.transformer(view))
                encoded.append(F.normalize((pooled@visual.proj).float(),dim=-1).half().cpu())
            output[f"masked{side}"].append(torch.cat(encoded).reshape(len(batch),len(indices),-1))
    result={k:torch.cat(v) for k,v in output.items()}
    save_tensor(result,cached)
    return result


def prepare_data(data,args,photo_paths,gallery_indices):
    data.window_embeddings=encode(args,photo_paths)
    data.gallery_indices=gallery_indices


class Index:
    def __init__(self,data):
        self.d=data
        self.weights=None
        self.last=None
        self.info={k:list(x.shape) for k,x in data.window_embeddings.items()}
        self.perm=torch.randperm(len(data.p),generator=torch.Generator().manual_seed(3401))

    def window_posterior(self,kind,side):
        key=kind,side
        if self.last==key:
            return self.weights
        self.weights=None
        mode="patch" if kind=="patch" else f"masked{side}"
        em=self.d.window_embeddings[mode]
        photo_ids=self.d.gallery_indices if kind!="shuffle" else self.d.gallery_indices[self.perm]
        output=torch.empty_like(self.d.a)
        for first in range(0,len(photo_ids),32):
            x=em[photo_ids[first:first+32]].cuda().float()
            if kind=="patch":
                # Compare the same spatial support with globally contextualized tokens.
                ix=windows(side)[:,1:].cuda()-1
                x=F.normalize(x[:,ix].mean(2),dim=-1)
            bank=x.flatten(0,1)
            dist=torch.cat([(self.d.scale*(batch@self.d.target.T)).softmax(1)
                           for batch in bank.split(384)])
            output[first:first+len(x)]=dist.reshape(len(x),x.size(1),-1).mean(1)
        self.weights=output
        self.last=key
        return output

    def score(self,c):
        if c.mix==0:
            return self.d.base
        region=self.d.query_emission@self.window_posterior(c.encoding,c.side).T
        return self.d.base.lerp(region,c.mix) if c.mix!=1 else region
