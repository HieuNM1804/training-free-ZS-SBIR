"""Encode regular partial observations of each independent query at original image resolution."""

import argparse
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import digest
from .data import CLIP_MEAN,CLIP_STD,configure_image_worker
from .encoding import load_frozen_clip,save_tensor
from .kaggle_unseen_features import BLOCK,DEFAULT_CACHE,SEED,index_file
from .replicated_simplex_study import read


SCALES=(.5,.75)
POINTS=((0.,0.),(1.,0.),(0.,1.),(1.,1.))


class Windows(Dataset):
    def __init__(self,paths):
        self.paths=paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as source:
            image=source.convert("RGB")
        width,height=image.size
        out=[]
        for frac in SCALES:
            a,b=max(1,round(frac*width)),max(1,round(frac*height))
            for x,y in POINTS:
                left,top=round(x*(width-a)),round(y*(height-b))
                crop=image.crop((left,top,left+a,top+b)).resize((224,224),Image.Resampling.BILINEAR)
                out.append(pil_to_tensor(crop))
        return torch.stack(out)


def path_for(cache,dataset,record,start):
    key=digest({"dataset":record["signature"],"scales":SCALES,"points":POINTS,
                "size":224,"interpolation":"PIL bilinear crop before resize","mirror":True})
    return cache/f"source-windows-{dataset}-{key[:16]}"/f"{start:06d}.pt"


@torch.inference_mode()
def run(cache,rank,world,workers,batch):
    if BLOCK%batch:raise ValueError("Window batch must divide encoding block")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    spec=read(cache/"manifest.json")
    model=load_frozen_clip(spec["checkpoint"],torch.device("cuda:0"))
    mean=torch.tensor(CLIP_MEAN,device="cuda")[None,:,None,None]
    std=torch.tensor(CLIP_STD,device="cuda")[None,:,None,None]
    dtype=model.visual.conv1.weight.dtype
    number=0
    for dataset in ("sketchy","tuberlin"):
        record=spec["datasets"][dataset]
        paths=torch.load(index_file(cache,dataset,record),weights_only=True)["sketch"]["paths"]
        jobs=[]
        for start in range(0,len(paths),BLOCK):
            file=path_for(cache,dataset,record,start)
            if number%world==rank and not file.is_file():jobs.append((start,min(start+BLOCK,len(paths)),file))
            number+=1
        if not jobs:continue
        files=[str(Path(record["root"])/p) for a,b,_ in jobs for p in paths[a:b]]
        batches=DataLoader(Windows(files),batch_size=batch,num_workers=workers,pin_memory=True,
                           worker_init_fn=configure_image_worker,prefetch_factor=2)
        it=iter(batches)
        for start,end,file in jobs:
            chunks=[]
            for off in range(start,end,batch):
                pixels=next(it).cuda(non_blocking=True)
                if len(pixels)!=min(batch,end-off):raise ValueError("Window ordering failed")
                n,v=pixels.shape[:2]
                x=((pixels.flatten(0,1).float()/255-mean)/std).to(dtype)
                views=[]
                for img in (x,x.flip((-1,))):
                    emb=F.normalize(model.encode_image(img).float(),dim=1)
                    views.append(emb.reshape(n,v,-1).half().cpu())
                chunks.append(torch.stack(views,dim=2))
            save_tensor({"start":start,"end":end,"signature":record["signature"],
                         "windows":torch.cat(chunks)},file)
            print(rank,dataset,start,end,flush=True)


def load_windows(cache,split):
    dataset="sketchy" if split.name.startswith("sketchy_") else "tuberlin"
    record=read(cache/"manifest.json")["datasets"][dataset]
    length=record["ranges"]["sketch"]
    values=[]
    for start in range(0,length,BLOCK):
        row=torch.load(path_for(cache,dataset,record,start),weights_only=True,mmap=True)
        if (row["start"],row["end"],row["signature"])!=(start,min(length,start+BLOCK),record["signature"]):
            raise ValueError("Window shard mismatch")
        values.append(row["windows"])
    return torch.cat(values)[split.query_ids].cuda()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--rank",type=int,required=True)
    parser.add_argument("--world",type=int,default=4)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--batch",type=int,default=64)
    args=parser.parse_args()
    run(args.cache,args.rank,args.world,args.workers,args.batch)
