"""TokenCut object proposals from OpenAI CLIP patch keys, with size-matched controls."""

import torch
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm.auto import tqdm

from .data import Images, normal_transform, threaded_batches
from .encoding import global_cache_name, load_frozen_clip, save_tensor


def key_projections(vision,pixels):
    _, hidden = vision.transformer.forward_intermediates(vision._embeds(pixels),indices=[5,10],stop_early=True)
    keys=[]
    for features, layer in zip(hidden,(6,11),strict=True):
        block=vision.transformer.resblocks[layer]
        w=block.attn.in_proj_weight.chunk(3)[1]
        b=block.attn.in_proj_bias.chunk(3)[1]
        keys.append(F.linear(block.ln_1(features[:,1:]),w,b).float())
    return keys


def spectral_box(keys,center):
    if center:
        keys=keys-keys.mean(1,keepdim=True)
    unit=F.normalize(keys,dim=-1)
    cosine=unit@unit.mT
    edges=torch.where(cosine>.2,1.,1.e-5)
    inverse=edges.sum(-1).rsqrt()
    symmetric=edges*inverse[:,:,None]*inverse[:,None,:]
    _,vectors=torch.linalg.eigh(symmetric)
    # Second smallest generalized Laplacian eigenvalue corresponds to the
    # second largest eigenvalue of the degree-normalized adjacency.
    y=vectors[:,:,-2]*inverse
    seed=y.abs().argmax(1)
    mask=y>y.mean(1,keepdim=True)
    mask=mask==mask.gather(1,seed[:,None])
    side=int(len(y[0])**.5)
    mask=mask.view(-1,1,side,side)
    component=F.one_hot(seed,side*side).view_as(mask).float()
    # Keep the connected component containing the foreground seed as in TokenCut.
    for _ in range(side*side-1):
        h=F.max_pool2d(component,(1,3),1,(0,1))
        v=F.max_pool2d(component,(3,1),1,(1,0))
        component=torch.maximum(h,v)*mask
    grid=torch.arange(side,device=y.device)[None]
    xs=component[:,0].amax(1).bool()
    ys=component[:,0].amax(2).bool()
    x0=torch.where(xs,grid,side).amin(1); y0=torch.where(ys,grid,side).amin(1)
    x1=torch.where(xs,grid,-1).amax(1)+1; y1=torch.where(ys,grid,-1).amax(1)+1
    box=torch.stack((x0,y0,x1,y1),1).float()/side
    # A complete graph has no identifiable foreground eigenvector.
    empty=(edges==1).all(dim=(1,2))
    return torch.where(empty[:,None],box.new_tensor([0,0,1,1]),box)


def roi_views(model,pixels):
    middle,last=key_projections(model.visual,pixels)
    mid=spectral_box(middle,True)
    late=spectral_box(last,True)
    original=spectral_box(last,False)
    margin=late.new_tensor([-1.,-1.,1.,1.])/7
    padded=(late+margin).clamp(0,1)
    span=(late[:,2:]-late[:,:2]).amax(1,keepdim=True)
    corner=torch.minimum(((late[:,:2]+late[:,2:]-span)*.5).clamp_min(0),1-span)
    square=torch.cat((corner,corner+span),1)
    modes=(late,padded,mid,original,square)
    def centered(box):
        size=box[:,2:]-box[:,:2]
        return torch.cat(((1-size)/2,(1+size)/2),1)
    variants=(*modes,late,*[centered(box) for box in modes],*[1.-box[:,[2,3,0,1]] for box in modes])
    crops,encoded=[],[]
    indices=torch.arange(len(pixels),device=pixels.device,dtype=torch.float32)[:,None]
    for i,box in enumerate(variants):
        regions=torch.cat((indices,box*224),1)
        crop=roi_align(pixels.float(),regions,(224,224),aligned=True,sampling_ratio=1).to(pixels.dtype)
        crops.append(crop.flip(-1) if i==5 else crop)
        if len(crops)==4 or i==len(variants)-1:
            encoded.append(F.normalize(model.encode_image(torch.cat(crops)).float(),dim=-1))
            crops.clear()
    return torch.cat(encoded).reshape(len(variants),len(pixels),-1).transpose(0,1),torch.stack(modes,1)


@torch.inference_mode()
def get_views(args,paths,mode):
    name=global_cache_name(args.root,paths,mode,"spectral-clip-box-v1","fp16",args.clip_checkpoint)
    file=args.cache/name
    if not file.is_file():
        model=load_frozen_clip(args.clip_checkpoint,torch.device("cuda"))
        batch=96
        stream=threaded_batches(Images(paths,normal_transform()),batch,8,True)
        stored,boxes=[],[]
        for pixels in tqdm(stream,total=(len(paths)+batch-1)//batch,desc=f"CLIP proposal {mode}",mininterval=2):
            features,regions=roi_views(model,pixels.to("cuda",dtype=model.visual.conv1.weight.dtype,non_blocking=True))
            stored.append(features.half().cpu()); boxes.append(regions.half().cpu())
        save_tensor({"features":torch.cat(stored),"boxes":torch.cat(boxes)},file)
    return torch.load(file,map_location="cpu",mmap=True,weights_only=True)["features"]
