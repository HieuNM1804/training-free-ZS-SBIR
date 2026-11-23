"""Keep physically indexed visual content fixed while varying CLIP's learned position labels."""
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Images,configure_image_worker,normal_transform
from .depiction_pivot_geometry import native_patches
from .encoding import save_tensor


def frame_indices(device):
    g=torch.arange(49,device=device).reshape(7,7)
    return torch.stack([g.flatten(),g.flip(1).flatten(),g.flip(0).flatten(),g.flip((0,1)).flatten()])


def explicit(vision,patches,position):
    cls=vision.class_embedding.to(dtype=patches.dtype).reshape(1,1,-1).expand(len(patches),-1,-1)
    grid=torch.cat([cls,patches],1)+position.to(dtype=patches.dtype)
    return describe(vision,vision.ln_pre(vision.patch_dropout(grid)))


def describe(vision,tokens):
    encoded=vision.transformer(tokens)
    return F.normalize((vision.ln_post(encoded[:,1:])@vision.proj).float(),dim=-1)


@torch.inference_mode()
def read_frames(w,paths,vision,indices):
    native=native_patches(paths,vision,w/'cache/depiction-pivot-geometry')
    stamp=hashlib.sha256(('openaiViTB32-axis-relabel-v1\n'+'\n'.join(map(str,paths))).encode()).hexdigest()
    folder=w/'cache/independent-patch-frames'/stamp
    files=[folder/f'frame-{j}.pt' for j in (1,2,3)]
    if not all(p.is_file() for p in files):
        # Evaluate each alternative with exactly the same convolutional patch content.
        arrays=[torch.empty_like(native) for _ in files]
        batches=DataLoader(Images(paths,normal_transform(),'rgb'),batch_size=160,
                num_workers=min(4,len(os.sched_getaffinity(0))),pin_memory=True,worker_init_fn=configure_image_worker)
        pos=vision.positional_embedding
        assert tuple(pos.shape)==(50,768)
        start=0
        for pixels in batches:
            px=pixels.cuda(non_blocking=True).to(vision.conv1.weight.dtype)
            patch=vision.conv1(px).flatten(2).transpose(1,2)
            if not start:
                ordinary=explicit(vision,patch[:4],pos)
                original=describe(vision,vision._embeds(px[:4]))
                if (ordinary-original).abs().max()>8e-4:raise ValueError('Different native encoder')
                if (1-(ordinary*native[:len(ordinary)].float().cuda()).sum(-1)).abs().max()>.003:
                    raise ValueError('Native patch cache disagrees')
                uniform=px[:2].float().mean((-1,-2),keepdim=True).to(px.dtype).expand_as(px[:2]).contiguous()
                uniform_patch=vision.conv1(uniform).flatten(2).transpose(1,2)
                ordinary_flat=explicit(vision,uniform_patch,pos)
            for j,target in enumerate(arrays,1):
                permuted=torch.cat((pos[:1],pos[1:][indices[j]]),0)
                target[start:start+len(px)]=explicit(vision,patch,permuted).half().cpu()
                if not start:
                    new=explicit(vision,uniform_patch,permuted)
                    if (1-(new*ordinary_flat[:,indices[j]]).sum(-1)).abs().max()>.003:
                        raise ValueError('Uniform positional equivariance failed')
            start+=len(px)
        if start!=len(paths):raise ValueError('Missing patches')
        for a,file in zip(arrays,files,strict=True):save_tensor(a,file)
        del arrays
    return [native.cuda(),*[torch.load(p,mmap=True,weights_only=True).cuda() for p in files]]


def affinities(q,g,scale):
    # Inputs hold the same physical patch indices, encoded under different position tables.
    # The independent mean is exactly the average over all 16 dot product matrices.
    # Keep averaged features unnormalized to preserve this identity.
    same=sum(a.float()[:,None]@b.float().transpose(-1,-2) for a,b in zip(q,g,strict=True))/len(q)
    mq=sum(t.float() for t in q)/len(q)
    mg=sum(t.float() for t in g)/len(g)
    independent=mq[:,None]@mg.transpose(-1,-2)
    residual=same-independent
    def patch_std(t):
        return t.std(dim=(-2,-1),keepdim=True,unbiased=False)
    matched_scale=independent*(patch_std(same)/patch_std(independent).clamp_min(1e-8))
    return {'native':scale*(q[0][:,None]@g[0].transpose(-1,-2)).float(),
            'same':scale*same, 'independent':scale*independent,
            'mismatch':scale*(independent-residual/(len(q)-1)), 'shared_component':scale*residual,
            'independent_scale':scale*matched_scale}


def mutual(logit):
    return (logit.softmax(-1)+logit.softmax(-2))/(2*logit.size(-1))
