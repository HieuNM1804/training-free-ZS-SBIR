"""Construct a frozen visual index from the attenuation of a printed name on each photo."""

import argparse
import hashlib
import json
from pathlib import Path

import open_clip
import PIL
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn.functional import interpolate, normalize
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

from .concept_benchmark import atomic_json, checksum
from .data import CLIP_MEAN, CLIP_STD, configure_image_worker
from .description_choice_blocks import PILOT, RawImages, inputs
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE
from .native_block_probe import SEED, deterministic
from .wordnet_renaming_pilot import SPLITS
from .wordnet_renaming_text import Sense, describe, encode


OUT = DEFAULT_CACHE.parent / "results" / "photo-name-interaction"
PARTS = 4
BOARD = 224
CONTENT = 192
STRIP = BOARD-CONTENT


def card(word):
    strip = Image.new("L", (BOARD, STRIP), 255)
    if word:
        font = ImageFont.load_default(size=22)
        x0, y0, x1, y1 = font.getbbox(word)
        glyph = Image.new("L", (max(1,x1-x0), max(1,y1-y0)), 255)
        ImageDraw.Draw(glyph).text((-x0,-y0), word, font=font, fill=0)
        glyph.thumbnail((BOARD-8, STRIP-4), Image.Resampling.LANCZOS)
        strip.paste(glyph, ((BOARD-glyph.width)//2,(STRIP-glyph.height)//2))
    return pil_to_tensor(strip)


class Encoder:
    def __init__(self, checkpoint):
        self.model = load_frozen_clip(checkpoint, torch.device("cuda"))
        self.visual = self.model.visual
        self.mean = torch.tensor(CLIP_MEAN,device="cuda")[None,:,None,None]
        self.std = torch.tensor(CLIP_STD,device="cuda")[None,:,None,None]
        self.gain = self.visual.ln_post.weight.float()
        self.projection = self.visual.proj.float()
        if self.visual.pool_type != "tok" or self.visual.attn_pool is not None:
            raise ValueError("Unexpected CLIP pooling")
        self.handle = self.visual.transformer.resblocks[-1].register_forward_hook(self.capture)
        self.cls = None

    def capture(self, module, args, value):
        self.cls = (value[:,0] if self.visual.transformer.batch_first else value[0]).float()

    def __call__(self, pixels):
        x = ((pixels - self.mean)/self.std).to(self.visual.conv1.weight.dtype)
        native = normalize(self.model.encode_image(x).float(),dim=1)
        cls = self.cls
        if cls.shape != (len(x),768):
            raise ValueError("Unexpected final CLS layout")
        self.cls = None
        linear = ((cls - cls.mean(1,keepdim=True))*self.gain) @ self.projection
        return native, linear

    def close(self):
        self.handle.remove()


def loader(paths, ids, base, batch, workers):
    images = RawImages([base/paths[i] for i in ids.tolist()])
    return DataLoader(images,batch_size=batch,num_workers=workers,pin_memory=True,
                      prefetch_factor=1,generator=torch.Generator().manual_seed(SEED),
                      worker_init_fn=configure_image_worker)


def record(path, tensor, revision, extra):
    save_tensor(tensor,path)
    atomic_json(path.with_suffix(".json"),dict(extra,revision=revision,sha256=checksum(path),
                                               seed=SEED,device=torch.cuda.get_device_name()))


def verified(path, revision):
    meta = json.loads(path.with_suffix(".json").read_bytes())
    if meta["revision"] != revision or checksum(path) != meta["sha256"]:
        raise ValueError(f"Invalid result shard {path}")
    return torch.load(path,mmap=True,weights_only=True)


def ordered_shards(paths, revision, ids):
    stores = [verified(p,revision) for p in paths]
    full_ids = torch.cat([z["ids"] for z in stores])
    order = full_ids.argsort()
    if not torch.equal(full_ids[order],ids):
        raise ValueError("Shards do not cover the requested IDs exactly once")
    return {k:torch.cat([z[k] for z in stores])[order] for k in stores[0] if k!="ids"}


def sense_bank(model):
    catalog = json.loads((PILOT/"inventory.json").read_bytes())
    senses = [Sense(**s) for s in catalog["senses"]]
    if len(senses)!=1135:
        raise ValueError("Wrong definition inventory")
    vectors = encode(model,open_clip.get_tokenizer("ViT-B-32"),[describe("drawing",s.anonymous) for s in senses])
    return vectors,torch.tensor([s.head for s in senses],device="cuda")


def assign_names(part,revision,batch,workers):
    meta,index,_ = inputs()
    paths = index["photo"]["paths"]
    all_ids = torch.arange(len(paths))[part::PARTS]
    encoder = Encoder(meta["checkpoint"])
    senses, heads = sense_bank(encoder.model)
    picked, vectors = [], []
    try:
        data = loader(paths,all_ids,Path(meta["datasets"]["sketchy"]["root"]),batch,workers)
        for pixels in data:
            native,_ = encoder(pixels.cuda(non_blocking=True).float().div_(255))
            picked.append(heads[(native@senses.T).argmax(1)].cpu())
            vectors.append(native.cpu())
    finally:
        encoder.close()
    record(OUT/f"names-{part}.pt",{"ids":all_ids,"name":torch.cat(picked),"native":torch.cat(vectors)},
           revision,{"pictures":len(all_ids),"stage":"nominations"})


def word_permutation(split, length):
    key = hashlib.sha256(f"{SEED}/photo-name-permutation/{split}".encode()).digest()
    rng = torch.Generator().manual_seed(int.from_bytes(key[:8],"little")%(2**63))
    return torch.randperm(length,generator=rng)


def blanks(strips, encoder, batch):
    refs = []
    for s in strips.split(batch):
        x = torch.ones((len(s),3,BOARD,BOARD),device="cuda")
        x[:,:,CONTENT:] = s
        refs.append(encoder(x)[1])
    return torch.cat(refs)


def build(split,part,revision,names_revision,batch,workers):
    meta,index,_ = inputs()
    paths = index["photo"]["paths"]
    nom = ordered_shards([OUT/f"names-{i}.pt" for i in range(PARTS)],names_revision,torch.arange(len(paths)))
    ids,_ = index["photo"]["views"][split]
    names = nom["name"][ids]
    wrong = names[word_permutation(split,len(ids))]
    take = torch.arange(part,len(ids),2)
    images = ids[take]
    text = torch.load(DEFAULT_CACHE/"text.pt",weights_only=True)
    words = text["words"]
    raw = torch.stack([card(word) for word in [*words,""]])
    card_hash = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    strips = raw.cuda().float().div_(255)
    encoder = Encoder(meta["checkpoint"])
    rows = {k:[] for k in ("interaction","wrong_interaction","canvas","joint","word_only")}
    try:
        ref = blanks(strips,encoder,batch)
        word_change = ref[:-1]-ref[-1]
        if not torch.isfinite(word_change).all() or word_change.norm(dim=1).min()<=0:
            raise ValueError("Undefined effect of a printed word")
        data = loader(paths,images,Path(meta["datasets"]["sketchy"]["root"]),batch,workers)
        offset = 0
        for x in data:
            subset = take[offset:offset+len(x)]
            a,b = names[subset].cuda(),wrong[subset].cuda()
            photo = x.cuda(non_blocking=True).float().div_(255)
            content = interpolate(photo,size=(CONTENT,CONTENT),mode="bilinear",align_corners=False,antialias=True)
            board = torch.ones((len(x),3,BOARD,BOARD),device="cuda")
            board[:,:,:CONTENT,(BOARD-CONTENT)//2:(BOARD+CONTENT)//2] = content
            bare = encoder(board)[1]
            board[:,:,CONTENT:] = strips[a]
            joint = encoder(board)[1]
            board[:,:,CONTENT:] = strips[b]
            decoy = encoder(board)[1]
            interaction = word_change[a]-(joint-bare)
            sham = word_change[b]-(decoy-bare)
            for key,value in (("interaction",interaction),("wrong_interaction",sham),("canvas",bare),
                              ("joint",joint),("word_only",word_change[a])):
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite {key}")
                rows[key].append(value.cpu())
            offset += len(x)
            if offset%(8*batch)==0:print(split,part,offset,"/",len(images),flush=True)
        if offset!=len(images):raise ValueError("Missing photo batch")
        cancellation = (word_change-(ref[:-1]-ref[-1])).abs().max().item()
    finally:
        encoder.close()
    tensors = {k:torch.cat(v) for k,v in rows.items()}
    tensors.update(ids=images,name=names[take],wrong_name=wrong[take],native=nom["native"][images])
    record(OUT/f"gallery-{split}-{part}.pt",tensors,revision,{"names_revision":names_revision,
           "pictures":len(images),"card_sha256":card_hash,"pillow_version":PIL.__version__,
           "blank_cancellation_max_abs":cancellation,"unchanged_names":int((names[take]==wrong[take]).sum())})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("names","gallery"),required=True)
    p.add_argument("--part",type=int,required=True)
    p.add_argument("--split",choices=SPLITS)
    p.add_argument("--revision",required=True)
    p.add_argument("--names-revision")
    p.add_argument("--batch",type=int,default=256)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    deterministic()
    with torch.inference_mode():
        if a.stage=="names" and a.part in range(PARTS):
            assign_names(a.part,a.revision,a.batch,a.workers)
        elif a.stage=="gallery" and a.part in (0,1) and a.split and a.names_revision:
            build(a.split,a.part,a.revision,a.names_revision,a.batch,a.workers)
        else:raise ValueError("Invalid part, stage, or missing dependency revision")
