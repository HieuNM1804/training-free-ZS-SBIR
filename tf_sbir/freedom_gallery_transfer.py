"""Transfer both stages of FreeDom to the existing unseen split and gallery protocol."""

from dataclasses import dataclass,replace

import torch

from .conditional_semantic_index import bare_words
from .foundation import load_words
from .reverse_prompt import nearest_words


@dataclass(frozen=True)
class Config:
    proxies: int = 20
    words_per_proxy: int = 7
    retained_words: int = 7
    search: str = "rgb"
    composition: str = "word-domain"
    shuffled: bool = False


def canonical():
    return Config()


def candidates():
    original=canonical()
    return [replace(original,proxies=k,search=s) for k in (1,8,20,48,96) for s in ("rgb","semantic")]


def controls(c):
    return {"primary":c,"original":canonical(),"query_only":replace(c,proxies=1),
            "rgb_search":replace(c,search="rgb"),"semantic_search":replace(c,search="semantic"),
            "prompt_ensemble":replace(c,composition="ensemble"),"bare_composition":replace(c,composition="bare"),
            "shuffle_composed_word_identity":replace(c,shuffled=True)}


def cache_key(c):
    return c.proxies,c.search,c.shuffled,c.composition


def prepare_data(data,args,words):
    data.bare=bare_words(args,words)[0]
    data.photo_composed=load_words(args,words,(("{} photo","{} photo"),("{} photo","{} photo")))[0].cuda()


class Index:
    def __init__(self,data):
        self.d=data
        self.vocab=len(data.bare)
        self.query_words=nearest_words(data.q,data.bare,7)[0]
        self.photo_words=nearest_words(data.p,data.bare,7)[0]
        self.retrieved={}
        self.perm=torch.randperm(self.vocab,generator=torch.Generator().manual_seed(819)).to(data.p.device)
        self.info={"visual_memory":"unlabeled retrieval gallery","raw_vocabulary":self.vocab,
                   "composed_text":"<term> photo","query_is_always_present":True}

    def get_neighbors(self,search):
        if search not in self.retrieved:
            scores=self.d.q@self.d.p.T if search=="rgb" else self.d.base
            self.retrieved[search]=scores.topk(95,dim=1).indices
        return self.retrieved[search]

    def score(self,c):
        if c.proxies==1:
            proxies=self.query_words[:,None,:c.words_per_proxy]
        else:
            ids=self.get_neighbors(c.search)[:,:c.proxies-1]
            proxies=torch.cat((self.query_words[:,None,:c.words_per_proxy],self.photo_words[ids,:c.words_per_proxy]),1)
        bank={"word-domain":self.d.photo_composed,"ensemble":self.d.target,"bare":self.d.bare}[c.composition]
        queries=[]
        for votes in proxies.flatten(1).split(256):
            totals=torch.zeros((len(votes),self.vocab),device=votes.device,dtype=torch.int32)
            totals.scatter_add_(1,votes,torch.ones_like(votes,dtype=torch.int32))
            order=torch.arange(votes.size(1),device=votes.device,dtype=torch.int32).expand_as(votes)
            seen=torch.full_like(totals,votes.size(1))
            seen.scatter_reduce_(1,votes,order,reduce="amin",include_self=True)
            # The reference's stable frequency sort breaks ties by first encounter.
            keys=totals*(votes.size(1)+1)+(votes.size(1)-seen)
            selected=keys.topk(c.retained_words,dim=1).indices
            counts=totals.gather(1,selected).float()
            weights=counts/counts.amax(1,keepdim=True)
            if c.shuffled:
                selected=self.perm[selected]
            queries.append(torch.bmm(weights[:,None],bank[selected]).squeeze(1))
        return torch.cat(queries)@self.d.p.T
