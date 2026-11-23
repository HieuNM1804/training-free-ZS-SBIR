"""Confirm the observed reversal for candidate scale selection alone and with global word retrieval."""
import torch

from .regional_held_word_attention import run


def controls(reader):
    l=reader.l
    a=(reader.prior.log()+reader.logl-l.mean(-1,keepdim=True).clamp_min(1e-30).log()).softmax(2)
    shared=a.mean(-1,keepdim=True)
    scale=[]
    for sl in (slice(0,9),slice(9,13)):
        positions=shared[:,:,sl]
        positions=positions/positions.sum(2,keepdim=True).clamp_min(1e-30)
        scale.append(a[:,:,sl].sum(2,keepdim=True)*positions)
    def read(gate):
        return (gate*l).sum(2).mean(1).clamp_min(1e-30).log()
    return {'self':read(a),'scale_only':read(torch.cat(scale,2)),'shared':read(shared)}


if __name__=='__main__':
    for dataset,start in [('sketchy',6912),('tuberlin',0)]:
        run(family='regional-scale-confirmation',alternate=controls,dataset=dataset,start=start,
            count=100_000,batch_size=24,contrasts=(('scale_only','self'),('scale_only','shared')))
