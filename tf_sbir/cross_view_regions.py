"""Select photo-specific sketch windows in one rendering and verify the selection in the other."""

import torch


class CandidateRegions:
    def __init__(self,full,flipped,crops,reverse,source,compact,bank,full_index,historical):
        gs=torch.stack((full,flipped.float()),1)@source.T
        zz=torch.stack((crops.float(),reverse.float()),1)@source[compact].T
        self.full=gs
        self.region=zz
        self.bank=bank
        self.full_index=full_index
        self.historical=historical

    def prepare(self,scale):
        g=self.full*scale
        # One query-wide offset preserves relative evidence across both poses.
        total=(g-g.amax((1,2),keepdim=True)).exp().mean(1)
        self.both=(total@self.full_index.T).clamp_min(1e-30).log()
        self.first=self.historical.clamp_min(1e-30).log()
        z=self.region*scale
        p=(z-z.amax((1,2,3),keepdim=True)).exp()
        self.l=(p.flatten(0,2)@self.bank.T).reshape(*p.shape[:3],-1).clamp_min_(1e-30)
        self.prior=self.l.new_tensor([1/18]*9+[1/8]*4)[None,None,:,None]
        self.logl=self.l.log()
        self.rolling=torch.tensor((3,4,5,6,7,8,0,1,2,11,12,9,10),device=p.device)

    def read(self,kind,temperature,normalization):
        l=self.l
        if kind=="native":return (l[:,0]*self.prior[:,0]).sum(1).clamp_min(1e-30).log()
        if kind=="uniform":return (l*self.prior).sum(2).mean(1).clamp_min(1e-30).log()
        if kind in ("ratio","ratio_wrong"):
            items=[]
            for subset in (slice(0,9),slice(9,13)):
                a,b=l[:,0,subset],l[:,1,subset]
                if kind=="ratio_wrong":b=b.roll(1,1)
                mean_a=a.mean(1);mean_b=b.mean(1)
                value=(a*b).mean(1)/(mean_a*mean_b).clamp_min(1e-30)
                items.append(.5*(mean_a+mean_b)*value.clamp_min(1e-8).pow(temperature))
            return ((items[0]+items[1])*.5).clamp_min(1e-30).log()
        if normalization=="gallery":
            named=self.logl-l.mean(3,keepdim=True).clamp_min(1e-30).log()
        elif normalization=="raw":named=self.logl
        else:raise ValueError(normalization)
        if kind=="global_selector":named=named.amax(3,keepdim=True)
        att=(self.prior.log()+temperature*named).softmax(2)
        if kind=="self":other=l
        elif kind=="wrong":other=l.flip((1,)).index_select(2,self.rolling)
        elif kind in ("cross","global_selector"):other=l.flip((1,))
        elif kind=="opposed":other=l.flip((1,));att=(self.prior.log()-temperature*named).softmax(2)
        else:raise ValueError(kind)
        return (att*other).sum(2).mean(1).clamp_min(1e-30).log()

    def check(self):
        err=0.
        for sl in (slice(0,9),slice(9,13)):
            a,b=self.l[:3,0,sl,:21].double(),self.l[:3,1,sl,:21].double()
            denom=a.mean(1)*b.mean(1)
            terms=[(a*b.roll(i,1)).mean(1)/denom for i in range(a.size(1))]
            err=max(err,float((torch.stack(terms).mean(0)-1).abs().max()))
        if err>5e-9:raise ValueError("Region permutation ratio is not normalized")
        return err
