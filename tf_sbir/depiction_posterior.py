"""Turn the uncertain observation of shared word coordinates into a predictive word readout."""

import torch

from .depiction_mode_diagnosis import centered,observation_modes,view_covariances


class DepictionModel:
    def __init__(self,photos,text,scale):
        within,between=view_covariances(photos,text,scale)
        n,axes=observation_modes(within,between)
        self.axes=axes.float()
        self.back=torch.linalg.pinv(axes).float()
        self.noise=n.float().clamp_min(1e-7)
        self.between=(1-n).float().clamp_min(1e-7)
        self.word_noise=within.diag().float().clamp_min(1e-7)
        self.word_between=between.diag().float().clamp_min(1e-7)
        total=text.new_zeros(len(text),dtype=torch.float64)
        for view in photos.split(256):
            raw=centered((view.cuda().float()@text.T)*scale)
            total+=raw.mean(1).sum(0,dtype=torch.float64)
        self.mean=(total/len(photos)).float()
        self.diagnostic={"retained_axes":len(n),"small_noise_fraction":float(n.min()),"large_noise_fraction":float(n.max()),
                         "basis_right_inverse_error":float((self.back@self.axes-torch.eye(len(n),device=axes.device)).abs().max())}

    def observe(self,query,zooms,text,scale):
        z=zooms.float()@text.T*scale
        small=z[:,:9].logsumexp(1)-z.new_tensor(9.).log()
        large=z[:,9:].logsumexp(1)-z.new_tensor(4.).log()
        view_mean=centered(torch.logaddexp(small,large)-z.new_tensor(2.).log())
        orig=centered(query@text.T*scale)
        z=centered(z)
        return {"native":orig,"zoom":view_mean,"variance":(z@self.axes).var(1,correction=1),
                "word_variance":z.var(1,correction=1),"zg":(view_mean-self.mean)@self.axes,
                "ng":(orig-self.mean)@self.axes}

    def likelihood(self,x,view,tau,eta,kind,geometry="gallery"):
        y=x[view]
        if kind=="native" or tau==0:return y
        if geometry=="word":
            noise,b=self.word_noise,self.word_between
            individual=x["word_variance"]
            co=y-self.mean
            back=None
        else:
            noise,b=self.noise,self.between
            individual=x["variance"]
            co=x["zg" if view=="zoom" else "ng"]
            back=self.back
            if geometry=="reversed":
                noise,b=noise.flip(0),b.flip(0)
            elif geometry=="spherical":
                noise=noise.mean().expand_as(noise)
                b=b.mean().expand_as(b)
                individual=individual.mean(1,keepdim=True).expand_as(individual)
            elif geometry=="scalar_query":
                individual=noise*(individual.sum(1,keepdim=True)/noise.sum())
            elif geometry=="shuffled_query":
                order=torch.randperm(len(noise),generator=torch.Generator().manual_seed(63453)).to(noise.device)
                individual=(individual/noise)[:,order]*noise
            elif geometry!="gallery":raise ValueError(geometry)
        observed=tau*((1-eta)*noise+eta*individual)
        gain=b/(b+observed).clamp_min(1e-20)
        delta=(gain-1)*co
        v=b*(1-gain)
        if back is not None:
            delta=delta@back
            v=v@back.square()
            prior_var=b@back.square()
        else:prior_var=b
        m=y+delta
        if kind=="mean":return m
        if kind=="predictive":return m+.5*v
        if kind=="ratio":return m-self.mean+.5*(v-prior_var)
        raise ValueError(kind)
