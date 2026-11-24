"""Does the compact word mixture of a photo support the same locations on the query?"""
import torch

from .regional_held_word_attention import run


def energies(response, measure, bank):
    mean = (response * measure).sum(2)
    moment = (response.square() * measure).sum(2)
    mixed = response @ bank.T
    candidate_mean = (mixed * measure).sum(2)
    joint = (mixed.square() * measure).sum(2) / candidate_mean.clamp_min(1e-60)
    separate = (moment / mean.clamp_min(1e-60)) @ bank.T
    # The weighted squared norm of the mean location profile cannot exceed its mean squared norm.
    if (joint > separate + 1e-10 * separate.abs() + 1e-22).any():
        raise ValueError('Word-location Jensen identity violated')
    return joint.mean(1).clamp_min(1e-30).log(), separate.mean(1).clamp_min(1e-30).log()


def read(reader):
    logits = reader.region * reader.scale
    response = (logits - logits.amax((1,2,3),keepdim=True)).exp().double()
    bank = reader.bank.double()
    prior = reader.prior.double()
    gallery_mean = reader.l.mean(-1,keepdim=True).clamp_min(1e-30).double()
    measure = prior / gallery_mean
    measure = measure / measure.sum(2,keepdim=True)
    # Only the word-level moments need high precision; computing a dense FP64 gallery product
    # is unnecessary. The gallery likelihood and native attention already carry the native joint term.
    original = reader.read('self',1.,'gallery').double()
    def moments(profiles, nu):
        mean = (profiles*nu).sum(2)
        second = (profiles.square()*nu).sum(2)
        return ((second/mean.clamp_min(1e-60)).float() @ reader.bank.T).mean(1).clamp_min(1e-30).log().double()
    separate = moments(response,measure)
    if (original-separate).max() > .005:
        raise ValueError('Gallery readout exceeded its separate-word upper bound')
    # Independently reconstruct the algebra on the first fixed 16 gallery IDs of every query.
    native_check, separate_check = energies(response,measure,bank[:16])
    if max((native_check-original[:,:16]).abs().max(),(separate_check-separate[:,:16]).abs().max())>.005:
        raise ValueError('Moment readout does not reproduce the native region scorer')
    gap = (separate-original).clamp_min(0)
    scale_mass = torch.cat([measure[:,:,sl].sum(2,keepdim=True)
                            for sl in (slice(0,9),slice(9,13))],2)
    scale_profiles = torch.stack([(response[:,:,sl]*measure[:,:,sl]).sum(2) / scale_mass[:,:,i]
                                  for i,sl in enumerate((slice(0,9),slice(9,13)))],2)
    coarse_words = moments(scale_profiles,scale_mass)
    coarse_gallery = scale_profiles.float() @ reader.bank.T
    coarse_measure = scale_mass.float()
    coarse_native = ((coarse_gallery.square()*coarse_measure).sum(2) /
                     (coarse_gallery*coarse_measure).sum(2).clamp_min(1e-30)).mean(1).clamp_min(1e-30).log()
    coarse_gap = (coarse_words-coarse_native).clamp_min(0)
    order = original.argsort(dim=1,stable=True)
    peer = torch.arange(order.size(1),device=order.device)^1
    peer.clamp_max_(order.size(1)-1)
    # Each row swaps gaps between adjacent native scores. Odd singleton stays fixed.
    shuffled_gap = torch.empty_like(gap).scatter_(1,order,gap.gather(1,order[:,peer]))
    return {'self':original.float(),'agreement':(original-gap).float(),'scale_agreement':(original-coarse_gap).float(),
            'near_score_sham':(original-shuffled_gap).float(),'separate_words':separate.float()}


if __name__ == '__main__':
    run(family='regional-word-location-agreement',alternate=read,start=1536,count=3072,batch_size=24,
        selected_windows=True,device_reference='regional-additive-device-reference',
        contrasts=(('agreement','self'),('agreement','scale_agreement'),
                   ('agreement','near_score_sham'),('separate_words','self')))
