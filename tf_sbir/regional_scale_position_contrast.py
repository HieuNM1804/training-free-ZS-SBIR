"""Disentangle standalone positional evidence from the scale-position interaction in a candidate's regional score."""
import torch

from .regional_held_word_attention import run


def decompose(reader):
    likelihood = reader.l
    evidence = reader.logl - likelihood.mean(-1, keepdim=True).clamp_min(1e-30).log()
    attention = (reader.prior.log() + evidence).softmax(2)
    shared = attention.mean(-1, keepdim=True)
    scale_gate, position_gate = [], []
    for rows in (slice(0, 9), slice(9, 13)):
        individual, average = attention[:, :, rows], shared[:, :, rows]
        scale = individual.sum(2, keepdim=True)
        shared_scale = average.sum(2, keepdim=True)
        scale_gate.append(scale * average / shared_scale.clamp_min(1e-30))
        position_gate.append(shared_scale * individual / scale.clamp_min(1e-30))

    def read(weights):
        return (weights * likelihood).sum(2).mean(1).clamp_min(1e-30).log()

    full = read(attention)
    scale = read(torch.cat(scale_gate, 2))
    position = read(torch.cat(position_gate, 2))
    base = read(shared)
    interaction = (full-scale) - (position-base)
    additive = scale + position - base
    if (full - (additive+interaction)).abs().max() > 2e-5:
        raise ValueError('Log factorial decomposition changed the full score')

    # Substitute another candidate's interaction without altering its per-query marginal distribution.
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(interaction.size(1), generator=generator).to(interaction.device)
    donor = torch.empty_like(order).scatter_(0, order, order.roll(1))
    return {'self': full, 'shared': base, 'scale': scale, 'position': position,
            'additive': additive, 'scale_interaction': scale+interaction,
            'permuted_interaction': scale+interaction[:,donor]}


if __name__ == '__main__':
    run(family='regional-scale-position-contrast', alternate=decompose, start=0, count=1536,
        batch_size=24, selected_windows=True, device_reference=True, contrasts=(('scale_interaction','self'),('scale_interaction','scale'),
        ('scale_interaction','permuted_interaction'),('self','additive'),('position','shared')))
