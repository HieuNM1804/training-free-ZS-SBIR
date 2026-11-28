"""Quantify whether the printed-name effect is attenuated more for the photo's own name."""

import argparse

import torch

from .description_block_results import Population
from .description_choice_blocks import inputs
from .native_block_probe import deterministic
from .photo_name_evaluation import gallery
from .photo_name_interaction import OUT, record
from .wordnet_renaming_pilot import SPLITS


def attenuation(interaction, reference):
    ref_energy=reference.square().sum(1)
    energy=interaction.square().sum(1)
    if ref_energy.min()<=0 or energy.min()<=0:raise ValueError("Undefined geometric control")
    dot=(interaction*reference).sum(1)
    # fraction of the isolated word response absent when the image is present
    return dot/ref_energy,dot.square()/(energy*ref_energy)


@torch.inference_mode()
def run(split,revision,gallery_revision):
    _,index,_=inputs()
    gi,classes=index["photo"]["views"][split]
    data=gallery(split,gi,gallery_revision)
    names,wrong=data["name"],data["wrong_name"]
    word=data["word_only"].double()
    refs=torch.empty((601,word.shape[1]),dtype=word.dtype,device="cuda")
    # Different shards may batch the same frozen word slightly differently.
    for n in names.unique():
        refs[n]=word[names==n].mean(0)
    deviation=(refs[names]-word).norm(dim=1)/word.norm(dim=1)
    if deviation.max()>.002:raise ValueError("Unexpected discrepancy among repeated name cards")
    own,own_energy=attenuation(data["interaction"].double(),word)
    other,other_energy=attenuation(data["wrong_interaction"].double(),refs[wrong])
    population=Population(classes,gi)
    quantiles=torch.tensor([.1,.5,.9],dtype=torch.float64,device="cuda")
    arrays={"own_attenuation":own,"shuffled_attenuation":other,"own_fraction_on_word_axis":own_energy,
            "shuffled_fraction_on_word_axis":other_energy}
    summaries={k:{"mean_percent":100*v.mean().item(),"p10_median_p90_percent":(100*torch.quantile(v,quantiles)).tolist()}
               for k,v in arrays.items()}
    record(OUT/f"attenuation-{split}.pt",{"photo_ids":gi,"classes":classes,**{k:v.cpu() for k,v in arrays.items()}},
           revision,{"split":split,"photos":len(gi),"gallery_revision":gallery_revision,"summaries":summaries,
                     "paired_own_minus_shuffled":population.effect((own-other).cpu()),
                     "max_relative_word_card_discrepancy":deviation.max().item()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",required=True,choices=SPLITS)
    p.add_argument("--revision",required=True)
    p.add_argument("--gallery-revision",required=True)
    a=p.parse_args()
    deterministic()
    run(a.split,a.revision,a.gallery_revision)
