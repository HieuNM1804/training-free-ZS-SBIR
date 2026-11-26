"""Retrospective source-identity and caption diagnostics. Never called by the retrieval map."""

import json

import open_clip
import torch
import torch.nn.functional as F

from .data import UNSEEN_CLASSES
from .encoding import load_frozen_clip
from .kaggle_unseen_features import DEFAULT_CACHE, load_visual_split
from .profiled_study import true_sources
from .replicated_simplex_study import read
from .semantic_boxable_prompt_search import SKETCHY


def subject_changes(split):
    source = true_sources(split, DEFAULT_CACHE)
    valid = source >= 0
    ids, groups = source[valid].unique(return_inverse=True)
    drawing = F.normalize(split.sketch[valid].float() + split.mirror[valid].float(), dim=1)
    # Match the original diagnostic by averaging independently over redraws of the same subject.
    subjects = F.normalize(torch.stack([drawing[groups == i].mean(0) for i in range(len(ids))]), dim=1)
    photo = F.normalize(split.photo[ids].float(), dim=1)
    return photo - subjects, split.photo_labels[ids.cpu()].cuda(), int(valid.sum()), (subjects * photo).sum(1).mean().item()


def interactions(delta, labels):
    rng = torch.Generator().manual_seed(42)
    parts = [[], []]
    sizes = []
    for cls in labels.unique():
        rows = torch.where(labels == cls)[0]
        sizes.append(len(rows))
        order = rows[torch.randperm(len(rows), generator=rng)]
        for half in range(2):
            parts[half].append(delta[order[half::2]].mean(0))
    left, right = (torch.stack(v).double().cpu() for v in parts)
    global_left, global_right = left.mean(0), right.mean(0)
    left, right = left - global_left, right - global_right
    covariance = (left * right).sum(1).mean()
    permutations = torch.stack([torch.randperm(len(right), generator=rng) for _ in range(4096)])
    null = (left[None] * right[permutations]).sum(2).mean(1)
    return {"classes": len(sizes), "min_source_photos_per_class": min(sizes),
            "global_shift_norm": ((global_left.norm()+global_right.norm())/2).item(),
            "class_dependent_cross_variance": covariance.item(),
            "estimated_class_dependent_fraction_of_squared_class_mean_displacement":
                (covariance/(covariance + global_left @ global_right)).item(),
            "class_identity_permutation95": null.quantile(torch.tensor([.025,.975], dtype=null.dtype)).tolist(),
            "p_perm": (1 + int((null >= covariance).sum())) / (1 + len(null)),
            "class_residual_split_half_cos": F.cosine_similarity(left, right).mean().item()}


def caption_diagnosis(split, delta, labels, model, tokenizer):
    classes = [name.replace("_", " ") for name in UNSEEN_CLASSES[split.name]]
    def encode(template):
        tokens = tokenizer([template.format(name) for name in classes]).cuda()
        return F.normalize(model.encode_text(tokens).float(), dim=1)
    text_shift = encode("a photo of a {}.") - encode("a drawing of a {}.")
    visual = torch.stack([delta[labels == i].mean(0) for i in range(len(classes))])
    v, t = visual - visual.mean(0), text_shift - text_shift.mean(0)
    matched = F.cosine_similarity(v, t)
    rng = torch.Generator().manual_seed(42)
    permutations = torch.stack([torch.randperm(len(classes), generator=rng) for _ in range(4096)]).cuda()
    matrix = F.normalize(v, dim=1) @ F.normalize(t, dim=1).T
    null = matrix[torch.arange(len(classes), device=v.device)[None], permutations].mean(1)
    return {"label_names_used_only_for_this_diagnostic": True, "n_classes": len(classes),
            "cosine_mean_global_shift": F.cosine_similarity(visual.mean(0)[None], text_shift.mean(0)[None]).item(),
            "mean_cosine_class_corrected_shift": matched.mean().item(), "n_positive": int((matched>0).sum()),
            "p_perm": (1 + int((null >= matched.mean()).sum()))/(1 + len(null)),
            "null95": null.quantile(torch.tensor([.025,.975], device=v.device)).tolist(),
            "text_mean_conditional_norm": t.norm(dim=1).mean().item(), "visual_mean_conditional_norm": v.norm(dim=1).mean().item()}


@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = load_frozen_clip(read(DEFAULT_CACHE/"manifest.json")["checkpoint"], torch.device("cuda")).float()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    result = {}
    for name in SKETCHY:
        split = load_visual_split(DEFAULT_CACHE, name)
        delta, labels, n, similarity = subject_changes(split)
        result[name] = {"visual": dict(interactions(delta,labels), paired_sketches=n, distinct_source_photos=len(delta),
                                       avg_source_pair_cos=similarity),
                        "captions": caption_diagnosis(split,delta,labels,model,tokenizer)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
