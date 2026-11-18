"""Full-gallery evaluation using the original KD-SBIR TorchMetrics protocol."""

import torch
from torch.nn import functional as F
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision
from tqdm.auto import tqdm


@torch.inference_mode()
def evaluate_rows(score_rows, query_labels, gallery_labels, dataset, description=None):
    """Evaluate one cosine or cosine-scaled score row per query."""
    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()
    ap = torch.zeros(len(query_labels))
    precision = torch.zeros_like(ap)
    map_k = 200 if dataset == "sketchy_2" else 0
    p_k = 200 if dataset in ("sketchy_2", "quickdraw") else 100
    for index, (similarities, label) in enumerate(zip(
        tqdm(score_rows, total=len(query_labels), desc=description, disable=description is None),
        query_labels, strict=True,
    )):
        cosine = similarities.cpu()
        score = ((cosine + 1.0) * 0.5).clamp(min=torch.finfo(cosine.dtype).eps, max=1.0)
        target = gallery_labels.eq(label)
        if map_k:
            ap[index] = retrieval_average_precision(score, target, top_k=min(map_k, len(gallery_labels)))
        else:
            ap[index] = retrieval_average_precision(score, target)
        precision[index] = retrieval_precision(score, target, top_k=p_k)
    return ap.mean(), precision.mean(), map_k, p_k


def cosine_rows(queries, gallery):
    for query in queries:
        yield F.cosine_similarity(query.unsqueeze(0), gallery)
