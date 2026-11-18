"""Full-gallery evaluation using the original KD-SBIR TorchMetrics protocol."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import torch
from torch.nn import functional as F
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_precision
from tqdm.auto import tqdm


@torch.inference_mode()
def evaluate_batch(batch, gallery_labels, map_k, p_k):
    values = torch.empty((len(batch), 2))
    for index, (similarities, label) in enumerate(batch):
        cosine = similarities.cpu()
        score = ((cosine + 1.0) * 0.5).clamp(min=torch.finfo(cosine.dtype).eps, max=1.0)
        target = gallery_labels.eq(label)
        if map_k:
            values[index, 0] = retrieval_average_precision(score, target, top_k=min(map_k, len(gallery_labels)))
        else:
            values[index, 0] = retrieval_average_precision(score, target)
        values[index, 1] = retrieval_precision(score, target, top_k=p_k)
    return values


def metric_batches(pairs, gallery_labels, map_k, p_k, workers):
    chunks = iter(lambda: list(islice(pairs, 32)), [])
    if workers == 1:
        for batch in chunks:
            yield evaluate_batch(batch, gallery_labels, map_k, p_k)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for batch in chunks:
            pending.append(pool.submit(evaluate_batch, batch, gallery_labels, map_k, p_k))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


@torch.inference_mode()
def evaluate_rows(score_rows, query_labels, gallery_labels, dataset, description=None, workers=1):
    """Apply the reference metrics per query, preserving their aggregation order."""
    query_labels = query_labels.cpu()
    gallery_labels = gallery_labels.cpu()
    map_k = 200 if dataset == "sketchy_2" else 0
    p_k = 200 if dataset in ("sketchy_2", "quickdraw") else 100
    ap = torch.zeros(len(query_labels))
    precision = torch.zeros_like(ap)
    pairs = zip(score_rows, query_labels, strict=True)
    start = 0
    with tqdm(total=len(query_labels), desc=description, disable=description is None, mininterval=1.0) as progress:
        for values in metric_batches(pairs, gallery_labels, map_k, p_k, workers):
            end = start + len(values)
            ap[start:end], precision[start:end] = values[:, 0], values[:, 1]
            progress.update(len(values))
            start = end
    return ap.mean(), precision.mean(), map_k, p_k


def cosine_batches(queries, gallery, batch_size):
    for batch in queries.split(batch_size):
        if len(batch) == 1:
            yield F.cosine_similarity(batch, gallery).unsqueeze(0)
        else:
            yield F.cosine_similarity(batch[:, None, :], gallery[None, :, :], dim=-1)


def cosine_rows(queries, gallery, batch_size=1):
    for scores in cosine_batches(queries, gallery, batch_size):
        yield from scores.cpu()
