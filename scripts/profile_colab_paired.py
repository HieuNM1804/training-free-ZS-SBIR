"""Profile real Sketchy inputs with the same frozen encoder, full gallery, and official evaluator."""

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from colab_paired import ROUTING, checkout


def measure(run, samples):
    durations = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    seconds = statistics.median(durations)
    return {"median_seconds": seconds, "items_per_second": samples / seconds,
            "peak_cuda_mib": round(torch.cuda.max_memory_reserved() / 2**20)}


@torch.inference_mode()
def main():
    work = Path("/content/sbir")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(checkout(work, repo, *ROUTING[:2])))
    from tf_sbir.benchmark import merge_inputs
    from tf_sbir.capture_query_metrics import evaluate_with_details
    from tf_sbir.data import SKETCHY_SPLITS, threaded_batches
    from tf_sbir.encoding import encode_clip, global_cache_name, load_frozen_clip
    from tf_sbir.foundation import load_words
    from tf_sbir.gallery_semantic_balance import query_evidence
    from tf_sbir.global_methods import centered
    from tf_sbir.mta import normalized_crops, pixel_lookup
    from tf_sbir.paired_neighborhood_scale import scale_index
    from tf_sbir.reflection_transfer import cache_reflection_features
    from tf_sbir.semantic import read_words
    from tf_sbir.transfer_benchmark import AnchorSketchViews
    from tf_sbir.contour_query_expansion import gallery_word_codes

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    root = Path(json.loads((work / "ready.json").read_text())["root"])
    cache = work / "features"
    checkpoint = str(work / "assets/ViT-B-32.pt")
    args = SimpleNamespace(cache=cache, device="cuda", clip_checkpoint=checkpoint, workers=6, batch_size=192)
    q_paths, qs = merge_inputs(root, SKETCHY_SPLITS, "sketch")
    p_paths, ps = merge_inputs(root, SKETCHY_SPLITS, "photo")
    ids, labels = qs["sketchy_1"]
    pids, plabels = ps["sketchy_1"]
    chosen = torch.linspace(0, len(ids)-1, 768).long()
    queries = [q_paths[int(i)] for i in ids[chosen]]
    photos = [p_paths[int(i)] for i in pids[torch.linspace(0, len(pids)-1, 768).long()]]
    result = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cpus": len(os.sched_getaffinity(0)),
              "graph_gallery": len(pids), "graph_queries": len(chosen), "repeats": 2,
              "revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()}
    model = load_frozen_clip(checkpoint, device)
    encode_clip(model, photos[:64], "rgb", 32, 0, device)
    for view in ("rgb", "contours"):
        result[view] = []
        for batch, workers in ((64, 2), (192, 6), (320, 8), (448, 10)):
            try:
                stats = measure(lambda: encode_clip(model, photos, view, batch, workers, device), len(photos))
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower() and "bus error" not in str(exc).lower():
                    raise
                stats = {"unavailable": type(exc).__name__}
            result[view].append({"batch": batch, "workers": workers, **stats})
    dataset = AnchorSketchViews(queries)
    lookup = pixel_lookup(device, model.visual.conv1.weight.dtype)
    def crops(batch, workers):
        for pixels in threaded_batches(dataset, batch // 4, workers, True):
            data = normalized_crops(pixels.flatten(0, 1), lookup)
            F.normalize(model.encode_image(data).float(), dim=-1).half().cpu()
    result["crop_views"] = []
    for batch, workers in ((192, 6), (384, 6), (384, 8), (512, 10)):
        result["crop_views"].append({"batch": batch, "workers": workers, **measure(lambda: crops(batch, workers), 4*len(dataset))})
    del model
    torch.cuda.empty_cache()

    def bank(paths, modality, view):
        name = global_cache_name(root, paths, modality, view, "fp16", checkpoint)
        return torch.load(cache/name, map_location="cpu", weights_only=True, mmap=True)
    raw = bank(q_paths, "sketch", "rgb")
    gallery = bank(p_paths, "photo", "rgb")
    outline = bank(p_paths, "photo", "contours")[pids].cuda()
    crop_bank = bank(q_paths, "sketch", "ink-scales-mirror-v1")["features"]
    mirrors = bank(q_paths, "sketch", "reflected-full-medium-small-v1")["features"]
    project = torch.load(next(cache.glob("iso-b32-150-50-*.pt")), map_location=device, weights_only=True)
    src = F.normalize(F.normalize(outline @ project, dim=-1).mean(1), dim=-1)
    center = src.mean(0)
    source = centered(src, center)
    photo = gallery[pids].cuda()
    target = F.normalize(photo @ project, dim=-1)
    target = centered(target, target.mean(0))
    words = read_words(work / "assets/open-images-v7-classes.csv")
    text = [x.cuda() for x in load_words(args, words)]
    code = gallery_word_codes(photo, text[1]).half().float()
    index = scale_index(photo, target, code, 192)
    with tempfile.TemporaryDirectory(dir=work) as temp:
        features = cache_reflection_features(Path(temp), ids[chosen], (raw, crop_bank, mirrors), source, target, center, project, *text)
        queries = [x.cuda() for x in features]
        def graph(batch):
            for start in range(0, len(chosen), batch):
                span = slice(start, start + batch)
                tail, head = query_evidence(*(q[span] for q in queries), index, modes=("text35",), compact=True)
                head["text35"].cpu()
                tail.cpu()
        graph(64)
        result["query_graph"] = [{"batch": b, **measure(lambda: graph(b), len(chosen))} for b in (64, 128, 192, 256, 384)]
    for split in SKETCHY_SPLITS:
        qi, labels = qs[split]
        pi, plabels = ps[split]
        ix = torch.linspace(0, len(qi)-1, 1024).long()
        scores = (raw[qi[ix]].cuda() @ gallery[pi].cuda().T).cpu()
        reference = evaluate_with_details(iter(scores), labels[ix], plabels, split, None, 1)[1]
        def evaluate(workers):
            current = evaluate_with_details(iter(scores), labels[ix], plabels, split, None, workers)[1]
            if not torch.equal(reference, current):
                raise ValueError(f"Worker count changed {split} metrics")
        result["evaluator_" + split] = [{"workers": n, **measure(lambda: evaluate(n), len(ix))} for n in (2, 4, 6, 8, 12)]
    output = work / "results/paired-throughput.json"
    output.write_text(json.dumps(result, indent=2))
    print(output, flush=True)


if __name__ == "__main__":
    main()
