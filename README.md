# Training-free zero-shot sketch-based image retrieval

Frozen OpenAI CLIP ViT-B/32 for category-level sketch-to-photo retrieval. The methods estimate gallery statistics or similarity scores without fitting a model. The two Sketchy unseen splits are the defaults. TU-Berlin and QuickDraw can be selected explicitly.

## Setup

Install a matching PyTorch and torchvision build for your machine, then install the remaining dependencies.

```powershell
python -m pip install -r requirements.txt
```

Supply a dataset root with this layout.

```text
Sketchy/
  photo/<class>/<image>
  sketch/<class>/<image>
```

## Reproduce the experiments

```powershell
python -m tf_sbir.benchmark --root <Sketchy> --output runs/global
python -m tf_sbir.patch_benchmark --root <Sketchy> --output runs/patch-fixed
```

The default runs evaluate both complete Sketchy splits. Use `--splits sketchy_1`,
`--splits sketchy_2`, `--splits tuberlin`, or `--splits quickdraw` with the
matching dataset root to run one split. Options for batch sizes, workers, device, and
cache location are available through `--help`. For an explicit local OpenAI CLIP
checkpoint, pass `--clip-checkpoint <ViT-B-32.pt>`. Models otherwise resolve through
OpenCLIP. `--encode-only` in the patch runner prepares spatial CLIP patch
caches for subsequent scoring. Each method writes a standalone JSON result. All encoders run under
inference mode with frozen weights. Batched cosine scoring uses the same
operation as the reference, and the patch sweeps reuse cached cosine rows.

| Global method | Score |
| --- | --- |
| `rgb` | Frozen CLIP cosine on the original inputs |
| `contour` | Sketch to the mean of two Canny views of each photo |
| `rrf` | Reciprocal rank fusion of RGB and contour |
| `centered` | Subtract the contour mean from the sketch side and the RGB mean from the photo side |
| `query_expansion` | Expand the centered query with RGB features of its eight nearest contour anchors |
| `local_residual` | Shift the centered query by paired RGB-minus-contour offsets from the same anchors |
| `procrustes` | Align centered contour features to their corresponding RGB features with an orthogonal map |

The patch runner uses `centered` to create a common shortlist. It evaluates uniform
query patches, ink-supported patches against RGB or contour, independent averaging
of RGB and contour matches, agreement at the same gallery patch, and agreement
after changing sketch line thickness. A contrast-weighted variant downweights
sketch patches that match the candidate pool uniformly. The centered patch
variants estimate one feature mean per gallery view and test whether removing
those offsets improves RGB matching or cross-view agreement. All variants use spatial tokens from the same frozen CLIP
visual encoder and the same gallery support positions. The reranker permutes the existing
scores within the shortlist and evaluates the complete gallery.

The fixed patch comparison uses a shortlist of 200 and a local score weight of
0.5. The following sweeps are exploratory because the configurations are
compared and selected using the official unseen splits. Labels are only
used in evaluation, and model weights stay frozen.

```powershell
python -m tf_sbir.benchmark --root <Sketchy> --output runs/global-exploratory --methods query_expansion local_residual --weights 0.25 0.75 1 --exploratory-unseen
python -m tf_sbir.benchmark --root <Sketchy> --output runs/global-exploratory --methods procrustes --weights 0.25 0.5 --exploratory-unseen
python -m tf_sbir.patch_benchmark --root <Sketchy> --output runs/patch-exploratory --methods patch_centered_ink patch_independent --top-k 100 400 --weights 0.25 0.75 --exploratory-unseen
python -m tf_sbir.patch_benchmark --root <Sketchy> --output runs/combined-exploratory --methods patch_centered_ink --shortlist-method local_residual --top-k 200 400 --weights 0.25 0.5 --exploratory-unseen
```

CUDA uses FP16 model inference and batched float32 cosine operations. The
TorchMetrics calls run in CPU workers and preserve the original order of
aggregation. Large score matrices and patch features are cached locally. Use
a fresh output directory when changing experiment inputs, or pass
`--resume` to skip configurations already completed with the same inputs.

## Evaluation

The ordered class lists, RGB resize to 224 × 224, CLIP normalization, and metric
conventions come from the [KD-SBIR reference repository](https://github.com/HieuNM1804/KD-SBIR).
The evaluator uses its positive score transform and query-averaged TorchMetrics
retrieval functions. Gallery positives are photos of the query class.

| Split | Unseen classes | mAP | Precision |
| --- | ---: | --- | --- |
| Sketchy-Ext-1 | 25 | mAP@all | P@100 |
| Sketchy-Ext-2 | 21 | mAP@200 | P@200 |
| TU-Berlin | 30 | mAP@all | P@100 |
| QuickDraw | 30 | mAP@all | P@200 |

The loader walks class folders in protocol order and sorts files within each
folder. It skips non-file entries inside class folders.

Run the local checks with `python -m unittest discover -s tests`.

## Colab

The [Sketchy to TU-Berlin report](docs/SKETCHY_TUBERLIN_TRANSFER_VI.md) gives results for parameters selected on both Sketchy splits and frozen before inference on all 30 TU-Berlin classes. The [per-class table](docs/results/tuberlin-per-class.csv) covers 2,130 queries against the full 27,262-photo gallery.

Clone the repository into the GPU runtime and install `requirements.txt`.
Point `--root` to the directory containing the dataset's `photo` and `sketch`
folders. Keep datasets, cache and run output under the runtime's local disk.
Reuse the same `--cache` across benchmarks to avoid repeated CLIP inference.

Tune `--batch-size` and `--workers` together on the runtime. Each source
photo in the contour encoder produces two model inputs. The crop encoder
interprets the batch size as a limit on the number of views. Use
`--crop-batch-size` and `--crop-workers` to tune crop inference separately
from the contour loader. Crop images move to the device as bytes and use a
table with the reference normalization values. For semantic
transport, `--score-batch` limits the number of simultaneous queries and
`--transport-buffer-mib` caps their gathered photo-view tensor. The memory used
by scoring and CLIP is additional to this buffer. The reference precision and
mAP functions continue to run through `--metric-workers` CPU threads.

Run `tf_sbir.semantic_benchmark --study partial-transport` to compare the
STiTch conditional score with SuperGlue-style partial matching on the same
shortlist and crop descriptors.


## Training-free research experiments

The [literature review in Vietnamese](docs/TRAINING_FREE_LITERATURE_VI.md)
covers cross-domain retrieval, modality alignment, vocabulary inversion,
test-time adaptation, and downstream segmentation. These experiments build
the index from unlabeled gallery photos and process each sketch independently.
The public vocabulary is the complete Open Images V7 class-name list. The
benchmark class lists are used for loading and scoring.

```powershell
Invoke-WebRequest -Uri 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv' -OutFile open-images-v7-classes.csv
python -m tf_sbir.manifold_benchmark --root <Sketchy> --output runs/paper-manifold-b32 --exploratory-unseen
python -m tf_sbir.manifold_benchmark --root <Sketchy> --output runs/paper-diffusion-b32 --study diffusion --exploratory-unseen
python -m tf_sbir.manifold_benchmark --root <Sketchy> --output runs/paper-gallery-scales-b32 --study scales --score-batch 384 --metric-workers 6 --exploratory-unseen
python -m tf_sbir.semantic_benchmark --root <Sketchy> --vocabulary open-images-v7-classes.csv --output runs/paper-semantic-b32 --exploratory-unseen
python -m tf_sbir.semantic_benchmark --root <Sketchy> --vocabulary open-images-v7-classes.csv --output runs/paper-semantic-walk-b32 --study manifold --exploratory-unseen
python -m tf_sbir.semantic_benchmark --root <Sketchy> --vocabulary open-images-v7-classes.csv --output runs/paper-agreement-b32 --study agreement --exploratory-unseen
python -m tf_sbir.semantic_benchmark --root <Sketchy> --vocabulary open-images-v7-classes.csv --output runs/paper-consensus-b32 --study consensus --exploratory-unseen
```

All research experiments use the same frozen OpenAI CLIP ViT-B/32 checkpoint. The core and semantic manifold
experiments use the original RGB gallery and sketches. The agreement experiment
adds a visual path initialized from paired contour calibration. All encoders
remain frozen, and all rows use the existing TorchMetrics evaluator.
Use a separate output directory for each study.

The fixed fusion rows `fuse-k160-rgb-w0.6` and `fuse-k320-rgb-w0.6` use the same
settings on both splits. Both scored above 70% on all four metrics in the
public-vocabulary study. These are exploratory results selected on the official
unseen splits.

## Conference-method transfers

The gallery-only benchmark includes shared-neighbor and spectral ranking (`context`),
intermediate visual blocks (`layers`), and paired contour–RGB anchors (`asif`). The
public-vocabulary benchmark includes gallery prototype voting (`paired-concepts` and
`cluster-mine`), score normalization (`hubness` and `querybank`), test-time view
aggregation (`mta`), and conditional transport (`transport` and `transport-views`).
The `dense-transport` study uses ClearCLIP's patch features on the same ViT-B/32 checkpoint.
The [literature review](docs/TRAINING_FREE_LITERATURE_VI.md#các-cơ-chế-từ-hội-nghị-được-kiểm-tra)
links the source papers and describes the adaptations.

```powershell
foreach ($study in @('context', 'layers', 'asif')) {
    python -m tf_sbir.manifold_benchmark --root <Sketchy> --output "runs/sota-$study-b32" --study $study --exploratory-unseen
}
foreach ($study in @('paired-concepts', 'hubness', 'querybank', 'cluster-mine', 'mta', 'transport', 'transport-views', 'dense-transport')) {
    python -m tf_sbir.semantic_benchmark --root <Sketchy> --vocabulary open-images-v7-classes.csv --output "runs/sota-$study-b32" --study $study --exploratory-unseen
}
```

Both augmentation studies cache the added views. On CUDA, use `--batch-size 224
--workers 8` for `mta` or `--batch-size 252 --workers 8` for `transport-views` if
memory permits. Batch size counts the added images sent to CLIP for these studies.
