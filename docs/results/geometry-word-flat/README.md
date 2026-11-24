# Seed-42 three-module ZS-SBIR baseline

The three modules are Cross-Domain Semantic Retrieval, Candidate-Conditioned Regional Matching and Geometry-Conditioned Word Evidence (GCWE). On a prepared Colab workspace, the commands below evaluate the full model, remove each of its three score terms and evaluate single channels. The computation uses the original KD-SBIR evaluator, frozen OpenAI CLIP ViT-B/32 and the public Open Images word bank. Historical rankings are optional and are consulted only after scoring to compare matching rows. [Full term ablation](../geometry-word-flat-terms/README.md) · [Research evidence](../geometry-word-flat-terms/research-status-vi.md)

Run each dataset in a fresh Python process. Set the environment before starting Python.

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m tf_sbir.geometry_word_flat sketchy --term-ablation
python -m tf_sbir.geometry_word_flat tuberlin --term-ablation
```

Seed 42 initializes Python, NumPy, PyTorch CPU and CUDA. The two-fold partition uses a private CPU generator initialized with 42 for each split and takes a permutation of the ordered gallery. Inference uses fixed views, deterministic PyTorch algorithms and disabled cuDNN benchmarking. Query and gallery indices are saved alongside the per-query metrics. To replay scoring, archive the completed results directory before launching again so no scoring shards are reused. Both runs can reuse the same frozen feature caches. Compare each method's tensors and both ID lists with `torch.equal`.

The default workspace is `/content/sbir`. To use a fresh results directory, pass `--workspace /content/your-sbir-workspace`. The runtime expects the corresponding benchmark data JSON (`ready.json` for Sketchy and `tuberlin-data.json` for TU), assets and the pinned KD reference module, plus a writable feature cache. The scorer builds the required regional and patch caches. New partial shards are keyed by method list, seed, source tree, ordered input paths, checkpoint, evaluator and runtime. Legacy completed seed-42 results are recognized by scoring revision, method names and saved IDs. The two original independent scoring runs agreed to under 0.00002 aggregate percentage points; a few per-query values differed. [Measured reproducibility](baseline-vi.md)

Omitting `--term-ablation` reproduces the original four-row study, whose third intervention turns diffusion off within both word paths. Passing `--diffusion-paths` instead runs the complete two-by-two ablation of global and regional diffusion. [Diffusion-path results](../geometry-word-flat-diffusion-paths/README.md)
