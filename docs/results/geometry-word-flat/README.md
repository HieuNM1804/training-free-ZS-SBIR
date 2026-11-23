# Ablation without the gallery hierarchy

Use the Colab workspace with the pinned OpenAI ViT-B/32 and Open Images dictionary from the preceding study. The full score combines RGB diffusion, candidate-specific regional word evidence and geometry-conditioned word feedback. The three masks remove one of these terms at a time. The previous hierarchical study's stored rankings are used only for an evaluation-time cross-seed comparison.

Run each dataset in a fresh Python process. Set the environment before starting Python.

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m tf_sbir.geometry_word_flat sketchy
python -m tf_sbir.geometry_word_flat tuberlin
```

Seed 42 initializes Python, NumPy, PyTorch CPU and CUDA. The two-fold partition uses a private CPU generator initialized with 42 for each split and takes a permutation of the ordered gallery. Inference uses fixed views, deterministic PyTorch algorithms and disabled cuDNN benchmarking. Query and gallery indices are saved alongside the per-query metrics. To replay scoring, archive the completed results directory before launching again so no scoring shards are reused. Both runs can reuse the same frozen feature caches. Compare each method's tensors and both ID lists with `torch.equal`.
