# Reproduce the T4 comparison on Colab

The full-gallery scoring code is pinned at `e896de55640222a72b44cb893568daa60a205fde`. The original pilot, confirmation, and full run each record their actual scoring commit in their output JSON. The procedure below replays all stages from the full-run commit. Use the project's prepared Colab workspace `/content/sbir`, with Sketchy and TU-Berlin paths in `ready.json` and `tuberlin-data.json`, the OpenAI ViT-B/32 checkpoint, the Open Images csv, and the original KD-SBIR reference `inquiry_kd_reference.py`. The metric function's source hash is checked before scoring. Existing frozen feature caches can be reused.

```bash
git clone --sparse --filter=blob:none --depth=1 -b research/regional-scale-position-contrast https://github.com/HieuNM1804/training-free-ZS-SBIR.git /content/region-replay
cd /content/region-replay
git fetch --depth=1 origin e896de55640222a72b44cb893568daa60a205fde
git checkout --detach FETCH_HEAD
git sparse-checkout set tf_sbir docs/results/geometry-word-flat
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
python - <<'PY'
import hashlib, zipfile
from pathlib import Path
p = Path('docs/results/geometry-word-flat/seed42-L4.zip')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '7e62624e62b7b9955deaa1f2f927c2c350be1cc983b2576adef3ebc018d82d4c'
folder = Path('/content/sbir/results/geometry-word-flat-first')
folder.mkdir(exist_ok=True, parents=True)
with zipfile.ZipFile(p) as archive:
    for split in ('sketchy_1', 'sketchy_2', 'tuberlin'):
        dest = folder / (split + '.pt')
        blob = archive.read('first/' + dest.name)
        if dest.exists():
            assert dest.read_bytes() == blob
        else:
            dest.write_bytes(blob)
PY
python -m tf_sbir.regional_position_device_reference
python -m tf_sbir.regional_scale_position_contrast
python -m tf_sbir.regional_position_device_reference --confirmation
python -m tf_sbir.regional_scale_position_contrast --confirmation
python -m tf_sbir.geometry_word_flat tuberlin --additive
python -m tf_sbir.geometry_word_flat sketchy --additive
```

The gallery-fold RNG for GCWE is a CPU generator seeded with 42 per split. Pilot and confirmation use disjoint sections of another seed-42 permutation. Both score every gallery photo. The full run uses all queries, writing atomic 256-query shards. Re-running the same command after interruption reuses its shards. Use a fresh results directory when measuring independent rerun variability. For the published analysis of untouched Sketchy queries, use `python -m tf_sbir.regional_additive_effects` at commit `f0a72b9501ef7855c71b8e182a4ce470b5867641` after all three split results have been generated.
