"""Cache the original 13 windows in both poses once per sketch across the splits."""
import argparse
from pathlib import Path

import torch

from .gallery_panel_resume import region_views, semantic_inputs


@torch.inference_mode()
def source_windows(workspace, dataset, inputs):
    cfg, paths, _, _, _, features, *_ = inputs
    rows = torch.arange(len(paths))
    packed, _ = region_views(workspace, 'loo-' + dataset, cfg, paths, rows, features)
    return packed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('sketchy', 'tuberlin'))
    dataset = parser.parse_args().dataset
    workspace = Path('/content/sbir')
    source_windows(workspace, dataset, semantic_inputs(workspace, dataset))
