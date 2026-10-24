"""Portable NumPy checkpoints including pruning masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aqua_nn.models import MLP


_LAYERS = ("layer1", "layer2", "layer3")


def save_model_checkpoint(model: MLP, path: str | Path, metadata: dict | None = None) -> Path:
    """Save weights, biases, masks, and JSON metadata in a compressed NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for index, layer_name in enumerate(_LAYERS, start=1):
        layer = getattr(model, layer_name)
        arrays[f"layer{index}_weight"] = layer.weight.data
        arrays[f"layer{index}_bias"] = layer.bias.data
        arrays[f"layer{index}_mask"] = layer.mask
    arrays["metadata_json"] = np.asarray(json.dumps(metadata or {}, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return path


def load_model_checkpoint(path: str | Path) -> tuple[MLP, dict]:
    """Load a checkpoint and reconstruct an MLP with the saved layer shapes."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = [
            "layer1_weight", "layer1_bias",
            "layer2_weight", "layer2_bias",
            "layer3_weight", "layer3_bias",
        ]
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"Checkpoint is missing arrays: {', '.join(missing)}")

        w1 = np.asarray(archive["layer1_weight"], dtype=np.float64)
        b1 = np.asarray(archive["layer1_bias"], dtype=np.float64)
        w2 = np.asarray(archive["layer2_weight"], dtype=np.float64)
        b2 = np.asarray(archive["layer2_bias"], dtype=np.float64)
        w3 = np.asarray(archive["layer3_weight"], dtype=np.float64)
        b3 = np.asarray(archive["layer3_bias"], dtype=np.float64)
        metadata = (
            json.loads(str(archive["metadata_json"].item()))
            if "metadata_json" in archive.files
            else {}
        )
        masks = [
            np.asarray(archive[f"layer{i}_mask"], dtype=np.float64)
            if f"layer{i}_mask" in archive.files
            else np.ones_like(np.asarray(archive[f"layer{i}_weight"]))
            for i in range(1, 4)
        ]

    if w1.ndim != 2 or w2.ndim != 2 or w3.ndim != 2:
        raise ValueError("Checkpoint weight arrays must be matrices")
    if w1.shape[1] != w2.shape[0] or w2.shape[1] != w3.shape[0]:
        raise ValueError("Checkpoint layer dimensions are incompatible")
    if b1.shape != (w1.shape[1],) or b2.shape != (w2.shape[1],) or b3.shape != (w3.shape[1],):
        raise ValueError("Checkpoint bias shapes do not match weight matrices")

    model = MLP(
        input_size=w1.shape[0],
        hidden1=w1.shape[1],
        hidden2=w2.shape[1],
        num_classes=w3.shape[1],
        seed=0,
    )
    for index, (layer_name, weight, bias, mask) in enumerate(
        zip(_LAYERS, (w1, w2, w3), (b1, b2, b3), masks), start=1
    ):
        layer = getattr(model, layer_name)
        if mask.shape != weight.shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"Invalid binary mask for layer {index}")
        layer.weight.data[...] = weight
        layer.bias.data[...] = bias
        layer.set_mask(mask)

    return model, metadata
