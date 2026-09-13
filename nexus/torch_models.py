from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


MODEL_HIDDEN_SIZES = {
    "time_predict": (16, 8),
    "sustantive_predict": (64, 32),
}


class BinaryMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU()))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def build_model(input_size: int, model_key: str) -> BinaryMLP:
    try:
        hidden_sizes = MODEL_HIDDEN_SIZES[model_key]
    except KeyError as error:
        raise ValueError(f"Modelo desconocido: {model_key}") from error
    return BinaryMLP(input_size, hidden_sizes)


def fit_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def apply_normalizer(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) - mean) / scale


def make_checkpoint(
    model: BinaryMLP,
    model_key: str,
    feature_cols: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "format": "nexus-torch-v1",
        "model_key": model_key,
        "feature_cols": feature_cols,
        "model_config": {
            "input_size": len(feature_cols),
            "hidden_sizes": list(MODEL_HIDDEN_SIZES[model_key]),
        },
        "normalizer": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
        "epoch": epoch,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
    }


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def save_normalization(
    path: str | Path,
    model_key: str,
    feature_cols: list[str],
    mean: np.ndarray | list[float],
    scale: np.ndarray | list[float],
    source_format: str,
    method: str = "standard_score",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "nexus-normalizer-v1",
        "method": method,
        "fit_split": "train",
        "model_key": model_key,
        "source_format": source_format,
        "feature_cols": feature_cols,
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "scale": np.asarray(scale, dtype=np.float32).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "nexus-torch-v1":
        raise ValueError(f"Checkpoint incompatible o invalido: {path}")

    feature_cols = checkpoint.get("feature_cols")
    model_key = checkpoint.get("model_key")
    if not isinstance(feature_cols, list) or not isinstance(model_key, str):
        raise ValueError(f"Checkpoint sin metadatos de entrada: {path}")
    model = build_model(len(feature_cols), model_key)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    normalizer = checkpoint.get("normalizer", {})
    checkpoint = dict(checkpoint)
    checkpoint["model"] = model
    checkpoint["normalizer"] = {
        "mean": np.asarray(normalizer["mean"], dtype=np.float32),
        "scale": np.asarray(normalizer["scale"], dtype=np.float32),
    }
    return checkpoint