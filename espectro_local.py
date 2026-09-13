"""Normaliza audio y ejecuta inferencia con el Mini-AST entrenado.

Ejemplos desde la raiz del proyecto:

    python espectro_local.py normalizar datos/channel_0/human/audio.wav
    python espectro_local.py evaluar datos/channel_0 --limite 20

La transformacion es la misma que se uso durante el entrenamiento en
``procesamiento_audio_mel_modelos.py``. Cada audio se normaliza de forma
independiente y termina como una matriz ``(64, 128)`` en el rango ``[0, 1]``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from procesamiento_audio_mel_modelos import (
    CLASS_NAMES,
    FIXED_FRAMES,
    N_MELS,
    RESULTS_DIR,
    MiniAST,
    extract_mel_feature,
    save_feature_image,
)


AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
DEFAULT_MODEL_PATH = RESULTS_DIR / "mini_ast_best.pt"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "inferencia"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "predicciones_mini_ast.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_one_audio(audio_path: Path) -> np.ndarray:
    """Obtiene la unica representacion normalizada que acepta el modelo."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"No existe el audio: {audio_path}")
    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Formato no soportado: {audio_path.suffix}")

    feature = extract_mel_feature(audio_path)
    expected_shape = (N_MELS, FIXED_FRAMES)
    if feature.shape != expected_shape:
        raise ValueError(
            f"La entrada normalizada tiene forma {feature.shape}; "
            f"se esperaba {expected_shape}."
        )
    if feature.dtype != np.float32:
        feature = feature.astype(np.float32)
    return np.clip(feature, 0.0, 1.0)


def save_normalized_feature(audio_path: Path, feature: np.ndarray, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{audio_path.stem}_mel.npy"
    image_path = output_dir / f"{audio_path.stem}_mel.png"
    np.save(feature_path, feature)
    save_feature_image(feature, image_path)
    return feature_path, image_path


def load_model(model_path: Path) -> torch.nn.Module:
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {model_path}")
    model = MiniAST(n_classes=len(CLASS_NAMES))
    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()
    return model


def predict_features(model: torch.nn.Module, features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not features:
        return np.empty(0, dtype=np.int64), np.empty((0, len(CLASS_NAMES)), dtype=np.float32)
    batch = np.stack(features).astype(np.float32)
    device = next(model.parameters()).device
    input_tensor = torch.from_numpy(batch[:, None, :, :]).to(device)
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = logits.argmax(dim=1).cpu().numpy()
    return predictions, probabilities


def find_audio_files(inputs: list[Path]) -> list[Path]:
    audio_paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            audio_paths.append(input_path)
            continue
        if input_path.is_dir():
            audio_paths.extend(
                path
                for path in sorted(input_path.rglob("*"))
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
            )
            continue
        raise FileNotFoundError(f"No existe la entrada: {input_path}")
    return sorted(set(audio_paths))


def label_from_path(audio_path: Path) -> str:
    parent_name = audio_path.parent.name.lower()
    return parent_name if parent_name in CLASS_NAMES else ""


def calculate_labeled_metrics(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    labeled_rows = [row for row in rows if row["label_real"] in CLASS_NAMES]
    if not labeled_rows:
        return None
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    y_true = np.asarray([class_to_index[row["label_real"]] for row in labeled_rows])
    y_pred = np.asarray([class_to_index[row["prediccion"]] for row in labeled_rows])
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for expected, predicted in zip(y_true, y_pred):
        matrix[expected, predicted] += 1
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_index in range(len(CLASS_NAMES)):
        true_positive = matrix[class_index, class_index]
        false_positive = matrix[:, class_index].sum() - true_positive
        false_negative = matrix[class_index, :].sum() - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))
    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1_scores)),
    }


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audio",
        "label_real",
        "prediccion",
        "prob_human",
        "prob_synthetic",
        "feature_shape",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def command_normalize(audio_path: Path, output_dir: Path) -> None:
    feature = normalize_one_audio(audio_path)
    feature_path, image_path = save_normalized_feature(audio_path, feature, output_dir)
    print(f"Audio: {audio_path}")
    print(f"Entrada normalizada: {feature.shape}, {feature.dtype}, rango [{feature.min():.3f}, {feature.max():.3f}]")
    print(f"Numpy guardado: {feature_path}")
    print(f"Imagen guardada: {image_path}")


def command_evaluate(
    inputs: list[Path],
    model_path: Path,
    output_path: Path,
    output_dir: Path,
    limit: int | None,
    save_features: bool,
) -> None:
    audio_paths = find_audio_files(inputs)
    if limit is not None:
        audio_paths = audio_paths[:limit]
    if not audio_paths:
        raise ValueError("No se encontraron audios para evaluar.")

    model = load_model(model_path)
    features: list[np.ndarray] = []
    valid_paths: list[Path] = []
    for audio_path in audio_paths:
        try:
            feature = normalize_one_audio(audio_path)
            features.append(feature)
            valid_paths.append(audio_path)
            if save_features:
                save_normalized_feature(audio_path, feature, output_dir)
        except Exception as exc:
            print(f"Se omite {audio_path}: {exc}")

    predictions, probabilities = predict_features(model, features)
    rows: list[dict[str, Any]] = []
    for audio_path, prediction, probability in zip(valid_paths, predictions, probabilities):
        rows.append(
            {
                "audio": str(audio_path),
                "label_real": label_from_path(audio_path),
                "prediccion": CLASS_NAMES[int(prediction)],
                "prob_human": f"{probability[0]:.6f}",
                "prob_synthetic": f"{probability[1]:.6f}",
                "feature_shape": f"{N_MELS}x{FIXED_FRAMES}",
            }
        )
    write_results(rows, output_path)
    print(f"Audios evaluados: {len(rows)}")
    print(f"Predicciones guardadas: {output_path}")
    metrics = calculate_labeled_metrics(rows)
    if metrics is not None:
        print(
            "Metricas con etiquetas de la carpeta: "
            f"accuracy={metrics['accuracy']:.3f}, "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normaliza audio y evalua el Mini-AST de resultados_audio.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser(
        "normalizar",
        help="Normaliza un solo audio y guarda la matriz que recibe el modelo.",
    )
    normalize_parser.add_argument("audio", type=Path, help="Archivo WAV/FLAC/OGG/MP3/M4A.")
    normalize_parser.add_argument(
        "--salida",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta de salida (predeterminada: {DEFAULT_OUTPUT_DIR}).",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluar",
        help="Evalua uno o varios audios, o todos los audios de una carpeta.",
    )
    evaluate_parser.add_argument(
        "entradas",
        type=Path,
        nargs="+",
        help="Archivos y/o carpetas que contienen audios.",
    )
    evaluate_parser.add_argument(
        "--modelo",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Checkpoint Mini-AST (predeterminado: {DEFAULT_MODEL_PATH}).",
    )
    evaluate_parser.add_argument(
        "--salida",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"CSV de resultados (predeterminado: {DEFAULT_RESULTS_PATH}).",
    )
    evaluate_parser.add_argument(
        "--guardar-features",
        action="store_true",
        help="Guarda tambien cada entrada normalizada como .npy y .png.",
    )
    evaluate_parser.add_argument(
        "--carpeta-features",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta para .npy/.png (predeterminada: {DEFAULT_OUTPUT_DIR}).",
    )
    evaluate_parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Evalua como maximo esta cantidad de audios.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "normalizar":
        command_normalize(args.audio, args.salida)
    else:
        if args.limite is not None and args.limite < 1:
            raise ValueError("--limite debe ser mayor que cero.")
        command_evaluate(
            args.entradas,
            args.modelo,
            args.salida,
            args.carpeta_features,
            args.limite,
            args.guardar_features,
        )


if __name__ == "__main__":
    main()