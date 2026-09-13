"""Flujo educativo de audio: tiempo, frecuencia, Mel y modelos.

Ejecutar desde la raiz del proyecto:
    python procesamiento_audio_mel_modelos.py

El script busca un WAV en datos/channel_0. Si no encuentra audio, genera una
senal sintetica para que las visualizaciones sigan siendo reproducibles.
Las figuras y, cuando PyTorch esta disponible, los pesos se guardan en
resultados_audio/.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import random
import shutil
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

try:
    torch_summary = getattr(importlib.import_module("torchinfo"), "summary")
    TORCHINFO_AVAILABLE = True
except Exception:
    torch_summary = None
    TORCHINFO_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except Exception:
    torchaudio = None
    TORCHAUDIO_AVAILABLE = False

try:
    from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    ASTConfig = None
    ASTFeatureExtractor = None
    ASTForAudioClassification = None
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    accuracy_score = None
    confusion_matrix = None
    precision_recall_fscore_support = None
    SKLEARN_AVAILABLE = False

SEED = 42
TARGET_SR = 16_000
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 256
N_MELS = 64
FMIN = 50.0
FMAX = 8_000.0
CLIP_SECONDS = 4.0
FIXED_FRAMES = 128
CLASS_NAMES = ["human", "synthetic"]

random.seed(SEED)
np.random.seed(SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(SEED)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "image.cmap": "magma",
    }
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "resultados_audio"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DATASET_DIR = RESULTS_DIR / "dataset_mel_imagenes"


def find_audio_file(root: Path) -> Path | None:
    search_groups = [
        root / "datos" / "channel_0",
        root / "altur-challenge-audio" / "audio",
        root / "audios",
    ]
    extensions = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    for folder in search_groups:
        if not folder.exists():
            continue
        candidates = sorted(
            path for path in folder.glob("**/*") if path.is_file() and path.suffix.lower() in extensions
        )
        if candidates:
            return candidates[0]
    return None


def synthetic_audio(sample_rate: int = 8_000, seconds: float = 5.0) -> tuple[np.ndarray, int]:
    time_axis = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 1.2 * time_axis) ** 2
    carrier = 0.35 * np.sin(2 * np.pi * 230 * time_axis)
    harmonic = 0.18 * np.sin(2 * np.pi * 460 * time_axis)
    chirp = 0.10 * signal.chirp(time_axis, f0=700, f1=2_000, t1=seconds, method="linear")
    noise = 0.025 * np.random.default_rng(SEED).normal(size=time_axis.size)
    return (envelope * (carrier + harmonic) + chirp + noise).astype(np.float32), sample_rate


def convert_to_float(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = max(abs(info.min), info.max)
        return array.astype(np.float32) / scale
    return array.astype(np.float32)


def load_audio(path: Path | None, max_seconds: float = 20.0) -> tuple[np.ndarray, int, str]:
    if path is None:
        y, sample_rate = synthetic_audio()
        return y, sample_rate, "senal sintetica"

    errors: list[str] = []
    if sf is not None:
        try:
            data, sample_rate = sf.read(str(path), always_2d=False)
            array = convert_to_float(data)
            if array.ndim == 2:
                array = array[:, 0]
            array = array[: int(sample_rate * max_seconds)]
            return array, int(sample_rate), str(path)
        except Exception as exc:
            errors.append(f"soundfile: {exc}")

    try:
        sample_rate, data = wavfile.read(path)
        array = convert_to_float(data)
        if array.ndim == 2:
            array = array[:, 0]
        array = array[: int(sample_rate * max_seconds)]
        return array, int(sample_rate), str(path)
    except Exception as exc:
        errors.append(f"scipy.io.wavfile: {exc}")

    warnings.warn("No se pudo leer el archivo; se usara una senal sintetica. " + " | ".join(errors))
    y, sample_rate = synthetic_audio()
    return y, sample_rate, "senal sintetica"


def normalize_peak(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y)))
    return y if peak == 0 else y / peak


def normalize_rms(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(y))))
    return y if rms == 0 else y * (target_rms / rms)


def resample_audio(y: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return y.astype(np.float32, copy=True)
    divisor = math.gcd(int(original_sr), int(target_sr))
    up = target_sr // divisor
    down = original_sr // divisor
    return signal.resample_poly(y, up, down).astype(np.float32)


def frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    frame_count = 1 + int(math.ceil((y.size - frame_length) / hop_length))
    padded_length = frame_length + (frame_count - 1) * hop_length
    padded = np.pad(y, (0, max(0, padded_length - y.size)))
    return np.stack(
        [padded[start : start + frame_length] for start in range(0, padded_length - frame_length + 1, hop_length)]
    )


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2_595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2_595.0) - 1.0)


def make_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    fmax = min(float(fmax), sample_rate / 2.0)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_numbers = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for mel_index in range(1, n_mels + 1):
        left, center, right = bin_numbers[mel_index - 1 : mel_index + 2]
        left = max(0, min(left, n_fft // 2))
        center = max(left + 1, min(center, n_fft // 2))
        right = max(center + 1, min(right, n_fft // 2 + 1))
        filterbank[mel_index - 1, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        filterbank[mel_index - 1, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    return filterbank


def stft_power(
    y: np.ndarray,
    sample_rate: int,
    n_fft: int = N_FFT,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, spectrum = signal.stft(
        y,
        fs=sample_rate,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(spectrum)
    power = magnitude**2
    return frequencies, times, magnitude, power


def mel_spectrogram_from_power(
    power: np.ndarray,
    sample_rate: int,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> tuple[np.ndarray, np.ndarray]:
    filterbank = make_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel_power = filterbank @ power
    mel_db = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    mel_db -= float(np.max(mel_db))
    return mel_power, mel_db


def crop_or_pad(signal_data: np.ndarray, target_length: int) -> np.ndarray:
    if signal_data.size >= target_length:
        return signal_data[:target_length]
    return np.pad(signal_data, (0, target_length - signal_data.size))


def resize_time(matrix: np.ndarray, target_frames: int) -> np.ndarray:
    if matrix.shape[1] >= target_frames:
        return matrix[:, :target_frames]
    pad_value = float(np.min(matrix))
    return np.pad(matrix, ((0, 0), (0, target_frames - matrix.shape[1])), constant_values=pad_value)


def extract_mel_feature(path: Path, duration: float = CLIP_SECONDS) -> np.ndarray:
    y, sample_rate, _ = load_audio(path, max_seconds=duration)
    y = normalize_peak(y)
    y = resample_audio(y, sample_rate, TARGET_SR)
    y = crop_or_pad(y, int(TARGET_SR * duration))
    frequencies, _, _, power = stft_power(y, TARGET_SR)
    _, mel_db = mel_spectrogram_from_power(power, TARGET_SR, n_fft=N_FFT, n_mels=N_MELS)
    feature = resize_time(mel_db, FIXED_FRAMES)
    return np.clip((feature + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)


def save_feature_image(feature: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, feature, cmap="gray", vmin=0.0, vmax=1.0)
    return output_path


def load_feature_image(image_path: Path) -> np.ndarray:
    image = plt.imread(image_path)
    if image.ndim == 3:
        image = image[..., 0]
    image = image.astype(np.float32)
    if image.max(initial=0.0) > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def save_figure(figure: plt.Figure, filename: str) -> Path:
    output_path = RESULTS_DIR / filename
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Figura guardada: {output_path}")
    return output_path


def collect_balanced_records(
    root: Path,
    train_per_class: int = 120,
    seed: int = SEED,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    train_records: list[tuple[Path, int]] = []
    remaining_records: list[tuple[Path, int]] = []
    rng = np.random.default_rng(seed)
    base = root / "datos" / "channel_0"
    for label, class_name in enumerate(CLASS_NAMES):
        class_folder = base / class_name
        paths = sorted(class_folder.glob("*.wav"))
        if len(paths) < train_per_class:
            raise ValueError(
                f"La clase {class_name} solo tiene {len(paths)} audios; "
                f"se necesitan {train_per_class}."
            )
        rng.shuffle(paths)
        train_records.extend((path, label) for path in paths[:train_per_class])
        remaining_records.extend((path, label) for path in paths[train_per_class:])
    rng.shuffle(train_records)
    rng.shuffle(remaining_records)
    return train_records, remaining_records


def prepare_dataset(
    root: Path,
    example_feature: np.ndarray,
    train_per_class: int,
) -> dict[str, np.ndarray]:
    train_records, remaining_records = collect_balanced_records(
        root,
        train_per_class=train_per_class,
    )
    shutil.rmtree(IMAGE_DATASET_DIR, ignore_errors=True)
    train_features: list[np.ndarray] = []
    train_labels: list[int] = []
    remaining_features: list[np.ndarray] = []
    remaining_labels: list[int] = []
    manifest_rows: list[dict[str, str]] = []

    def process_records(records: list[tuple[Path, int]], split_name: str) -> None:
        feature_list = train_features if split_name == "train" else remaining_features
        label_list = train_labels if split_name == "train" else remaining_labels
        for path, label in records:
            try:
                feature = extract_mel_feature(path)
                image_path = IMAGE_DATASET_DIR / CLASS_NAMES[label] / f"{path.stem}.png"
                save_feature_image(feature, image_path)
                feature_list.append(load_feature_image(image_path))
                label_list.append(label)
                manifest_rows.append(
                    {
                        "split": split_name,
                        "label": CLASS_NAMES[label],
                        "audio": str(path.relative_to(root)),
                        "image": str(image_path.relative_to(SCRIPT_DIR)),
                    }
                )
            except Exception as exc:
                print(f"Se omite {path.name}: {exc}")

    process_records(train_records, "train")
    process_records(remaining_records, "remaining")

    expected_train_count = train_per_class * len(CLASS_NAMES)
    if len(train_features) != expected_train_count:
        raise RuntimeError(
            f"El entrenamiento no quedo balanceado: se obtuvieron {len(train_features)} "
            f"imagenes, se esperaban {expected_train_count}."
        )
    if not train_features:
        for label in range(len(CLASS_NAMES)):
            for index in range(train_per_class):
                image_path = IMAGE_DATASET_DIR / CLASS_NAMES[label] / f"demo_{label}_{index:03d}.png"
                save_feature_image(example_feature, image_path)
                train_features.append(load_feature_image(image_path))
                train_labels.append(label)
        print("No se encontro un dataset etiquetado; se usa un conjunto sintetico de demostracion.")

    manifest_path = RESULTS_DIR / "dataset_split.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=["split", "label", "audio", "image"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Split guardado: {manifest_path}")

    train_x = np.stack(train_features).astype(np.float32)
    train_y = np.asarray(train_labels, dtype=np.int64)
    remaining_x = (
        np.stack(remaining_features).astype(np.float32)
        if remaining_features
        else np.empty((0, N_MELS, FIXED_FRAMES), dtype=np.float32)
    )
    remaining_y = np.asarray(remaining_labels, dtype=np.int64)
    return {
        "train": train_x,
        "y_train": train_y,
        "remaining": remaining_x,
        "y_remaining": remaining_y,
    }


def show_original_audio(y: np.ndarray, sample_rate: int, source: str) -> None:
    time_axis = np.arange(y.size) / sample_rate
    duration = y.size / sample_rate
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)))
    print(f"Fuente: {source}")
    print(f"Frecuencia original: {sample_rate:,} Hz")
    print(f"Duracion analizada: {duration:.2f} s | muestras: {y.size:,}")
    print(f"Amplitud maxima: {peak:.4f} | RMS: {rms:.4f}")
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(time_axis, y, color="#176b87", linewidth=0.7)
    axis.set_title("Señal original en el dominio del tiempo")
    axis.set_xlabel("Tiempo (s)")
    axis.set_ylabel("Amplitud")
    axis.set_xlim(0, duration)
    save_figure(figure, "01_onda_original.png")


def show_resampling(y: np.ndarray, sample_rate: int, normalized: np.ndarray, resampled: np.ndarray) -> None:
    original_time = np.arange(normalized.size) / sample_rate
    target_time = np.arange(resampled.size) / TARGET_SR
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    axes[0].plot(original_time, normalized, color="#c44900", linewidth=0.7)
    axes[0].set_title(f"Normalizada, antes del remuestreo ({sample_rate:,} Hz)")
    axes[0].set_ylabel("Amplitud")
    axes[1].plot(target_time, resampled, color="#287271", linewidth=0.7)
    axes[1].set_title(f"Normalizada y remuestreada ({TARGET_SR:,} Hz)")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Amplitud")
    print(f"Muestras antes: {normalized.size:,} -> despues: {resampled.size:,}")
    save_figure(figure, "02_normalizacion_resampling.png")


def show_framing(resampled: np.ndarray) -> None:
    frame_ms = 32.0
    hop_ms = 16.0
    frame_length = int(round(TARGET_SR * frame_ms / 1_000.0))
    hop_length = int(round(TARGET_SR * hop_ms / 1_000.0))
    frames = frame_signal(resampled, frame_length, hop_length)
    hann_window = np.hanning(frame_length)
    hamming_window = np.hamming(frame_length)
    first_frame = frames[0]
    time_axis = np.arange(frame_length) / TARGET_SR * 1_000.0
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(time_axis, hann_window, label="Hann", color="#e76f51")
    axes[0].plot(time_axis, hamming_window, label="Hamming", color="#264653")
    axes[0].set_title("Funciones de ventana")
    axes[0].set_xlabel("Tiempo (ms)")
    axes[0].legend()
    axes[1].plot(time_axis, first_frame, color="#777777", label="trama sin ventana")
    axes[1].plot(time_axis, first_frame * hann_window, color="#e76f51", label="Hann")
    axes[1].plot(time_axis, first_frame * hamming_window, color="#264653", label="Hamming")
    axes[1].set_title("Una trama de 32 ms")
    axes[1].set_xlabel("Tiempo (ms)")
    axes[1].legend(fontsize=8)
    visible_frames = frames[: min(8, len(frames))] * hann_window
    axes[2].imshow(
        visible_frames.T,
        origin="lower",
        aspect="auto",
        extent=[0, visible_frames.shape[0] - 1, 0, frame_ms],
        cmap="viridis",
    )
    axes[2].set_title("Tramas consecutivas con Hann")
    axes[2].set_xlabel("Indice de trama")
    axes[2].set_ylabel("Posicion dentro de la trama (ms)")
    print(f"Framing: {frame_length} muestras ({frame_ms:.0f} ms), salto {hop_length} muestras ({hop_ms:.0f} ms)")
    save_figure(figure, "03_framing_windowing.png")


def show_stft(resampled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, magnitude, power = stft_power(resampled, TARGET_SR)
    stft_db = 10.0 * np.log10(np.maximum(power, 1e-10))
    stft_db -= float(np.max(stft_db))
    figure, axis = plt.subplots(figsize=(12, 5))
    image = axis.pcolormesh(times, frequencies, stft_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
    axis.set_title("STFT: espectrograma lineal en escala de frecuencia")
    axis.set_xlabel("Tiempo (s)")
    axis.set_ylabel("Frecuencia (Hz)")
    axis.set_ylim(0, TARGET_SR / 2)
    figure.colorbar(image, ax=axis, label="Potencia relativa (dB)")
    print(f"STFT: magnitud {magnitude.shape}, potencia {power.shape}, eje de frecuencia {frequencies.size}")
    save_figure(figure, "04_stft_espectrograma.png")
    return frequencies, times, magnitude, power


def show_mel_spectrogram(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mel_power, mel_db = mel_spectrogram_from_power(power, TARGET_SR)
    mel_frequencies = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS))
    time_axis = np.arange(mel_db.shape[1]) * HOP_LENGTH / TARGET_SR
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    first = axes[0].pcolormesh(time_axis, mel_frequencies, mel_power, shading="auto", cmap="viridis")
    axes[0].set_title("Mel-espectrograma de potencia")
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel("Frecuencia Mel aproximada (Hz)")
    figure.colorbar(first, ax=axes[0], label="Potencia")
    second = axes[1].pcolormesh(time_axis, mel_frequencies, mel_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
    axes[1].set_title(r"Escala logaritmica: $10\log_{10}(P)$")
    axes[1].set_xlabel("Tiempo (s)")
    figure.colorbar(second, ax=axes[1], label="dB relativos")
    print(f"Mel: potencia {mel_power.shape}; rango dB [{mel_db.min():.1f}, {mel_db.max():.1f}]")
    save_figure(figure, "05_mel_espectrograma_db.png")
    return mel_power, mel_db


def draw_architecture_diagram() -> None:
    pipelines = {
        "CNN 2D": ["Mel T x F", "Conv2d", "Pool", "Conv2d", "GAP", "Logits"],
        "AST": ["Mel T x F", "Patches", "Embeddings", "Attention", "Pooling", "Logits"],
        "CRNN": ["Mel T x F", "Conv2d", "Pool F", "Secuencia", "GRU/LSTM", "Logits"],
    }
    colors = ["#e9c46a", "#f4a261", "#2a9d8f", "#457b9d", "#8ab17d", "#e76f51"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    for axis, (title, stages) in zip(axes, pipelines.items()):
        axis.set_title(title)
        axis.axis("off")
        for index, stage in enumerate(stages):
            y_position = 0.78 - index * 0.13
            axis.text(
                0.5,
                y_position,
                stage,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.45", "facecolor": colors[index], "edgecolor": "#333333"},
            )
            if index < len(stages) - 1:
                axis.annotate(
                    "",
                    xy=(0.5, y_position - 0.055),
                    xytext=(0.5, y_position - 0.095),
                    arrowprops={"arrowstyle": "->", "color": "#555555"},
                )
    figure.suptitle("Tres formas de integrar un Mel-espectrograma en un modelo")
    save_figure(figure, "06_arquitecturas_modelos.png")


if TORCH_AVAILABLE:

    class MelCNN(nn.Module):
        def __init__(self, n_classes: int = 2) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(64, n_classes)

        def forward(self, x: Any) -> Any:
            features = self.features(x)
            return self.classifier(features.flatten(1))


    class MiniAST(nn.Module):
        def __init__(self, n_classes: int = 2, embed_dim: int = 64, patch_size: int = 8) -> None:
            super().__init__()
            self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=128,
                batch_first=True,
                dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.classifier = nn.Linear(embed_dim, n_classes)

        def forward(self, x: Any) -> Any:
            patches = self.patch_embed(x).flatten(2).transpose(1, 2)
            encoded = self.encoder(patches)
            return self.classifier(encoded.mean(dim=1))


    class MelCRNN(nn.Module):
        def __init__(self, n_classes: int = 2, recurrent: str = "GRU") -> None:
            super().__init__()
            self.convolution = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
            )
            recurrent_class = nn.LSTM if recurrent.upper() == "LSTM" else nn.GRU
            self.recurrent = recurrent_class(input_size=32, hidden_size=64, batch_first=True)
            self.classifier = nn.Linear(64, n_classes)

        def forward(self, x: Any) -> Any:
            features = self.convolution(x).mean(dim=2).transpose(1, 2)
            sequence, _ = self.recurrent(features)
            return self.classifier(sequence[:, -1])


def demonstrate_models(example_feature: np.ndarray) -> dict[str, object]:
    draw_architecture_diagram()
    if not TORCH_AVAILABLE:
        print("PyTorch no esta instalado: se conserva el diagrama y se omite la inferencia numerica.")
        return {}

    sample = torch.from_numpy(example_feature[None, None, :, :]).float()
    models = {
        "CNN 2D": MelCNN(len(CLASS_NAMES)),
        "Mini-AST": MiniAST(len(CLASS_NAMES)),
        "CRNN": MelCRNN(len(CLASS_NAMES)),
    }
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            logits = model(sample)
        print(f"{name}: entrada {tuple(sample.shape)} -> logits {tuple(logits.shape)}")
        if name == "CNN 2D" and TORCHINFO_AVAILABLE:
            print(torch_summary(model, input_size=tuple(sample.shape), verbose=0))
    if TRANSFORMERS_AVAILABLE and ASTFeatureExtractor is not None:
        print("Transformers disponible: ASTFeatureExtractor y ASTForAudioClassification pueden sustituir Mini-AST.")
        try:
            ast_config = ASTConfig(
                num_labels=len(CLASS_NAMES),
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=128,
            )
            print(f"Configuracion AST entrenable creada: hidden_size={ast_config.hidden_size}")
        except Exception as exc:
            print(f"No se pudo construir la configuracion AST: {exc}")
    else:
        print("Transformers no esta instalado; Mini-AST muestra la idea de patches y autoatencion sin descargar un checkpoint.")
    return models


def run_epoch(model: Any, loader: Any, loss_function: Any, optimizer: object | None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in loader:
        if training:
            optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_function(logits, batch_y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * batch_x.size(0)
        correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total += batch_x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, float, float, float, np.ndarray]:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for expected, predicted in zip(y_true, y_pred):
        matrix[int(expected), int(predicted)] += 1
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_index in range(n_classes):
        true_positive = matrix[class_index, class_index]
        false_positive = matrix[:, class_index].sum() - true_positive
        false_negative = matrix[class_index, :].sum() - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return accuracy, float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1_scores)), matrix


def evaluate_model_on_features(model: Any, features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    if len(labels) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matrix": np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int),
            "y_true": labels,
            "predictions": np.empty(0, dtype=np.int64),
            "probabilities": np.empty((0, len(CLASS_NAMES)), dtype=np.float32),
            "logits": np.empty((0, len(CLASS_NAMES)), dtype=np.float32),
        }
    model.eval()
    input_tensor = torch.from_numpy(features[:, None, :, :]).float()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = logits.argmax(dim=1).cpu().numpy()
    metrics = calculate_metrics(labels, predictions, len(CLASS_NAMES))
    return {
        "accuracy": metrics[0],
        "precision": metrics[1],
        "recall": metrics[2],
        "f1": metrics[3],
        "matrix": metrics[4],
        "y_true": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "logits": logits.cpu().numpy(),
    }


def train_models(models: dict[str, object], splits: dict[str, np.ndarray], epochs: int = 10) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, object]]]:
    if not TORCH_AVAILABLE or not models:
        return {}, {}
    x_train = torch.from_numpy(splits["train"][:, None, :, :]).float()
    y_train = torch.from_numpy(splits["y_train"]).long()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=min(16, len(x_train)), shuffle=True)
    histories: dict[str, dict[str, list[float]]] = {}
    evaluations: dict[str, dict[str, object]] = {}
    loss_function = nn.CrossEntropyLoss()
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = {"train_loss": [], "train_accuracy": []}
        best_state: dict[str, Any] | None = None
        best_loss = float("inf")
        best_epoch = 0
        for epoch in range(epochs):
            train_loss, train_accuracy = run_epoch(model, loader, loss_function, optimizer)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            print(
                f"{name} epoca {epoch + 1}/{epochs}: "
                f"loss={train_loss:.3f}, acc={train_accuracy:.2f}"
            )
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if best_state is not None:
            model.load_state_dict(best_state)
        evaluation = evaluate_model_on_features(
            model,
            splits["remaining"],
            splits["y_remaining"],
        )
        evaluation["best_epoch"] = best_epoch
        evaluation["best_train_loss"] = best_loss
        evaluations[name] = evaluation
        histories[name] = history
        print(
            f"{name}: checkpoint elegido en epoca {best_epoch} "
            f"(perdida de entrenamiento={best_loss:.3f}); "
            f"evaluacion restante acc={evaluation['accuracy']:.3f}"
        )
    return histories, evaluations


def save_training_and_metrics(histories: dict[str, dict[str, list[float]]], evaluations: dict[str, dict[str, object]]) -> None:
    if not histories:
        return
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, history in histories.items():
        axes[0].plot(history["train_loss"], marker="o", label=f"{name} train")
        axes[1].plot(history["train_accuracy"], marker="o", label=name)
    axes[0].set_title("Perdida durante el entrenamiento")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Exactitud durante el entrenamiento")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Exactitud")
    axes[1].legend(fontsize=8)
    save_figure(figure, "07_curvas_entrenamiento.png")

    figure, axes = plt.subplots(1, len(evaluations), figsize=(5 * len(evaluations), 4), squeeze=False)
    for axis, (name, result) in zip(axes[0], evaluations.items()):
        image = axis.imshow(result["matrix"], cmap="Blues", vmin=0)
        axis.set_title(f"{name}\neval acc={result['accuracy']:.2f}")
        axis.set_xlabel("Predicha")
        axis.set_ylabel("Real")
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30)
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        for row in range(len(CLASS_NAMES)):
            for column in range(len(CLASS_NAMES)):
                axis.text(column, row, result["matrix"][row, column], ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046)
    save_figure(figure, "08_matrices_confusion.png")
    print("\nComparacion de modelos")
    print(f"{'Modelo':<12} {'Accuracy':>9} {'Precision':>10} {'Recall':>9} {'F1':>9}")
    for name, result in evaluations.items():
        print(
            f"{name:<12} {result['accuracy']:>9.3f} {result['precision']:>10.3f} "
            f"{result['recall']:>9.3f} {result['f1']:>9.3f}"
        )
        print(
            f"  mejor epoca={result['best_epoch']} | "
            f"audios evaluados={len(result['y_true'])}"
        )


def extract_activation_and_prediction(models: dict[str, object], example_feature: np.ndarray, evaluations: dict[str, dict[str, object]]) -> None:
    if not TORCH_AVAILABLE or "CNN 2D" not in models:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.imshow(example_feature, origin="lower", aspect="auto", cmap="magma")
        axis.set_title("Mel-espectrograma usado como entrada")
        axis.set_xlabel("Tiempo")
        axis.set_ylabel("Frecuencia Mel")
        save_figure(figure, "09_activaciones_predicciones.png")
        return
    activations: dict[str, Any] = {}

    def capture_hook(_module: object, _inputs: tuple[object, ...], output: Any) -> None:
        activations["conv1"] = output.detach().cpu()

    handle = models["CNN 2D"].features[0].register_forward_hook(capture_hook)
    input_tensor = torch.from_numpy(example_feature[None, None, :, :]).float()
    with torch.no_grad():
        example_logits = models["CNN 2D"](input_tensor)
        probabilities = torch.softmax(example_logits, dim=1).numpy()[0]
    handle.remove()
    maps = activations.get("conv1")
    if maps is None:
        return
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, activation in zip(axes, maps[0, :4]):
        axis.imshow(activation, origin="lower", aspect="auto", cmap="viridis")
        axis.set_title("Mapa Conv2d")
        axis.set_xlabel("Tiempo")
        axis.set_ylabel("Frecuencia")
    save_figure(figure, "09_activaciones_conv2d.png")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(example_feature, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("Entrada Mel normalizada")
    axes[0].set_xlabel("Tiempo")
    axes[0].set_ylabel("Frecuencia")
    axes[1].bar(CLASS_NAMES, probabilities, color=["#287271", "#e76f51"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Softmax de la prediccion")
    axes[1].set_ylabel("Probabilidad")
    save_figure(figure, "10_prediccion_softmax.png")
    print("Probabilidades CNN 2D:", dict(zip(CLASS_NAMES, probabilities.round(4))))


def predict_audio_from_image(model: Any, audio_path: Path) -> dict[str, float]:
    feature = extract_mel_feature(audio_path)
    image_path = RESULTS_DIR / "inferencia" / f"{audio_path.stem}_mel.png"
    save_feature_image(feature, image_path)
    image_feature = load_feature_image(image_path)
    input_tensor = torch.from_numpy(image_feature[None, None, :, :]).float()
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_index = int(np.argmax(probabilities))
    result = dict(zip(CLASS_NAMES, probabilities.astype(float)))
    print(f"Audio clasificado desde imagen: {audio_path.name}")
    print(f"Imagen usada por la CNN: {image_path}")
    print(f"Prediccion: {CLASS_NAMES[predicted_index]} | probabilidades: {result}")
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def ctc_greedy_decode(logits: np.ndarray, blank_index: int = 0) -> list[int]:
    best_path = np.argmax(logits, axis=-1)
    decoded: list[int] = []
    previous = blank_index
    for token in best_path:
        token = int(token)
        if token != blank_index and token != previous:
            decoded.append(token)
        previous = token
    return decoded


def show_classification_and_ctc() -> None:
    vocabulary = ["<blank>", "a", "b", "c", "o"]
    target_path = [1, 1, 0, 2, 2, 0, 3, 4, 4, 0, 1]
    rng = np.random.default_rng(SEED)
    logits = rng.normal(-3.0, 0.35, size=(len(target_path), len(vocabulary)))
    for time_index, token in enumerate(target_path):
        logits[time_index, token] = 3.0
    probabilities = softmax(logits)
    decoded_ids = ctc_greedy_decode(logits, blank_index=0)
    decoded_text = "".join(vocabulary[index] for index in decoded_ids)
    print("Clasificacion discreta: Softmax convierte logits en probabilidades que suman", probabilities[0].sum())
    print("CTC greedy: ruta", target_path, "->", decoded_ids, "->", decoded_text)
    if TORCH_AVAILABLE:
        print("En entrenamiento ASR se puede usar nn.CTCLoss(blank=0) con logits [tiempo, lote, clases].")
    figure, axis = plt.subplots(figsize=(12, 4))
    image = axis.imshow(probabilities.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_title("CTC: probabilidades por instante y colapso de repetidos/blancos")
    axis.set_xlabel("Instante temporal")
    axis.set_ylabel("Token")
    axis.set_yticks(range(len(vocabulary)), vocabulary)
    figure.colorbar(image, ax=axis, label="Probabilidad")
    save_figure(figure, "11_clasificacion_softmax_ctc.png")


def save_weights(models: dict[str, object], histories: dict[str, dict[str, list[float]]]) -> None:
    if not TORCH_AVAILABLE:
        return
    best_name = None
    best_loss = float("inf")
    for name, model in models.items():
        filename = name.lower().replace(" ", "_").replace("-", "_") + "_best.pt"
        path = RESULTS_DIR / filename
        torch.save(model.state_dict(), path)
        print(f"Pesos guardados: {path}")
        model_loss = min(histories.get(name, {}).get("train_loss", [float("inf")]))
        if model_loss < best_loss:
            best_loss = model_loss
            best_name = name
    if best_name is not None:
        best_path = RESULTS_DIR / "best_model_by_training_loss.pt"
        torch.save(models[best_name].state_dict(), best_path)
        print(f"Mejor arquitectura por perdida de entrenamiento: {best_name}")
        print(f"Checkpoint principal guardado: {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explora audio, Mel-espectrogramas y modelos.")
    parser.add_argument("--max-per-class", type=int, default=120, help="Numero de WAV por clase; el valor predeterminado usa 240 audios.")
    parser.add_argument("--epochs", type=int, default=10, help="Epocas cortas para la demostracion.")
    parser.add_argument("--skip-training", action="store_true", help="No entrenar los modelos PyTorch.")
    parser.add_argument("--predict", type=Path, default=None, help="Audio adicional que se clasificara desde su imagen Mel.")
    args = parser.parse_args()

    audio_path = find_audio_file(SCRIPT_DIR)
    y_original, original_sr, source = load_audio(audio_path)
    print("=== 1. Audio original ===")
    show_original_audio(y_original, original_sr, source)

    print("=== 2. Normalizacion y resampling ===")
    normalized = normalize_peak(y_original)
    resampled = resample_audio(normalized, original_sr, TARGET_SR)
    show_resampling(y_original, original_sr, normalized, resampled)
    if sf is not None:
        sf.write(RESULTS_DIR / "audio_normalizado_16khz.wav", resampled, TARGET_SR)

    print("=== 3. Framing y windowing ===")
    show_framing(resampled)

    print("=== 4. STFT y Mel-espectrograma ===")
    _, _, _, power = show_stft(resampled)
    _, example_mel_db = show_mel_spectrogram(power)
    example_feature = np.clip((resize_time(example_mel_db, FIXED_FRAMES) + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)

    print("=== 5. Dataset y tensores ===")
    splits = prepare_dataset(SCRIPT_DIR, example_feature, args.max_per_class)
    all_features = np.concatenate(
        (splits["train"], splits["remaining"]),
        axis=0,
    )
    all_labels = np.concatenate(
        (splits["y_train"], splits["y_remaining"]),
        axis=0,
    )
    print(f"Dataset completo: X={all_features.shape}, y={all_labels.shape}")
    print(f"Tensor de entrada esperado por los modelos: [lote, canal, frecuencia, tiempo] = [N, 1, {N_MELS}, {FIXED_FRAMES}]")
    print(f"Imagenes Mel etiquetadas para la CNN: {IMAGE_DATASET_DIR}")
    print({name: value.shape for name, value in splits.items()})

    print("=== 6. CNN 2D, AST y CRNN ===")
    models = demonstrate_models(example_feature)
    histories: dict[str, dict[str, list[float]]] = {}
    evaluations: dict[str, dict[str, object]] = {}
    if models and not args.skip_training:
        histories, evaluations = train_models(models, splits, epochs=args.epochs)
        save_training_and_metrics(histories, evaluations)
    elif args.skip_training:
        print("Entrenamiento omitido por --skip-training.")
    extract_activation_and_prediction(models, example_feature, evaluations)
    prediction_path = args.predict if args.predict is not None else audio_path
    if models and "CNN 2D" in models and prediction_path is not None and prediction_path.exists():
        predict_audio_from_image(models["CNN 2D"], prediction_path)
    elif args.predict is not None:
        print(f"No existe el audio indicado para predecir: {args.predict}")
    save_weights(models, histories)

    print("=== 7. Clasificacion y decodificacion ===")
    show_classification_and_ctc()
    print(f"Proceso terminado. Revisa las figuras y pesos en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
