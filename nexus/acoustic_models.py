from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor


WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MINI_AST_PATH = WORKSPACE_DIR / "resultados_audio" / "mini_ast_best.pt"
WAV2VEC_DIR = WORKSPACE_DIR / "wav2vec2_finetuned_model_run"
TARGET_SAMPLE_RATE = 16_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from procesamiento_audio_mel_modelos import (  # noqa: E402
    CLASS_NAMES,
    CLIP_SECONDS,
    FIXED_FRAMES,
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    TARGET_SR,
    MiniAST,
    crop_or_pad,
    hz_to_mel,
    mel_spectrogram_from_power,
    normalize_peak,
    resample_audio,
    resize_time,
    stft_power,
)


def resolve_training_audio(anon_id: str, label: str) -> Path:
    candidates = [
        WORKSPACE_DIR / "datos" / "channel_0" / label / f"{anon_id}__ch0.wav",
        WORKSPACE_DIR / "altur-challenge-audio" / "audio" / f"{anon_id}.wav",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No se encontro WAV para {anon_id}: {candidates}")


def resolve_audio_by_id(anon_id: str) -> Path:
    candidates = [
        WORKSPACE_DIR / "datos" / "channel_0" / label / f"{anon_id}__ch0.wav"
        for label in ("human", "synthetic")
    ]
    candidates.append(WORKSPACE_DIR / "altur-challenge-audio" / "audio" / f"{anon_id}.wav")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No se encontro un WAV local para {anon_id}")


def load_mini_ast_model(model_path: Path = MINI_AST_PATH) -> torch.nn.Module:
    model = MiniAST(n_classes=len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_wav2vec_models(
    model_dir: Path = WAV2VEC_DIR,
) -> tuple[Wav2Vec2Processor, Wav2Vec2ForSequenceClassification]:
    processor = Wav2Vec2Processor.from_pretrained(model_dir, local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to(DEVICE)
    model.eval()
    return processor, model


def _load_mono_16k(audio_path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    return _prepare_mono_16k(audio[:, 0], sample_rate)


def _prepare_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = resample_poly(mono, TARGET_SAMPLE_RATE, sample_rate)
    return np.nan_to_num(mono.astype(np.float32, copy=False))


def _mel_feature_from_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = normalize_peak(np.asarray(audio, dtype=np.float32))
    audio = resample_audio(audio, sample_rate, TARGET_SR)
    audio = crop_or_pad(audio, int(TARGET_SR * CLIP_SECONDS))
    _, _, _, power = stft_power(audio, TARGET_SR, n_fft=N_FFT)
    _, mel_db = mel_spectrogram_from_power(
        power,
        TARGET_SR,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    feature = resize_time(mel_db, FIXED_FRAMES)
    return np.clip((feature + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)


def _synthetic_index(labels: dict[int | str, str]) -> int:
    for index, label in labels.items():
        if str(label).lower() == "synthetic":
            return int(index)
    raise ValueError(f"El modelo no tiene una etiqueta synthetic: {labels}")


def predict_acoustic_audio(
    audio: np.ndarray,
    sample_rate: int,
    mini_ast_model: torch.nn.Module,
    wav2vec_processor: Wav2Vec2Processor,
    wav2vec_model: Wav2Vec2ForSequenceClassification,
) -> dict[str, float]:
    audio = _prepare_mono_16k(audio, sample_rate)
    mel_feature = _mel_feature_from_audio(audio, TARGET_SAMPLE_RATE)
    mini_device = next(mini_ast_model.parameters()).device
    mini_input = torch.from_numpy(mel_feature[None, None, :, :]).float().to(mini_device)
    with torch.inference_mode():
        mini_probabilities = torch.softmax(mini_ast_model(mini_input), dim=1)[0]
    mini_synthetic = float(mini_probabilities[CLASS_NAMES.index("synthetic")].item())

    wav2vec_inputs = wav2vec_processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )
    wav2vec_device = next(wav2vec_model.parameters()).device
    wav2vec_inputs = {
        key: value.to(wav2vec_device) if torch.is_tensor(value) else value
        for key, value in wav2vec_inputs.items()
    }
    with torch.inference_mode():
        wav2vec_probabilities = torch.softmax(
            wav2vec_model(**wav2vec_inputs).logits,
            dim=-1,
        )[0]
    synthetic_index = _synthetic_index(wav2vec_model.config.id2label)
    return {
        "mini_ast_probability": mini_synthetic,
        "wav2vec_probability": float(wav2vec_probabilities[synthetic_index].item()),
    }


def predict_acoustic_file(
    audio_path: Path,
    mini_ast_model: torch.nn.Module,
    wav2vec_processor: Wav2Vec2Processor,
    wav2vec_model: Wav2Vec2ForSequenceClassification,
) -> dict[str, float]:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    return predict_acoustic_audio(
        audio[:, 0],
        sample_rate,
        mini_ast_model,
        wav2vec_processor,
        wav2vec_model,
    )