"""Extract caller_0 intervals from a WAV and save them as mono 16 kHz audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
CALLER_KEYS = ("speaker", "label", "channel", "source", "caller", "name", "id")
INTERVAL_COLLECTION_KEYS = ("turns", "segments", "intervals")


def _normalise_value(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _is_caller_0(turn: dict[str, Any]) -> bool:
    for key in CALLER_KEYS:
        if key not in turn:
            continue
        value = _normalise_value(turn[key])
        if value in {"caller_0", "caller0", "channel_0", "channel0", "0"}:
            return True
    return False


def _read_turns(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        turns = payload
    elif isinstance(payload, dict):
        turns = None
        for key in INTERVAL_COLLECTION_KEYS:
            collection = payload.get(key)
            if isinstance(collection, list):
                turns = collection
                break

        if turns is None:
            for key in ("caller_0", "caller0"):
                collection = payload.get(key)
                if isinstance(collection, list):
                    turns = [dict(item, speaker=key) for item in collection]
                    break

        if turns is None:
            raise ValueError(
                f"No se encontraron intervalos en {json_path}. "
                "Se esperaba 'turns', 'segments' o 'intervals'."
            )
    else:
        raise ValueError(f"El JSON debe contener una lista o un objeto: {json_path}")

    if not all(isinstance(turn, dict) for turn in turns):
        raise ValueError("Cada intervalo del JSON debe ser un objeto.")
    return turns


def _get_interval(turn: dict[str, Any], index: int) -> tuple[float, float]:
    start_value = turn.get("start", turn.get("start_s"))
    end_value = turn.get("end", turn.get("end_s"))

    if start_value is None or end_value is None:
        raise ValueError(f"El intervalo {index} no tiene 'start' y 'end'.")

    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"El intervalo {index} tiene tiempos inválidos.") from error

    if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"El intervalo {index} debe cumplir 0 <= start < end.")

    return start, end


def _load_mono_audio(wav_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)

    if audio.shape[1] == 0:
        raise ValueError(f"El WAV no contiene canales: {wav_path}")

    # En los audios del reto, el canal 0 es caller_0. Elegirlo evita mezclar al agente.
    mono_audio = audio[:, 0]
    return mono_audio, sample_rate


def normalizar_audio(
    wav_path: str | Path,
    json_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Create a mono 16 kHz WAV containing only the caller_0 intervals."""
    wav_path = Path(wav_path)
    json_path = Path(json_path)
    if output_path is None:
        output_path = wav_path.with_name(f"{wav_path.stem}_caller_0_16k.wav")
    output_path = Path(output_path)

    turns = _read_turns(json_path)
    caller_intervals = [
        _get_interval(turn, index)
        for index, turn in enumerate(turns)
        if _is_caller_0(turn)
    ]
    caller_intervals.sort(key=lambda interval: (interval[0], interval[1]))

    if not caller_intervals:
        raise ValueError("El JSON no contiene intervalos de caller_0.")

    mono_audio, source_sample_rate = _load_mono_audio(wav_path)
    if source_sample_rate <= 0:
        raise ValueError(f"Frecuencia de muestreo inválida: {source_sample_rate}")

    if source_sample_rate == TARGET_SAMPLE_RATE:
        audio_16k = mono_audio
    else:
        audio_16k = resample_poly(
            mono_audio,
            TARGET_SAMPLE_RATE,
            source_sample_rate,
        ).astype(np.float32, copy=False)

    pieces = []
    for start, end in caller_intervals:
        start_sample = max(0, int(round(start * TARGET_SAMPLE_RATE)))
        end_sample = min(len(audio_16k), int(round(end * TARGET_SAMPLE_RATE)))
        if start_sample < end_sample:
            pieces.append(audio_16k[start_sample:end_sample])

    if not pieces:
        raise ValueError("Los intervalos caller_0 están fuera de la duración del WAV.")

    output_audio = np.clip(np.concatenate(pieces), -1.0, 1.0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, output_audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae caller_0 de un WAV usando sus intervalos JSON."
    )
    parser.add_argument("wav", type=Path, help="Ruta al archivo WAV de entrada.")
    parser.add_argument(
        "json",
        type=Path,
        nargs="?",
        help="Ruta al JSON. Por defecto, usa el mismo nombre base que el WAV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Ruta del WAV de salida. Se genera una por defecto si se omite.",
    )
    args = parser.parse_args()

    json_path = args.json or args.wav.with_suffix(".json")
    output_path = normalizar_audio(args.wav, json_path, args.output)
    print(f"Audio guardado en: {output_path}")


if __name__ == "__main__":
    main()