import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CALL_FEATURE_COLS = [
    "reaction_mean",
    "reaction_std",
    "reaction_min",
    "reaction_max",
    "reaction_median",
]

FULL_FEATURE_COLS = [
    "reaction_mean",
    "reaction_std",
    "reaction_min",
    "reaction_max",
    "reaction_median",
    "agent_reaction_mean",
    "agent_reaction_std",
    "agent_reaction_min",
    "agent_reaction_max",
    "agent_reaction_median",
    "caller_turn_dur_mean",
    "caller_turn_dur_std",
    "caller_turn_dur_min",
    "caller_turn_dur_max",
    "caller_turn_dur_median",
    "agent_turn_dur_mean",
    "agent_turn_dur_std",
    "agent_turn_dur_min",
    "agent_turn_dur_max",
    "agent_turn_dur_median",
    "n_turns_caller",
    "n_turns_agent",
    "overlap_count",
    "overlap_ratio",
]

ACOUSTIC_FEATURE_COLS = [
    "mini_ast_probability",
    "wav2vec_probability",
]

PREDICTION_FEATURE_COLS = [*FULL_FEATURE_COLS, *ACOUSTIC_FEATURE_COLS]


def _sort_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(turn) for turn in turns), key=lambda turn: turn["start"])


def load_turns(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        turns = data.get("turns")
    elif isinstance(data, list):
        turns = data
    else:
        turns = None
    if not isinstance(turns, list):
        raise ValueError("El JSON debe contener una lista de turnos en 'turns'.")
    return _sort_turns(turns)


def merge_turns(turns, gap_threshold=0.6):
    turns = sorted(turns, key=lambda t: t["start"])
    merged = []
    for t in turns:
        if merged and merged[-1]["channel"] == t["channel"] and \
           t["start"] - merged[-1]["end"] <= gap_threshold:
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
        else:
            merged.append(dict(t))
    return merged


def _stats(arr, name):
    arr = np.array(arr) if len(arr) else np.array([0.0])
    return {
        f"{name}_mean": float(arr.mean()),
        f"{name}_std": float(arr.std()),
        f"{name}_min": float(arr.min()),
        f"{name}_max": float(arr.max()),
        f"{name}_median": float(np.median(arr)),
    }


def extract_features(turns, gap_threshold=0.6):
    turns = merge_turns(turns, gap_threshold=gap_threshold)
    caller_turns = [t for t in turns if t["channel"] == 0]
    agent_turns = [t for t in turns if t["channel"] == 1]

    reaction_times = []
    agent_reaction_times = []
    overlaps = 0
    for i in range(1, len(turns)):
        prev, curr = turns[i - 1], turns[i]
        if prev["channel"] != curr["channel"]:
            gap = curr["start"] - prev["end"]
            if curr["channel"] == 0:
                reaction_times.append(gap)
            else:
                agent_reaction_times.append(gap)
            if gap < 0:
                overlaps += 1

    durations_caller = [t["end"] - t["start"] for t in caller_turns]
    durations_agent = [t["end"] - t["start"] for t in agent_turns]

    feats = {}
    feats.update(_stats(reaction_times, "reaction"))
    feats.update(_stats(agent_reaction_times, "agent_reaction"))
    feats.update(_stats(durations_caller, "caller_turn_dur"))
    feats.update(_stats(durations_agent, "agent_turn_dur"))
    feats["n_turns_caller"] = len(caller_turns)
    feats["n_turns_agent"] = len(agent_turns)
    feats["overlap_count"] = overlaps
    denom = max(1, len(reaction_times) + len(agent_reaction_times))
    feats["overlap_ratio"] = overlaps / denom
    return feats


def extract_turn_features(turns, gap_threshold=0.6):
    """Un feature-vector por CADA turno individual del caller, en vez de
    estadisticas agregadas de toda la llamada. Da ~14 muestras por llamada
    en lugar de 1, a costa de perder contexto de la llamada completa (por
    eso el split train/val debe seguir siendo por llamada, no por turno,
    o un mismo caller aparece en ambos lados)."""
    merged = merge_turns(turns, gap_threshold=gap_threshold)
    call_duration = merged[-1]["end"] if merged else 0.0

    rows = []
    for i, t in enumerate(merged):
        if t["channel"] != 0:
            continue
        if i == 0:
            reaction_time = t["start"]
            is_after_agent = 0
            prev_turn_end = None
        else:
            prev = merged[i - 1]
            reaction_time = t["start"] - prev["end"]
            is_after_agent = 1 if prev["channel"] == 1 else 0
            prev_turn_end = prev["end"]
        rows.append({
            "turn_number": i,
            "turn_start": t["start"],
            "turn_end": t["end"],
            "prev_turn_end": prev_turn_end,
            "reaction_time": reaction_time,
            "is_after_agent": is_after_agent,
            "overlap": 1 if reaction_time < 0 else 0,
            "turn_duration": t["end"] - t["start"],
            "call_progress": t["start"] / call_duration if call_duration > 0 else 0.0,
        })
    return rows


def _identifier_column(columns: pd.Index) -> str | None:
    for candidate in ("anon_id", "call_id", "id"):
        if candidate in columns:
            return candidate
    return None


def _group_frame(frame: pd.DataFrame, path: Path) -> list[tuple[str, pd.DataFrame]]:
    identifier_column = _identifier_column(frame.columns)
    if identifier_column is None:
        return [(path.stem, frame)]
    return [
        (str(identifier), group)
        for identifier, group in frame.groupby(
            identifier_column, sort=False, dropna=False
        )
    ]


def _turns_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"channel", "start", "end"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"El CSV de turnos no contiene las columnas: {missing}")
    turns = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        turns.append({
            "channel": int(values["channel"]),
            "start": float(values["start"]),
            "end": float(values["end"]),
        })
    return _sort_turns(turns)


def _feature_rows_from_frame(frame: pd.DataFrame, path: Path) -> list[tuple[str, dict[str, float]]]:
    identifier_column = _identifier_column(frame.columns)
    rows = []
    for index, row in frame.iterrows():
        identifier = path.stem if identifier_column is None else str(row[identifier_column])
        features = {
            column: float(row[column])
            for column in PREDICTION_FEATURE_COLS
            if column in frame.columns and pd.notna(row[column])
        }
        rows.append((identifier if identifier else f"{path.stem}_{index}", features))
    return rows


def load_prediction_inputs(path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """Carga una entrada de prediccion JSON o CSV en un formato comun."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict) and isinstance(data.get("turns"), list):
            identifier = str(data.get("anon_id") or data.get("call_id") or path.stem)
            return [(identifier, {"turns": _sort_turns(data["turns"])})]
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            if data and all({"channel", "start", "end"}.issubset(item) for item in data):
                return [(path.stem, {"turns": _sort_turns(data)})]
            if all("turns" in item for item in data):
                return [
                    (
                        str(item.get("anon_id") or item.get("call_id") or f"{path.stem}_{index}"),
                        {"turns": _sort_turns(item["turns"])},
                    )
                    for index, item in enumerate(data)
                ]
            if data and all(set(CALL_FEATURE_COLS).issubset(item) for item in data):
                return [
                    (
                        f"{path.stem}_{index}",
                        {"features": {column: float(item[column]) for column in CALL_FEATURE_COLS}},
                    )
                    for index, item in enumerate(data)
                ]
        raise ValueError("El JSON debe contener turnos o características de una llamada.")

    if path.suffix.lower() != ".csv":
        raise ValueError("La entrada debe ser un archivo .json o .csv.")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"El CSV no contiene filas: {path}")

    if {"channel", "start", "end"}.issubset(frame.columns):
        return [
            (identifier, {"turns": _turns_from_frame(group)})
            for identifier, group in _group_frame(frame, path)
        ]

    if set(CALL_FEATURE_COLS).issubset(frame.columns):
        return [
            (identifier, {"features": features})
            for identifier, features in _feature_rows_from_frame(frame, path)
        ]

    if "reaction_time" in frame.columns:
        records = []
        for identifier, group in _group_frame(frame, path):
            reaction_times = group["reaction_time"].dropna().astype(float).tolist()
            if not reaction_times:
                raise ValueError(f"No hay reaction_time valido para {identifier}.")
            records.append((identifier, {"features": _stats(reaction_times, "reaction")}))
        return records

    raise ValueError(
        "El CSV debe tener channel/start/end, reaction_time, "
        "o las columnas de características agregadas."
    )
