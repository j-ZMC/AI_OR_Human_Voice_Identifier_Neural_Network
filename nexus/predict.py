import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from features import (
    ACOUSTIC_FEATURE_COLS,
    CALL_FEATURE_COLS,
    FULL_FEATURE_COLS,
    extract_features,
    extract_turn_features,
    load_prediction_inputs,
    load_turns,
)
from acoustic_models import (
    MINI_AST_PATH,
    WAV2VEC_DIR,
    load_mini_ast_model,
    load_wav2vec_models,
    predict_acoustic_audio,
    predict_acoustic_file,
    resolve_audio_by_id,
)
from torch_models import apply_normalizer, load_checkpoint

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest" / "model.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model" / "model.pt"
TIME_MODEL_PATH = MODELS_DIR / "time_predict" / "model.pt"
LEGACY_MODEL_PATH = MODELS_DIR / "legacy_model" / "model.joblib"
MODEL_PATH = BEST_MODEL_PATH


def load_model(model_path: str | Path | None = None):
    if model_path:
        path = Path(model_path)
    else:
        candidates = (
            RANDOM_FOREST_PATH,
            MODEL_PATH,
            LEGACY_MODEL_PATH,
            BASE_DIR / "best_model.pt",
            BASE_DIR / "model.joblib",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(f"No existe el modelo: {path}")
    if path.suffix.lower() == ".pt":
        return load_checkpoint(path)
    return joblib.load(path)


def _reaction_features_from_turns(turns):
    turn_rows = extract_turn_features(sorted(turns, key=lambda t: t["start"]))
    if not turn_rows:
        raise ValueError("La llamada no tiene ningun turno del caller (channel=0)")
    reaction_times = np.array([row["reaction_time"] for row in turn_rows])
    return {
        "reaction_mean": float(reaction_times.mean()),
        "reaction_std": float(reaction_times.std()),
        "reaction_min": float(reaction_times.min()),
        "reaction_max": float(reaction_times.max()),
        "reaction_median": float(np.median(reaction_times)),
    }, len(turn_rows)


def _features_from_turns(turns, feature_cols):
    reaction_features, n_turns = _reaction_features_from_turns(turns)
    full_features = extract_features(turns)
    all_features = reaction_features if set(feature_cols).issubset(CALL_FEATURE_COLS) else full_features
    missing = [column for column in feature_cols if column not in all_features]
    if missing:
        raise ValueError(f"Faltan características para el modelo: {', '.join(missing)}")
    return {column: all_features[column] for column in feature_cols}, n_turns


def _torch_prediction(features, model_bundle):
    feature_cols = model_bundle["feature_cols"]
    missing = [column for column in feature_cols if column not in features]
    if missing:
        raise ValueError(f"El CSV no tiene las columnas: {', '.join(missing)}")
    values = np.array(
        [[float(features[column]) for column in feature_cols]], dtype=np.float32
    )
    normalizer = model_bundle["normalizer"]
    normalized = apply_normalizer(
        values,
        normalizer["mean"],
        normalizer["scale"],
    )
    with torch.no_grad():
        logits = model_bundle["model"](torch.from_numpy(normalized))
        confidence = float(torch.sigmoid(logits).item())
    return confidence


def _load_random_forest_base_models(model_bundle):
    base_models = {}
    for model_key, relative_path in model_bundle["base_model_paths"].items():
        path = Path(relative_path)
        if not path.is_absolute():
            path = MODELS_DIR / path
        base_models[model_key] = load_model(path)
    return base_models


def _resolve_model_path(path_value: str | Path | None, default: Path) -> Path:
    path = default if path_value is None else Path(path_value)
    return path if path.is_absolute() else MODELS_DIR / path


@lru_cache(maxsize=4)
def _load_random_forest_acoustic_models(
    mini_ast_path: str,
    wav2vec_dir: str,
):
    mini_ast_model = load_mini_ast_model(Path(mini_ast_path))
    wav2vec_processor, wav2vec_model = load_wav2vec_models(Path(wav2vec_dir))
    return mini_ast_model, wav2vec_processor, wav2vec_model


def _acoustic_predictions_from_inputs(
    model_bundle,
    acoustic_audio: np.ndarray | None = None,
    acoustic_sample_rate: int = 16_000,
    acoustic_audio_path: Path | None = None,
) -> dict[str, float]:
    configured_paths = model_bundle.get("acoustic_model_paths", {})
    mini_ast_path = _resolve_model_path(
        configured_paths.get("mini_ast"), MINI_AST_PATH
    )
    wav2vec_dir = _resolve_model_path(
        configured_paths.get("wav2vec"), WAV2VEC_DIR
    )
    mini_ast_model, wav2vec_processor, wav2vec_model = _load_random_forest_acoustic_models(
        str(mini_ast_path),
        str(wav2vec_dir),
    )
    if acoustic_audio_path is not None:
        return predict_acoustic_file(
            acoustic_audio_path,
            mini_ast_model,
            wav2vec_processor,
            wav2vec_model,
        )
    if acoustic_audio is None:
        raise ValueError("Se necesita audio para ejecutar los modelos acusticos.")
    return predict_acoustic_audio(
        acoustic_audio,
        acoustic_sample_rate,
        mini_ast_model,
        wav2vec_processor,
        wav2vec_model,
    )


def _optional_audio_path(identifier: str, model_bundle) -> Path | None:
    if model_bundle.get("format") != "nexus-random-forest-v1":
        return None
    try:
        return resolve_audio_by_id(identifier)
    except FileNotFoundError:
        return None


def _random_forest_prediction_from_base_features(
    time_feature_map,
    substantive_feature_map,
    model_bundle,
    n_turns=0,
    acoustic_features: dict[str, float] | None = None,
    acoustic_audio: np.ndarray | None = None,
    acoustic_sample_rate: int = 16_000,
    acoustic_audio_path: Path | None = None,
):
    base_models = _load_random_forest_base_models(model_bundle)
    time_features = {
        column: time_feature_map[column] for column in CALL_FEATURE_COLS
    }
    substantive_features = {
        column: substantive_feature_map[column] for column in FULL_FEATURE_COLS
    }
    time_result = predict_from_features(
        time_features, model_bundle=base_models["time_predict"], n_turns=n_turns
    )
    substantive_result = predict_from_features(
        substantive_features,
        model_bundle=base_models["sustantive_predict"],
        n_turns=n_turns,
    )
    meta_features = {
        "time_predict_probability": time_result["confidence"],
        "sustantive_predict_probability": substantive_result["confidence"],
    }
    acoustic_features = dict(acoustic_features or {})
    required_acoustic = [
        column for column in ACOUSTIC_FEATURE_COLS
        if column in model_bundle["feature_cols"]
    ]
    missing_acoustic = [
        column for column in required_acoustic if column not in acoustic_features
    ]
    if missing_acoustic and (acoustic_audio is not None or acoustic_audio_path is not None):
        acoustic_features.update(
            _acoustic_predictions_from_inputs(
                model_bundle,
                acoustic_audio=acoustic_audio,
                acoustic_sample_rate=acoustic_sample_rate,
                acoustic_audio_path=acoustic_audio_path,
            )
        )
    for column in required_acoustic:
        meta_features[column] = float(acoustic_features.get(column, 0.5))
    meta_frame = pd.DataFrame([meta_features])[model_bundle["feature_cols"]]
    confidence = float(model_bundle["forest"].predict_proba(meta_frame)[0, 1])
    return {
        "label": "synthetic" if confidence > 0.5 else "human",
        "confidence": confidence,
        "n_turns": int(n_turns),
    }


def _random_forest_prediction(
    features,
    model_bundle,
    n_turns=0,
    acoustic_features: dict[str, float] | None = None,
):
    return _random_forest_prediction_from_base_features(
        {column: features[column] for column in CALL_FEATURE_COLS},
        {column: features[column] for column in FULL_FEATURE_COLS},
        model_bundle,
        n_turns=n_turns,
        acoustic_features=acoustic_features,
    )


def _random_forest_prediction_from_turns(
    turns,
    model_bundle,
    acoustic_audio: np.ndarray | None = None,
    acoustic_sample_rate: int = 16_000,
    acoustic_audio_path: Path | None = None,
):
    base_models = _load_random_forest_base_models(model_bundle)
    time_features, n_turns = _features_from_turns(
        turns, base_models["time_predict"]["feature_cols"]
    )
    substantive_features, _ = _features_from_turns(
        turns, base_models["sustantive_predict"]["feature_cols"]
    )
    return _random_forest_prediction_from_base_features(
        time_features,
        substantive_features,
        model_bundle,
        n_turns=n_turns,
        acoustic_audio=acoustic_audio,
        acoustic_sample_rate=acoustic_sample_rate,
        acoustic_audio_path=acoustic_audio_path,
    )


def predict_from_features(features, model_bundle=None, n_turns=0):
    if model_bundle is None:
        model_bundle = load_model()
    if model_bundle.get("format") == "nexus-random-forest-v1":
        acoustic_features = {
            column: float(features[column])
            for column in ACOUSTIC_FEATURE_COLS
            if column in features
        }
        return _random_forest_prediction(
            features,
            model_bundle,
            n_turns=n_turns,
            acoustic_features=acoustic_features,
        )
    if model_bundle.get("format") == "nexus-torch-v1":
        confidence = _torch_prediction(features, model_bundle)
    else:
        feature_cols = model_bundle["feature_cols"]
        missing = [column for column in feature_cols if column not in features]
        if missing:
            raise ValueError(f"El CSV no tiene las columnas: {', '.join(missing)}")
        frame = pd.DataFrame([[features[column] for column in feature_cols]], columns=feature_cols)
        confidence = float(model_bundle["pipeline"].predict_proba(frame)[0, 1])
    return {
        "label": "synthetic" if confidence > 0.5 else "human",
        "confidence": confidence,
        "n_turns": int(n_turns),
    }


def predict_from_turns(
    turns,
    model_bundle=None,
    acoustic_audio: np.ndarray | None = None,
    acoustic_sample_rate: int = 16_000,
    acoustic_audio_path: Path | None = None,
):
    """turns: lista de {"channel": 0|1, "start": float, "end": float}."""
    if model_bundle is None:
        model_bundle = load_model()

    if model_bundle.get("format") == "nexus-random-forest-v1":
        return _random_forest_prediction_from_turns(
            turns,
            model_bundle,
            acoustic_audio=acoustic_audio,
            acoustic_sample_rate=acoustic_sample_rate,
            acoustic_audio_path=acoustic_audio_path,
        )
    if model_bundle.get("format") == "nexus-torch-v1":
        features, n_turns = _features_from_turns(turns, model_bundle["feature_cols"])
        return predict_from_features(features, model_bundle=model_bundle, n_turns=n_turns)

    features, n_turns = _reaction_features_from_turns(turns)
    return predict_from_features(features, model_bundle=model_bundle, n_turns=n_turns)


def predict_call(turns_path, model_bundle=None):
    turns = load_turns(turns_path)
    if model_bundle is None:
        model_bundle = load_model()
    result = predict_from_turns(
        turns,
        model_bundle=model_bundle,
        acoustic_audio_path=_optional_audio_path(Path(turns_path).stem, model_bundle),
    )
    return {"anon_id": Path(turns_path).stem, **result}


def _model_for_features(features, model_bundle, model_path=None):
    if model_bundle.get("format") == "nexus-random-forest-v1":
        if set(FULL_FEATURE_COLS).issubset(features):
            return model_bundle
        if model_path is not None:
            raise ValueError("El bosque requiere las 24 características para este CSV.")
        return load_model(TIME_MODEL_PATH)
    if set(model_bundle["feature_cols"]).issubset(features):
        return model_bundle
    if model_path is not None:
        raise ValueError("El modelo indicado requiere columnas que no trae el CSV.")
    reaction_path = TIME_MODEL_PATH
    if reaction_path.exists():
        reaction_model = load_model(reaction_path)
        if set(reaction_model["feature_cols"]).issubset(features):
            return reaction_model
    raise ValueError("La entrada no contiene las características requeridas por el modelo.")


def predict_input(input_path, model_path=None):
    records = load_prediction_inputs(input_path)
    model_bundle = load_model(model_path)
    results = []
    for identifier, payload in records:
        if "turns" in payload:
            result = predict_from_turns(
                payload["turns"],
                model_bundle=model_bundle,
                acoustic_audio_path=_optional_audio_path(identifier, model_bundle),
            )
        else:
            selected_model = _model_for_features(
                payload["features"], model_bundle, model_path=model_path
            )
            result = predict_from_features(
                payload["features"], model_bundle=selected_model
            )
        results.append({"anon_id": identifier, **result})
    return results[0] if len(results) == 1 else results


def main():
    parser = argparse.ArgumentParser(description="Predecir desde un JSON o CSV de Nexus.")
    parser.add_argument("input_path", help="Ruta a un archivo .json o .csv")
    parser.add_argument("--model", type=Path, default=None, help="Checkpoint .pt opcional")
    args = parser.parse_args()

    result = predict_input(args.input_path, model_path=args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
